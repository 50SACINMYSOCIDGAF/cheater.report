# worker.py
import logging
import threading
import asyncio
import time
from typing import Dict, List, Any, Optional
import traceback

from bungio_client import BungioClient
from graph_manager import GraphManager
from bayesian_detector import BayesianDetector

logger = logging.getLogger("cheat_detector")


class Worker(threading.Thread):
    def __init__(self, worker_id: int, api_client: BungioClient, graph_manager: GraphManager,
                 bayesian_detector: BayesianDetector, webhook_url: str, min_confidence: float = 0.9):
        super().__init__(name=f"Worker-{worker_id}")
        self.worker_id = worker_id
        self.api_client = api_client
        self.graph_manager = graph_manager
        self.bayesian_detector = bayesian_detector
        self.webhook_url = webhook_url
        self.min_confidence = min_confidence

        self.daemon = True
        self.running = True
        self.loop = None

        self.stats = {
            "matches_processed": 0,
            "cheaters_detected": 0,
            "api_errors": 0,
            "start_time": time.time(),
            "last_activity": time.time()
        }

    def run(self):
        """Thread main function"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.worker_loop())
        except Exception as e:
            logger.error(f"Worker {self.worker_id} crashed: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            if self.loop:
                self.loop.close()

    async def worker_loop(self):
        """Main worker loop for processing matches"""
        logger.info(f"Worker {self.worker_id} started")

        while self.running:
            try:
                # Get next match to process
                match_id = self.graph_manager.get_next_queued_match(timeout=0.1)

                if not match_id:
                    # If no matches in queue, get matches from frontier
                    next_matches = self.graph_manager.get_next_matches(1)
                    if next_matches:
                        match_id = next_matches[0]
                    else:
                        # If frontier is empty, look for connected matches
                        connected = self.graph_manager.get_connected_matches(suspicion_threshold=0.3)
                        if connected:
                            match_id = connected[0]
                        else:
                            # Nothing to do, wait a bit
                            await asyncio.sleep(0.5)
                            continue

                # Process match
                success = await self.process_match(match_id)

                if success:
                    # Occasionally check for cheaters
                    if self.stats["matches_processed"] % 5 == 0:
                        await self.detect_and_report_cheaters()

                    # Cleanup attention window occasionally
                    if self.stats["matches_processed"] % 10 == 0:
                        self.graph_manager.cleanup_attention_window()

                # Prevent CPU hogging when processing lots of cached data
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)

    async def process_match(self, match_id: str) -> bool:
        """Process a single match"""
        try:
            self.stats["last_activity"] = time.time()

            # Get PGCR data
            pgcr_data = await self.api_client.get_pgcr(match_id)

            if not pgcr_data:
                logger.warning(f"No PGCR data found for match {match_id}")
                self.stats["api_errors"] += 1
                return False

            # Extract player metrics
            player_metrics = await self.api_client.extract_player_metrics(pgcr_data)

            if not player_metrics:
                logger.warning(f"No player metrics found for match {match_id}")
                return False

            # Update graph
            self.graph_manager.add_match_data(match_id, player_metrics)

            # Find and queue connected matches
            await self.queue_connected_matches(match_id, player_metrics)

            self.stats["matches_processed"] += 1
            logger.debug(f"Worker {self.worker_id} processed match {match_id} with {len(player_metrics)} players")

            return True

        except Exception as e:
            logger.error(f"Error processing match {match_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    async def queue_connected_matches(self, current_match_id: str, player_metrics: Dict[str, Dict[str, Any]]) -> None:
        """Find and queue matches connected to players in the current match"""
        try:
            # First get player suspicion scores to prioritize
            max_player_suspicion = 0.0
            for player in player_metrics:
                suspicion = self.graph_manager.player_suspicion.get(player, 0.0)
                max_player_suspicion = max(max_player_suspicion, suspicion)

            # For each player, look through their recent history
            for player_id in player_metrics:
                # Skip if player not suspicious enough
                player_suspicion = self.graph_manager.player_suspicion.get(player_id, 0.0)
                if player_suspicion < 0.1 and max_player_suspicion > 0.4:
                    continue

                if "#" not in player_id:
                    continue

                display_name, membership_id = player_id.split("#", 1)
                try:
                    # Create BungIO user
                    from bungio.models import DestinyUser, BungieMembershipType

                    # Try different membership types
                    for membership_type in [
                        BungieMembershipType.TIGER_STEAM,
                        BungieMembershipType.TIGER_PSN,
                        BungieMembershipType.TIGER_XBOX
                    ]:
                        try:
                            user = DestinyUser(
                                membership_id=membership_id,
                                membership_type=membership_type
                            )

                            # Get activities for all PvP modes
                            activities = await self.api_client.get_player_recent_activities(
                                user=user,
                                modes=list(self.api_client.game_mode_types.values()),
                                count=10
                            )

                            if activities:
                                # Queue activities for exploration
                                for activity in activities:
                                    instance_id = activity.get("activityDetails", {}).get("instanceId")
                                    if instance_id and instance_id != current_match_id:
                                        # Higher suspicion = higher priority (negative for priority queue)
                                        priority = -player_suspicion
                                        self.graph_manager.queue_match_for_analysis(instance_id, priority)

                                # Found activities for this user, move on
                                break

                        except Exception as e:
                            logger.debug(
                                f"Error getting activities for {player_id} with type {membership_type}: {str(e)}")

                except Exception as e:
                    logger.warning(f"Error processing connected matches for {player_id}: {str(e)}")

        except Exception as e:
            logger.error(f"Error in queue_connected_matches: {str(e)}")

    async def detect_and_report_cheaters(self) -> None:
        """Detect cheaters and report to Discord"""
        try:
            # Update suspicion scores
            confirmed_cheaters = self.graph_manager.update_suspicion_scores()

            # Report new cheaters
            for player, score in confirmed_cheaters:
                if score >= self.min_confidence:
                    # Get evidence
                    evidence_matches, flags = self.graph_manager.get_player_evidence(player)

                    # Send Discord alert
                    await self.api_client.send_discord_alert(
                        self.webhook_url,
                        player,
                        evidence_matches[:5],
                        score,
                        flags
                    )

                    self.stats["cheaters_detected"] += 1
                    logger.info(f"Worker {self.worker_id} detected cheater: {player} with score {score:.3f}")

        except Exception as e:
            logger.error(f"Error in cheater detection: {str(e)}")
            logger.error(traceback.format_exc())

    def stop(self):
        """Signal worker to stop"""
        self.running = False

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        stats = self.stats.copy()
        stats["runtime_seconds"] = time.time() - stats["start_time"]
        stats["idle_seconds"] = time.time() - stats["last_activity"]
        return stats