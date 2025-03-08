# worker.py
import logging
import threading
import asyncio
import time
from typing import Dict, List, Any, Optional
import traceback

import aiohttp
from bungio.models import DestinyPostGameCarnageReportData, BungieMembershipType, DestinyUser

from bungio_client import BungioClient
from graph_manager import GraphManager
from bayesian_detector import BayesianDetector

logger = logging.getLogger("cheat_detector")


class Worker(threading.Thread):
    def __init__(self, worker_id: int, api_client: BungioClient, graph_manager: GraphManager,
                 bayesian_detector: BayesianDetector, webhook_url: str, min_confidence: float = 0.9):
        super().__init__(name=f"Worker-{worker_id}")
        self.worker_id = worker_id
        self.api_client = api_client  # Original client for reference
        self.graph_manager = graph_manager
        self.bayesian_detector = bayesian_detector
        self.webhook_url = webhook_url
        self.min_confidence = min_confidence

        self.daemon = True
        self.running = True
        self.loop = None
        self.worker_client = None  # Worker-specific API client

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
            # Create a new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # Initialize worker's own HTTP client
            self.loop.run_until_complete(self._init_worker_client())

            # Run the main worker loop
            self.loop.run_until_complete(self.worker_loop())
        except Exception as e:
            logger.error(f"Worker {self.worker_id} crashed: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            if self.loop:
                self.loop.close()

    async def _init_worker_client(self):
        """Initialize worker-specific resources"""
        # Clone necessary settings from main client
        self.worker_client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        logger.info(f"Worker {self.worker_id} initialized with dedicated HTTP client")

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
                    # Check for cheaters after every few matches
                    if self.stats["matches_processed"] % 5 == 0:
                        await self.detect_and_report_cheaters()

                    # Cleanup attention window occasionally
                    if self.stats["matches_processed"] % 10 == 0:
                        self.graph_manager.cleanup_attention_window()

                # Prevent CPU hogging
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)

    async def process_match(self, match_id: str) -> bool:
        """Process a single match"""
        try:
            self.stats["last_activity"] = time.time()

            # Get PGCR data using direct API call with worker's session
            pgcr_raw = await self.api_client.get_pgcr_direct(match_id, self.worker_client)

            if not pgcr_raw:
                logger.warning(f"No PGCR data found for match {match_id}")
                self.stats["api_errors"] += 1
                return False

            # Convert raw PGCR to the expected format
            pgcr_data = self._convert_pgcr(pgcr_raw, match_id)

            # Extract player metrics
            player_metrics = await self.api_client.extract_player_metrics(pgcr_data)

            if not player_metrics:
                logger.warning(f"No player metrics found for match {match_id}")
                return False

            # Update graph
            self.graph_manager.add_match_data(match_id, player_metrics)

            # Check for high win rates in the most suspicious players
            await self.analyze_win_rates(player_metrics)

            # Find and queue connected matches
            await self.queue_connected_matches(match_id, player_metrics)

            self.stats["matches_processed"] += 1
            logger.debug(f"Worker {self.worker_id} processed match {match_id} with {len(player_metrics)} players")

            return True

        except Exception as e:
            logger.error(f"Error processing match {match_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _convert_pgcr(self, pgcr_raw: Dict, activity_id: str) -> DestinyPostGameCarnageReportData:
        """Convert raw PGCR data to the expected format"""
        pgcr = DestinyPostGameCarnageReportData(
            activity_details=pgcr_raw.get("activityDetails", {}),
            activity_was_started_from_beginning=pgcr_raw.get("activityWasStartedFromBeginning", True),
            entries=pgcr_raw.get("entries", []),
            period=pgcr_raw.get("period", ""),
            starting_phase_index=pgcr_raw.get("startingPhaseIndex", 0),
            teams=pgcr_raw.get("teams", [])
        )

        # Set additional properties that might not be required by the constructor
        pgcr.activity_duration_seconds = pgcr_raw.get("activityDurationSeconds", 0)
        return pgcr

    async def queue_connected_matches(self, current_match_id: str, player_metrics: Dict[str, Dict[str, Any]]) -> None:
        """Find and queue matches connected to players in the current match using branching search"""
        try:
            # Sort players by suspicion
            players_with_suspicion = []
            for player_id in player_metrics:
                if "#" in player_id:
                    suspicion = self.graph_manager.player_suspicion.get(player_id, 0.0)
                    players_with_suspicion.append((player_id, suspicion))

            # Sort by suspicion (highest first)
            players_with_suspicion.sort(key=lambda x: x[1], reverse=True)

            # Determine branching factor based on max suspicion
            max_suspicion = players_with_suspicion[0][1] if players_with_suspicion else 0
            branching_factor = 2  # Default low branching

            if max_suspicion > 0.5:
                branching_factor = 4  # Increase for suspicious players
            if max_suspicion > 0.8:
                branching_factor = 6  # High branching for likely cheaters

            # Only process top N players based on branching factor
            for player_id, suspicion in players_with_suspicion[:branching_factor]:
                try:
                    # Extract membership info
                    display_name, membership_id = player_id.split("#", 1)

                    # Try different membership types
                    for membership_type in [3, 2, 1]:  # Steam, PSN, Xbox
                        try:
                            # Direct API call for recent activities
                            headers = {"X-API-Key": self.api_client.api_key}

                            # First get character IDs
                            profile_url = f"https://www.bungie.net/Platform/Destiny2/{membership_type}/Profile/{membership_id}/?components=Characters"

                            async with self.worker_client.get(profile_url, headers=headers, timeout=10) as response:
                                if response.status != 200:
                                    continue

                                profile_data = await response.json()

                                if profile_data.get("ErrorCode", 0) != 1:
                                    continue

                                characters_data = profile_data.get("Response", {}).get("characters", {}).get("data", {})

                                if not characters_data:
                                    continue

                                # Get activity count based on suspicion
                                activity_count = int(5 + suspicion * 15)  # 5-20 activities
                                matches_found = 0

                                # Check each character
                                for character_id in characters_data:
                                    # For each relevant mode
                                    for mode in [84, 69, 80]:  # trials, competitive, elimination
                                        try:
                                            history_url = f"https://www.bungie.net/Platform/Destiny2/{membership_type}/Account/{membership_id}/Character/{character_id}/Stats/Activities/"
                                            params = {
                                                "count": activity_count,
                                                "mode": mode,
                                                "page": 0
                                            }

                                            async with self.worker_client.get(history_url, headers=headers,
                                                                              params=params,
                                                                              timeout=10) as history_response:
                                                if history_response.status != 200:
                                                    continue

                                                history_data = await history_response.json()

                                                if history_data.get("ErrorCode", 0) != 1:
                                                    continue

                                                activities = history_data.get("Response", {}).get("activities", [])

                                                for activity in activities:
                                                    instance_id = activity.get("activityDetails", {}).get("instanceId")

                                                    if instance_id and str(instance_id) != str(current_match_id):
                                                        # Calculate priority - higher suspicion = higher priority
                                                        priority = -suspicion  # Negative for priority queue

                                                        # Prioritize by mode (trials first)
                                                        activity_mode = activity.get("activityDetails", {}).get("mode",
                                                                                                                0)
                                                        if activity_mode == 84:  # Trials
                                                            priority -= 0.3
                                                        elif activity_mode == 69:  # Competitive
                                                            priority -= 0.1

                                                        self.graph_manager.queue_match_for_analysis(str(instance_id),
                                                                                                    priority)
                                                        matches_found += 1

                                                        if matches_found >= activity_count:
                                                            break

                                            if matches_found >= activity_count:
                                                break

                                        except Exception as e:
                                            logger.debug(
                                                f"Error getting activities for character {character_id}, mode {mode}: {str(e)}")

                                    if matches_found >= activity_count:
                                        break

                                if matches_found > 0:
                                    logger.debug(
                                        f"Queued {matches_found} matches from {player_id} (suspicion: {suspicion:.2f})")
                                    break  # Found valid profile, stop checking other platforms

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
                if (score >= self.min_confidence and
                        player not in self.graph_manager.reported_cheaters):

                    # Get evidence
                    evidence_matches, flags = self.graph_manager.get_player_evidence(player)

                    # Add win rate flags if available
                    for metric_name, values in self.graph_manager.player_metrics.get(player, {}).items():
                        if "_win_rate" in metric_name and values:
                            mode = metric_name.replace("_win_rate", "")
                            flags[metric_name] = f"{values[0]:.1%}"

                    # Send Discord alert
                    if evidence_matches:
                        await self.api_client.send_discord_alert(
                            self.webhook_url,
                            player,
                            evidence_matches[:5],
                            score,
                            flags
                        )

                        # Mark as reported
                        with self.graph_manager.graph_lock:
                            self.graph_manager.reported_cheaters.add(player)

                        self.stats["cheaters_detected"] += 1
                        logger.info(f"Worker {self.worker_id} detected cheater: {player} with score {score:.3f}")

        except Exception as e:
            logger.error(f"Error in cheater detection: {str(e)}")

    def stop(self):
        """Signal worker to stop"""
        self.running = False

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        stats = self.stats.copy()
        stats["runtime_seconds"] = time.time() - stats["start_time"]
        stats["idle_seconds"] = time.time() - stats["last_activity"]
        return stats