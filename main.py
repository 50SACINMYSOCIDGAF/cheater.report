# main.py
import asyncio
import logging
import argparse
import sys
import yaml
import time
import json
import os
from typing import List, Dict, Any, Optional
import signal
import threading

from bungio.models import DestinyUser

from bungio_client import BungioClient
from data_processor import DataProcessor
from queue_mananger import QueueManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cheat_detector.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("cheat_detector")


class CheatDetectorPipeline:
    def __init__(self, config_path: str = "config.yml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.api_client = BungioClient(config_path)
        self.data_processor = DataProcessor(config_path)
        self.queue_manager = QueueManager(config_path)

        # Settings
        self.worker_count = self.config["processing"].get("worker_threads", 4)
        self.known_cheaters = self.config.get("known_cheaters", [])
        self.data_output_dir = self.config["processing"].get("data_output_dir", "data")

        # Runtime state
        self.running = False
        self.shutdown_event = threading.Event()
        self.stats = {
            "start_time": time.time(),
            "games_processed": 0,
            "games_queued": 0,
            "api_errors": 0,
            "last_update": time.time()
        }

        # Create output directory if it doesn't exist
        os.makedirs(self.data_output_dir, exist_ok=True)

    async def initialize(self):
        """Initialize the pipeline by finding seed matches"""
        # Initialize API client
        logger.info("Initializing Bungie API client...")
        if not await self.api_client.initialize():
            logger.error("Failed to initialize API client!")
            return False

        # Find seed matches from known cheaters
        logger.info(f"Finding seed matches from known cheaters...")
        seed_matches = await self.api_client.get_seed_matches_from_cheaters(self.known_cheaters)

        # Queue seed matches
        logger.info(f"Queueing {len(seed_matches)} seed matches...")
        count = self.queue_manager.queue_games(seed_matches, priority=1.0)
        logger.info(f"Successfully queued {count} seed matches")

        return count > 0

    async def process_game(self, game_id: str) -> bool:
        """Process a single game, extract data, and save to disk"""
        try:
            logger.info(f"Processing game {game_id}...")

            # Get PGCR data
            pgcr_data = await self.api_client.get_pgcr(game_id)

            if not pgcr_data:
                logger.warning(f"No PGCR data found for game {game_id}")
                self.stats["api_errors"] += 1
                return False

            # Extract player metrics
            player_metrics = self.data_processor.extract_player_metrics(pgcr_data)

            if not player_metrics:
                logger.warning(f"No player metrics found for game {game_id}")
                return False

            # Save data to disk
            self.save_game_data(game_id, player_metrics)

            # Queue connected games
            await self.queue_connected_games(game_id, player_metrics)

            # Mark as processed
            self.queue_manager.mark_game_processed(game_id)

            # Update stats
            self.stats["games_processed"] += 1

            logger.info(f"Successfully processed game {game_id} with {len(player_metrics)} players")
            return True

        except Exception as e:
            logger.error(f"Error processing game {game_id}: {str(e)}")
            self.queue_manager.mark_game_failed(game_id)
            return False

    def save_game_data(self, game_id: str, player_metrics: Dict[str, Dict[str, Any]]) -> bool:
        """Save game data to disk"""
        try:
            # Create output file path
            file_path = os.path.join(self.data_output_dir, f"game_{game_id}.json")

            # Create data structure
            game_data = {
                "game_id": game_id,
                "timestamp": time.time(),
                "player_metrics": player_metrics
            }

            # Write to file
            with open(file_path, "w") as f:
                json.dump(game_data, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Error saving game data for {game_id}: {str(e)}")
            return False

    async def queue_connected_games(self, game_id: str, player_metrics: Dict[str, Dict[str, Any]]) -> int:
        """Find and queue games connected to players in the current game"""
        queued_count = 0

        try:
            # For each player, get their recent games
            for player_id in player_metrics:
                # Skip if invalid player ID format
                if "#" not in player_id:
                    continue

                try:
                    # Parse player ID from format: "DisplayName#MembershipID"
                    display_name, membership_id_str = player_id.split("#", 1)

                    # Check if the part after # is a numeric membership ID (not a Bungie name code)
                    if membership_id_str.isdigit() and len(membership_id_str) > 10:  # Membership IDs are long numbers
                        # Use get_player_by_membership_id instead of trying each platform manually
                        # This will search for the correct platform based on membership ID
                        user, membership_type = await self.api_client.get_player_by_membership_id(membership_id_str)

                        if user and membership_type:
                            logger.info(
                                f"Found correct platform {membership_type.name} for player {display_name} with ID {membership_id_str}")

                            # Get activities using the correct platform
                            activities = await self.api_client.get_player_recent_activities(
                                user=user,
                                modes=list(self.api_client.game_mode_types.values()),
                                count=10
                            )

                            if activities:
                                # Queue activities for processing
                                new_games = []
                                for activity in activities:
                                    instance_id = activity.get("activityDetails", {}).get("instanceId")
                                    if instance_id and str(instance_id) != str(game_id):
                                        new_games.append(str(instance_id))

                                # Queue new games
                                if new_games:
                                    count = self.queue_manager.queue_games(new_games, priority=0.5)
                                    queued_count += count
                                    self.stats["games_queued"] += count
                        else:
                            logger.warning(
                                f"Could not find valid platform for player {display_name} with ID {membership_id_str}")

                    else:
                        # This is likely a regular Bungie name format (DisplayName#Code)
                        bungie_name = player_id.split("#")[0]  # Just use the display name part

                        # Get player using standard search
                        user, membership_type = await self.api_client.search_player_by_name(bungie_name)

                        if user and membership_type:
                            # Get activities for found player
                            activities = await self.api_client.get_player_recent_activities(
                                user=user,
                                modes=list(self.api_client.game_mode_types.values()),
                                count=10
                            )

                            if activities:
                                # Queue activities for processing
                                new_games = []
                                for activity in activities:
                                    instance_id = activity.get("activityDetails", {}).get("instanceId")
                                    if instance_id and str(instance_id) != str(game_id):
                                        new_games.append(str(instance_id))

                                # Queue new games
                                if new_games:
                                    count = self.queue_manager.queue_games(new_games, priority=0.5)
                                    queued_count += count
                                    self.stats["games_queued"] += count
                        else:
                            logger.warning(f"Could not find player by Bungie name: {bungie_name}")

                except Exception as e:
                    logger.warning(f"Error processing connected games for {player_id}: {str(e)}")

        except Exception as e:
            logger.error(f"Error in queue_connected_games: {str(e)}")

        return queued_count



    async def worker(self, worker_id: int) -> None:
        """Worker coroutine for processing games"""
        logger.info(f"Worker {worker_id} started")

        while self.running and not self.shutdown_event.is_set():
            try:
                # Get next game to process
                game_id, _ = self.queue_manager.get_next_game()

                if not game_id:
                    # No games in queue, wait a bit
                    await asyncio.sleep(1)
                    continue

                # Process game
                await self.process_game(game_id)

                # Prevent CPU hogging
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                await asyncio.sleep(1)

        logger.info(f"Worker {worker_id} stopped")

    async def stats_reporter(self) -> None:
        """Periodically report statistics"""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Calculate runtime
                runtime = time.time() - self.stats["start_time"]
                hours = int(runtime // 3600)
                minutes = int((runtime % 3600) // 60)
                seconds = int(runtime % 60)

                # Log statistics
                logger.info(f"=== Stats after {hours:02d}:{minutes:02d}:{seconds:02d} ===")
                logger.info(f"Games processed: {self.stats['games_processed']}")
                logger.info(f"Games queued: {self.stats['games_queued']}")
                logger.info(f"Queue size: {self.queue_manager.queue_size()}")
                logger.info(f"API errors: {self.stats['api_errors']}")

                if runtime > 0:
                    rate = self.stats['games_processed'] / runtime
                    logger.info(f"Processing rate: {rate:.2f} games/second")

                # Wait for next report
                for _ in range(30):  # Report every 5 minutes, check shutdown every 10 seconds
                    if self.shutdown_event.is_set():
                        break
                    await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error in stats reporter: {str(e)}")
                await asyncio.sleep(60)

    def setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown"""

        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.running = False
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def run(self) -> None:
        """Main pipeline entry point"""
        logger.info("Starting Destiny 2 Cheat Detector Pipeline")

        self.setup_signal_handlers()
        self.running = True

        try:
            # Initialize
            if not await self.initialize():
                logger.error("Initialization failed!")
                return

            # Start workers
            workers = []
            for i in range(self.worker_count):
                worker_task = asyncio.create_task(self.worker(i))
                workers.append(worker_task)

            # Start stats reporter
            stats_task = asyncio.create_task(self.stats_reporter())

            # Wait for shutdown signal
            while self.running and not self.shutdown_event.is_set():
                await asyncio.sleep(1)

            logger.info("Shutting down...")

            # Cancel all tasks
            for worker_task in workers:
                worker_task.cancel()

            stats_task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(*workers, stats_task, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error in main pipeline: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        finally:
            self.running = False
            logger.info("Cheat detector pipeline shutdown complete")


async def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Destiny 2 Cheat Detector")
    parser.add_argument("--config", default="config.yml", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set log level
    if args.debug:
        logging.getLogger("cheat_detector").setLevel(logging.DEBUG)

    # Create and run pipeline
    pipeline = CheatDetectorPipeline(args.config)
    await pipeline.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPipeline stopped by user")