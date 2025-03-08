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

from bungio_client import BungioClient
from bayesian_detector import BayesianDetector
from graph_manager import GraphManager
from worker import Worker

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


class CheatDetector:
    def __init__(self, config_path: str = "config.yml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.api_client = BungioClient(config_path)
        self.bayesian_detector = BayesianDetector(config_path)
        self.graph_manager = GraphManager(config_path, self.bayesian_detector)

        # Settings
        self.webhook_url = self.config["discord_webhook"]["url"]
        self.min_confidence = self.config["discord_webhook"]["min_confidence_to_report"]
        self.worker_count = self.config["processing"]["worker_threads"]
        self.known_cheaters = self.config.get("known_cheaters", [])

        # Worker threads
        self.workers = []

        # Runtime state
        self.running = False
        self.shutdown_event = threading.Event()
        self.global_stats = {
            "start_time": time.time(),
            "matches_processed": 0,
            "cheaters_detected": 0,
            "api_errors": 0,
            "last_update": time.time()
        }

    async def initialize(self):
        """Initialize the detector by loading manifest and finding seed matches"""
        # Initialize API client (load manifest)
        await self.api_client.initialize()

        # Find seed matches from known cheaters
        seed_matches = await self.api_client.get_seed_matches_from_cheaters(self.known_cheaters)

        # Add seed matches to frontier
        self.graph_manager.update_frontier(seed_matches)

        # Queue seed matches with high priority
        for match_id in seed_matches:
            self.graph_manager.queue_match_for_analysis(match_id, priority=-0.9)

        logger.info(f"Initialized with {len(seed_matches)} seed matches")

    def start_workers(self):
        """Start worker threads"""
        for i in range(self.worker_count):
            worker = Worker(
                i,
                self.api_client,
                self.graph_manager,
                self.bayesian_detector,
                self.webhook_url,
                self.min_confidence
            )
            worker.start()
            self.workers.append(worker)
            logger.info(f"Started worker {i}")

    def stop_workers(self):
        """Stop all worker threads"""
        for worker in self.workers:
            worker.stop()

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)

        logger.info("All workers stopped")

    def update_global_stats(self):
        """Update global statistics from workers"""
        self.global_stats["matches_processed"] = sum(w.stats["matches_processed"] for w in self.workers)
        self.global_stats["cheaters_detected"] = sum(w.stats["cheaters_detected"] for w in self.workers)
        self.global_stats["api_errors"] = sum(w.stats["api_errors"] for w in self.workers)
        self.global_stats["last_update"] = time.time()
        self.global_stats["runtime_seconds"] = time.time() - self.global_stats["start_time"]

        # Calculate processing rate
        runtime = self.global_stats["runtime_seconds"]
        if runtime > 0:
            self.global_stats["matches_per_second"] = self.global_stats["matches_processed"] / runtime
        else:
            self.global_stats["matches_per_second"] = 0

    def print_stats(self):
        """Print runtime statistics"""
        self.update_global_stats()

        runtime = self.global_stats["runtime_seconds"]
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        seconds = int(runtime % 60)

        logger.info(f"=== Stats after {hours:02d}:{minutes:02d}:{seconds:02d} ===")
        logger.info(f"Matches processed: {self.global_stats['matches_processed']}")
        logger.info(f"Cheaters detected: {self.global_stats['cheaters_detected']}")
        logger.info(f"API errors: {self.global_stats['api_errors']}")
        logger.info(f"Processing rate: {self.global_stats['matches_per_second']:.2f} matches/second")

        # Print graph statistics
        self.graph_manager.print_statistics()

        # Print worker details
        active_workers = sum(1 for w in self.workers if time.time() - w.stats["last_activity"] < 60)
        logger.info(f"Workers: {active_workers} active out of {len(self.workers)}")

    def save_state(self, filename: str = "detector_state.json"):
        """Save detector state to file"""
        try:
            # Get suspicious players
            suspicious_players = self.graph_manager.get_suspicious_player_metrics(min_suspicion=0.5)

            # Prepare state
            state = {
                "timestamp": time.time(),
                "stats": self.global_stats,
                "suspicious_players": suspicious_players,
                "worker_stats": [w.get_stats() for w in self.workers]
            }

            # Write to file
            with open(filename, "w") as f:
                json.dump(state, f, indent=2)

            logger.info(f"State saved to {filename}")

        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")

    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""

        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.running = False
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def stats_reporter(self):
        """Periodically report statistics"""
        while self.running and not self.shutdown_event.is_set():
            self.print_stats()
            self.save_state()

            try:
                # Check every 10 seconds for shutdown, report every 5 minutes
                for _ in range(30):
                    if self.shutdown_event.is_set():
                        break
                    await asyncio.sleep(10)
            except:
                break

    async def run(self):
        """Main detector loop"""
        logger.info("Starting Destiny 2 Cheat Detector")

        self.setup_signal_handlers()
        self.running = True

        try:
            # Initialize
            await self.initialize()

            # Start workers
            self.start_workers()

            # Start stats reporter
            asyncio.create_task(self.stats_reporter())

            # Wait for shutdown signal
            while self.running and not self.shutdown_event.is_set():
                await asyncio.sleep(1)

            logger.info("Shutting down...")

        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        finally:
            self.running = False
            self.stop_workers()
            self.save_state("detector_final_state.json")
            logger.info("Cheat detector shutdown complete")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Destiny 2 Cheat Detector")
    parser.add_argument("--config", default="config.yml", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set log level
    if args.debug:
        logging.getLogger("cheat_detector").setLevel(logging.DEBUG)

    # Create and run detector
    detector = CheatDetector(args.config)
    await detector.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDetector stopped by user")