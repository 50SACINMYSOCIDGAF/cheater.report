# queue_manager.py
import logging
import time
import asyncio
import yaml
from typing import Dict, List, Set, Tuple, Any, Optional
import heapq
import traceback
import threading
from collections import deque

logger = logging.getLogger("cheat_detector")


class QueueManager:
    def __init__(self, config_path: str = "config.yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.max_queue_size = config["processing"].get("max_queue_size", 1000)
        self.max_retry_count = config["processing"].get("max_retry_count", 3)

        # Initialize data structures
        self._priority_queue = []  # Heap queue for games to process
        self._visited_games = set()  # Set of processed game IDs
        self._game_metadata = {}  # Metadata for queued games
        self._retry_counts = {}  # Track retry attempts

        # Locks for thread safety
        self._queue_lock = threading.Lock()
        self._visited_lock = threading.Lock()
        self._metadata_lock = threading.Lock()

    def queue_game(self, game_id: str, priority: float = 0.0, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a game to the processing queue with optional metadata"""
        # Ensure game_id is a string
        game_id = str(game_id)

        with self._visited_lock:
            # Check if already processed
            if game_id in self._visited_games:
                return False

        with self._queue_lock:
            # Check if already in queue
            for _, gid in self._priority_queue:
                if gid == game_id:
                    return False

            # Add to queue with priority (negate for min-heap behavior)
            heapq.heappush(self._priority_queue, (-priority, game_id))

            # Trim queue if too large
            if len(self._priority_queue) > self.max_queue_size:
                # Remove lowest priority item
                self._priority_queue.sort()
                self._priority_queue = self._priority_queue[:self.max_queue_size]
                heapq.heapify(self._priority_queue)

            # Store metadata if provided
            if metadata:
                with self._metadata_lock:
                    self._game_metadata[game_id] = metadata

            return True

    def queue_games(self, game_ids: List[str], priority: float = 0.0) -> int:
        """Add multiple games to the queue with the same priority"""
        count = 0
        for game_id in game_ids:
            if self.queue_game(game_id, priority):
                count += 1
        return count

    def get_next_game(self) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Get the next game from the queue with highest priority"""
        with self._queue_lock:
            if not self._priority_queue:
                return None, None

            _, game_id = heapq.heappop(self._priority_queue)

        # Get metadata if available
        metadata = None
        with self._metadata_lock:
            if game_id in self._game_metadata:
                metadata = self._game_metadata[game_id]

        return game_id, metadata

    def mark_game_processed(self, game_id: str) -> None:
        """Mark a game as successfully processed"""
        game_id = str(game_id)
        with self._visited_lock:
            self._visited_games.add(game_id)

        # Clean up metadata
        with self._metadata_lock:
            if game_id in self._game_metadata:
                del self._game_metadata[game_id]

        # Clean up retry count
        if game_id in self._retry_counts:
            del self._retry_counts[game_id]

    def mark_game_failed(self, game_id: str, requeue: bool = True, priority: float = -0.5) -> bool:
        """Mark a game as failed and optionally requeue it"""
        game_id = str(game_id)

        # Increment retry count
        retry_count = self._retry_counts.get(game_id, 0) + 1
        self._retry_counts[game_id] = retry_count

        # Requeue if requested and under max retry limit
        if requeue and retry_count <= self.max_retry_count:
            # Get existing metadata
            metadata = None
            with self._metadata_lock:
                if game_id in self._game_metadata:
                    metadata = self._game_metadata[game_id]

            # Requeue with decreased priority
            adjusted_priority = priority * (1.0 - (retry_count * 0.2))
            return self.queue_game(game_id, adjusted_priority, metadata)
        else:
            # Clean up metadata
            with self._metadata_lock:
                if game_id in self._game_metadata:
                    del self._game_metadata[game_id]
            return False

    def is_game_visited(self, game_id: str) -> bool:
        """Check if a game has been processed already"""
        game_id = str(game_id)
        with self._visited_lock:
            return game_id in self._visited_games

    def queue_size(self) -> int:
        """Get current queue size"""
        with self._queue_lock:
            return len(self._priority_queue)

    def visited_count(self) -> int:
        """Get count of processed games"""
        with self._visited_lock:
            return len(self._visited_games)

    def clear_queue(self) -> None:
        """Clear the processing queue"""
        with self._queue_lock:
            self._priority_queue = []

        with self._metadata_lock:
            self._game_metadata = {}