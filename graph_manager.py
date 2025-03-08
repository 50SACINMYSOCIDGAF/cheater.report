# graph_manager.py
import logging
import yaml
import threading
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, deque
import queue
import random

from bayesian_detector import BayesianDetector

logger = logging.getLogger("cheat_detector")


class GraphManager:
    def __init__(self, config_path: str = "config.yml", bayesian_detector: Optional[BayesianDetector] = None):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.attention_window_size = config["processing"]["attention_window_size"]
        self.suspicion_decay_rate = config["processing"]["suspicion_decay_rate"]
        self.confidence_threshold = config["processing"]["confidence_threshold"]

        self.exploration_config = config["processing"]["exploration_strategy"]
        self.breadth_depth_ratio = self.exploration_config["breadth_depth_ratio"]
        self.suspicion_weight = self.exploration_config["suspicion_weight"]
        self.novelty_weight = self.exploration_config["novelty_weight"]
        self.max_frontier_size = self.exploration_config["max_frontier_size"]

        self.bayesian_detector = bayesian_detector

        # Graph structures
        self.graph_lock = threading.RLock()
        self.frontier_lock = threading.RLock()

        self.player_to_matches = defaultdict(set)
        self.match_to_players = defaultdict(set)
        self.player_suspicion = {}
        self.player_evidence = defaultdict(list)
        self.player_metrics = defaultdict(lambda: defaultdict(list))
        self.player_activities = defaultdict(list)

        # Exploration data
        self.match_frontier = []
        self.visited_matches = set()
        self.attention_window = deque(maxlen=self.attention_window_size)

        # Queue for multi-threaded exploration
        self.match_queue = queue.PriorityQueue()

    def add_match_data(self, match_id: str, player_metrics: Dict[str, Dict[str, Any]]) -> None:
        """Add match data to the graph"""
        with self.graph_lock:
            players = list(player_metrics.keys())
            self.match_to_players[match_id] = set(players)

            for player in players:
                self.player_to_matches[player].add(match_id)
                self.player_activities[player].append(match_id)

                # Update metrics
                for metric, value in player_metrics[player].items():
                    self.player_metrics[player][metric].append(value)

                # Add to attention window if new
                if player not in self.player_suspicion:
                    self.player_suspicion[player] = 0.0
                    if len(self.attention_window) < self.attention_window_size:
                        self.attention_window.append(player)

            self.visited_matches.add(match_id)

    def get_connected_matches(self, suspicion_threshold: float = 0.3) -> List[str]:
        """Get unvisited matches connected to suspicious players"""
        connected_matches = set()

        with self.graph_lock:
            for player, score in self.player_suspicion.items():
                if score >= suspicion_threshold and player in self.attention_window:
                    for match_id in self.player_to_matches[player]:
                        # Find matches where we don't have all players
                        all_players = self.match_to_players[match_id]
                        if not all(p in self.player_metrics for p in all_players):
                            connected_matches.add(match_id)

        return list(connected_matches - self.visited_matches)

    def update_frontier(self, new_matches: List[str]) -> None:
        """Add new matches to the frontier"""
        with self.frontier_lock:
            for match_id in new_matches:
                if match_id not in self.visited_matches and match_id not in self.match_frontier:
                    self.match_frontier.append(match_id)

            # Trim frontier if too large
            if len(self.match_frontier) > self.max_frontier_size:
                prioritized = self.prioritize_frontier()
                self.match_frontier = [match_id for _, match_id in prioritized[:self.max_frontier_size]]

    def prioritize_frontier(self) -> List[Tuple[float, str]]:
        """Prioritize matches in the frontier"""
        frontier_scores = []

        with self.frontier_lock:
            for match_id in self.match_frontier:
                if match_id in self.visited_matches:
                    continue

                # Calculate match priority

                # 1. Average suspicion of known players
                players = self.match_to_players.get(match_id, set())
                known_players = [p for p in players if p in self.player_suspicion]

                if not known_players:
                    avg_suspicion = 0.0
                else:
                    avg_suspicion = sum(self.player_suspicion.get(p, 0.0) for p in known_players) / len(known_players)

                # 2. Novelty - ratio of unknown players
                if players:
                    novelty = 1.0 - (len(known_players) / len(players))
                else:
                    novelty = 0.0

                # 3. Combined score
                score = (avg_suspicion * self.suspicion_weight) + (novelty * self.novelty_weight)

                # 4. Add to prioritized list
                frontier_scores.append((-score, match_id))  # Negative for priority queue (lowest first)

            return sorted(frontier_scores)

    def get_next_matches(self, count: int = 10) -> List[str]:
        """Get next matches to explore using prioritized selection"""
        with self.frontier_lock:
            if not self.match_frontier:
                return []

            prioritized = self.prioritize_frontier()

            # Blend breadth and depth exploration
            depth_count = max(1, int(count * (1 - self.breadth_depth_ratio)))
            breadth_count = count - depth_count

            # Depth - highest scoring matches
            depth_matches = [match_id for _, match_id in prioritized[:depth_count]]

            # Breadth - random selection from remaining frontier
            remaining = prioritized[depth_count:]
            breadth_matches = []

            if remaining and breadth_count > 0:
                indices = random.sample(range(len(remaining)), min(breadth_count, len(remaining)))
                breadth_matches = [remaining[i][1] for i in indices]

            # Remove selected matches from frontier
            for match_id in depth_matches + breadth_matches:
                if match_id in self.match_frontier:
                    self.match_frontier.remove(match_id)

            return depth_matches + breadth_matches

    def update_suspicion_scores(self) -> List[Tuple[str, float]]:
        """Update suspicion scores for all players in attention window"""
        if not self.bayesian_detector:
            logger.error("No Bayesian detector provided for suspicion score updates")
            return []

        confirmed_cheaters = []

        with self.graph_lock:
            players_to_process = list(self.attention_window)
            evidence_batch = []

            for player in players_to_process:
                # Skip if player has fewer than 3 matches
                if len(self.player_to_matches.get(player, set())) < 3:
                    continue

                # Prepare metrics for Bayesian processing
                metrics = {}
                for metric, values in self.player_metrics[player].items():
                    if values:
                        if metric == "headshot_ratio_by_weapon":
                            # Special handling for nested structure
                            metrics[metric] = values
                        else:
                            metrics[metric] = values

                evidence_batch.append(metrics)

            # Batch process with Bayesian model
            if evidence_batch:
                suspicion_scores = self.bayesian_detector.batch_process(evidence_batch)

                # Update player suspicion scores
                for player, new_score in zip(players_to_process, suspicion_scores):
                    old_score = self.player_suspicion.get(player, 0.0)

                    # Apply decay if score decreasing
                    if new_score < old_score:
                        new_score = max(new_score, old_score * (1 - self.suspicion_decay_rate))

                    self.player_suspicion[player] = new_score

                    # If player above threshold, add to confirmed list
                    if new_score >= self.confidence_threshold:
                        confirmed_cheaters.append((player, new_score))
                        # Remove from attention window
                        if player in self.attention_window:
                            self.attention_window.remove(player)

            return confirmed_cheaters

    def get_player_evidence(self, player: str) -> Tuple[List[str], Dict[str, Any]]:
        """Get evidence and flags for a player"""
        with self.graph_lock:
            # Get activity IDs
            activity_ids = self.player_activities.get(player, [])

            # Get metrics
            metrics = {}
            for metric, values in self.player_metrics.get(player, {}).items():
                metrics[metric] = values

            if not metrics or not activity_ids:
                return [], {}

            # Generate evidence
            if self.bayesian_detector:
                evidence_matches = self.bayesian_detector.get_evidence_matches(metrics, activity_ids)
                flags = self.bayesian_detector.detect_anomalies(metrics)

                # Get unique match IDs from evidence
                match_ids = []
                for match_id, _ in evidence_matches:
                    if match_id not in match_ids:
                        match_ids.append(match_id)

                return match_ids, flags

            return [], {}

    def queue_match_for_analysis(self, match_id: str, priority: float = 0.0) -> None:
        """Add match to priority queue for analysis"""
        if match_id not in self.visited_matches:
            self.match_queue.put((priority, match_id))

    def get_next_queued_match(self, timeout: float = 0.1) -> Optional[str]:
        """Get next match from queue"""
        try:
            _, match_id = self.match_queue.get(timeout=timeout)
            return match_id
        except queue.Empty:
            return None

    def cleanup_attention_window(self) -> None:
        """Remove low-suspicion players from attention window"""
        with self.graph_lock:
            players_to_remove = []

            for player in self.attention_window:
                # Remove if suspicion score is low and we have seen enough matches
                if (self.player_suspicion.get(player, 0.0) < 0.3 and
                        len(self.player_to_matches.get(player, set())) >= 10):
                    players_to_remove.append(player)

            for player in players_to_remove:
                if player in self.attention_window:
                    self.attention_window.remove(player)

            if players_to_remove:
                logger.debug(f"Removed {len(players_to_remove)} players from attention window")

    def get_suspicious_player_metrics(self, min_suspicion: float = 0.5) -> Dict[str, Dict[str, Any]]:
        """Get metrics for suspicious players"""
        result = {}

        with self.graph_lock:
            for player, suspicion in self.player_suspicion.items():
                if suspicion >= min_suspicion:
                    aggregated_metrics = {}

                    # Get raw metrics
                    raw_metrics = self.player_metrics.get(player, {})

                    # Aggregate metrics across games
                    for metric, values in raw_metrics.items():
                        if not values:
                            continue

                        if metric == "headshot_ratio_by_weapon":
                            # Special handling for weapon headshots
                            weapon_ratios = defaultdict(list)

                            for game_data in values:
                                if isinstance(game_data, dict):
                                    for weapon_type, ratio in game_data.items():
                                        weapon_ratios[weapon_type].append(ratio)

                            aggregated_metrics[metric] = {
                                weapon: {
                                    "avg": np.mean(ratios),
                                    "max": np.max(ratios),
                                    "games": len(ratios)
                                }
                                for weapon, ratios in weapon_ratios.items() if ratios
                            }
                        else:
                            # Standard numeric metrics
                            aggregated_metrics[metric] = {
                                "avg": np.mean(values),
                                "max": np.max(values),
                                "min": np.min(values),
                                "games": len(values)
                            }

                    aggregated_metrics["suspicion_score"] = suspicion
                    aggregated_metrics["match_count"] = len(self.player_to_matches.get(player, set()))

                    result[player] = aggregated_metrics

            return result

    def print_statistics(self):
        """Print statistics about the graph"""
        with self.graph_lock:
            player_count = len(self.player_to_matches)
            match_count = len(self.match_to_players)
            attention_count = len(self.attention_window)
            frontier_count = len(self.match_frontier)
            suspicious_count = sum(1 for score in self.player_suspicion.values() if score >= 0.5)

            logger.info(
                f"Graph stats: {player_count} players, {match_count} matches, {attention_count} in attention window")
            logger.info(f"Frontier: {frontier_count} matches, Suspicious players: {suspicious_count}")