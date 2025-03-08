# bayesian_detector.py
import logging
import os

import yaml
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

logger = logging.getLogger("cheat_detector")


class BayesianDetector:
    def __init__(self, config_path: str = "config.yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.metrics_config = config["suspicious_metrics"]

        # Set device for PyTorch operations with CUDA 12.4 support
        # First try to get the device from environment variable to support different CUDA versions
        if torch.cuda.is_available():
            cuda_device_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            try:
                # Check if the specified CUDA device is available
                device_count = torch.cuda.device_count()
                device_idx = int(cuda_device_idx)
                if device_idx < device_count:
                    self.device = torch.device(f"cuda:{device_idx}")
                    logger.info(f"Using CUDA device {device_idx} for Bayesian analysis")
                else:
                    logger.warning(f"CUDA device {device_idx} not available. Falling back to CPU")
                    self.device = torch.device("cpu")
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Error setting CUDA device: {str(e)}. Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU for Bayesian analysis")

        # Initialize Bayesian model parameters
        self.initialize_model()

    def initialize_model(self):
        """Initialize Bayesian network parameters"""
        # Prior probabilities: [not_cheating, cheating]
        self.prior = torch.tensor([0.99, 0.01], device=self.device)

        # Conditional probability tables (CPTs)
        # Format: P(evidence | cheating_status)

        # Heavy ammo CPT - P(heavy_kills | cheating_status)
        # Buckets: [0, 1-2, 3, 4+]
        self.heavy_cpt = torch.tensor([
            [0.4, 0.5, 0.09, 0.01],  # Not cheating
            [0.1, 0.2, 0.3, 0.4]  # Cheating
        ], device=self.device)

        # Headshot CPT - P(headshot_ratio | cheating_status)
        # Buckets: [0-0.3, 0.3-0.6, 0.6-0.8, 0.8-0.9, 0.9+]
        self.headshot_cpt = torch.tensor([
            [0.2, 0.4, 0.3, 0.09, 0.01],  # Not cheating
            [0.05, 0.15, 0.25, 0.25, 0.3]  # Cheating
        ], device=self.device)

        # Super kills CPT - P(super_kills | cheating_status)
        # Buckets: [0, 1-2, 3-4, 5+]
        self.super_cpt = torch.tensor([
            [0.2, 0.6, 0.18, 0.02],  # Not cheating
            [0.1, 0.2, 0.3, 0.4]  # Cheating
        ], device=self.device)

        # Game duration CPT - P(duration | cheating_status)
        # Buckets: [<4min, 4-6min, 6-8min, 8+min]
        self.duration_cpt = torch.tensor([
            [0.01, 0.09, 0.3, 0.6],  # Not cheating
            [0.3, 0.5, 0.15, 0.05]  # Cheating
        ], device=self.device)

        # Kill streak CPT - P(streak | cheating_status)
        # Buckets: [0-5, 5-10, 10-15, 15-20, 20+]
        self.streak_cpt = torch.tensor([
            [0.3, 0.4, 0.2, 0.09, 0.01],  # Not cheating
            [0.05, 0.15, 0.3, 0.3, 0.2]  # Cheating
        ], device=self.device)

    def _discretize_evidence(self, evidence: Dict[str, Any]) -> Dict[str, int]:
        """Convert continuous evidence values to discrete buckets"""
        discrete = {}

        # Heavy ammo kills
        if "heavy_kills" in evidence:
            heavy_kills = evidence["heavy_kills"]
            if isinstance(heavy_kills, list):
                heavy_kills = np.mean(heavy_kills)

            if heavy_kills == 0:
                discrete["heavy_bucket"] = 0
            elif heavy_kills <= 2:
                discrete["heavy_bucket"] = 1
            elif heavy_kills <= 3:
                discrete["heavy_bucket"] = 2
            else:
                discrete["heavy_bucket"] = 3

        # Headshot ratio (use the most suspicious weapon)
        if "headshot_ratio_by_weapon" in evidence:
            ratios = evidence["headshot_ratio_by_weapon"]
            max_suspicion = 0.0

            if isinstance(ratios, dict):
                # Single game data
                for weapon_type, ratio in ratios.items():
                    if weapon_type in self.metrics_config["headshot_rate"]["thresholds"]:
                        threshold = self.metrics_config["headshot_rate"]["thresholds"][weapon_type]
                        if ratio > threshold:
                            suspicion = min(1.0, (ratio - threshold) / (1 - threshold) + 0.5)
                            max_suspicion = max(max_suspicion, suspicion)
            elif isinstance(ratios, list):
                # Multiple games data
                for game_data in ratios:
                    if isinstance(game_data, dict):
                        for weapon_type, ratio in game_data.items():
                            if weapon_type in self.metrics_config["headshot_rate"]["thresholds"]:
                                threshold = self.metrics_config["headshot_rate"]["thresholds"][weapon_type]
                                if ratio > threshold:
                                    suspicion = min(1.0, (ratio - threshold) / (1 - threshold) + 0.5)
                                    max_suspicion = max(max_suspicion, suspicion)

            # Discretize max suspicion
            if max_suspicion <= 0.2:
                discrete["headshot_bucket"] = 0
            elif max_suspicion <= 0.4:
                discrete["headshot_bucket"] = 1
            elif max_suspicion <= 0.6:
                discrete["headshot_bucket"] = 2
            elif max_suspicion <= 0.8:
                discrete["headshot_bucket"] = 3
            else:
                discrete["headshot_bucket"] = 4

        # Super kills
        if "super_kills" in evidence:
            super_kills = evidence["super_kills"]
            if isinstance(super_kills, list):
                super_kills = np.mean(super_kills)

            if super_kills == 0:
                discrete["super_bucket"] = 0
            elif super_kills <= 2:
                discrete["super_bucket"] = 1
            elif super_kills <= 4:
                discrete["super_bucket"] = 2
            else:
                discrete["super_bucket"] = 3

        # Game duration
        if "game_duration" in evidence:
            duration = evidence["game_duration"]
            if isinstance(duration, list):
                duration = np.mean(duration)

            if duration < 4:
                discrete["duration_bucket"] = 0
            elif duration < 6:
                discrete["duration_bucket"] = 1
            elif duration < 8:
                discrete["duration_bucket"] = 2
            else:
                discrete["duration_bucket"] = 3

        # Kill streak
        if "longest_streak" in evidence:
            streak = evidence["longest_streak"]
            if isinstance(streak, list):
                streak = np.max(streak)

            if streak < 5:
                discrete["streak_bucket"] = 0
            elif streak < 10:
                discrete["streak_bucket"] = 1
            elif streak < 15:
                discrete["streak_bucket"] = 2
            elif streak < 20:
                discrete["streak_bucket"] = 3
            else:
                discrete["streak_bucket"] = 4

        return discrete

    def calculate_suspicion(self, evidence: Dict[str, Any]) -> float:
        """Calculate suspicion score using Bayesian inference"""
        try:
            # Discretize evidence
            discrete = self._discretize_evidence(evidence)

            # Start with prior probabilities
            posterior = self.prior.clone()

            # Update with each piece of evidence
            if "heavy_bucket" in discrete:
                bucket = discrete["heavy_bucket"]
                if 0 <= bucket < self.heavy_cpt.shape[1]:
                    posterior = posterior * self.heavy_cpt[:, bucket]

            if "headshot_bucket" in discrete:
                bucket = discrete["headshot_bucket"]
                if 0 <= bucket < self.headshot_cpt.shape[1]:
                    posterior = posterior * self.headshot_cpt[:, bucket]

            if "super_bucket" in discrete:
                bucket = discrete["super_bucket"]
                if 0 <= bucket < self.super_cpt.shape[1]:
                    posterior = posterior * self.super_cpt[:, bucket]

            if "duration_bucket" in discrete:
                bucket = discrete["duration_bucket"]
                if 0 <= bucket < self.duration_cpt.shape[1]:
                    posterior = posterior * self.duration_cpt[:, bucket]

            if "streak_bucket" in discrete:
                bucket = discrete["streak_bucket"]
                if 0 <= bucket < self.streak_cpt.shape[1]:
                    posterior = posterior * self.streak_cpt[:, bucket]

            # Normalize
            posterior = posterior / posterior.sum()

            # Return probability of cheating (second element)
            return posterior[1].item()

        except Exception as e:
            logger.error(f"Error in Bayesian calculation: {str(e)}")
            return 0.0

    def batch_process(self, evidence_batch: List[Dict[str, Any]]) -> List[float]:
        """Process multiple players' evidence in parallel"""
        try:
            batch_size = len(evidence_batch)
            if batch_size == 0:
                return []

            # Prepare tensor for batch processing
            heavy_buckets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            headshot_buckets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            super_buckets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            duration_buckets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            streak_buckets = torch.zeros(batch_size, dtype=torch.long, device=self.device)

            # Flags for which evidence is available
            has_heavy = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            has_headshot = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            has_super = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            has_duration = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            has_streak = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            # Discretize each player's evidence
            for i, evidence in enumerate(evidence_batch):
                discrete = self._discretize_evidence(evidence)

                if "heavy_bucket" in discrete:
                    heavy_buckets[i] = discrete["heavy_bucket"]
                    has_heavy[i] = True

                if "headshot_bucket" in discrete:
                    headshot_buckets[i] = discrete["headshot_bucket"]
                    has_headshot[i] = True

                if "super_bucket" in discrete:
                    super_buckets[i] = discrete["super_bucket"]
                    has_super[i] = True

                if "duration_bucket" in discrete:
                    duration_buckets[i] = discrete["duration_bucket"]
                    has_duration[i] = True

                if "streak_bucket" in discrete:
                    streak_buckets[i] = discrete["streak_bucket"]
                    has_streak[i] = True

            # Batch Bayesian inference
            posteriors = self.prior.repeat(batch_size, 1)  # Shape: [batch_size, 2]

            # Update with heavy evidence
            for i in range(batch_size):
                if has_heavy[i] and 0 <= heavy_buckets[i] < self.heavy_cpt.shape[1]:
                    posteriors[i] = posteriors[i] * self.heavy_cpt[:, heavy_buckets[i]]

            # Update with headshot evidence
            for i in range(batch_size):
                if has_headshot[i] and 0 <= headshot_buckets[i] < self.headshot_cpt.shape[1]:
                    posteriors[i] = posteriors[i] * self.headshot_cpt[:, headshot_buckets[i]]

            # Update with super evidence
            for i in range(batch_size):
                if has_super[i] and 0 <= super_buckets[i] < self.super_cpt.shape[1]:
                    posteriors[i] = posteriors[i] * self.super_cpt[:, super_buckets[i]]

            # Update with duration evidence
            for i in range(batch_size):
                if has_duration[i] and 0 <= duration_buckets[i] < self.duration_cpt.shape[1]:
                    posteriors[i] = posteriors[i] * self.duration_cpt[:, duration_buckets[i]]

            # Update with streak evidence
            for i in range(batch_size):
                if has_streak[i] and 0 <= streak_buckets[i] < self.streak_cpt.shape[1]:
                    posteriors[i] = posteriors[i] * self.streak_cpt[:, streak_buckets[i]]

            # Normalize
            posteriors = posteriors / posteriors.sum(dim=1, keepdim=True)

            # Extract cheating probabilities
            cheating_probs = posteriors[:, 1].cpu().numpy()

            return cheating_probs.tolist()

        except Exception as e:
            logger.error(f"Error in batch Bayesian processing: {str(e)}")
            return [0.0] * len(evidence_batch)

    def detect_anomalies(self, metrics: Dict[str, List]) -> Dict[str, Any]:
        """Detect specific anomalies in metrics and return flagged issues"""
        flags = {}

        # Heavy ammo anomalies
        if "heavy_kills" in metrics and metrics["heavy_kills"]:
            heavy_kills = metrics["heavy_kills"]
            threshold = self.metrics_config["heavy_ammo"]["threshold_per_game"]
            impossible = self.metrics_config["heavy_ammo"]["impossible_threshold"]

            suspicious_count = sum(1 for kills in heavy_kills if kills >= threshold)
            impossible_count = sum(1 for kills in heavy_kills if kills >= impossible)

            if suspicious_count > 0:
                flags["heavy_ammo_anomalies"] = {
                    "suspicious_games": f"{suspicious_count}/{len(heavy_kills)}",
                    "avg_heavy_kills": f"{np.mean(heavy_kills):.2f}",
                    "max_heavy_kills": f"{np.max(heavy_kills)}"
                }

                if impossible_count > 0:
                    flags["heavy_ammo_anomalies"]["impossible_heavy_games"] = f"{impossible_count}"

        # Headshot anomalies
        if "headshot_ratio_by_weapon" in metrics and metrics["headshot_ratio_by_weapon"]:
            headshot_data = metrics["headshot_ratio_by_weapon"]
            headshot_flags = {}

            if isinstance(headshot_data, dict):
                # Single game data
                for weapon_type, ratio in headshot_data.items():
                    if weapon_type in self.metrics_config["headshot_rate"]["thresholds"]:
                        threshold = self.metrics_config["headshot_rate"]["thresholds"][weapon_type]
                        if ratio > threshold:
                            headshot_flags[weapon_type] = f"{ratio:.2f} ratio (threshold: {threshold:.2f})"

            elif isinstance(headshot_data, list):
                # Multiple games data
                weapon_ratios = defaultdict(list)

                for game_data in headshot_data:
                    if isinstance(game_data, dict):
                        for weapon_type, ratio in game_data.items():
                            weapon_ratios[weapon_type].append(ratio)

                for weapon_type, ratios in weapon_ratios.items():
                    if weapon_type in self.metrics_config["headshot_rate"]["thresholds"]:
                        threshold = self.metrics_config["headshot_rate"]["thresholds"][weapon_type]
                        avg_ratio = np.mean(ratios)
                        max_ratio = np.max(ratios)

                        if avg_ratio > threshold:
                            headshot_flags[
                                weapon_type] = f"Avg: {avg_ratio:.2f}, Max: {max_ratio:.2f} (threshold: {threshold:.2f})"

            if headshot_flags:
                flags["headshot_anomalies"] = headshot_flags

        # Game duration anomalies
        if "game_duration" in metrics and metrics["game_duration"]:
            durations = metrics["game_duration"]
            threshold = self.metrics_config["game_duration"]["suspicion_threshold_minutes"]
            impossible = self.metrics_config["game_duration"]["impossible_threshold_minutes"]

            fast_count = sum(1 for d in durations if d < threshold)
            impossible_count = sum(1 for d in durations if d < impossible)

            if fast_count > 0:
                flags["game_duration_anomalies"] = {
                    "fast_games": f"{fast_count}/{len(durations)}",
                    "avg_duration": f"{np.mean(durations):.2f} minutes",
                    "min_duration": f"{np.min(durations):.2f} minutes"
                }

                if impossible_count > 0:
                    flags["game_duration_anomalies"]["impossible_fast_games"] = f"{impossible_count}"

        # Kill streak anomalies
        if "longest_streak" in metrics and metrics["longest_streak"]:
            streaks = metrics["longest_streak"]
            threshold = self.metrics_config["kill_streaks"]["suspicion_threshold"]

            high_streaks = sum(1 for s in streaks if s >= threshold)

            if high_streaks > 0:
                flags["kill_streak_anomalies"] = {
                    "high_streak_games": f"{high_streaks}/{len(streaks)}",
                    "avg_streak": f"{np.mean(streaks):.2f}",
                    "max_streak": f"{np.max(streaks)}"
                }

        return flags

    def get_evidence_matches(self, metrics: Dict[str, List], activity_ids: List[str]) -> List[Tuple[str, str]]:
        """Generate evidence matches with reasons from metrics"""
        evidence = []

        if not activity_ids:
            return evidence

        # Match metrics with activity IDs
        for metric_name, values in metrics.items():
            if not isinstance(values, list) or len(values) == 0:
                continue

            # Make sure we don't exceed available activity IDs
            values = values[:len(activity_ids)]

            if metric_name == "heavy_kills":
                threshold = self.metrics_config["heavy_ammo"]["threshold_per_game"]
                impossible = self.metrics_config["heavy_ammo"]["impossible_threshold"]

                for i, kills in enumerate(values):
                    if kills >= impossible:
                        evidence.append((activity_ids[i], f"Impossible heavy kills: {kills}"))
                    elif kills >= threshold:
                        evidence.append((activity_ids[i], f"Suspicious heavy kills: {kills}"))

            elif metric_name == "headshot_ratio_by_weapon" and isinstance(values, list):
                for i, game_data in enumerate(values):
                    if isinstance(game_data, dict):
                        for weapon_type, ratio in game_data.items():
                            if weapon_type in self.metrics_config["headshot_rate"]["thresholds"]:
                                threshold = self.metrics_config["headshot_rate"]["thresholds"][weapon_type]
                                if ratio > threshold + 0.15:  # Significantly over threshold
                                    evidence.append(
                                        (activity_ids[i], f"Suspicious {weapon_type} headshot ratio: {ratio:.2f}"))

            elif metric_name == "game_duration":
                threshold = self.metrics_config["game_duration"]["suspicion_threshold_minutes"]
                impossible = self.metrics_config["game_duration"]["impossible_threshold_minutes"]

                for i, duration in enumerate(values):
                    if duration < impossible:
                        evidence.append((activity_ids[i], f"Impossible game duration: {duration:.2f} minutes"))
                    elif duration < threshold:
                        evidence.append((activity_ids[i], f"Suspicious game duration: {duration:.2f} minutes"))

            elif metric_name == "longest_streak":
                threshold = self.metrics_config["kill_streaks"]["suspicion_threshold"]

                for i, streak in enumerate(values):
                    if streak >= threshold:
                        evidence.append((activity_ids[i], f"Suspicious kill streak: {streak}"))

        return evidence