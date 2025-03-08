# data_processor.py
import logging
import os
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback
from collections import defaultdict

logger = logging.getLogger("cheat_detector")


class DataProcessor:
    def __init__(self, config_path: str = "config.yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.metrics_config = config.get("suspicious_metrics", {})

    def extract_player_metrics(self, pgcr_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract relevant metrics from PGCR data for cheat detection"""
        if not pgcr_data or "entries" not in pgcr_data:
            logger.warning("Invalid PGCR data structure")
            return {}

        try:
            metrics = {}

            # Get game duration
            game_duration_seconds = pgcr_data.get("activityDurationSeconds", 0)
            if not game_duration_seconds and "period" in pgcr_data and "activityEndTimeOffset" in pgcr_data:
                # Calculate from period and end time offset
                try:
                    end_time_offset = pgcr_data["activityEndTimeOffset"]
                    game_duration_seconds = end_time_offset
                except Exception as e:
                    logger.debug(f"Error calculating game duration: {str(e)}")
                    game_duration_seconds = 0

            game_duration_minutes = game_duration_seconds / 60.0

            # Get game mode
            mode_hash = pgcr_data.get("activityDetails", {}).get("directorActivityHash", 0)
            mode_type = self._get_game_mode(mode_hash)  # This would need to be implemented or fetched

            for entry in pgcr_data["entries"]:
                try:
                    player = entry.get("player", {})
                    if not player or "destinyUserInfo" not in player:
                        continue

                    # Create unique player ID
                    player_id = f"{player['destinyUserInfo'].get('displayName', 'Unknown')}#{player['destinyUserInfo'].get('membershipId', '0')}"

                    # Extract basic stats
                    values = entry.get("values", {})
                    metrics[player_id] = {
                        "heavy_kills": 0,
                        "super_kills": 0,
                        "game_duration": game_duration_minutes,
                        "game_mode": mode_type,
                        "headshot_ratio_by_weapon": {},
                        "score": self._get_stat_value(entry, "score"),
                        "kills": self._get_stat_value(values, "kills"),
                        "deaths": self._get_stat_value(values, "deaths"),
                        "assists": self._get_stat_value(values, "assists"),
                        "kill_death_ratio": self._get_stat_value(values, "killsDeathsRatio"),
                        "efficiency": self._get_stat_value(values, "efficiency"),
                        "longest_streak": self._get_stat_value(entry.get("extended", {}).get("values", {}),
                                                               "longestKillSpree"),
                    }

                    # Extract weapon-specific stats
                    if "extended" in entry and "weapons" in entry["extended"]:
                        for weapon in entry["extended"]["weapons"]:
                            try:
                                # Get weapon type
                                weapon_hash = weapon.get("referenceId", 0)
                                weapon_type = self._get_weapon_type(weapon_hash)  # This would need to be implemented

                                # Extract precision stats
                                weapon_values = weapon.get("values", {})
                                precision_kills = self._get_stat_value(weapon_values, "precisionKills")
                                total_kills = self._get_stat_value(weapon_values, "uniqueWeaponKills")

                                if total_kills > 0:
                                    headshot_ratio = precision_kills / total_kills
                                    metrics[player_id]["headshot_ratio_by_weapon"][weapon_type] = headshot_ratio

                                # Check if this is a heavy weapon
                                if weapon.get("isHeavyWeapon", False):
                                    metrics[player_id]["heavy_kills"] += total_kills
                            except Exception as e:
                                logger.warning(f"Error processing weapon: {str(e)}")

                    # Extract ability/super kills
                    if "extended" in entry and "values" in entry["extended"]:
                        ext_values = entry["extended"]["values"]

                        # Check for super kills
                        super_kills = self._get_stat_value(ext_values, "weaponKillsSuper")
                        if not super_kills:
                            # Try alternative stats
                            super_kills = self._get_stat_value(ext_values, "weaponKillsAbility")

                        metrics[player_id]["super_kills"] = super_kills

                        # If heavy kills not set from weapons, try to get from stats
                        if metrics[player_id]["heavy_kills"] == 0:
                            heavy_kills = self._get_stat_value(ext_values, "weaponKillsHeavy")
                            metrics[player_id]["heavy_kills"] = heavy_kills

                except Exception as e:
                    logger.error(f"Error extracting metrics for player in PGCR: {str(e)}")
                    logger.error(traceback.format_exc())

            return metrics

        except Exception as e:
            logger.error(f"Error extracting player metrics: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _get_stat_value(self, data: Dict[str, Any], stat_name: str) -> float:
        """Helper method to extract stat values safely"""
        try:
            if not data or stat_name not in data:
                return 0.0

            stat = data[stat_name]
            if isinstance(stat, dict) and "basic" in stat:
                return stat["basic"].get("value", 0.0)
            elif isinstance(stat, dict):
                return stat.get("value", 0.0)
            else:
                return float(stat)
        except:
            return 0.0

    def _get_weapon_type(self, weapon_hash: int) -> str:
        """Get weapon type from hash"""
        # Would need implementation - for now using placeholder
        return "unknown"

    def _get_game_mode(self, mode_hash: int) -> str:
        """Get game mode from hash"""
        # Would need implementation - for now using placeholder
        return "unknown"