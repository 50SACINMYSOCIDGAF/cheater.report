# bungio_client.py
import logging
import asyncio
import os
import yaml
import time
import json
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime
import aiohttp
import sqlite3
from collections import defaultdict, deque

import bungio
from bungio.client import Client
from bungio.models import (
    BungieMembershipType,
    DestinyActivityModeType,
    DestinyUser,
    DestinyPostGameCarnageReportData,
    DestinyHistoricalStatsAccountResult
)

# Import SQLAlchemy for AsyncEngine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
import sqlalchemy

from manifest_manager import ManifestManager

logger = logging.getLogger("cheat_detector")


class BungioClient:
    def __init__(self, config_path: str = "config.yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.api_key = config["api_settings"]["api_key"]
        self.manifest_path = config["api_settings"]["manifest_storage"]
        self.max_cache_size = config["api_settings"]["max_cache_size"]
        self.rate_limit = config["api_settings"]["rate_limit_per_second"]

        # Create SQLAlchemy async engine for manifest storage
        # Convert file path to sqlite URL
        manifest_url = f"sqlite+aiosqlite:///{self.manifest_path}"
        self.engine = create_async_engine(manifest_url)

        # Initialize BungIO client with the engine
        self.client = Client(
            bungie_token=self.api_key,
            bungie_client_id="",  # Not needed for read-only API operations
            bungie_client_secret="",  # Not needed for read-only API operations
            manifest_storage=self.engine  # Pass engine instead of string path
        )

        # Initialize manifest manager
        self.manifest_manager = ManifestManager(self.api_key, self.manifest_path)

        # Set up caches
        self.pgcr_cache = {}
        self.player_cache = {}

        # Game mode mapping
        self.game_mode_names = config["processing"]["game_mode_names"]

        # Print all available enum values to help with debugging
        self._print_available_enums()

        # Fixed: Use correct DestinyActivityModeType enum values
        self.game_mode_types = {
            "trials": DestinyActivityModeType.TRIALS_OF_OSIRIS,
            "competitive": DestinyActivityModeType.PV_P_COMPETITIVE,  # Fixed enum name
            "iron_banner": DestinyActivityModeType.IRON_BANNER,
            "control": DestinyActivityModeType.CONTROL,
            "rumble": DestinyActivityModeType.RUMBLE,
            "elimination": DestinyActivityModeType.ELIMINATION
        }

        # Rate limiting
        self.request_limiter = asyncio.Semaphore(self.rate_limit)
        self.last_requests = deque(maxlen=self.rate_limit)

    def _print_available_enums(self):
        """Print all available DestinyActivityModeType enum values"""
        logger.info("All available DestinyActivityModeType values:")
        for name, value in DestinyActivityModeType.__members__.items():
            logger.info(f"  {name}: {value.value}")

    async def initialize(self):
        """Initialize the client by loading manifest data"""
        await self.manifest_manager.load_manifest()

    async def _rate_limited_request(self, coro):
        """Execute a coroutine with rate limiting"""
        now = time.time()

        # Check if we need to wait for rate limit
        if len(self.last_requests) >= self.rate_limit:
            oldest = self.last_requests[0]
            time_since_oldest = now - oldest

            if time_since_oldest < 1.0:
                wait_time = 1.0 - time_since_oldest
                await asyncio.sleep(wait_time)

        # Add current request time
        self.last_requests.append(time.time())

        # Execute the coroutine with semaphore
        async with self.request_limiter:
            return await coro

    async def search_player_by_name(self, name: str) -> Optional[DestinyUser]:
        """Search for a player by their Bungie name"""
        if name in self.player_cache:
            return self.player_cache[name]

        try:
            # Parse Bungie name
            if "#" in name:
                display_name, display_name_code = name.split("#", 1)
            else:
                display_name = name
                display_name_code = "0"

            # List of membership types to try
            # 3=Steam, 1=Xbox, 2=PSN, 6=Epic Games, 5=Stadia
            membership_types = [3, 1, 2, 6, 5]

            # Try each membership type
            for membership_type in membership_types:
                try:
                    headers = {"X-API-Key": self.api_key}

                    # Construct the request body
                    body = {
                        "displayName": display_name,
                        "displayNameCode": display_name_code
                    }

                    # Use aiohttp to make the direct API call
                    async with aiohttp.ClientSession() as session:
                        url = f"https://www.bungie.net/Platform/Destiny2/SearchDestinyPlayerByBungieName/{membership_type}/"
                        async with session.post(url, json=body, headers=headers) as response:
                            if response.status != 200:
                                logger.debug(
                                    f"Player search API for platform {membership_type} returned status {response.status}")
                                continue

                            data = await response.json()

                            if data.get("ErrorCode", 0) != 1 or not data.get("Response"):
                                logger.debug(f"Player {name} not found on platform {membership_type}")
                                continue

                            # Try each result until we find an active Destiny 2 account
                            for player_data in data["Response"]:
                                # Create DestinyUser object
                                user = DestinyUser(
                                    membership_id=player_data["membershipId"],
                                    membership_type=BungieMembershipType(player_data["membershipType"])
                                )

                                # Verify account exists by checking for characters
                                try:
                                    # Check if profile has characters
                                    profile = await self._rate_limited_request(
                                        user.get_profile(components=["Characters"])
                                    )

                                    if profile and profile.characters and profile.characters.data:
                                        # Valid profile found
                                        logger.info(
                                            f"Found player {name} with ID {user.membership_id} on platform {membership_type}")

                                        # Cache result
                                        self.player_cache[name] = user
                                        return user

                                except Exception as e:
                                    logger.debug(
                                        f"Profile check failed for {name} on platform {membership_type}: {str(e)}")
                                    continue

                except Exception as e:
                    logger.debug(f"Error searching for player {name} on platform {membership_type}: {str(e)}")
                    continue

            logger.warning(f"Could not find active Destiny 2 account for {name} on any platform")
            return None

        except Exception as e:
            logger.error(f"Error searching for player {name}: {str(e)}")
            return None

    async def get_pgcr(self, activity_id: str) -> Optional[DestinyPostGameCarnageReportData]:
        """Get post-game carnage report for an activity"""
        if activity_id in self.pgcr_cache:
            return self.pgcr_cache[activity_id]

        try:
            # Get PGCR data
            pgcr = await self._rate_limited_request(
                self.client.api.get_post_game_carnage_report(activity_id=activity_id)
            )

            if not pgcr or not pgcr.response:
                logger.warning(f"No PGCR found for activity {activity_id}")
                return None

            # Cache result
            if len(self.pgcr_cache) >= self.max_cache_size:
                # Remove oldest entry
                self.pgcr_cache.pop(next(iter(self.pgcr_cache)))

            self.pgcr_cache[activity_id] = pgcr.response

            return pgcr.response

        except Exception as e:
            logger.error(f"Error getting PGCR {activity_id}: {str(e)}")
            return None

    async def get_player_recent_activities(self,
                                           user: DestinyUser,
                                           modes: Optional[List[DestinyActivityModeType]] = None,
                                           count: int = 25) -> List[Dict]:
        """Get recent activities for a player across specified modes"""
        activities = []

        try:
            # Get profile to access characters
            profile = await self._rate_limited_request(
                user.get_profile(components=["Characters"])
            )

            if not profile or not profile.characters or not profile.characters.data:
                logger.warning(f"No characters found for user {user.membership_id}")
                return []

            # Default to all PvP modes if none specified
            if not modes:
                modes = [DestinyActivityModeType.ALL_PVP]

            # For each character, get activity history for each mode
            for character_id in profile.characters.data:
                for mode in modes:
                    try:
                        # Use the proper API call for activity history
                        response = await self._rate_limited_request(
                            self.client.api.get_activity_history(
                                character_id=character_id,
                                destiny_membership_id=user.membership_id,
                                membership_type=user.membership_type,
                                count=count // len(modes),  # Distribute count across modes
                                mode=mode,
                                page=0
                            )
                        )

                        # Process activities if we got a result
                        if response and hasattr(response, "activities"):
                            for activity in response.activities:
                                # Extract relevant details
                                activity_data = {
                                    "activityDetails": {
                                        "instanceId": activity.activity_details.instance_id,
                                        "mode": activity.activity_details.mode,
                                        "referenceId": activity.activity_details.reference_id,
                                        "directorActivityHash": activity.activity_details.director_activity_hash,
                                        "activityDurationSeconds": getattr(activity, "activity_duration_seconds", 0)
                                    },
                                    "period": str(activity.period)
                                }
                                activities.append(activity_data)

                                # Stop if we've reached our target count
                                if len(activities) >= count:
                                    break

                    except Exception as e:
                        logger.warning(f"Error getting activities for character {character_id}, mode {mode}: {str(e)}")

            return activities

        except Exception as e:
            logger.error(f"Error getting recent activities: {str(e)}")
            return []

    async def get_seed_matches_from_cheaters(self, cheater_names: List[str]) -> List[str]:
        """Get recent match IDs from a list of known cheaters"""
        seed_matches = set()
        successful_players = 0

        # Include known PGCRs with confirmed cheaters
        known_pgcrs = ["15967863965", "15968884445"]
        seed_matches.update(known_pgcrs)
        logger.info(f"Added {len(known_pgcrs)} known PGCR IDs with confirmed cheaters")

        # Add known public cheaters - use the proper enum value for Steam
        known_cheaters = [
            ("4611686018540389245", BungieMembershipType.TIGER_STEAM),  # Known public cheater on Steam
            ("4611686018512740742", BungieMembershipType.TIGER_STEAM)  # Known public cheater on Steam
        ]

        for membership_id, membership_type in known_cheaters:
            try:
                user = DestinyUser(
                    membership_id=membership_id,
                    membership_type=membership_type
                )

                activities = await self.get_player_recent_activities(
                    user=user,
                    modes=[self.game_mode_types["trials"]],
                    count=25
                )

                for activity in activities:
                    instance_id = activity.get("activityDetails", {}).get("instanceId")
                    if instance_id:
                        seed_matches.add(instance_id)

                successful_players += 1
                if activities:
                    logger.info(f"Found {len(activities)} activities for known cheater {membership_id}")

            except Exception as e:
                logger.error(f"Error getting activities for known cheater {membership_id}: {str(e)}")

        # Try the list of potentially private accounts
        for name in cheater_names:
            try:
                # Find player
                player = await self.search_player_by_name(name)

                if not player:
                    logger.warning(f"Could not find player: {name}")
                    continue

                successful_players += 1
                logger.info(f"Found player {name} (ID: {player.membership_id})")

                # Try getting activities
                activities = await self.get_player_recent_activities(
                    user=player,
                    modes=[self.game_mode_types["trials"]],
                    count=25
                )

                # Add activity IDs to seed matches
                for activity in activities:
                    instance_id = activity.get("activityDetails", {}).get("instanceId")
                    if instance_id:
                        seed_matches.add(instance_id)

                if activities:
                    logger.info(f"Found {len(activities)} activities for {name}")

            except Exception as e:
                logger.error(f"Error processing player {name}: {str(e)}")

        logger.info(f"Found {len(seed_matches)} seed matches from {successful_players} players")

        # Include fallback match IDs if we need more
        if len(seed_matches) < 5:
            logger.info("Adding fallback Trials match IDs")
            fallback_matches = [
                "14676642267", "14676638309", "14676633805",
                "14676628655", "14676622923"
            ]
            seed_matches.update(fallback_matches)

        return list(seed_matches)

    # Extract player metrics from PGCR
    async def extract_player_metrics(self, pgcr: DestinyPostGameCarnageReportData) -> Dict[str, Dict[str, Any]]:
        """Extract relevant metrics from PGCR data for cheat detection"""
        if not pgcr or not pgcr.entries:
            return {}

        metrics = {}
        game_duration_seconds = getattr(pgcr, "activity_duration_seconds", 0)
        if not game_duration_seconds and hasattr(pgcr, "period") and hasattr(pgcr, "activity_end_time_offset_seconds"):
            # Calculate from period and end time offset
            try:
                end_time_offset = pgcr.activity_end_time_offset_seconds
                period = datetime.fromisoformat(str(pgcr.period).replace('Z', '+00:00'))
                game_duration_seconds = end_time_offset
            except:
                game_duration_seconds = 0

        game_duration_minutes = game_duration_seconds / 60.0

        # Get game mode
        mode_hash = getattr(pgcr, "activity_details", {}).get("director_activity_hash", 0)
        mode_type = self.manifest_manager.get_game_mode(mode_hash)

        for entry in pgcr.entries:
            try:
                player = entry.player
                if not player or not player.destiny_user_info:
                    continue

                # Create unique player ID
                player_id = f"{player.destiny_user_info.display_name}#{player.destiny_user_info.membership_id}"

                # Extract basic stats
                metrics[player_id] = {
                    "heavy_kills": 0,
                    "super_kills": 0,
                    "game_duration": game_duration_minutes,
                    "game_mode": mode_type,
                    "headshot_ratio_by_weapon": {},
                    "score": getattr(entry.score, "basic", {}).get("value", 0),
                    "kills": getattr(entry.values.get("kills", {}), "basic", {}).get("value", 0),
                    "deaths": getattr(entry.values.get("deaths", {}), "basic", {}).get("value", 0),
                    "assists": getattr(entry.values.get("assists", {}), "basic", {}).get("value", 0),
                    "kill_death_ratio": getattr(entry.values.get("killsDeathsRatio", {}), "basic", {}).get("value", 0),
                    "efficiency": getattr(entry.values.get("efficiency", {}), "basic", {}).get("value", 0),
                    "longest_streak": getattr(entry.extended.get("values", {}).get("longestKillSpree", {}), "basic",
                                              {}).get("value", 0),
                }

                # Extract weapon-specific stats
                if hasattr(entry, "extended") and hasattr(entry.extended, "weapons"):
                    for weapon in entry.extended.weapons:
                        try:
                            # Get weapon type from manifest
                            weapon_hash = weapon.reference_id
                            weapon_type = self.manifest_manager.get_weapon_type(weapon_hash)

                            # Extract precision stats
                            precision_kills = getattr(weapon.values.get("precisionKills", {}), "basic", {}).get("value",
                                                                                                                0)
                            total_kills = getattr(weapon.values.get("uniqueWeaponKills", {}), "basic", {}).get("value",
                                                                                                               0)

                            if total_kills > 0:
                                headshot_ratio = precision_kills / total_kills
                                metrics[player_id]["headshot_ratio_by_weapon"][weapon_type] = headshot_ratio

                            # Check if this is a heavy weapon
                            if weapon.is_heavy_weapon:
                                metrics[player_id]["heavy_kills"] += total_kills
                        except Exception as e:
                            logger.warning(f"Error processing weapon: {str(e)}")

                # Extract ability/super kills
                if hasattr(entry, "extended") and hasattr(entry.extended, "values"):
                    values = entry.extended.values

                    # Check for super kills from various stats
                    super_kills = getattr(values.get("weaponKillsSuper", {}), "basic", {}).get("value", 0)
                    if not super_kills:
                        # Try alternative stats
                        super_kills = getattr(values.get("weaponKillsAbility", {}), "basic", {}).get("value", 0)

                    metrics[player_id]["super_kills"] = super_kills

                    # If heavy kills not set from weapons, try to get from stats
                    if metrics[player_id]["heavy_kills"] == 0:
                        heavy_kills = getattr(values.get("weaponKillsHeavy", {}), "basic", {}).get("value", 0)
                        metrics[player_id]["heavy_kills"] = heavy_kills

            except Exception as e:
                logger.error(f"Error extracting metrics for player in PGCR: {str(e)}")

        return metrics

    async def send_discord_alert(self, webhook_url: str, player_name: str, evidence_pgcrs: List[str],
                                 suspicion_score: float, flags: Dict[str, Any]) -> None:
        """Send a Discord webhook alert for a detected cheater"""
        try:
            # Create embed with evidence
            embed = {
                "title": f"🚨 Possible Cheater Detected: {player_name}",
                "color": 16711680,  # Red
                "description": f"Suspicion Score: {suspicion_score:.2f}/1.00\n\n**Suspicious Activities:**",
                "fields": []
            }

            # Add flags as fields
            for flag_type, details in flags.items():
                if isinstance(details, dict):
                    value = "\n".join(f"- {k}: {v}" for k, v in details.items())
                else:
                    value = str(details)

                embed["fields"].append({
                    "name": flag_type.replace("_", " ").title(),
                    "value": value,
                    "inline": True
                })

            # Add evidence links
            evidence_links = "\n".join(f"[Match #{i + 1}](https://www.bungie.net/en/PGCR/{pgcr})"
                                       for i, pgcr in enumerate(evidence_pgcrs[:5]))

            embed["fields"].append({
                "name": "Evidence (PGCRs)",
                "value": evidence_links if evidence_links else "No specific evidence links",
                "inline": False
            })

            payload = {
                "content": f"Possible cheater detected: {player_name}",
                "embeds": [embed]
            }

            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as resp:
                    if resp.status != 204:
                        logger.error(f"Failed to send Discord alert: {resp.status}")
                    else:
                        logger.info(f"Sent cheater alert for {player_name}")

        except Exception as e:
            logger.error(f"Error sending Discord alert: {str(e)}")