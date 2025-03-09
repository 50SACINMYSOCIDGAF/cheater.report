# bungio_client.py
import logging
import asyncio
import os
import yaml
import time
from typing import Dict, Any, Optional, List, Tuple
import aiohttp
from collections import OrderedDict

import bungio
from bungio.client import Client
from bungio.models import (
    BungieMembershipType,
    DestinyActivityModeType,
    DestinyUser,
    DestinyPostGameCarnageReportData
)
# Add SQLAlchemy imports
from sqlalchemy.ext.asyncio import create_async_engine

logger = logging.getLogger("cheat_detector")


class BungioClient:
    def __init__(self, config_path: str = "config.yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.api_key = config["api_settings"]["api_key"]
        self.manifest_path = config["api_settings"]["manifest_storage"]
        self.max_cache_size = config["api_settings"]["max_cache_size"]
        self.rate_limit = config["api_settings"]["rate_limit_per_second"]

        # Create AsyncEngine for manifest storage
        if self.manifest_path.startswith(('sqlite://', 'sqlite+aiosqlite://')):
            # It's already a SQLAlchemy URL
            manifest_engine = create_async_engine(self.manifest_path)
        else:
            # It's a file path, convert to SQLAlchemy URL
            manifest_engine = create_async_engine(f"sqlite+aiosqlite:///{self.manifest_path}")

        # Initialize client with the AsyncEngine
        self.client = Client(
            bungie_token=self.api_key,
            bungie_client_id="",  # Not needed for read-only API operations
            bungie_client_secret="",  # Not needed for read-only API operations
            manifest_storage=manifest_engine
        )

        # Set up caches with OrderedDict for FIFO behavior
        self.pgcr_cache = OrderedDict()
        self.player_cache = OrderedDict()

        # Game mode mapping - fixed PVP_COMPETITIVE to PV_P_COMPETITIVE
        self.game_mode_types = {
            "trials": DestinyActivityModeType.TRIALS_OF_OSIRIS,
            "competitive": DestinyActivityModeType.PV_P_COMPETITIVE,  # Fixed naming
            "iron_banner": DestinyActivityModeType.IRON_BANNER,
            "control": DestinyActivityModeType.CONTROL,
            "rumble": DestinyActivityModeType.RUMBLE,
            "elimination": DestinyActivityModeType.ELIMINATION
        }

        # Rate limiting with lock to ensure thread safety
        self.request_times = []
        self.request_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the client by loading manifest data"""
        try:
            await self.client.api.get_destiny_manifest()
            logger.info("Successfully loaded Destiny manifest")
            return True
        except Exception as e:
            logger.error(f"Failed to load Destiny manifest: {str(e)}")
            return False

    async def _rate_limited_request(self, coro):
        """Execute a coroutine with rate limiting"""
        async with self.request_lock:
            now = time.time()

            # Remove request times older than 1 second
            self.request_times = [t for t in self.request_times if now - t < 1.0]

            # Wait if at rate limit
            if len(self.request_times) >= self.rate_limit:
                wait_time = 1.0 - (now - self.request_times[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            # Add current request time
            self.request_times.append(time.time())

        try:
            return await coro
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            raise

    async def search_player_by_name(self, name: str) -> Tuple[Optional[DestinyUser], Optional[BungieMembershipType]]:
        """Search for a player by their Bungie name and return both the user and the correct membership type"""
        # Check if this is actually a membership ID instead of a name
        if name.isdigit() and len(name) > 10:
            return await self.get_player_by_membership_id(name)

        # Check cache for regular name search
        if name in self.player_cache:
            cached_user, cached_membership_type = self.player_cache[name]
            return cached_user, cached_membership_type

        try:
            # Parse Bungie name - handle multiple formats
            display_name = name
            display_name_code = "0"

            if "#" in name:
                name_parts = name.split("#", 1)
                display_name = name_parts[0]
                # Only use the code part if it looks like a Bungie name code (not a membership ID)
                if len(name_parts[1]) <= 5 and name_parts[1].isdigit():
                    display_name_code = name_parts[1]

            # Use BungieMembershipType.All (value 254) to search across all platforms at once
            try:
                headers = {"X-API-Key": self.api_key}

                # Construct the request body
                body = {
                    "displayName": display_name,
                    "displayNameCode": display_name_code
                }

                # Use aiohttp to make the direct API call
                async with aiohttp.ClientSession() as session:
                    # Search across all platforms with BungieMembershipType.All (254)
                    url = f"https://www.bungie.net/Platform/Destiny2/SearchDestinyPlayerByBungieName/254/"
                    async with session.post(url, json=body, headers=headers) as response:
                        if response.status != 200:
                            logger.debug(f"Player search API returned status {response.status}")
                            return None, None

                        data = await response.json()

                        if data.get("ErrorCode", 0) != 1 or not data.get("Response"):
                            logger.debug(f"Player {name} not found on any platform")
                            return None, None

                        # Try each result until we find an active Destiny 2 account
                        for player_data in data["Response"]:
                            # Get the actual membership type from the response
                            actual_membership_type = BungieMembershipType(player_data["membershipType"])

                            # Create DestinyUser object with the correct membership type
                            user = DestinyUser(
                                membership_id=player_data["membershipId"],
                                membership_type=actual_membership_type
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
                                        f"Found player {name} with ID {user.membership_id} on platform {actual_membership_type.name}")

                                    # Cache result with correct membership type
                                    self.player_cache[name] = (user, actual_membership_type)

                                    # Maintain cache size
                                    if len(self.player_cache) > self.max_cache_size:
                                        self.player_cache.popitem(last=False)

                                    return user, actual_membership_type

                            except Exception as e:
                                logger.debug(
                                    f"Profile check failed for {name} on platform {actual_membership_type.name}: {str(e)}")
                                continue

            except Exception as e:
                logger.debug(f"Error searching for player {name} using BungieMembershipType.All: {str(e)}")

            # If BungieMembershipType.All approach fails, fall back to the platform-by-platform approach
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
                                # Get the actual membership type from the response
                                actual_membership_type = BungieMembershipType(player_data["membershipType"])

                                # Create DestinyUser object with the correct membership type
                                user = DestinyUser(
                                    membership_id=player_data["membershipId"],
                                    membership_type=actual_membership_type
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
                                            f"Found player {name} with ID {user.membership_id} on platform {actual_membership_type.name}")

                                        # Cache result with correct membership type
                                        self.player_cache[name] = (user, actual_membership_type)

                                        # Maintain cache size
                                        if len(self.player_cache) > self.max_cache_size:
                                            self.player_cache.popitem(last=False)

                                        return user, actual_membership_type

                                except Exception as e:
                                    logger.debug(
                                        f"Profile check failed for {name} on platform {actual_membership_type.name}: {str(e)}")
                                    continue

                except Exception as e:
                    logger.debug(f"Error searching for player {name} on platform {membership_type}: {str(e)}")
                    continue

            logger.warning(f"Could not find active Destiny 2 account for {name} on any platform")
            return None, None

        except Exception as e:
            logger.error(f"Error searching for player {name}: {str(e)}")
            return None, None

    async def get_pgcr(self, activity_id: str) -> Optional[Dict[str, Any]]:
        """Get post-game carnage report for an activity"""
        # Ensure activity_id is a string
        activity_id = str(activity_id)

        if activity_id in self.pgcr_cache:
            return self.pgcr_cache[activity_id]

        try:
            # Make direct API call instead of using bungio to ensure proper handling
            headers = {"X-API-Key": self.api_key}
            url = f"https://www.bungie.net/Platform/Destiny2/Stats/PostGameCarnageReport/{activity_id}/"

            async with aiohttp.ClientSession() as session:
                async with self.request_lock:
                    # Implement rate limiting
                    now = time.time()
                    self.request_times = [t for t in self.request_times if now - t < 1.0]

                    if len(self.request_times) >= self.rate_limit:
                        wait_time = 1.0 - (now - self.request_times[0])
                        if wait_time > 0:
                            await asyncio.sleep(wait_time)

                    self.request_times.append(time.time())

                    # Make the request
                    async with session.get(url, headers=headers) as response:
                        if response.status != 200:
                            logger.warning(f"PGCR API returned status {response.status} for activity {activity_id}")
                            return None

                        data = await response.json()

                        if data.get("ErrorCode", 0) != 1 or not data.get("Response"):
                            logger.warning(f"No PGCR found for activity {activity_id}")
                            return None

                        # Cache result
                        self.pgcr_cache[activity_id] = data["Response"]

                        # Maintain cache size
                        if len(self.pgcr_cache) > self.max_cache_size:
                            self.pgcr_cache.popitem(last=False)

                        return data["Response"]

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
                modes = [DestinyActivityModeType.ALL_PV_P]

            # For each character, get activity history for each mode
            for character_id in profile.characters.data:
                for mode in modes:
                    try:
                        # Use direct API call to ensure proper handling
                        headers = {"X-API-Key": self.api_key}

                        # Ensure we're using the correct membership type that matches the provided user
                        membership_type_value = user.membership_type.value

                        url = f"https://www.bungie.net/Platform/Destiny2/{membership_type_value}/Account/{user.membership_id}/Character/{character_id}/Stats/Activities/"
                        params = {
                            "mode": mode.value,
                            "count": count // len(modes),
                            "page": 0
                        }

                        logger.debug(
                            f"Requesting activities for user {user.membership_id} with membership type {user.membership_type.name}")

                        async with aiohttp.ClientSession() as session:
                            async with self.request_lock:
                                # Implement rate limiting
                                now = time.time()
                                self.request_times = [t for t in self.request_times if now - t < 1.0]

                                if len(self.request_times) >= self.rate_limit:
                                    wait_time = 1.0 - (now - self.request_times[0])
                                    if wait_time > 0:
                                        await asyncio.sleep(wait_time)

                                self.request_times.append(time.time())

                                # Make the request
                                async with session.get(url, headers=headers, params=params) as response:
                                    if response.status != 200:
                                        logger.warning(f"Activity history API returned status {response.status}")
                                        continue

                                    data = await response.json()

                                    if data.get("ErrorCode", 0) != 1 or not data.get("Response") or not data[
                                        "Response"].get("activities"):
                                        logger.debug(f"No activities found for character {character_id}, mode {mode}")
                                        continue

                                    # Process activities
                                    for activity in data["Response"]["activities"]:
                                        # Extract relevant details
                                        activity_data = {
                                            "activityDetails": {
                                                "instanceId": activity["activityDetails"]["instanceId"],
                                                "mode": activity["activityDetails"]["mode"],
                                                "referenceId": activity["activityDetails"]["referenceId"],
                                                "directorActivityHash": activity["activityDetails"].get(
                                                    "directorActivityHash", 0),
                                            },
                                            "period": activity["period"]
                                        }

                                        # Add activity duration if available
                                        if "activityDurationSeconds" in activity:
                                            activity_data["activityDetails"]["activityDurationSeconds"] = activity[
                                                "activityDurationSeconds"]

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

    async def get_player_by_membership_id(self, membership_id: str) -> Tuple[
        Optional[DestinyUser], Optional[BungieMembershipType]]:
        """Try to find a valid platform for a membership ID by trying each platform type"""

        # Check cache first
        cache_key = f"mid:{membership_id}"
        if cache_key in self.player_cache:
            cached_user, cached_membership_type = self.player_cache[cache_key]
            return cached_user, cached_membership_type

        try:
            # First try to use the GetMembershipsById endpoint to get all memberships
            headers = {"X-API-Key": self.api_key}
            url = f"https://www.bungie.net/Platform/User/GetMembershipsById/{membership_id}/254/"

            async with aiohttp.ClientSession() as session:
                async with self.request_lock:
                    # Implement rate limiting
                    now = time.time()
                    self.request_times = [t for t in self.request_times if now - t < 1.0]

                    if len(self.request_times) >= self.rate_limit:
                        wait_time = 1.0 - (now - self.request_times[0])
                        if wait_time > 0:
                            await asyncio.sleep(wait_time)

                    self.request_times.append(time.time())

                    # Make the request
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()

                            if data.get("ErrorCode", 0) == 1 and data.get("Response"):
                                # Get the destiny memberships
                                destiny_memberships = data["Response"].get("destinyMemberships", [])

                                # Try each membership
                                for membership in destiny_memberships:
                                    platform = BungieMembershipType(membership["membershipType"])

                                    # Create a user with this platform type
                                    user = DestinyUser(
                                        membership_id=membership["membershipId"],
                                        membership_type=platform
                                    )

                                    # Verify if this combination is valid
                                    try:
                                        profile = await self._rate_limited_request(
                                            user.get_profile(components=["Characters"])
                                        )

                                        if profile and profile.characters and profile.characters.data:
                                            logger.info(
                                                f"Found valid platform {platform.name} for membership ID {membership_id}")

                                            # Cache the result
                                            self.player_cache[cache_key] = (user, platform)

                                            # Maintain cache size
                                            if len(self.player_cache) > self.max_cache_size:
                                                self.player_cache.popitem(last=False)

                                            return user, platform
                                    except Exception as e:
                                        logger.debug(
                                            f"Platform {platform.name} invalid for membership ID {membership_id}: {str(e)}")
                                        continue
        except Exception as e:
            logger.debug(f"Error using GetMembershipsById for {membership_id}: {str(e)}")

        # If the above approach fails, fall back to trying each platform individually
        # Try each membership type
        for platform in [
            BungieMembershipType.TIGER_STEAM,  # 3
            BungieMembershipType.TIGER_XBOX,  # 1
            BungieMembershipType.TIGER_PSN,  # 2
            BungieMembershipType.TIGER_EPIC,  # 6
            BungieMembershipType.TIGER_STADIA  # 5
        ]:
            try:
                # Create a user with this platform type
                user = DestinyUser(
                    membership_id=membership_id,
                    membership_type=platform
                )

                # Try to get profile to verify if this combination is valid
                try:
                    profile = await self._rate_limited_request(
                        user.get_profile(components=["Characters"])
                    )

                    if profile and profile.characters and profile.characters.data:
                        logger.info(f"Found valid platform {platform.name} for membership ID {membership_id}")

                        # Cache the result
                        self.player_cache[cache_key] = (user, platform)

                        # Maintain cache size
                        if len(self.player_cache) > self.max_cache_size:
                            self.player_cache.popitem(last=False)

                        return user, platform
                except Exception as e:
                    logger.debug(f"Platform {platform.name} invalid for membership ID {membership_id}: {str(e)}")
                    continue

            except Exception as e:
                logger.debug(f"Error checking membership ID {membership_id} on platform {platform.name}: {str(e)}")

        logger.warning(f"Could not find valid platform for membership ID {membership_id}")
        return None, None

    async def get_seed_matches_from_cheaters(self, cheater_names: List[str]) -> List[str]:
        """Get recent match IDs from a list of known cheaters"""
        seed_matches = set()
        successful_players = 0

        # Include known PGCRs with confirmed cheaters
        known_pgcrs = ["15967863965", "15968884445"]
        seed_matches.update(known_pgcrs)
        logger.info(f"Added {len(known_pgcrs)} known PGCR IDs with confirmed cheaters")

        # Add known public cheaters - using proper enum values
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
                        seed_matches.add(str(instance_id))

                successful_players += 1
                if activities:
                    logger.info(f"Found {len(activities)} activities for known cheater {membership_id}")

            except Exception as e:
                logger.error(f"Error getting activities for known cheater {membership_id}: {str(e)}")

        # Try the list of potentially private accounts
        for name in cheater_names:
            try:
                # Find player with correct membership type
                player, membership_type = await self.search_player_by_name(name)

                if not player:
                    logger.warning(f"Could not find player: {name}")
                    continue

                successful_players += 1
                logger.info(f"Found player {name} (ID: {player.membership_id}) on platform {membership_type.name}")

                # Try getting activities with correct membership type
                activities = await self.get_player_recent_activities(
                    user=player,
                    modes=[self.game_mode_types["trials"]],
                    count=25
                )

                # Add activity IDs to seed matches
                for activity in activities:
                    instance_id = activity.get("activityDetails", {}).get("instanceId")
                    if instance_id:
                        seed_matches.add(str(instance_id))

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

    async def get_player_weapon_stats(self, user: DestinyUser, character_id: str) -> Dict[str, Any]:
        """Get detailed weapon stats for a specific character of a player"""
        try:
            headers = {"X-API-Key": self.api_key}
            membership_type_value = user.membership_type.value
            membership_id = user.membership_id

            url = f"https://www.bungie.net/Platform/Destiny2/{membership_type_value}/Account/{membership_id}/Character/{character_id}/Stats/UniqueWeapons/"

            logger.debug(f"Fetching weapon stats for character {character_id} of player {membership_id}")

            async with aiohttp.ClientSession() as session:
                async with self.request_lock:
                    # Implement rate limiting
                    now = time.time()
                    self.request_times = [t for t in self.request_times if now - t < 1.0]

                    if len(self.request_times) >= self.rate_limit:
                        wait_time = 1.0 - (now - self.request_times[0])
                        if wait_time > 0:
                            await asyncio.sleep(wait_time)

                    self.request_times.append(time.time())

                    # Make the request
                    async with session.get(url, headers=headers) as response:
                        if response.status != 200:
                            logger.warning(f"Weapon stats API returned status {response.status}")
                            return {}

                        data = await response.json()

                        if data.get("ErrorCode", 0) != 1 or not data.get("Response"):
                            logger.warning(f"No weapon stats found for character {character_id}")
                            return {}

                        return data["Response"]

        except Exception as e:
            logger.error(f"Error getting weapon stats for character {character_id}: {str(e)}")
            return {}