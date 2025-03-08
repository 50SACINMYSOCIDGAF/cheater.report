# manifest_manager.py
import logging
import os
import json
import asyncio
import aiohttp
import sqlite3
from typing import Dict, Any, Optional, List, Set, Tuple
import time

logger = logging.getLogger("cheat_detector")


class ManifestManager:
    def __init__(self, api_key: str, manifest_path: str = "manifest.sqlite3"):
        self.api_key = api_key
        self.manifest_path = manifest_path
        self.base_url = "https://www.bungie.net"
        self.manifest_loaded = False
        self.weapon_type_map = {}
        self.game_mode_map = {}
        self.item_definitions = {}
        self.activity_definitions = {}
        self.mode_definitions = {}
        self._setup_database()

    def _setup_database(self):
        """Setup SQLite database for caching manifest data"""
        # Check if manifest file exists
        if os.path.exists(self.manifest_path):
            # Check if it's less than 24 hours old
            if time.time() - os.path.getmtime(self.manifest_path) < 24 * 60 * 60:
                logger.info(f"Using existing manifest cache: {self.manifest_path}")
                self.manifest_loaded = True
                self._load_cached_maps()
                return

        # Create new database
        self.conn = sqlite3.connect(self.manifest_path)
        cursor = self.conn.cursor()

        # Create tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS manifest_info (
            key TEXT PRIMARY KEY,
            value TEXT,
            timestamp INTEGER
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS weapon_types (
            hash INTEGER PRIMARY KEY,
            name TEXT,
            type TEXT,
            timestamp INTEGER
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS game_modes (
            hash INTEGER PRIMARY KEY,
            name TEXT,
            type TEXT,
            timestamp INTEGER
        )
        """)

        self.conn.commit()

    def _load_cached_maps(self):
        """Load cached weapon and game mode maps from database"""
        self.conn = sqlite3.connect(self.manifest_path)
        cursor = self.conn.cursor()

        # Load weapon types
        cursor.execute("SELECT hash, type FROM weapon_types")
        for hash_id, weapon_type in cursor.fetchall():
            self.weapon_type_map[hash_id] = weapon_type

        # Load game modes
        cursor.execute("SELECT hash, type FROM game_modes")
        for hash_id, mode_type in cursor.fetchall():
            self.game_mode_map[hash_id] = mode_type

        logger.info(f"Loaded {len(self.weapon_type_map)} weapons and {len(self.game_mode_map)} game modes from cache")

    async def get_manifest_urls(self) -> Dict[str, str]:
        """Get manifest URLs from Bungie API"""
        async with aiohttp.ClientSession() as session:
            headers = {"X-API-Key": self.api_key}
            async with session.get(f"{self.base_url}/Platform/Destiny2/Manifest/", headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to get manifest: {response.status}")
                    return {}

                data = await response.json()
                if data.get("ErrorCode", 0) != 1:
                    logger.error(f"API error: {data.get('ErrorStatus')} - {data.get('Message')}")
                    return {}

                return data["Response"]["jsonWorldComponentContentPaths"]["en"]

    async def _fetch_definition(self, session: aiohttp.ClientSession, url: str) -> Dict:
        """Fetch a definition JSON from Bungie"""
        async with session.get(f"{self.base_url}{url}") as response:
            if response.status != 200:
                logger.error(f"Failed to fetch definition: {response.status} - {url}")
                return {}

            return await response.json()

    async def load_manifest(self):
        """Load necessary manifest definitions"""
        if self.manifest_loaded:
            logger.info("Manifest already loaded")
            return True

        logger.info("Loading Destiny 2 manifest...")

        try:
            # Get manifest URLs
            manifest_urls = await self.get_manifest_urls()
            if not manifest_urls:
                logger.error("Failed to get manifest URLs")
                return False

            async with aiohttp.ClientSession() as session:
                # Load inventory items (for weapons)
                if "DestinyInventoryItemDefinition" in manifest_urls:
                    logger.info("Loading weapon definitions...")
                    item_url = manifest_urls["DestinyInventoryItemDefinition"]
                    self.item_definitions = await self._fetch_definition(session, item_url)

                    # Process weapons
                    cursor = self.conn.cursor()
                    for hash_id, item in self.item_definitions.items():
                        try:
                            hash_id = int(hash_id)
                            # Only process weapons
                            if item.get("itemType") == 3:  # Weapon
                                weapon_type = self._determine_weapon_type(item)
                                if weapon_type:
                                    self.weapon_type_map[hash_id] = weapon_type

                                    # Cache in database
                                    cursor.execute(
                                        "INSERT OR REPLACE INTO weapon_types (hash, name, type, timestamp) VALUES (?, ?, ?, ?)",
                                        (hash_id, item.get("displayProperties", {}).get("name", "Unknown"), weapon_type,
                                         int(time.time()))
                                    )
                        except Exception as e:
                            logger.error(f"Error processing weapon {hash_id}: {str(e)}")

                    self.conn.commit()
                    logger.info(f"Processed {len(self.weapon_type_map)} weapons")

                # Load activity modes
                if "DestinyActivityModeDefinition" in manifest_urls:
                    logger.info("Loading game mode definitions...")
                    mode_url = manifest_urls["DestinyActivityModeDefinition"]
                    self.mode_definitions = await self._fetch_definition(session, mode_url)

                    # Process game modes
                    cursor = self.conn.cursor()
                    for hash_id, mode in self.mode_definitions.items():
                        try:
                            hash_id = int(hash_id)
                            mode_name = mode.get("displayProperties", {}).get("name", "").lower()
                            mode_type = self._determine_game_mode(mode_name, mode)

                            if mode_type:
                                self.game_mode_map[hash_id] = mode_type

                                # Cache in database
                                cursor.execute(
                                    "INSERT OR REPLACE INTO game_modes (hash, name, type, timestamp) VALUES (?, ?, ?, ?)",
                                    (hash_id, mode_name, mode_type, int(time.time()))
                                )
                        except Exception as e:
                            logger.error(f"Error processing game mode {hash_id}: {str(e)}")

                    self.conn.commit()
                    logger.info(f"Processed {len(self.game_mode_map)} game modes")

            # Mark as loaded
            self.manifest_loaded = True

            # Store last update timestamp
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO manifest_info (key, value, timestamp) VALUES (?, ?, ?)",
                ("last_update", str(time.time()), int(time.time()))
            )
            self.conn.commit()

            return True

        except Exception as e:
            logger.error(f"Error loading manifest: {str(e)}")
            return False

    def _determine_weapon_type(self, item: Dict) -> Optional[str]:
        """Determine weapon type from item definition"""
        try:
            item_name = item.get("displayProperties", {}).get("name", "").lower()
            item_type = item.get("itemTypeDisplayName", "").lower()

            # Map to our standardized weapon types
            if "auto rifle" in item_type:
                return "auto_rifle"
            elif "pulse rifle" in item_type:
                return "pulse_rifle"
            elif "scout rifle" in item_type:
                return "scout_rifle"
            elif "hand cannon" in item_type:
                return "hand_cannon"
            elif "submachine gun" in item_type or "smg" in item_type:
                return "submachine_gun"
            elif "sidearm" in item_type:
                return "sidearm"
            elif "shotgun" in item_type:
                return "shotgun"
            elif "sniper rifle" in item_type:
                return "sniper_rifle"
            elif "fusion rifle" in item_type and "linear" not in item_type:
                return "fusion_rifle"
            elif "linear fusion rifle" in item_type:
                return "linear_fusion_rifle"
            elif "rocket launcher" in item_type:
                return "rocket_launcher"
            elif "grenade launcher" in item_type:
                return "grenade_launcher"
            elif "machine gun" in item_type:
                return "machine_gun"
            elif "sword" in item_type:
                return "sword"
            elif "bow" in item_type:
                return "bow"
            elif "trace rifle" in item_type:
                return "trace_rifle"
            elif "glaive" in item_type:
                return "glaive"

            return None

        except Exception as e:
            logger.error(f"Error determining weapon type: {str(e)}")
            return None

    def _determine_game_mode(self, mode_name: str, mode: Dict) -> Optional[str]:
        """Determine game mode type from mode definition"""
        try:
            # Map to our standardized game modes
            if "trials" in mode_name:
                return "trials"
            elif "competitive" in mode_name:
                return "competitive"
            elif "iron banner" in mode_name:
                return "iron_banner"
            elif "control" == mode_name:
                return "control"
            elif "rumble" == mode_name:
                return "rumble"
            elif "elimination" == mode_name:
                return "elimination"

            return None

        except Exception as e:
            logger.error(f"Error determining game mode: {str(e)}")
            return None

    def get_weapon_type(self, hash_id: int) -> str:
        """Get weapon type from hash ID"""
        return self.weapon_type_map.get(hash_id, "unknown")

    def get_game_mode(self, hash_id: int) -> str:
        """Get game mode from hash ID"""
        return self.game_mode_map.get(hash_id, "unknown")

    def get_game_mode_id(self, mode_name: str) -> Optional[int]:
        """Get game mode ID from name"""
        for hash_id, mode in self.game_mode_map.items():
            if mode == mode_name:
                return hash_id
        return None

    def is_trials_mode(self, hash_id: int) -> bool:
        """Check if mode is Trials of Osiris"""
        return self.get_game_mode(hash_id) == "trials"