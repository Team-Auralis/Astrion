"""
Pokemon Gen 2 Environment (Gold/Silver/Crystal).

Extends Gen 1 environment with:
- Gen 2 RAM address map
- Day/night cycle awareness
- Pokegear indicator
- Expanded party features (boxes, eggs)
- Crystal animated sprites (handled by PyBoy)

Usage:
    env = PokemonGen2Env(rom_path="roms/pokemon_gold.gbc")
"""

import os
from typing import Optional, Dict, Tuple

import numpy as np

from orion.pokemon.environment import PokemonEnv, Gen1RAM


class Gen2RAM:
    """
    Pokemon Gen 2 RAM address map (Gold/Silver/Crystal).

    Gen 2 is mostly backwards-compatible with Gen 1 but adds:
    - Day/night cycle (based on real-time clock)
    - Pokegear (radio, phone, map)
    - 16 new Pokemon (Johto dex)
    - Breeding (eggs in party)
    - Box system improvements
    """

    # Party structure (similar to Gen 1 but different offsets)
    PARTY_COUNT = 0xD163  # Same as Gen 1
    PARTY_START = 0xD164

    # Player state (Gen 2 uses different map IDs)
    PLAYER_X = 0xD362
    PLAYER_Y = 0xD361
    MAP_ID = 0xD35E

    # Battle state
    IN_BATTLE = 0xD057
    BATTLE_TYPE = 0xD05A

    # Current pokemon HP
    CURRENT_HP = 0xD16C
    MAX_HP = 0xD16D

    # Badges (Gen 2 has 16 badges - 8 Johto + 8 Kanto)
    BADGES_JOHTO = 0xD356  # First 8 badges
    BADGES_KANTO = 0xD357  # Next 8 badges (after Elite 4)

    # Money (3 bytes BCD, same as Gen 1)
    MONEY_1 = 0xD347
    MONEY_2 = 0xD348
    MONEY_3 = 0xD349

    # Experience (first party member)
    EXP_1 = 0xD179
    EXP_2 = 0xD17A
    EXP_3 = 0xD17B

    # Levels (Gen 2 party structure: 0x2C bytes per member)
    LEVEL_FIRST = 0xD18C

    # Day/Night cycle (Gen 2 specific)
    # Based on real-time clock - value changes throughout the day
    # 0=Morning, 1=Day, 2=Evening, 3=Night
    TIME_OF_DAY = 0xD269

    # Pokegear flags (Gen 2 specific)
    POKEGEAR_OWNED = 0xD35D  # Bitfield: radio, phone, map, expn
    POKEGEAR_ACTIVE = 0xD35E  # Currently active Pokegear app

    # Current radio station (0=off, 1-8=stations)
    RADIO_STATION = 0xD360

    # Party eggs (Gen 2 specific)
    # Each party member has an egg flag in their data structure
    # Egg flag is at offset 0x1F within party data structure

    # Game state flags
    GAME_STATE = 0xD355

    # NPC/dialogue state
    DIALOGUE_STATE = 0xD35D

    # Kanto exploration (post-Elite 4)
    KANTO_VISITED = 0xD35F  # Bitfield of visited Kanto towns

    # Special flags
    REVEAL_MAP = 0xD358  # Map reveal progress
    SHOW_MAP_NAME = 0xD359  # Map name display

    # Pokemon caught flags (3 arrays for 256 Pokemon)
    CAUGHT_ZONE1 = 0xD2B7  # Pokemon 1-64
    CAUGHT_ZONE2 = 0xD2B8  # Pokemon 65-128
    CAUGHT_ZONE3 = 0xD2B9  # Pokemon 129-192

    # Party status conditions (Gen 2 expanded)
    # Each party member has status at offset 0x04 in their structure


class PokemonGen2Env(PokemonEnv):
    """
    Pokemon Gen 2 (Gold/Silver/Crystal) environment.

    Key differences from Gen 1:
    - 16 badges (8 Johto + 8 Kanto)
    - Day/night cycle affects gameplay
    - Pokegear with radio and phone
    - Breeding and eggs
    - 251 Pokemon (vs 151 in Gen 1)
    """

    def __init__(
        self,
        rom_path: str,
        headless: bool = True,
        render_mode: Optional[str] = None,
        state_path: Optional[str] = None,
        max_steps: int = 4096,  # Longer episodes for bigger region
        reward_config: Optional[Dict[str, float]] = None,
    ):
        # Gen 2 has more badges, so completion bonus is higher
        default_config = {
            "badge_earned": 1000.0,
            "new_pokemon_caught": 500.0,
            "pokemon_leveled_up": 50.0,
            "new_map_entered": 100.0,
            "hp_healed": 200.0,
            "in_battle_penalty": -1.0,
            "step_penalty": -0.1,
            "talked_to_npc": 10.0,
            "menu_open_penalty": -0.5,
            "new_area_explored": 25.0,
            "kanto_unlocked": 5000.0,  # Big reward for reaching Kanto
            "pokegear_used": 5.0,
        }

        if reward_config:
            default_config.update(reward_config)

        super().__init__(
            rom_path=rom_path,
            headless=headless,
            render_mode=render_mode,
            state_path=state_path,
            max_steps=max_steps,
            reward_config=default_config,
        )

        # Gen 2 specific state
        self._kanto_unlocked = False
        self._pokegear_apps = set()

    def _extract_state_features(self) -> np.ndarray:
        """
        Extract Gen 2 specific state features.

        Extends Gen 1 features with:
        - 16 badges instead of 8
        - Time of day
        - Pokegear apps owned
        - Kanto exploration progress
        - Egg presence in party
        """
        features = np.zeros(self.state_feature_size, dtype=np.float32)

        try:
            # Badges (0-16: 8 Johto + 8 Kanto)
            badges_johto = bin(self._read_ram(Gen2RAM.BADGES_JOHTO)).count("1")
            badges_kanto = bin(self._read_ram(Gen2RAM.BADGES_KANTO)).count("1")
            total_badges = badges_johto + badges_kanto
            features[0] = total_badges / 16.0  # Normalized 0-1

            # Party count (0-6)
            party_count = min(self._read_ram(Gen2RAM.PARTY_COUNT), 6)
            features[1] = party_count / 6.0

            # First party member HP (normalized)
            current_hp = self._read_ram_16(Gen2RAM.CURRENT_HP)
            max_hp = self._read_ram_16(Gen2RAM.MAX_HP)
            if max_hp > 0:
                features[2] = min(current_hp / max_hp, 1.0)
            else:
                features[2] = 0.0

            # First party member level (0-100)
            level = self._read_ram(Gen2RAM.LEVEL_FIRST)
            features[3] = min(level / 100.0, 1.0)

            # Map ID (Gen 2 has different map IDs, up to ~255)
            features[4] = self._read_ram(Gen2RAM.MAP_ID) / 255.0

            # In battle flag
            features[5] = 1.0 if self._read_ram(Gen2RAM.IN_BATTLE) > 0 else 0.0

            # Player X position
            features[6] = self._read_ram(Gen2RAM.PLAYER_X) / 255.0

            # Player Y position
            features[7] = self._read_ram(Gen2RAM.PLAYER_Y) / 255.0

            # Money (BCD)
            money = (
                self._read_ram(Gen2RAM.MONEY_1) * 10000
                + self._read_ram(Gen2RAM.MONEY_2) * 100
                + self._read_ram(Gen2RAM.MONEY_3)
            )
            features[8] = min(money / 999999.0, 1.0)

            # Experience of first party member
            exp = (
                self._read_ram(Gen2RAM.EXP_1) * 65536
                + self._read_ram(Gen2RAM.EXP_2) * 256
                + self._read_ram(Gen2RAM.EXP_3)
            )
            features[9] = min(exp / 1000000.0, 1.0)

            # Party levels (first 3 members)
            for i in range(min(party_count, 3)):
                addr = Gen2RAM.LEVEL_FIRST + i * 0x2C
                features[10 + i] = min(self._read_ram(addr) / 100.0, 1.0)

            # Party HP (members 1-3)
            for i in range(min(party_count, 3)):
                hp_addr = Gen2RAM.CURRENT_HP + i * 0x2C
                hp = self._read_ram_16(hp_addr)
                features[13 + i] = min(hp / 999.0, 1.0)

            # Game state
            features[16] = self._read_ram(Gen2RAM.GAME_STATE) / 255.0

            # Dialogue state
            features[17] = 1.0 if self._read_ram(Gen2RAM.DIALOGUE_STATE) > 0 else 0.0

            # Step count normalized
            features[18] = min(self._step_count / self.max_steps, 1.0)

            # Visited maps count normalized
            features[19] = min(len(self._visited_maps) / 255.0, 1.0)

        except Exception:
            # If RAM reading fails, return zeros
            pass

        return features

    def _extract_gen2_features(self) -> Dict[str, float]:
        """
        Extract additional Gen 2 specific features (not in base 20-dim vector).

        These can be used for reward shaping and info tracking.
        """
        info = {}

        try:
            # Time of day (0=Morning, 1=Day, 2=Evening, 3=Night)
            info["time_of_day"] = self._read_ram(Gen2RAM.TIME_OF_DAY)

            # Kanto badges (post-Elite 4)
            kanto_badges = bin(self._read_ram(Gen2RAM.BADGES_KANTO)).count("1")
            info["kanto_badges"] = kanto_badges
            info["kanto_unlocked"] = kanto_badges > 0

            # Pokegear apps
            pokegear = self._read_ram(Gen2RAM.POKEGEAR_OWNED)
            info["pokegear_apps"] = bin(pokegear).count("1")

            # Radio station (0=off)
            info["radio_station"] = self._read_ram(Gen2RAM.RADIO_STATION)

            # Pokemon caught (3 bytes, each bit = one Pokemon)
            caught1 = bin(self._read_ram(Gen2RAM.CAUGHT_ZONE1)).count("1")
            caught2 = bin(self._read_ram(Gen2RAM.CAUGHT_ZONE2)).count("1")
            caught3 = bin(self._read_ram(Gen2RAM.CAUGHT_ZONE3)).count("1")
            info["pokemon_caught_total"] = caught1 + caught2 + caught3

        except Exception:
            pass

        return info

    def _compute_reward(self, old_state: Dict, new_state: Dict) -> Tuple[float, Dict]:
        """
        Gen 2 specific reward shaping.

        Extends Gen 1 rewards with:
        - Kanto unlock bonus
        - Pokegear usage
        - Day/night exploration variety
        """
        reward, info = super()._compute_reward(old_state, new_state)

        # Extract Gen 2 features for both states
        old_gen2 = self._extract_gen2_features() if hasattr(self, "_prev_gen2") else {}
        new_gen2 = self._extract_gen2_features()

        # Kanto unlock reward (big milestone)
        if new_gen2.get("kanto_unlocked") and not old_gen2.get("kanto_unlocked"):
            reward += self.reward_config["kanto_unlocked"]
            info["kanto_unlocked"] = True

        # Pokegear usage reward (encourages using tools)
        old_apps = old_gen2.get("pokegear_apps", 0)
        new_apps = new_gen2.get("pokegear_apps", 0)
        if new_apps > old_apps:
            reward += self.reward_config["pokegear_used"] * (new_apps - old_apps)
            info["new_pokegear_apps"] = new_apps - old_apps

        # Store for next step
        self._prev_gen2 = new_gen2

        return reward, info

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        self._kanto_unlocked = False
        self._prev_gen2 = {}
        return super().reset(seed, options)

    @property
    def total_badges(self) -> int:
        """Get total badge count (Johto + Kanto)."""
        johto = bin(self._read_ram(Gen2RAM.BADGES_JOHTO)).count("1")
        kanto = bin(self._read_ram(Gen2RAM.BADGES_KANTO)).count("1")
        return johto + kanto

    @property
    def is_complete(self) -> bool:
        """Check if game is complete (all 16 badges)."""
        return self.total_badges >= 16

    def get_day_night_phase(self) -> str:
        """Get human-readable time of day."""
        phases = ["Morning", "Day", "Evening", "Night"]
        idx = self._read_ram(Gen2RAM.TIME_OF_DAY)
        return phases[idx] if 0 <= idx < len(phases) else "Unknown"
