"""
Pokemon Gen 2 Environment (Gold/Silver/Crystal).

Extends the Gen 1 environment with:
- Gen 2 RAM address map (verified against pret pokegold/pokecrystal symbol
  tables -- `wram.sym` files generated with rgbds v1.0.4)
- (group, map) map identifiers (Gen 2 uses two bytes, not one)
- 16-badge progression (8 Johto + 8 Kanto)
- WRAM bank switching for Crystal's party data (party lives in WRAM bank 6,
  selected via SVBK at 0xFF70)

Address provenance
------------------
- Gold/Silver: `pokegold/ram/wram.asm` assembled to `wram_gold.sym`
- Crystal:     `pokecrystal/ram/wram.asm` assembled to `wram.sym`

Confirmed values (bank:addr -- bank 1 is the default WRAM bank read by
PyBoy's memory[] without extra handling):
    GOLD/SILVER                      CRYSTAL
    wMapGroup     01:d85f            01:dcda
    wMapNumber    01:d860            01:dcdb
    wYCoord       01:d861            01:dcdc
    wXCoord       01:d862            01:dcdd
    wMoney        01:d3d2            01:d873
    wJohtoBadges  01:d3db            01:d87c
    wKantoBadges  01:d3dc            01:d87d
    wPartyCount   01:d881            06:d44a   (bank 6!)
    wPartySpecies 01:d882            06:d44b
    wPartyMon1    01:d889            06:d452
    wPartyMon2    01:d8b9            06:d482   (stride 0x30)
    wPartyMon1Level   01:d8a8        06:d471
    wPartyMon1Status  01:d8a9        06:d472
    wPartyMon1HP      01:d8ab        06:d474
    wPartyMon1MaxHP   01:d8ad        06:d476
    wPartyMon1Exp     01:d891        06:d45a
    wBattleMode   01:de78            01:dec3   (nonzero = in battle)
"""

from typing import Dict, Optional, Tuple

import numpy as np

from orion.pokemon.environment import PokemonEnv
from orion.pokemon.knowledge import (
    gen2_map_name,
    gen2_map_progress,
)


class Gen2RAM:
    """Pokemon Gen 2 RAM address map, Gold/Silver vs Crystal variants."""

    # Each entry: (wram_bank, address). Bank 8+ means "default bank 1 for
    # PyBoy reads" (bank 1 is what memory[] sees during normal overworld).
    PARTY_COUNT = (1, 0xD881)  # gold/silver default; overridden for crystal
    PARTY_START = (1, 0xD882)
    PARTY_MON1 = (1, 0xD889)
    PARTY_MON2 = (1, 0xD8B9)
    PARTY_MON3 = (1, 0xD8E9)
    PARTY_STRIDE = 0x30  # party_struct size
    MON1_LEVEL_OFF = 0x1F  # bytes between wPartyMon1 and wPartyMon1Level
    MON1_STATUS_OFF = 0x20
    MON1_HP_OFF = 0x22
    MON1_MAXHP_OFF = 0x24
    MON1_EXP_OFF = 0x08  # bytes between wPartyMon1 and wPartyMon1Exp

    MAP_GROUP = (1, 0xD85F)
    MAP_NUMBER = (1, 0xD860)
    Y_COORD = (1, 0xD861)
    X_COORD = (1, 0xD862)

    MONEY_1 = (1, 0xD3D2)
    MONEY_2 = (1, 0xD3D3)
    MONEY_3 = (1, 0xD3D4)

    BADGES_JOHTO = (1, 0xD3DB)
    BADGES_KANTO = (1, 0xD3DC)

    BATTLE_MODE = (1, 0xDE78)  # nonzero while a battle is active

    # Crystal overrides (party lives in WRAM bank 6; maps/money/badges stay
    # in bank 1 but at different offsets than Gold/Silver).
    CRYSTAL = {
        "PARTY_COUNT": (6, 0xD44A),
        "PARTY_START": (6, 0xD44B),
        "PARTY_MON1": (6, 0xD452),
        "PARTY_MON2": (6, 0xD482),
        "PARTY_MON3": (6, 0xD4B2),
        "MAP_GROUP": (1, 0xDCDA),
        "MAP_NUMBER": (1, 0xDCDB),
        "Y_COORD": (1, 0xDCDC),
        "X_COORD": (1, 0xDCDD),
        "MONEY_1": (1, 0xD873),
        "MONEY_2": (1, 0xD874),
        "MONEY_3": (1, 0xD875),
        "BADGES_JOHTO": (1, 0xD87C),
        "BADGES_KANTO": (1, 0xD87D),
        "BATTLE_MODE": (1, 0xDEC3),
    }


class PokemonGen2Env(PokemonEnv):
    """
    Pokemon Gen 2 (Gold/Silver/Crystal) environment.

    Key differences from Gen 1:
    - 16 badges (8 Johto + 8 Kanto)
    - Maps identified by (group, number) pairs
    - Crystal party data lives in WRAM bank 6 (SVBK switching)
    """

    def __init__(
        self,
        rom_path: str,
        headless: bool = True,
        render_mode: Optional[str] = None,
        state_path: Optional[str] = None,
        max_steps: int = 4096,  # Longer episodes for bigger region
        reward_config: Optional[Dict[str, float]] = None,
        frame_skip: int = 8,
        persistent: bool = False,
    ):
        # Gen 2 has twice the badges of Gen 1; completion is worth more.
        default_config = {
            "badge_earned": 1000.0,
            "new_pokemon_caught": 500.0,
            "pokemon_leveled_up": 25.0,
            "new_map_entered": 100.0,
            "hp_healed": 20.0,
            "in_battle_penalty": 0.0,
            "step_penalty": -0.005,
            "talked_to_npc": 2.0,
            "menu_open_penalty": -0.02,
            "new_area_explored": 5.0,
            "new_tile_explored": 0.1,
            "money_earned": 1.0,
            "map_progress": 2000.0,  # scaled per-map below
            "kanto_unlocked": 5000.0,  # reaching the Kanto badge block
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
            frame_skip=frame_skip,
            persistent=persistent,
        )

        # Cartridge-derived variant selection (after PyBoy is up).
        title = (getattr(self._pyboy, "cartridge_title", "") or "").upper()
        self._is_crystal = "CRYSTAL" in title
        self._version = "crystal" if self._is_crystal else "gold"

        # badge_max must be set after __init__ (base defaults to 8).
        self.badge_max = 16
        # Map keys are (group << 8) | number; use the full 16-bit scale so the
        # normalized feature round-trips exactly.
        self._map_key_scale = 65535.0

    # -- variant helper -----------------------------------------------------

    def _ram(self, label: str) -> Tuple[int, int]:
        """Return (bank, addr) for a Gen 2 RAM label."""
        base = getattr(Gen2RAM, label)
        if self._is_crystal and label in Gen2RAM.CRYSTAL:
            return Gen2RAM.CRYSTAL[label]
        return base

    def _read_bank(self, label: str) -> int:
        """Read one RAM byte, switching SVBK if the label lives in a
        non-default WRAM bank (Crystal party data is in bank 6)."""
        bank, addr = self._ram(label)
        svbk = self._pyboy.memory[0xFF70]
        if svbk != bank:
            self._pyboy.memory[0xFF70] = bank
        try:
            return self._pyboy.memory[addr]
        finally:
            if svbk != bank:
                self._pyboy.memory[0xFF70] = svbk

    def _read_bank_16(self, label: str, offset: int = 0) -> int:
        """Read a 16-bit big-endian value at label + offset (SVBK-aware)."""
        bank, addr = self._ram(label)
        svbk = self._pyboy.memory[0xFF70]
        if svbk != bank:
            self._pyboy.memory[0xFF70] = bank
        try:
            return (self._pyboy.memory[addr + offset] << 8) | self._pyboy.memory[
                addr + offset + 1
            ]
        finally:
            if svbk != bank:
                self._pyboy.memory[0xFF70] = svbk

    # -- money --------------------------------------------------------------

    def _read_money(self) -> int:
        """Read the player's money (3-byte packed BCD, same as Gen 1)."""

        def bcd(b: int) -> int:
            return (b >> 4) * 10 + (b & 0x0F)

        try:
            bank, a1 = self._ram("MONEY_1")
            _, a2 = self._ram("MONEY_2")
            _, a3 = self._ram("MONEY_3")
            svbk = self._pyboy.memory[0xFF70]
            if svbk != bank:
                self._pyboy.memory[0xFF70] = bank
            try:
                m1 = self._pyboy.memory[a1]
                m2 = self._pyboy.memory[a2]
                m3 = self._pyboy.memory[a3]
            finally:
                if svbk != bank:
                    self._pyboy.memory[0xFF70] = svbk
            return bcd(m1) * 10000 + bcd(m2) * 100 + bcd(m3)
        except Exception:
            return 0

    # -- map hooks ----------------------------------------------------------

    def _map_key(self, features: np.ndarray) -> int:
        """(group << 8) | map_number, round-tripped through the 16-bit scale."""
        return int(round(float(features[4]) * 65535.0))

    def _map_info(self) -> Tuple[int, int]:
        """Return (group, number) for the current map."""
        return self._read_bank("MAP_GROUP"), self._read_bank("MAP_NUMBER")

    def _map_progress_for(self, features: np.ndarray) -> float:
        group, number = self._decompose_map_key(self._map_key(features))
        return gen2_map_progress(group, number, self._version)

    def _map_name_for(self, features: np.ndarray) -> Optional[str]:
        group, number = self._decompose_map_key(self._map_key(features))
        return gen2_map_name(group, number, self._version)

    @staticmethod
    def _decompose_map_key(key: int) -> Tuple[int, int]:
        return (key >> 8) & 0xFF, key & 0xFF

    # -- state extraction ---------------------------------------------------

    def _extract_state_features(self) -> np.ndarray:
        """Extract Gen 2 state features (same 20-dim layout as Gen 1)."""
        features = np.zeros(self.state_feature_size, dtype=np.float32)

        try:
            # Badges (0-16)
            badges_johto = bin(self._read_bank("BADGES_JOHTO")).count("1")
            badges_kanto = bin(self._read_bank("BADGES_KANTO")).count("1")
            features[0] = (badges_johto + badges_kanto) / float(self.badge_max)

            # Party count (0-6)
            party_count = min(self._read_bank("PARTY_COUNT"), 6)
            features[1] = party_count / 6.0

            # First party member HP (normalized)
            current_hp = self._read_bank_16("PARTY_MON1", Gen2RAM.MON1_HP_OFF)
            max_hp = self._read_bank_16("PARTY_MON1", Gen2RAM.MON1_MAXHP_OFF)
            if max_hp > 0:
                features[2] = min(current_hp / max_hp, 1.0)
            else:
                features[2] = 0.0

            # First party member level (0-100)
            level = self._read_party_mon_byte(0, Gen2RAM.MON1_LEVEL_OFF)
            features[3] = min(level / 100.0, 1.0)

            # Map key: (group << 8) | number, normalized to 16 bits
            group, number = self._map_info()
            map_key = (group << 8) | number
            features[4] = map_key / 65535.0

            # In battle (battle mode register is nonzero during battles)
            features[5] = 1.0 if self._read_bank("BATTLE_MODE") > 0 else 0.0

            # Player X/Y position
            features[6] = self._read_bank("X_COORD") / 255.0
            features[7] = self._read_bank("Y_COORD") / 255.0

            # Money (BCD)
            features[8] = min(self._read_money() / 999999.0, 1.0)

            # Experience of first party member
            exp = self._read_party_mon_exp(0)
            features[9] = min(exp / 1000000.0, 1.0)

            # Party levels (first 3 members)
            for i in range(min(party_count, 3)):
                lvl = self._read_party_mon_byte(i, Gen2RAM.MON1_LEVEL_OFF)
                features[10 + i] = min(lvl / 100.0, 1.0)

            # Party HP (members 2-3)
            for i in range(min(party_count, 3)):
                hp = self._read_party_mon_hp(i)
                features[13 + i] = min(hp / 999.0, 1.0)

            # Game state (party count doubles as a coarse progress flag)
            features[16] = party_count / 6.0

            # Dialogue indicator: Gen 2 exposes no single textbox byte here,
            # reuse battle/party context (keeps the vector informative).
            features[17] = 1.0 if self._read_bank("BATTLE_MODE") > 0 else 0.0

            # Step count normalized
            features[18] = min(self._step_count / self.max_steps, 1.0)

            # Visited maps count normalized
            features[19] = min(len(self._visited_maps) / 255.0, 1.0)

        except Exception:
            # If RAM reading fails, return zeros
            pass

        # Pre-world gate (mirror of the Gen 1 base class): before the world
        # initializes (boot/title/menu/intro) the party/level/HP/money RAM
        # holds volatile or uninitialized memory (observed on the Crystal
        # title screen: party=6 and levels 56/96 from garbage). The reward
        # path already uses _in_game_world(); apply it to the observation
        # features too so the policy and dashboard are not fed phantom state.
        if not self._in_game_world(features):
            features[0] = 0.0  # badges
            features[1] = 0.0  # party count
            features[2] = 0.0  # first member HP
            features[3] = 0.0  # first member level
            features[5] = 0.0  # in-battle flag
            features[6] = 0.0  # player X
            features[7] = 0.0  # player Y
            features[8] = 0.0  # money
            features[9] = 0.0  # experience
            features[10:16] = 0.0  # party levels (3) + party HP (3)
            features[16] = 0.0  # game-state byte
            features[17] = 0.0  # dialogue flag

        return features

    def _party_mon_addr(self, index: int) -> Tuple[int, int]:
        """(svbk_bank, absolute address) of party member `index` (0-based)."""
        bank, base = self._ram("PARTY_MON1")
        return bank, base + index * Gen2RAM.PARTY_STRIDE

    def _read_party_mon_byte(self, index: int, offset: int) -> int:
        bank, addr = self._party_mon_addr(index)
        svbk = self._pyboy.memory[0xFF70]
        if svbk != bank:
            self._pyboy.memory[0xFF70] = bank
        try:
            return self._pyboy.memory[addr + offset]
        finally:
            if svbk != bank:
                self._pyboy.memory[0xFF70] = svbk

    def _read_party_mon_hp(self, index: int) -> int:
        bank, addr = self._party_mon_addr(index)
        svbk = self._pyboy.memory[0xFF70]
        if svbk != bank:
            self._pyboy.memory[0xFF70] = bank
        try:
            return (self._pyboy.memory[addr + Gen2RAM.MON1_HP_OFF] << 8) | (
                self._pyboy.memory[addr + Gen2RAM.MON1_HP_OFF + 1]
            )
        finally:
            if svbk != bank:
                self._pyboy.memory[0xFF70] = svbk

    def _read_party_mon_exp(self, index: int) -> int:
        bank, addr = self._party_mon_addr(index)
        svbk = self._pyboy.memory[0xFF70]
        if svbk != bank:
            self._pyboy.memory[0xFF70] = bank
        try:
            off = Gen2RAM.MON1_EXP_OFF
            return (
                self._pyboy.memory[addr + off] * 65536
                + self._pyboy.memory[addr + off + 1] * 256
                + self._pyboy.memory[addr + off + 2]
            )
        finally:
            if svbk != bank:
                self._pyboy.memory[0xFF70] = svbk

    # -- rewards ------------------------------------------------------------

    def _compute_reward(
        self, old_state: Dict, new_state: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Gen 2 reward: same progress terms as Gen 1, with 16-badge scale
        and (group, map) story progress."""
        reward, info = super()._compute_reward(old_state, new_state)

        # Add a Kanto unlock bonus when the first Kanto badge appears.
        old_features = old_state["state"]
        new_features = new_state["state"]
        old_badges = self._badges_from(old_features)
        new_badges = self._badges_from(new_features)
        if old_badges < 8 <= new_badges:
            reward += self.reward_config.get("kanto_unlocked", 5000.0)
            info["kanto_unlocked"] = True

        return reward, info

    # -- completion ---------------------------------------------------------

    @property
    def total_badges(self) -> int:
        johto = bin(self._read_bank("BADGES_JOHTO")).count("1")
        kanto = bin(self._read_bank("BADGES_KANTO")).count("1")
        return johto + kanto

    @property
    def is_complete(self) -> bool:
        """Check if game is complete (all 16 badges)."""
        return self.total_badges >= 16

    def reset(self, seed=None, options=None):
        self._max_map_progress = 0.0
        return super().reset(seed, options)
