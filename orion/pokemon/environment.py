"""
Pokemon Gen 1/2 Gymnasium Environment for ORION.

Wraps PyBoy emulator into a standard Gym interface with:
- 160x144 grayscale screen observation
- 8-button Game Boy action space (A, B, Start, Select, Up, Down, Left, Right)
- Milestone-based reward shaping (badges, party, levels, exploration)
- RAM-based state extraction for Pokemon-specific features
"""

import os
import hashlib
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

from orion.pokemon.knowledge import map_progress_by_id, map_name

# Game Boy screen dimensions
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 144

# Game Boy button mapping
BUTTONS = ["a", "b", "start", "select", "up", "down", "left", "right"]
NUM_ACTIONS = len(BUTTONS)

# PokeRL-style action space (RESEARCH_FASTER_TRAINING.md 4.2): Start/Select
# are pure menu spam that produce no state change, so they are dropped and
# movement buttons are tapped (press+release) so a held direction cannot
# re-engage its toggle. Kept separate from BUTTONS/NUM_ACTIONS so the
# default 8-action path (mock/evaluate/play and the supervisor fleet) is
# unchanged.
POKERL_BUTTONS = ["a", "b", "up", "down", "left", "right"]
MOVEMENT_BUTTONS = frozenset(("up", "down", "left", "right"))

# Per-step cap on the combined anti-loop penalty (pokerl_mode only). A fully
# stuck agent would otherwise stack all five layers (visit -0.05 + loop -0.2
# + pattern -0.05 + no-change up to -0.20 + spam -0.02 = ~-0.52/step at a
# 64-step identical streak) with no bound, bleeding ~-0.4..-0.5/step from the
# episode reward. Clamping the *applied* penalty at -0.5/step bounds the
# worst-case per-episode bleed while preserving each layer's signal (the info
# keys still report the raw per-layer contributions for diagnostics).
POKERL_ANTILOOP_CAP = -0.5

# PokeRL hierarchical reward table (RESEARCH_FASTER_TRAINING.md 4.3), payable
# only in pokerl_mode: micro (per-tile exploration / distance), meso (area and
# event milestones: maps, regions, grass, battle start), macro (battle win,
# catch). Default mode (pokerl_mode=False) is byte-identical -- the whole
# _pokerl_rewards block is skipped and none of these weights are consulted.
POKERL_REWARD_WEIGHTS = {
    "new_tile": 1.0,  # micro: first time this (map, x, y) is touched this episode
    "first_visit": 0.5,  # micro: tile never seen before in this env instance's life
    "distance_progress": 0.2,  # micro: L1 distance-to-goal decrease (curriculum)
    "new_map_entry": 5.0,  # meso: first entry into a map this episode
    "map_transition": 10.0,  # meso: any map_id change
    "unexplored_region": 2.0,  # meso: first map-name family this episode
    "tall_grass": 20.0,  # meso: first tall-grass step this episode (one-shot)
    "battle_start": 10.0,  # meso: in_battle rising edge
    "battle_win": 50.0,  # macro: battle ends with EXP gained since it started
    "pokemon_caught": 50.0,  # macro: party count rises (reuses info pokemon_caught)
}

# Tall-grass tile id in the Gen 1 overworld tilemap, mined from pret/pokered
# (not guessed): the overworld tileset header declares its grass tile as $52
# (data/tilesets/tileset_headers.asm, "tileset Overworld, ..., $52"), matching
# wGrassTile -- the exact byte the game compares the tile under the player's
# feet against when rolling wild encounters (engine/battle/wild_encounters.asm,
# `hlcoord 9, 9`). $52 is the only id this milestone cares about; the
# companion block-set edge tile $53 (gfx/blocksets/overworld.bst) is solid
# (not walkable) and never appears under the feet, so it is not listed.
# Viridian Forest ($20) and Indigo Plateau ($45) declare their own grass tiles
# in their tileset headers and are deliberately excluded: the meso tall-grass
# reward is the overworld "tall grass at the town edge" milestone, not the
# forest floor.
POKERL_GRASS_TILES = frozenset((0x52,))


# Pokemon Red/Blue/Yellow RAM addresses (Gen 1)
# These are well-documented in the speedrunning/emulation community
#
# NOTE (Pokemon Yellow): Yellow's WRAM layout is shifted by -1 relative to
# Red/Blue for (almost) all addresses in these regions -- confirmed against
# Data Crystal's Yellow RAM map, the pret/pokeyellow disassembly, and
# empirical value-set fingerprinting of Red vs Yellow ROMs. The environment
# applies this shift automatically via PokemonEnv._A() after detecting the
# cartridge title "POKEMON YELLOW".
class Gen1RAM:
    """Pokemon Gen 1 RAM address map for state extraction (Red/Blue base)."""

    # Party (6 pokemon, each has specific offsets)
    PARTY_COUNT = 0xD163
    PARTY_START = 0xD164  # First party pokemon species index

    # Player state
    PLAYER_X = 0xD362
    PLAYER_Y = 0xD361
    MAP_ID = 0xD35E

    # Map buffer
    TILE_MAP = 0xC4A0  # wTileMap (20x18 = 360 bytes); 0xC55D is the
    # "feet" byte (row 9, col 9), the tile under the sprite that the game
    # checks for wild-encounter grass (`hlcoord 9, 9` in TryDoWildEncounter)
    W_OVERWORLD_MAP = 0xC6E8  # wOverworldMap (block buffer, 16x15 for Pallet)

    # Battle state
    IN_BATTLE = 0xD057  # non-zero = in battle
    BATTLE_TYPE = 0xD05A

    # Current pokemon HP (active pokemon in battle or first in party)
    # Party mon struct starts at PARTY_START (0xD16B): HP at +1 (u16),
    # level at +0x21, max HP at +0x22 (verified against PyBoy's
    # GameWrapperPokemonGen1 and pret/pokered party_struct layout).
    CURRENT_HP = 0xD16C  # HP of first party member (2 bytes big-endian)
    MAX_HP = 0xD18D  # max HP of first party member (0xD16B + 0x22)

    # Badges (bitfield)
    BADGES = 0xD356  # Each bit = one badge

    # Money (3 bytes BCD)
    MONEY_1 = 0xD347
    MONEY_2 = 0xD348
    MONEY_3 = 0xD349

    # Experience of first party member (3 bytes)
    EXP_1 = 0xD179
    EXP_2 = 0xD17A
    EXP_3 = 0xD17B

    # Levels (each party member has a level at specific offset)
    LEVEL_FIRST = 0xD18C  # Level of 1st party member
    # Party structure: each pokemon block is 44 bytes starting at 0xD16B
    # Levels: offset +0x1C within each block? Actually different for Gen1.
    # Gen1 party levels are at: 0xD18C + i*0x2C for i in 0..5

    # Game state flags
    GAME_STATE = 0xD355  # General game state

    # Text box indicator (non-zero when text/menu is showing)
    TEXT_BOX = 0xD355

    # NPC/dialogue state
    DIALOGUE_STATE = 0xD35D


class PokemonEnv:
    """
    Gymnasium-compatible Pokemon Gen 1 environment using PyBoy.

    Observation: 160x144 grayscale screen (uint8) + state vector
    Action: 0-7 corresponding to Game Boy buttons
    Reward: Shaped based on milestones (badges, party, levels, map exploration)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Screen downsampling factor for the neural network
    DOWNSAMPLE = 2  # 160x144 -> 80x72

    def __init__(
        self,
        rom_path: str,
        headless: bool = True,
        render_mode: Optional[str] = None,
        state_path: Optional[str] = None,
        max_steps: int = 2048,
        reward_config: Optional[Dict[str, float]] = None,
        frame_skip: int = 8,
        persistent: bool = False,
        pokerl_mode: bool = False,
    ):
        self.rom_path = rom_path
        self.headless = headless
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.state_path = state_path
        # Frames emulated per decision. A larger skip advances more game time
        # per policy step, which is what a long-horizon game like Pokemon needs.
        self.frame_skip = max(1, int(frame_skip))
        # Persistent mode: when True, reset()/soft_reset() keep the current
        # game state instead of rewinding to the boot snapshot, so the agent
        # can actually travel across the whole game rather than replaying the
        # intro every episode.
        self.persistent = persistent
        # PokeRL-style input hygiene: drop Start/Select from the decision
        # space and tap movement buttons (see POKERL_BUTTONS/MOVEMENT_BUTTONS).
        # `pokerl_mode` only changes THIS instance; module-level BUTTONS and
        # NUM_ACTIONS stay 8 so default consumers are unaffected.
        self.pokerl_mode = pokerl_mode
        self._action_buttons = list(POKERL_BUTTONS) if pokerl_mode else list(BUTTONS)

        if not os.path.exists(rom_path):
            raise FileNotFoundError(
                f"ROM not found: {rom_path}\n"
                f"Place your Pokemon Red/Blue/Yellow ROM at this path."
            )

        # Reward configuration.
        #
        # These defaults are tuned for *actual game progress* (badges > party >
        # levels > exploration) rather than milking the old healing/step terms.
        # In particular the step penalty is tiny (a 2048-step episode costs
        # ~20 reward, not ~205) and the in-battle penalty is gone so the agent
        # is not discouraged from fighting the battles it must win.
        self.reward_config = reward_config or {
            "badge_earned": 1000.0,  # scaled by new badge count (escalating)
            "new_pokemon_caught": 500.0,
            "pokemon_leveled_up": 25.0,
            "new_map_entered": 100.0,
            "hp_healed": 20.0,
            "in_battle_penalty": 0.0,
            "step_penalty": -0.005,
            "talked_to_npc": 2.0,
            "menu_open_penalty": -0.02,
            "new_area_explored": 5.0,
            "new_tile_explored": 0.1,  # dense exploration signal
            "money_earned": 1.0,  # proxy for winning trainer battles
            "exp_gained": 0.15,  # direct battle-win signal: EXP only rises from wins
            "map_progress": 2000.0,  # advance toward later-story maps
        }

        self._pyboy = None
        self._step_count = 0
        self._prev_state = None
        self._visited_maps = set()
        self._prev_party_info = None

        # State tracking for rewards
        self._badges_seen = 0
        self._party_count = 0
        self._max_levels = [0] * 6
        self._total_reward = 0.0
        self._visited_tiles = set()  # (map_id, x, y) visited this game
        self._prev_money_raw = 0
        self._max_map_progress = 0.0  # deepest story position reached

        # PokeRL-style anti-loop / anti-spam bookkeeping (used only in
        # pokerl_mode; see _pokerl_antiloop / RESEARCH_FASTER_TRAINING.md
        # 4.1/4.2). Cleared on every episode boundary (reset/soft_reset).
        self._tile_visit_counts = {}  # (map, x, y) -> visits this episode
        self._position_history = deque(maxlen=32)  # recent (map, x, y)
        self._action_history = deque(maxlen=8)  # recent action indices
        self._no_change_count = 0  # consecutive decisions with no state change
        self._last_position = None  # last (map, x, y, in_battle)

        # PokeRL hierarchical-reward bookkeeping (pokerl_mode only; see
        # POKERL_REWARD_WEIGHTS and _pokerl_rewards). The _ep_* episode sets
        # are cleared on every episode boundary (reset/soft_reset);
        # _lifetime_tiles persists for the life of THIS env instance -- in a
        # long-running fleet "first-ever visit" therefore spans every episode
        # that this worker env plays, which is the intended meaning per
        # RESEARCH_FASTER_TRAINING.md 4.3. The battle EXP snapshot tracks the
        # game's battle state (which can legitimately span a soft_reset
        # boundary), so it is not episode-cleared either.
        self._ep_tiles = set()  # (map, x, y) tiles touched this episode
        self._lifetime_tiles = set()  # (map, x, y) tiles ever seen (this env)
        self._ep_maps = set()  # map ids entered this episode
        self._ep_map_families = set()  # map-name families entered this episode
        self._ep_grass_paid = False  # tall_grass one-shot paid this episode (C1)
        self._battle_exp_at_start = None  # first-member EXP at battle start
        # Curriculum goal for the distance-to-goal micro reward. NEVER
        # hard-coded to a route: supplied by the operator through the config
        # dict (reward_config["goal"] = (map_id, x, y)) or per-episode
        # reset(options={"goal": ...}). None = the distance term is 0. The
        # config goal is validated here (before _init_pyboy, so a malformed
        # config fails fast without booting or leaking PyBoy); a reset that
        # omits `goal` restores this default rather than leaking the previous
        # override (W4).
        self._default_goal = self._validate_goal(
            self.reward_config.get("goal"), "reward_config['goal']"
        )
        self._goal = self._default_goal

        # Generation parameters. Gen 1 has 8 badges and map ids keyed 0..255;
        # Gen 2 overrides these (16 badges, (group,map) pairs).
        self.badge_max = 8
        self._map_key_scale = 255.0

        self._init_pyboy()

    # -- Generation hooks ---------------------------------------------------
    # These let a subclass (e.g. Gen 2) change how badges / maps are read from
    # the normalized state vector without touching the reward logic.

    def _badges_from(self, features: np.ndarray) -> int:
        return int(round(float(features[0]) * self.badge_max))

    def _map_key(self, features: np.ndarray) -> int:
        """Return a compact integer key for the current map."""
        return int(round(float(features[4]) * self._map_key_scale))

    def _map_progress_for(self, features: np.ndarray) -> float:
        """Story-progress fraction (0..1) for the map in `features`."""
        return map_progress_by_id(self._map_key(features))

    def _map_name_for(self, features: np.ndarray) -> Optional[str]:
        """Human-readable name for the map in `features`."""
        return map_name(self._map_key(features))

    def _in_game_world(self, features: np.ndarray) -> bool:
        """True when the game is in a real world state (shaped rewards eligible).

        Before the world initializes (boot logos, title screen, clock set,
        intro) the map/party/money/level RAM addresses hold uninitialized or
        volatile memory. Without this gate those phantom values get shaped
        rewards -- observed on the Gen 2 Crystal title screen (party=6,
        lvl=56, reward ~900k over 24 rollouts) because its bank-6 party
        reads oscillate. A map is "real" when the knowledge tables can name
        it; garbage groups (e.g. 195) are not nameable.
        """
        try:
            return self._map_name_for(features) is not None
        except Exception:
            # A hook failure should not silently kill shaped rewards in an
            # unknown generation; fall back to allowing them (old behavior).
            return True

    def _pokerl_controllable_world(self, features: np.ndarray) -> bool:
        """True when the game is in a controllable overworld state (W5).

        Boot logos / title / clock set / intro read map id 0 (PALLET_TOWN, a
        nameable map) with x/y = (0,0), so `_in_game_world` alone lets phantom
        shaped rewards through. The intro sequence never reads a non-origin
        player position before the first controllable map loads
        (REDS_HOUSE_1F at (2,3)), and every real map keeps tile (0,0) inside
        its collision border, so requiring a non-origin position rejects the
        zeroed boot bootstrap cheaply -- reusing the existing map-reading
        signal rather than inventing a new boot heuristic.
        """
        if not self._in_game_world(features):
            return False
        try:
            x = int(round(float(features[6]) * 255))
            y = int(round(float(features[7]) * 255))
        except Exception:
            return False
        return not (x == 0 and y == 0)

    @staticmethod
    def _validate_goal(goal, key):
        """Validate a curriculum goal, returning a normalized tuple or None.

        A goal is either None (unset: the distance-to-goal term is 0) or a
        (map_id, x, y) tuple/list of three integers. Anything else raises
        ValueError naming the offending config key, so a typo fails loudly at
        construction/reset instead of silently disabling the term (W4).
        """
        if goal is None:
            return None
        if (
            not isinstance(goal, (tuple, list))
            or len(goal) != 3
            or not all(isinstance(v, (int, np.integer)) for v in goal)
        ):
            raise ValueError(
                f"{key} must be None or a (map_id, x, y) tuple of 3 integers; "
                f"got {goal!r}"
            )
        return (int(goal[0]), int(goal[1]), int(goal[2]))

    def _init_pyboy(self):
        import io
        from pyboy import PyBoy

        if self.headless:
            self._pyboy = PyBoy(self.rom_path, window="null", sound_emulated=False)
        else:
            self._pyboy = PyBoy(self.rom_path, window="SDL2", sound_emulated=False)

        self._pyboy.set_emulation_speed(0)  # Unlimited speed

        # Pokemon Yellow's WRAM is shifted by -1 vs Red/Blue (see Gen1RAM note).
        title = getattr(self._pyboy, "cartridge_title", "") or ""
        self._is_yellow = "YELLOW" in title.upper()
        self._ram_offset = -1 if self._is_yellow else 0

        # Load state if provided
        if self.state_path and os.path.exists(self.state_path):
            with open(self.state_path, "rb") as f:
                self._pyboy.load_state(f)
            # Snapshot the loaded state so reset() returns here
            buf = io.BytesIO()
            self._pyboy.save_state(buf)
            self._initial_state = buf.getvalue()
        else:
            # Boot the ROM past the Game Boy logo / intro to a stable state
            for _ in range(600):  # ~10 seconds at 60fps
                self._pyboy.tick()
            # Snapshot this boot state so every reset() starts from here
            buf = io.BytesIO()
            self._pyboy.save_state(buf)
            self._initial_state = buf.getvalue()

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        """Downsampled screen shape: (C, H, W)."""
        h = SCREEN_HEIGHT // self.DOWNSAMPLE
        w = SCREEN_WIDTH // self.DOWNSAMPLE
        return (1, h, w)

    @property
    def state_feature_size(self) -> int:
        """Size of the extracted state feature vector."""
        return 20  # badges, party_count, party_hp, levels, map_id, etc.

    @property
    def observation_space(self):
        import gymnasium as gym

        h = SCREEN_HEIGHT // self.DOWNSAMPLE
        w = SCREEN_WIDTH // self.DOWNSAMPLE
        return gym.spaces.Dict(
            {
                "screen": gym.spaces.Box(low=0, high=255, shape=(h, w), dtype=np.uint8),
                "state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.state_feature_size,),
                    dtype=np.float32,
                ),
            }
        )

    @property
    def action_space(self):
        import gymnasium as gym

        return gym.spaces.Discrete(len(self._action_buttons))

    def _read_ram(self, addr: int) -> int:
        return self._pyboy.memory[addr]

    def _read_ram_16(self, addr: int) -> int:
        return (self._pyboy.memory[addr] << 8) | self._pyboy.memory[addr + 1]

    def _A(self, addr: int) -> int:
        """Resolve a Red/Blue base RAM address for the loaded ROM.

        Returns addr unchanged for Red/Blue, addr-1 for Yellow.
        """
        return addr + getattr(self, "_ram_offset", 0)

    def _money_addrs(self) -> Tuple[int, int, int]:
        """Return the three packed-BCD money address bytes for this game."""
        return (
            self._A(Gen1RAM.MONEY_1),
            self._A(Gen1RAM.MONEY_2),
            self._A(Gen1RAM.MONEY_3),
        )

    def _read_money(self) -> int:
        """Read the player's money (stored as packed BCD in Gen 1)."""

        def bcd(b: int) -> int:
            return (b >> 4) * 10 + (b & 0x0F)

        try:
            m1, m2, m3 = self._money_addrs()
            m1 = self._read_ram(m1)
            m2 = self._read_ram(m2)
            m3 = self._read_ram(m3)
            return bcd(m1) * 10000 + bcd(m2) * 100 + bcd(m3)
        except Exception:
            return 0

    def _extract_state_features(self) -> np.ndarray:
        """Extract Pokemon-specific state features from emulator RAM."""
        features = np.zeros(self.state_feature_size, dtype=np.float32)

        try:
            # Badges (0-8)
            badges_byte = self._read_ram(self._A(Gen1RAM.BADGES))
            badge_count = bin(badges_byte).count("1")
            features[0] = badge_count / float(self.badge_max)  # Normalized 0-1

            # Party count (0-6)
            party_count = min(self._read_ram(self._A(Gen1RAM.PARTY_COUNT)), 6)
            features[1] = party_count / 6.0

            # First party member HP (normalized)
            current_hp = self._read_ram_16(self._A(Gen1RAM.CURRENT_HP))
            max_hp = self._read_ram_16(self._A(Gen1RAM.MAX_HP))
            if max_hp > 0:
                features[2] = min(current_hp / max_hp, 1.0)
            else:
                features[2] = 0.0

            # First party member level (0-100)
            level = self._read_ram(self._A(Gen1RAM.LEVEL_FIRST))
            features[3] = min(level / 100.0, 1.0)

            # Map ID (normalized)
            features[4] = self._read_ram(self._A(Gen1RAM.MAP_ID)) / 255.0

            # In battle flag
            features[5] = 1.0 if self._read_ram(self._A(Gen1RAM.IN_BATTLE)) > 0 else 0.0

            # Player X position
            features[6] = self._read_ram(self._A(Gen1RAM.PLAYER_X)) / 255.0

            # Player Y position
            features[7] = self._read_ram(self._A(Gen1RAM.PLAYER_Y)) / 255.0

            # Money (packed BCD, up to 999999)
            features[8] = min(self._read_money() / 999999.0, 1.0)

            # Experience of first party member
            exp = (
                self._read_ram(self._A(Gen1RAM.EXP_1)) * 65536
                + self._read_ram(self._A(Gen1RAM.EXP_2)) * 256
                + self._read_ram(self._A(Gen1RAM.EXP_3))
            )
            features[9] = min(exp / 1000000.0, 1.0)

            # Party levels (first 3 members)
            for i in range(min(party_count, 3)):
                addr = self._A(Gen1RAM.LEVEL_FIRST + i * 0x2C)
                features[10 + i] = min(self._read_ram(addr) / 100.0, 1.0)

            # Party HP (members 2 and 3, normalized by estimated max)
            # Simplified: just store raw HP normalized
            for i in range(min(party_count, 3)):
                hp_addr = self._A(Gen1RAM.CURRENT_HP + i * 0x2C)
                hp = self._read_ram_16(hp_addr)
                features[13 + i] = min(hp / 999.0, 1.0)

            # Game state
            features[16] = self._read_ram(self._A(Gen1RAM.GAME_STATE)) / 255.0

            # Dialogue state
            features[17] = (
                1.0 if self._read_ram(self._A(Gen1RAM.DIALOGUE_STATE)) > 0 else 0.0
            )

            # Step count normalized
            features[18] = min(self._step_count / self.max_steps, 1.0)

            # Visited maps count normalized
            features[19] = min(len(self._visited_maps) / 255.0, 1.0)

        except Exception:
            # If RAM reading fails (wrong game, etc.), return zeros
            pass

        # Pre-world gate: before the world initializes (boot logos, title,
        # clock set, intro) the party/level/money/HP RAM holds volatile or
        # uninitialized memory (observed on the Crystal title screen:
        # party=6, lvl=96). The reward path already applies this gate; apply
        # it to the observation features too so the policy and the dashboard
        # are not fed phantom game state.
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

    def _get_screen(self) -> np.ndarray:
        """Get current screen as grayscale numpy array."""
        # PyBoy 2.x: screen.ndarray returns (144, 160, 4) RGBA
        screen = self._pyboy.screen.ndarray
        if screen.ndim == 3:
            # Convert RGBA to grayscale
            gray = (
                screen[:, :, 0] * 0.299
                + screen[:, :, 1] * 0.587
                + screen[:, :, 2] * 0.114
            ).astype(np.uint8)
        else:
            gray = screen.astype(np.uint8)

        # Downsample by DOWNSAMPLE factor
        if self.DOWNSAMPLE > 1:
            h, w = gray.shape
            new_h, new_w = h // self.DOWNSAMPLE, w // self.DOWNSAMPLE
            gray = gray[: new_h * self.DOWNSAMPLE, : new_w * self.DOWNSAMPLE]
            gray = (
                gray.reshape(new_h, self.DOWNSAMPLE, new_w, self.DOWNSAMPLE)
                .mean(axis=(1, 3))
                .astype(np.uint8)
            )

        return gray

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "screen": self._get_screen(),
            "state": self._extract_state_features(),
        }

    def _compute_reward(
        self, old_state: Dict, new_state: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Compute shaped reward based on state changes."""
        reward = 0.0
        info = {}

        old_features = old_state["state"]
        new_features = new_state["state"]

        # Shaped bonuses only apply while the game is actually in a real
        # world map. Before the world loads (boot/title/clock/intro) the RAM
        # reads are garbage; rewarding them made the Gen 2 Crystal trainer
        # chase phantom "progress" (party=6, lvl=56, ~900k reward while
        # still on the title screen). Out of world, only the step penalty
        # applies (the observed ~-10/rollout baseline).
        if not (
            self._in_game_world(old_features) and self._in_game_world(new_features)
        ):
            info["not_in_world"] = True
            reward += self.reward_config["step_penalty"]
            return reward, info

        # Badge reward (escalating: later badges are worth more because they
        # require progressively more of the game to have been completed).
        old_badges = self._badges_from(old_features)
        new_badges = self._badges_from(new_features)
        if new_badges > old_badges:
            gained = new_badges - old_badges
            reward += self.reward_config["badge_earned"] * gained * new_badges
            info["badges_earned"] = gained

        # Level up reward
        old_level = old_features[3] * 100
        new_level = new_features[3] * 100
        if new_level > old_level:
            levels_gained = int(new_level - old_level)
            reward += self.reward_config["pokemon_leveled_up"] * levels_gained
            info["levels_gained"] = levels_gained

        # New map exploration reward
        old_map = self._map_key(old_features)
        new_map = self._map_key(new_features)
        if new_map != old_map and new_map not in self._visited_maps:
            self._visited_maps.add(new_map)
            reward += self.reward_config["new_map_entered"]
            info["new_map"] = new_map

        # HP healing reward
        old_hp = old_features[2]
        new_hp = new_features[2]
        if new_hp > old_hp + 0.05:  # Significant healing
            reward += self.reward_config["hp_healed"]

        # Party growth reward
        old_party = int(old_features[1] * 6)
        new_party = int(new_features[1] * 6)
        if new_party > old_party:
            reward += self.reward_config["new_pokemon_caught"] * (new_party - old_party)
            info["pokemon_caught"] = new_party - old_party

        # Battle penalty (encourages progress over grinding)
        if new_features[5] > 0:
            reward += self.reward_config["in_battle_penalty"]

        # Step penalty (encourages efficiency)
        reward += self.reward_config["step_penalty"]

        # Dialogue reward (encourages talking to NPCs for info)
        if new_features[17] > 0 and old_features[17] == 0:
            reward += self.reward_config["talked_to_npc"]

        # Dense exploration bonus: reaching a new (map, x, y) tile. The
        # milestone rewards are far too sparse in a game this long, so this
        # provides the gradient that actually points "outward".
        try:
            tile = (
                self._map_key(new_features),
                int(round(new_features[6] * 255)),
                int(round(new_features[7] * 255)),
            )
            if tile not in self._visited_tiles:
                self._visited_tiles.add(tile)
                reward += self.reward_config["new_tile_explored"]
        except Exception:
            pass

        # Money gained. Money only rises when the player wins a trainer battle
        # or sells an item, so a positive delta is a decent proxy for having
        # actually won a fight (which has no direct RAM signal).
        money = self._read_money()
        if money > self._prev_money_raw:
            reward += self.reward_config["money_earned"] * (
                (money - self._prev_money_raw) / 100.0
            )
        self._prev_money_raw = money

        # EXP gained. Gen 1 EXP only increases when a battle is WON (each
        # surviving party member earns EXP from a defeated opponent; running
        # or catching awards none). This is the dense "you won this fight"
        # signal that was missing: `money_earned` only fires for trainer
        # battles, and the tile/map exploration bonuses actively discourage
        # committing to wild encounters -- which is why the fleet stalled at
        # Lv 6 in Oak's Lab. Paying per EXP point turns the level counter on.
        old_exp = float(old_features[9]) * 1000000.0
        new_exp = float(new_features[9]) * 1000000.0
        if new_exp > old_exp:
            gained = int(new_exp - old_exp)
            reward += self.reward_config["exp_gained"] * gained
            info["exp_gained"] = gained

        # Story-progress reward. Using the pret disassembly's map table we can
        # score how far into the intended progression each map sits, and pay
        # the agent the *first* time it pushes that frontier further. This is
        # monotonic (income never repeats), so it cannot be farmed.
        try:
            progress = self._map_progress_for(new_features)
            if progress > self._max_map_progress:
                reward += self.reward_config["map_progress"] * (
                    progress - self._max_map_progress
                )
                info["story_progress"] = round(progress, 3)
                self._max_map_progress = progress
        except Exception:
            pass

        # -- PokeRL hierarchical rewards (pokerl_mode only) -----------------
        # RESEARCH_FASTER_TRAINING.md 4.3: dense micro (tiles / distance),
        # meso (map entry / transition / region / tall grass / battle start)
        # and macro (win / catch) rewards layered on top of the milestone
        # shaping above; EXP stays as a sub-signal (the block above). All of
        # it is gated on pokerl_mode so the default mode is byte-identical.
        # The in-world gate at the top already returns early for off-world
        # steps; the micro tile terms are additionally gated on being in a
        # controllable overworld state (_pokerl_controllable_world) so the
        # boot bootstrap (map 0, x/y = (0,0)) can no longer pay a phantom
        # spawn-tile +1.5 before the intro (W5).
        if self.pokerl_mode:
            reward, info = self._pokerl_rewards(
                old_features, new_features, reward, info
            )

        return reward, info

    # -- PokeRL anti-loop / anti-spam (pokerl_mode only) -------------------

    def _clear_antiloop(self):
        """Reset per-episode PokeRL bookkeeping.

        Called from reset() and soft_reset() -- both are real episode
        boundaries (soft_reset starts each rollout in persistent mode).
        Clears the anti-loop / anti-spam state and the hierarchical-reward
        episode sets (_ep_tiles/_ep_maps/_ep_map_families) plus the one-shot
        tall-grass flag (_ep_grass_paid). _lifetime_tiles and
        _battle_exp_at_start are deliberately NOT cleared: the first spans the
        env instance's life, and the battle snapshot tracks the game's battle
        state which can legitimately span a soft_reset.
        """
        self._tile_visit_counts = {}
        self._position_history.clear()
        self._action_history.clear()
        self._no_change_count = 0
        self._last_position = None
        self._ep_tiles = set()
        self._ep_maps = set()
        self._ep_map_families = set()
        self._ep_grass_paid = False

    def _seed_episode_start(self, features: np.ndarray) -> None:
        """Seed the per-episode map/family sets with the episode's start map.

        Called from reset()/soft_reset() (real episode boundaries) so the map
        the agent is already standing on when an episode begins counts as
        already entered: leaving and re-entering it pays the +10 transition
        only, not the +5 first-entry / +2 unexplored-region terms (W3).
        Deliberately separate from _clear_antiloop(): direct _clear_antiloop()
        calls (diagnostics) keep empty-set semantics. No-op unless
        pokerl_mode, so the default reward path is untouched.
        """
        if not self.pokerl_mode:
            return
        try:
            start_map = self._map_key(features)
        except Exception:
            return
        self._ep_maps.add(start_map)
        self._ep_map_families.add(self._map_family_for(start_map))

    # -- PokeRL hierarchical rewards (pokerl_mode only) --------------------

    def _map_family_for(self, map_key: int) -> str:
        """Return the 'region family' for a map id, derived from the map-name
        string conventions in knowledge.py (pret disassembly). Maps that share
        a family are one 'unexplored region' for the meso +2 reward: all
        ROUTE_* maps are one family, all *_CITY, *_TOWN, *_GYM, *_LAB, ... are
        their own families. Also true for the game's themed interiors
        (towers, caves, tunnels, mansions). Unknown ids (garbage maps) get a
        unique UNKNOWN_<id> family so they can never collide with a real one.
        """
        name = map_name(map_key)
        if name is None:
            return "UNKNOWN_%d" % int(map_key)
        for needle, family in (
            ("ROUTE_", "ROUTE"),
            ("_CITY", "CITY"),
            ("_TOWN", "TOWN"),
            ("_GYM", "GYM"),
            ("_LAB", "LAB"),
            ("_HOUSE", "HOUSE"),
            ("TOWER", "TOWER"),
            ("_GATE", "GATE"),
            ("FOREST", "FOREST"),
            ("CAVE", "CAVE"),
            ("TUNNEL", "TUNNEL"),
            ("MANSION", "MANSION"),
            ("PLATEAU", "PLATEAU"),
        ):
            if name.startswith(needle) or needle in name:
                return family
        return name

    def _pokerl_standing_on_grass(self, features: np.ndarray) -> bool:
        """True when the player's feet tile is tall grass (POKERL_GRASS_TILES).

        Reads the same RAM byte the game itself checks for wild encounters:
        wTileMap row 9, col 9 (0xC55D, the tile under the player's sprite;
        `hlcoord 9, 9` in TryDoWildEncounter). This is game-authoritative --
        unlike the VRAM map buffer, it reflects the real tile under the feet,
        not a camera-clamped approximation. Only $52 is in
        POKERL_GRASS_TILES: the overworld tileset's wGrassTile, the id the
        game compares against for wild encounters and the one that appears
        under the player's feet on real overworld grass. Forest ($20) and
        plateau ($45) declare their own grass tiles and are deliberately
        excluded: the meso tall-grass reward is the overworld 'tall grass at
        the town edge' milestone, not the forest floor. False when the RAM
        read fails (e.g. during battle, when the overworld tilemap is hidden).
        """
        try:
            feet = self._read_ram(self._A(Gen1RAM.TILE_MAP) + 9 * 20 + 9)
            return int(feet) in POKERL_GRASS_TILES
        except Exception:
            return False

    def _pokerl_rewards(
        self,
        old_features: np.ndarray,
        new_features: np.ndarray,
        reward: float,
        info: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """Apply the PokeRL hierarchical rewards (RESEARCH_FASTER_TRAINING.md
        4.3) on top of the existing shaped reward. Only called from
        _compute_reward when pokerl_mode is on; the in-world gate has already
        returned for off-world steps. Every fired term reports its contribution
        through info (same convention as _pokerl_antiloop), so the sum of the
        new keys equals the reward delta added here.
        """
        w = POKERL_REWARD_WEIGHTS
        old_map = self._map_key(old_features)
        new_map = self._map_key(new_features)
        nx = int(round(new_features[6] * 255))
        ny = int(round(new_features[7] * 255))

        # --- micro: new tile / first-ever visit ---------------------------
        # New tile +1.0 the first time (map,x,y) is touched this episode, then
        # +0.5 extra the first time it is ever seen by this env instance.
        # Gated on a controllable overworld state (W5): at boot the map reads
        # 0 (PALLET_TOWN, nameable) with x/y = (0,0), which would otherwise
        # pay the phantom spawn tile (0,0,0) +1.5 on the first pre-world step.
        if self._pokerl_controllable_world(new_features):
            try:
                tile = (new_map, nx, ny)
                if tile not in self._ep_tiles:
                    self._ep_tiles.add(tile)
                    reward += w["new_tile"]
                    info["new_tile"] = w["new_tile"]
                    if tile not in self._lifetime_tiles:
                        self._lifetime_tiles.add(tile)
                        reward += w["first_visit"]
                        info["first_visit"] = w["first_visit"]
            except Exception:
                pass

        # --- micro: distance-to-goal --------------------------------------
        # +0.2 * (previous L1 distance - current L1 distance), i.e. positive
        # when moving closer to the configured goal. Only meaningful on the
        # goal's own map (per-map x/y tile coords are not comparable across
        # maps), and only when a goal is configured -- never hard-coded.
        if self._goal is not None:
            gmap, gx, gy = self._goal
            if old_map == gmap and new_map == gmap:
                ox = int(round(old_features[6] * 255))
                oy = int(round(old_features[7] * 255))
                prev_d = abs(ox - gx) + abs(oy - gy)
                cur_d = abs(nx - gx) + abs(ny - gy)
                progress = prev_d - cur_d
                if progress:
                    gained = w["distance_progress"] * progress
                    reward += gained
                    info["distance_progress"] = round(gained, 4)

        # --- meso: map transition / first entry / unexplored region -------
        # Any map_id change = +10; first entry into a map this episode = +5
        # (so first entry totals +15); first map-name family this episode = +2.
        if new_map != old_map:
            reward += w["map_transition"]
            info["map_transition"] = w["map_transition"]
            if new_map not in self._ep_maps:
                self._ep_maps.add(new_map)
                reward += w["new_map_entry"]
                info["new_map_entry"] = w["new_map_entry"]
                family = self._map_family_for(new_map)
                if family not in self._ep_map_families:
                    self._ep_map_families.add(family)
                    reward += w["unexplored_region"]
                    info["unexplored_region"] = w["unexplored_region"]

        # --- meso: tall grass (one-shot) ----------------------------------
        # +20 the first step this episode whose feet tile (the same RAM byte
        # the game checks for wild encounters) is tall grass, then 0 for the
        # rest of the episode: per RESEARCH_FASTER_TRAINING.md 4.3 the meso
        # milestones are reached once, so standing on grass must not pay per
        # step. Later grass steps still report tall_grass (0.0) so the sum of
        # the info keys always equals the reward added here (C1). Suppressed
        # during battle: battles are priced by their own start/win terms and
        # the battle screen hides the overworld tilemap anyway.
        new_battle = int(new_features[5] > 0)
        if new_battle == 0 and self._pokerl_standing_on_grass(new_features):
            if not self._ep_grass_paid:
                self._ep_grass_paid = True
                reward += w["tall_grass"]
                info["tall_grass"] = w["tall_grass"]
            else:
                info["tall_grass"] = 0.0

        # --- meso/macro: battle start / win -------------------------------
        # Start (rising edge) +10 and snapshot first-member EXP. End (falling
        # edge) +50 only when EXP rose since the snapshot -- the Gen 1 signal
        # that the battle was actually won (EXP only comes from wins).
        old_battle = int(old_features[5] > 0)
        new_exp = float(new_features[9]) * 1000000.0
        if new_battle and not old_battle:
            reward += w["battle_start"]
            info["battle_start"] = w["battle_start"]
            self._battle_exp_at_start = new_exp
        elif not new_battle and old_battle:
            won = (
                self._battle_exp_at_start is not None
                and new_exp > self._battle_exp_at_start
            )
            self._battle_exp_at_start = None
            if won:
                reward += w["battle_win"]
                info["battle_win"] = w["battle_win"]
        elif not new_battle:
            # Steady out-of-battle: keep no stale snapshot around.
            self._battle_exp_at_start = None

        # --- macro: catch (+50 when the party count rises) ----------------
        # Shares the existing info["pokemon_caught"] key (party delta) set by
        # the party-growth block above; only the extra reward is pokerl-only.
        old_party = int(old_features[1] * 6)
        new_party = int(new_features[1] * 6)
        if new_party > old_party:
            reward += w["pokemon_caught"] * (new_party - old_party)

        return reward, info

    def _pokerl_antiloop(
        self, action: int, features: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Apply PokeRL-style anti-loop + anti-spam penalties.

        Ported from PokeRL (RESEARCH_FASTER_TRAINING.md 4.1/4.2): three
        additive anti-loop layers -- position-visit, position-loop, and
        repeated-action / no-state-change -- plus an anti-spam streak
        penalty and a small diversity bonus for varying actions. Updates
        the per-episode bookkeeping and returns (penalty, info) for the
        current step. Returns (0.0, {}) outside pokerl_mode so the default
        reward path is byte-identical.
        """
        if not self.pokerl_mode:
            return 0.0, {}

        penalty = 0.0
        info: Dict[str, float] = {}

        try:
            tile = (
                self._map_key(features),
                int(round(features[6] * 255)),
                int(round(features[7] * 255)),
            )
            in_battle = int(features[5] > 0) if len(features) > 5 else 0
        except Exception:
            tile, in_battle = None, 0

        # W1: suppress anti-loop penalties during combat. Any battle keeps
        # map/x/y (and in_battle) constant, so the position-visit, position-
        # loop and no-change layers would all fire while the agent is
        # legitimately fighting. Battles are already rewarded through the
        # EXP/money win signals in _compute_reward and must not be double-
        # penalized, so the whole block (spam/diversity included) is skipped
        # while in_battle -- this also guarantees the no-change counter cannot
        # escalate on a fight that legitimately takes >8 decisions.
        if in_battle:
            return 0.0, {}

        # W2/W5: gate anti-loop on being in a controllable overworld state --
        # the same map-reading signal _compute_reward uses, plus the x/y !=
        # (0,0) boot check. During boot/title/clock/intro the map reads 0
        # (PALLET_TOWN, nameable) with x/y = (0,0), so _in_game_world alone
        # would let the visit/loop/no-change layers charge a large fixed
        # negative per-step baseline (~-0.2..-0.4) on every pre-world decision
        # of a non-persistent episode. Bookkeeping is deliberately NOT updated
        # before the world is controllable: such a tile key would carry an
        # unnameable/garbage map that can never collide with a real world
        # tile, so skipping is equivalent to updating with garbage but keeps
        # the position/action history scoped to actual world play. Penalties
        # are 0 and no info keys are emitted (the merged reward_info still
        # reports not_in_world from _compute_reward; no second convention is
        # introduced).
        if not self._pokerl_controllable_world(features):
            return 0.0, {}

        # Layer 1: position-visit penalty. A tile visited more than 3 times
        # in an episode is being camped; more than 5 is a hard stall. The
        # penalty is charged per step while the agent sits in the tile.
        if tile is not None:
            self._tile_visit_counts[tile] = self._tile_visit_counts.get(tile, 0) + 1
            visits = self._tile_visit_counts[tile]
            if visits > 5:
                penalty -= 0.05
                info["visit_penalty"] = -0.05
            elif visits > 3:
                penalty -= 0.01
                info["visit_penalty"] = -0.01

        # Layer 2: position-loop detection. The current tile appearing >= 6
        # times in the recent 32-position window means the agent is orbiting
        # a single spot rather than exploring.
        if tile is not None:
            self._position_history.append(tile)
            if self._position_history.count(tile) >= 6:
                penalty -= 0.2
                info["loop_penalty"] = -0.2

        # Layer 3: repeated action / no state change. Identical last-8
        # actions is a degenerate policy; so is a run of decisions where
        # map/x/y/in_battle never changes. The no-change penalty escalates
        # at every doubling of the streak: 8 -> -0.05, 16 -> -0.10,
        # 32 -> -0.15, ... (per step while the streak holds).
        self._action_history.append(action)
        recent = list(self._action_history)
        if len(recent) >= 8 and all(a == action for a in recent):
            penalty -= 0.05
            info["pattern_penalty"] = info.get("pattern_penalty", 0.0) - 0.05

        if tile is not None:
            sig = (tile[0], tile[1], tile[2], in_battle)
        else:
            sig = None
        if sig is not None and sig == self._last_position:
            self._no_change_count += 1
        else:
            self._no_change_count = 1
            self._last_position = sig
        if self._no_change_count >= 8:
            level = self._no_change_count.bit_length() - 3  # 8->1, 16->2, ...
            penalty -= 0.05 * level
            info["pattern_penalty"] = info.get("pattern_penalty", 0.0) - 0.05 * level

        # Anti-spam streak (same action 5x in a row) + diversity bonus
        # (a different action from the previous step is rewarded). The
        # reward sign matters: these cancel the lock-in pressure that
        # pushes a policy to repeat one button forever.
        if len(recent) >= 5 and all(a == action for a in recent[-5:]):
            penalty -= 0.02
            info["spam_penalty"] = -0.02
        if len(recent) >= 2 and recent[-1] != recent[-2]:
            penalty += 0.01
            info["diversity_bonus"] = 0.01

        # W3: per-step cap. The five layers are additive by design, but a
        # fully stuck agent stacks them ~-0.4..-0.5/step with no bound; clamp
        # the applied penalty at POKERL_ANTILOOP_CAP (-0.5/step). The info
        # keys keep their raw per-layer values so diagnostics still show what
        # fired; only the reward applied to the episode is bounded.
        if penalty < POKERL_ANTILOOP_CAP:
            penalty = POKERL_ANTILOOP_CAP

        return penalty, info

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        import io

        if seed is not None:
            np.random.seed(seed)

        # Reset PyBoy by loading a snapshot of the initial boot state
        if options and "state_path" in options:
            with open(options["state_path"], "rb") as f:
                self._pyboy.load_state(f)
        elif self.state_path and os.path.exists(self.state_path):
            with open(self.state_path, "rb") as f:
                self._pyboy.load_state(f)
        elif hasattr(self, "_initial_state") and self._initial_state:
            self._pyboy.load_state(io.BytesIO(self._initial_state))
        else:
            # Fallback: re-boot (should not happen after _init_pyboy)
            for _ in range(600):
                self._pyboy.tick()

        self._step_count = 0
        self._visited_maps = set()
        self._visited_tiles = set()
        # Baseline money at the loaded state so a rich save does not pay a
        # fake "money gained" reward on the very first step of a reboot.
        self._prev_money_raw = self._read_money()
        self._max_map_progress = 0.0
        self._total_reward = 0.0
        # Curriculum goal override (pokerl_mode distance-to-goal): a step or
        # call-site can point the episode at a specific (map_id, x, y) via
        # options; otherwise the config-dict goal from __init__ stands.
        if options and "goal" in options:
            self._goal = self._validate_goal(options["goal"], "options['goal']")
        else:
            # No per-episode goal: restore the config default so a goal set by
            # an earlier reset does not leak into this episode (W4).
            self._goal = self._default_goal
        self._clear_antiloop()

        obs = self._get_obs()
        self._prev_state = obs
        self._seed_episode_start(obs["state"])

        return obs, {"step_count": 0}

    def soft_reset(self) -> Dict[str, np.ndarray]:
        """Reset per-episode bookkeeping WITHOUT restarting the game.

        Used in persistent mode: the emulator keeps its current state (and the
        explored-tile/map sets are preserved so exploration bonuses cannot be
        farmed by walking back and forth), but the step counter is rewound so
        the next rollout behaves like a fresh episode.
        """
        self._step_count = 0
        self._total_reward = 0.0
        self._clear_antiloop()
        obs = self._get_obs()
        self._prev_state = obs
        self._seed_episode_start(obs["state"])
        return obs

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert 0 <= action < len(self._action_buttons), f"Invalid action {action}"

        button = self._action_buttons[action]

        # Press then release the button, spending `frame_skip` frames total.
        # More frames per decision = more game time travelled per policy step.
        #
        # In pokerl_mode, movement buttons are tapped: hold them only 1-2
        # frames then idle the rest of frame_skip. Holding a direction for
        # frame_skip//2 frames re-engages the on-screen toggle each step,
        # which causes held-direction jitter; a tap moves exactly one tile per
        # decision. Non-movement buttons (a/b) keep the default press behavior.
        if self.pokerl_mode and button in MOVEMENT_BUTTONS:
            # Reserve at least 1 released frame so consecutive identical taps
            # are not read by the game as a continuous hold (PyBoy samples
            # input at frame boundaries). frame_skip=1 is unavoidably degenerate.
            hold = max(1, min(2, self.frame_skip - 1))
            release = max(0, self.frame_skip - hold)
        else:
            hold = max(1, self.frame_skip // 2)
            release = max(1, self.frame_skip - hold)
        self._pyboy.button_press(button)
        for _ in range(hold):
            self._pyboy.tick()
        self._pyboy.button_release(button)
        for _ in range(release):
            self._pyboy.tick()

        self._step_count += 1

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(self._prev_state, obs)
        if self.pokerl_mode:
            # Anti-loop / anti-spam penalties are additive on top of the
            # shaped reward and report through the same info dict.
            anti_penalty, anti_info = self._pokerl_antiloop(action, obs["state"])
            reward += anti_penalty
            reward_info.update(anti_info)
        self._total_reward += reward
        self._prev_state = obs

        # Check if episode is done
        terminated = False
        truncated = self._step_count >= self.max_steps

        # Check for game completion (all badges earned)
        badges = self._badges_from(obs["state"])
        if badges >= self.badge_max:
            terminated = True
            reward += 5000.0  # Big completion bonus

        info = {
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "badges": badges,
            "party_count": int(obs["state"][1] * 6),
            "level": int(obs["state"][3] * 100),
            "map_id": self._map_key(obs["state"]),
            "map_name": self._map_name_for(obs["state"]),
            "story_progress": round(self._max_map_progress, 3),
            "in_battle": bool(obs["state"][5] > 0),
            "visited_maps": len(self._visited_maps),
            **reward_info,
        }

        if self.render_mode == "rgb_array":
            info["rgb_frame"] = self._pyboy.screen.ndarray

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._pyboy.screen.ndarray
        elif self.render_mode == "human":
            pass  # PyBoy window handles rendering

    def close(self):
        if self._pyboy is not None:
            try:
                self._pyboy.stop()
            except Exception:
                pass
            self._pyboy = None

    def save_state(self, path: str):
        with open(path, "wb") as f:
            self._pyboy.save_state(f)

    def load_state(self, path: str):
        with open(path, "rb") as f:
            self._pyboy.load_state(f)

    def get_map_tiles(self) -> np.ndarray:
        """
        Extract the current map's tile IDs from VRAM for exploration tracking.
        Returns the 10x9 visible map area as tile IDs.
        """
        try:
            # Gen 1: Tile map is at 0x9C00 in VRAM (background map)
            tiles = np.zeros((9, 10), dtype=np.uint8)
            for r in range(9):
                for c in range(10):
                    addr = 0x9C00 + r * 32 + c
                    tiles[r, c] = self._pyboy.memory[addr]
            return tiles
        except Exception:
            return np.zeros((9, 10), dtype=np.uint8)
