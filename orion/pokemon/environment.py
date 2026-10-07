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
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

from orion.pokemon.knowledge import map_progress_by_id, map_name

# Game Boy screen dimensions
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 144

# Game Boy button mapping
BUTTONS = ["a", "b", "start", "select", "up", "down", "left", "right"]
NUM_ACTIONS = len(BUTTONS)


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

        return gym.spaces.Discrete(NUM_ACTIONS)

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

        return reward, info

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

        obs = self._get_obs()
        self._prev_state = obs

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
        obs = self._get_obs()
        self._prev_state = obs
        return obs

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert 0 <= action < NUM_ACTIONS, f"Invalid action {action}"

        button = BUTTONS[action]

        # Press then release the button, spending `frame_skip` frames total.
        # More frames per decision = more game time travelled per policy step.
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
