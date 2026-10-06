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

# Game Boy screen dimensions
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 144

# Game Boy button mapping
BUTTONS = ["a", "b", "start", "select", "up", "down", "left", "right"]
NUM_ACTIONS = len(BUTTONS)


# Pokemon Red/Blue/Yellow RAM addresses (Gen 1)
# These are well-documented in the speedrunning/emulation community
class Gen1RAM:
    """Pokemon Gen 1 RAM address map for state extraction."""

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
    CURRENT_HP = 0xD16C  # HP of first party member (2 bytes big-endian)
    MAX_HP = 0xD16D

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
    ):
        self.rom_path = rom_path
        self.headless = headless
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.state_path = state_path

        if not os.path.exists(rom_path):
            raise FileNotFoundError(
                f"ROM not found: {rom_path}\n"
                f"Place your Pokemon Red/Blue/Yellow ROM at this path."
            )

        # Reward configuration
        self.reward_config = reward_config or {
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

        self._init_pyboy()

    def _init_pyboy(self):
        from pyboy import PyBoy

        if self.headless:
            self._pyboy = PyBoy(self.rom_path, window="null", sound_emulated=False)
        else:
            self._pyboy = PyBoy(self.rom_path, window="SDL2", sound_emulated=False)

        self._pyboy.set_emulation_speed(0)  # Unlimited speed

        # Load state if provided
        if self.state_path and os.path.exists(self.state_path):
            with open(self.state_path, "rb") as f:
                self._pyboy.load_state(f)

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

    def _extract_state_features(self) -> np.ndarray:
        """Extract Pokemon-specific state features from emulator RAM."""
        features = np.zeros(self.state_feature_size, dtype=np.float32)

        try:
            # Badges (0-8)
            badges_byte = self._read_ram(Gen1RAM.BADGES)
            badge_count = bin(badges_byte).count("1")
            features[0] = badge_count / 8.0  # Normalized 0-1

            # Party count (0-6)
            party_count = min(self._read_ram(Gen1RAM.PARTY_COUNT), 6)
            features[1] = party_count / 6.0

            # First party member HP (normalized)
            current_hp = self._read_ram_16(Gen1RAM.CURRENT_HP)
            max_hp = self._read_ram_16(Gen1RAM.MAX_HP)
            if max_hp > 0:
                features[2] = min(current_hp / max_hp, 1.0)
            else:
                features[2] = 0.0

            # First party member level (0-100)
            level = self._read_ram(Gen1RAM.LEVEL_FIRST)
            features[3] = min(level / 100.0, 1.0)

            # Map ID (normalized)
            features[4] = self._read_ram(Gen1RAM.MAP_ID) / 255.0

            # In battle flag
            features[5] = 1.0 if self._read_ram(Gen1RAM.IN_BATTLE) > 0 else 0.0

            # Player X position
            features[6] = self._read_ram(Gen1RAM.PLAYER_X) / 255.0

            # Player Y position
            features[7] = self._read_ram(Gen1RAM.PLAYER_Y) / 255.0

            # Money (BCD, up to 999999)
            money = (
                self._read_ram(Gen1RAM.MONEY_1) * 10000
                + self._read_ram(Gen1RAM.MONEY_2) * 100
                + self._read_ram(Gen1RAM.MONEY_3)
            )
            features[8] = min(money / 999999.0, 1.0)

            # Experience of first party member
            exp = (
                self._read_ram(Gen1RAM.EXP_1) * 65536
                + self._read_ram(Gen1RAM.EXP_2) * 256
                + self._read_ram(Gen1RAM.EXP_3)
            )
            features[9] = min(exp / 1000000.0, 1.0)

            # Party levels (first 3 members)
            for i in range(min(party_count, 3)):
                addr = Gen1RAM.LEVEL_FIRST + i * 0x2C
                features[10 + i] = min(self._read_ram(addr) / 100.0, 1.0)

            # Party HP (members 2 and 3, normalized by estimated max)
            # Simplified: just store raw HP normalized
            for i in range(min(party_count, 3)):
                hp_addr = Gen1RAM.CURRENT_HP + i * 0x2C
                hp = self._read_ram_16(hp_addr)
                features[13 + i] = min(hp / 999.0, 1.0)

            # Game state
            features[16] = self._read_ram(Gen1RAM.GAME_STATE) / 255.0

            # Dialogue state
            features[17] = 1.0 if self._read_ram(Gen1RAM.DIALOGUE_STATE) > 0 else 0.0

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

        # Badge reward
        old_badges = int(old_features[0] * 8)
        new_badges = int(new_features[0] * 8)
        if new_badges > old_badges:
            badge_reward = self.reward_config["badge_earned"] * (
                new_badges - old_badges
            )
            reward += badge_reward
            info["badges_earned"] = new_badges - old_badges

        # Level up reward
        old_level = old_features[3] * 100
        new_level = new_features[3] * 100
        if new_level > old_level:
            levels_gained = int(new_level - old_level)
            reward += self.reward_config["pokemon_leveled_up"] * levels_gained
            info["levels_gained"] = levels_gained

        # New map exploration reward
        old_map = int(old_features[4] * 255)
        new_map = int(new_features[4] * 255)
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

        return reward, info

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        if seed is not None:
            np.random.seed(seed)

        # Reset PyBoy by reinitializing or loading a save state
        if options and "state_path" in options:
            with open(options["state_path"], "rb") as f:
                self._pyboy.load_state(f)
        elif self.state_path and os.path.exists(self.state_path):
            with open(self.state_path, "rb") as f:
                self._pyboy.load_state(f)
        else:
            # Soft reset: just re-run from current state
            self._pyboy.reset()

        self._step_count = 0
        self._visited_maps = set()
        self._total_reward = 0.0

        obs = self._get_obs()
        self._prev_state = obs

        return obs, {"step_count": 0}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert 0 <= action < NUM_ACTIONS, f"Invalid action {action}"

        button = BUTTONS[action]

        # Press and release button
        self._pyboy.button_press(button)
        for _ in range(4):  # Hold for a few frames
            self._pyboy.tick()
        self._pyboy.button_release(button)
        for _ in range(4):  # Release frames
            self._pyboy.tick()

        self._step_count += 1

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(self._prev_state, obs)
        self._total_reward += reward
        self._prev_state = obs

        # Check if episode is done
        terminated = False
        truncated = self._step_count >= self.max_steps

        # Check for game completion (all 8 badges in Gen 1)
        badges = int(obs["state"][0] * 8)
        if badges >= 8:
            terminated = True
            reward += 5000.0  # Big completion bonus

        info = {
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "badges": badges,
            "party_count": int(obs["state"][1] * 6),
            "level": int(obs["state"][3] * 100),
            "map_id": int(obs["state"][4] * 255),
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
