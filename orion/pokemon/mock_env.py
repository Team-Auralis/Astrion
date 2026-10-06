"""
Mock Pokemon environment for testing the ORION pipeline without a ROM.

Generates synthetic observations that mimic Pokemon screen/state patterns
so you can verify training works end-to-end before having a ROM.
"""

import numpy as np
from typing import Dict, Tuple, Optional

from orion.pokemon.environment import (
    PokemonEnv,
    NUM_ACTIONS,
    BUTTONS,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
)


class MockPokemonEnv(PokemonEnv):
    """
    Drop-in replacement for PokemonEnv that generates synthetic data.

    Observation: random screen patterns + state features that evolve
    Action: affects state progression (simulates learning)
    Reward: mimics the real reward structure
    """

    def __init__(
        self,
        headless: bool = True,
        max_steps: int = 2048,
        seed: int = 42,
        reward_config: Optional[dict] = None,
    ):
        # Don't call PokemonEnv.__init__ (no ROM needed)
        self.headless = headless
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        self.reward_config = reward_config or {
            "badge_earned": 1000.0,
            "new_pokemon_caught": 500.0,
            "pokemon_leveled_up": 50.0,
            "new_map_entered": 100.0,
            "step_penalty": -0.1,
        }

        self._step_count = 0
        self._pyboy = None
        self._visited_maps = set()
        self._total_reward = 0.0

        # Simulated game state
        self._badges = 0
        self._party_count = 1
        self._level = 5
        self._hp = 20
        self._max_hp = 20
        self._map_id = 0
        self._x = 0
        self._y = 0
        self._in_battle = False
        self._money = 500
        self._dialogue = False
        self._steps_in_map = 0

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        h = SCREEN_HEIGHT // 2
        w = SCREEN_WIDTH // 2
        return (1, h, w)

    @property
    def state_feature_size(self) -> int:
        return 20

    @property
    def observation_space(self):
        import gymnasium as gym

        h = SCREEN_HEIGHT // 2
        w = SCREEN_WIDTH // 2
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

    def _generate_screen(self) -> np.ndarray:
        """Generate a synthetic Game Boy-looking screen."""
        h, w = SCREEN_HEIGHT // 2, SCREEN_WIDTH // 2
        screen = np.full((h, w), 255, dtype=np.uint8)  # White background

        # Add some tile-like patterns based on position
        tile_h, tile_w = 4, 4
        for ty in range(0, h, tile_h):
            for tx in range(0, w, tile_w):
                tile_hash = (self._x + tx // tile_w) * 31 + (self._y + ty // tile_h) * 7
                if (tile_hash % 5) == 0:
                    screen[ty : ty + tile_h, tx : tx + tile_w] = 180  # Light gray
                elif (tile_hash % 7) == 0:
                    screen[ty : ty + tile_h, tx : tx + tile_w] = 100  # Dark gray

        # Player sprite in center
        cy, cx = h // 2, w // 2
        screen[cy : cy + 6, cx : cx + 6] = 0  # Black player dot

        # HUD at top
        screen[0:8, :] = 220  # Light HUD bar

        # Add some noise
        noise = self.rng.randint(0, 20, (h, w), dtype=np.uint8)
        screen = np.clip(
            screen.astype(np.int16) + noise.astype(np.int16) - 10, 0, 255
        ).astype(np.uint8)

        return screen

    def _extract_state_features(self) -> np.ndarray:
        features = np.zeros(20, dtype=np.float32)
        features[0] = self._badges / 8.0
        features[1] = self._party_count / 6.0
        features[2] = self._hp / max(self._max_hp, 1)
        features[3] = self._level / 100.0
        features[4] = self._map_id / 255.0
        features[5] = 1.0 if self._in_battle else 0.0
        features[6] = self._x / 255.0
        features[7] = self._y / 255.0
        features[8] = self._money / 999999.0
        features[9] = min(self._level**3 / 1000000.0, 1.0)  # Fake exp curve
        features[10] = self._level / 100.0  # First pokemon level
        features[13] = self._hp / max(self._max_hp, 1)  # First pokemon HP
        features[16] = 0.0  # Game state
        features[17] = 1.0 if self._dialogue else 0.0
        features[18] = min(self._step_count / self.max_steps, 1.0)
        features[19] = min(len(self._visited_maps) / 255.0, 1.0)
        return features

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "screen": self._generate_screen(),
            "state": self._extract_state_features(),
        }

    def _compute_reward(self, old_state: Dict, new_state: Dict) -> Tuple[float, Dict]:
        reward = 0.0
        info = {}
        old_f = old_state["state"]
        new_f = new_state["state"]

        # Badge reward
        old_b = int(old_f[0] * 8)
        new_b = int(new_f[0] * 8)
        if new_b > old_b:
            reward += self.reward_config["badge_earned"]
            info["badges_earned"] = new_b - old_b

        # Level reward
        old_l = old_f[3] * 100
        new_l = new_f[3] * 100
        if new_l > old_l:
            reward += self.reward_config["pokemon_leveled_up"] * int(new_l - old_l)
            info["levels_gained"] = int(new_l - old_l)

        # Map exploration
        old_m = int(old_f[4] * 255)
        new_m = int(new_f[4] * 255)
        if new_m != old_m and new_m not in self._visited_maps:
            self._visited_maps.add(new_m)
            reward += self.reward_config["new_map_entered"]

        # Step penalty
        reward += self.reward_config["step_penalty"]

        return reward, info

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self._step_count = 0
        self._visited_maps = set()
        self._total_reward = 0.0
        self._badges = 0
        self._party_count = 1
        self._level = 5
        self._hp = 20
        self._max_hp = 20
        self._map_id = 0
        self._x = 10
        self._y = 10
        self._in_battle = False
        self._money = 500
        self._dialogue = False
        self._steps_in_map = 0

        obs = self._get_obs()
        self._prev_state = obs
        return obs, {"step_count": 0}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert 0 <= action < NUM_ACTIONS

        button = BUTTONS[action]
        self._step_count += 1
        self._steps_in_map += 1

        # Simulate game progression based on action
        old_obs = self._prev_state

        # Movement
        if button == "up":
            self._y = max(0, self._y - 1)
        elif button == "down":
            self._y = min(255, self._y + 1)
        elif button == "left":
            self._x = max(0, self._x - 1)
        elif button == "right":
            self._x = min(255, self._x + 1)

        # New map after enough steps in current map
        if self._steps_in_map > 30 and self.rng.random() < 0.05:
            self._map_id = (self._map_id + 1) % 255
            self._steps_in_map = 0

        # Random level up (slow)
        if self.rng.random() < 0.002 and self._level < 100:
            self._level += 1
            self._max_hp += 3
            self._hp = self._max_hp

        # Random badge (much rarer)
        if self._badges < 8 and self.rng.random() < 0.0005:
            self._badges += 1

        # Random encounter
        self._in_battle = self.rng.random() < 0.05

        # A button advances dialogue / interacts
        if button == "a":
            self._dialogue = self.rng.random() < 0.3
        else:
            self._dialogue = False

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(old_obs, obs)
        self._total_reward += reward
        self._prev_state = obs

        terminated = self._badges >= 8
        truncated = self._step_count >= self.max_steps

        if terminated:
            reward += 5000.0

        info = {
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "badges": self._badges,
            "party_count": self._party_count,
            "level": self._level,
            "map_id": self._map_id,
            "in_battle": self._in_battle,
            "visited_maps": len(self._visited_maps),
            **reward_info,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        pass

    def save_state(self, path: str):
        pass

    def load_state(self, path: str):
        pass
