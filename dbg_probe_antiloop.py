"""Probe: verify PokeRL anti-loop + anti-spam penalties (pokerl_mode only).

Checks the three anti-loop layers (position-visit, position-loop,
repeated-action / no-state-change) and the anti-spam streak + diversity
bonus from RESEARCH_FASTER_TRAINING.md 4.1/4.2:
- a forced 10-step same-position loop fires every penalty with the
  expected sign and magnitude,
- a varied rollout (different actions + tiles) keeps the anti-loop
  penalties at ~0 while the diversity bonus accrues,
- default mode (pokerl_mode=False) exposes none of the new info keys and
  the anti-loop hook is a byte-identical no-op.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from orion.pokemon.environment import PokemonEnv

ROM = r"D:\roms\Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb"
NEW_INFO_KEYS = (
    "visit_penalty",
    "loop_penalty",
    "pattern_penalty",
    "spam_penalty",
    "diversity_bonus",
)


def in_world(x=5, y=5, in_battle=0):
    """Synthetic in-world state vector (map 40 = OAKS_LAB, party 1, lvl 6)."""
    s = np.zeros(20, dtype=np.float32)
    s[4] = 40 / 255.0
    s[1] = 1 / 6.0
    s[3] = 6 / 100.0
    s[6] = x / 255.0
    s[7] = y / 255.0
    s[5] = float(in_battle)
    return s


# 1) Default mode: none of the new info keys; helper is a byte-identical no-op.
e = PokemonEnv(rom_path=ROM, headless=True, state_path=None, pokerl_mode=False)
obs, _ = e.reset()
_, _, _, _, info = e.step(0)
leaked = [k for k in NEW_INFO_KEYS if k in info]
assert not leaked, f"default mode leaked anti-loop keys: {leaked}"
pen, pinfo = e._pokerl_antiloop(0, in_world())
assert pen == 0.0 and pinfo == {}, (pen, pinfo)
print("default mode: no anti-loop keys, helper returns 0.0/{}")
e.close()

# 2) pokerl_mode: forced 10-step same-position loop (same tile, same action).
e = PokemonEnv(rom_path=ROM, headless=True, state_path=None, pokerl_mode=True)
obs, _ = e.reset()
state = in_world()
totals = {k: 0.0 for k in NEW_INFO_KEYS}
total_pen = 0.0
for _ in range(10):
    pen, pinfo = e._pokerl_antiloop(0, state)
    total_pen += pen
    for k, v in pinfo.items():
        totals[k] += v
    last_info = pinfo

assert abs(last_info["visit_penalty"] - (-0.05)) < 1e-6, last_info
assert abs(last_info["loop_penalty"] - (-0.2)) < 1e-6, last_info
assert abs(last_info["spam_penalty"] - (-0.02)) < 1e-6, last_info
assert abs(last_info["pattern_penalty"] - (-0.10)) < 1e-3, last_info
assert totals["visit_penalty"] <= -0.2, totals
assert abs(totals["loop_penalty"] - (-1.0)) < 1e-6, totals
assert totals["spam_penalty"] <= -0.1, totals
assert totals["pattern_penalty"] <= -0.25, totals
# Per-key sums must equal the actual penalty applied to the reward.
assert abs(sum(totals.values()) - total_pen) < 1e-6, (totals, total_pen)
print(
    f"10-step loop: total penalty={total_pen:.3f}  per-key={ {k: round(v, 3) for k, v in totals.items()} }"
)

# 3) pokerl_mode: varied rollout (different actions + tiles) -> ~no penalties.
e.reset()
totals = {k: 0.0 for k in NEW_INFO_KEYS}
total_pen = 0.0
for i in range(12):
    state = in_world(x=(i % 6) + 1, y=1)
    pen, pinfo = e._pokerl_antiloop(i % 6, state)
    total_pen += pen
    for k, v in pinfo.items():
        totals[k] += v
    for k in ("visit_penalty", "loop_penalty", "pattern_penalty", "spam_penalty"):
        assert k not in pinfo, (i, pinfo)
for k in ("visit_penalty", "loop_penalty", "pattern_penalty", "spam_penalty"):
    assert abs(totals[k]) < 1e-9, (k, totals[k])
assert 0.10 <= totals["diversity_bonus"] <= 0.12, totals["diversity_bonus"]
print(
    f"varied rollout: anti-loop penalties ~0, diversity_bonus={totals['diversity_bonus']:.2f}"
)

# 4) Real emulator step in pokerl_mode: integration smoke (no penalties on step 1).
obs, _ = e.reset()
_, _, _, _, info = e.step(0)
assert info["step_count"] == 1
leaked = [k for k in NEW_INFO_KEYS if k in info]
assert not leaked, f"pokerl step 1 leaked penalties: {leaked}"
print("pokerl live step: OK, no penalties on step 1")

e.close()
print("PROBE OK")
