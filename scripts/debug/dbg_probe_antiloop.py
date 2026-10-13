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

Plus the adversarial-review fixes (W1/W2/W3):
- W1: during combat (in_battle=True) no position/loop/no-change penalties
  fire, and the no-change counter cannot escalate on a long battle;
- W2: off-world (not in the game world) steps produce 0 anti-loop
  penalties and no info keys;
- W3: the per-step cap (-0.5) bounds a fully stuck agent (64-step
  identical streak), while uncapped reference steps are untouched.
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


def in_world(x=5, y=5, in_battle=0, map_id=40):
    """Synthetic state vector. map 40 (OAKS_LAB) is nameable -> in-world;
    map 250 is not in the Gen 1 map knowledge tables -> off-world.
    """
    s = np.zeros(20, dtype=np.float32)
    s[4] = map_id / 255.0
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
# Per-key sums must equal the actual penalty applied to the reward (no cap
# engaged: the 10-step loop peaks at -0.37/step, under POKERL_ANTILOOP_CAP).
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

# 4) W1: during combat (in_battle=True) nothing fires. A 12-step battle keeps
#    map/x/y constant -- without the W1 gate this would stack visit/loop/
#    pattern/no-change/spam penalties and escalate the no-change counter.
e.reset()
battle_state = in_world(in_battle=1)
total_pen = 0.0
for i in range(12):
    pen, pinfo = e._pokerl_antiloop(0, battle_state)
    assert pen == 0.0, (i, pen)
    assert pinfo == {}, (i, pinfo)
    total_pen += pen
assert abs(total_pen) < 1e-9, total_pen
print("in-battle: 0.0 penalty and no info keys across 12 constant steps")

# 5) W2: off-world steps (unnameable map = not in the game world) are never
#    penalized, even on a long identical streak that would fire every layer
#    if it were in-world.
e.reset()
off_world = in_world(map_id=250)  # 250 is not a known Gen 1 map id
total_pen = 0.0
for i in range(12):
    pen, pinfo = e._pokerl_antiloop(0, off_world)
    assert pen == 0.0, (i, pen)
    assert pinfo == {}, (i, pinfo)
    total_pen += pen
assert abs(total_pen) < 1e-9, total_pen
assert not e._in_game_world(off_world), "probe vector must be off-world"
print("off-world: 0.0 penalty and no info keys across 12 constant steps")

# 6) W3: per-step cap. A 64-step identical streak stacks all five layers
#    (visit -0.05 + loop -0.2 + pattern -0.05 + no-change -0.20 + spam -0.02
#    = -0.52 at step 64); the -0.5/step cap must clamp the applied penalty
#    while leaving the uncapped reference step (8th) at its raw -0.37.
e.reset()
state = in_world()
cap_hit = False
for i in range(64):
    pen, pinfo = e._pokerl_antiloop(0, state)
    assert pen >= -0.5, (i, pen)
    if i == 7:  # 8th step: visits+loop+pattern+no-change(level1)+spam
        assert abs(pen - (-0.37)) < 1e-6, (i, pen)
    if i == 63:  # 64th step: raw -0.52 must be clamped to exactly -0.5
        assert pen == -0.5, (i, pen)
        cap_hit = True
assert cap_hit, "cap never engaged on a 64-step identical streak"
print("per-step cap: engaged at -0.5/step on step 64; uncapped step 8 intact")

# 7) Real emulator step in pokerl_mode: integration smoke (no penalties on
#    step 1 -- the boot snapshot is off-world, so W2 must gate it out).
obs, _ = e.reset()
_, _, _, _, info = e.step(0)
assert info["step_count"] == 1
leaked = [k for k in NEW_INFO_KEYS if k in info]
assert not leaked, f"pokerl step 1 leaked penalties: {leaked}"
print("pokerl live step: OK, no penalties on step 1")

e.close()
print("PROBE OK")
