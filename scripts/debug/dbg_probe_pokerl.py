"""Probe: verify PokeRL dense hierarchical rewards (pokerl_mode only).

RESEARCH_FASTER_TRAINING.md 4.3 reward table:
  micro: new tile +1.0, first-ever visit +0.5, distance-to-goal +0.2
  meso:  first entry into a new map +5, any map transition +10,
         unexplored region +2, tall grass +20, battle start +10
  macro: battle win +50, catch +50

Checks, on synthesized feature deltas:
- new tile fires +1.0 then 0 on revisiting the same tile in the episode;
  first-ever visit adds +0.5;
- map change 0->1 fires transition +10 + first-entry +5 = +15 on first entry
  (with the region family pre-seen) and +10 on re-entry;
- battle start fires +10 on the rising edge and not on steady 1;
  battle win fires +50 on 1->0 with EXP gained since battle start and 0
  without;
- W3: the episode start map is pre-seeded, so leaving and re-entering the
  start map pays the +10 transition only (no +5 entry / +2 region);
- W4: a malformed goal raises ValueError naming the config key, and a reset
  that omits `goal` restores the config default (no leak);
- W5: boot-screen steps (map 0, x/y = (0,0)) pay 0 new_tile / first_visit /
  anti-loop penalties;
- C1: tall grass +20 is one-shot per episode -- the first grass step pays,
  later grass steps in the same episode report tall_grass 0.0;
- tall grass against the REAL ROM: the intro is played, the player walks
  onto Pallet Town's genuine tall grass, and the game-authoritative RAM feet
  byte (wTileMap[9][9] = 0xC55D, the same byte TryDoWildEncounter checks)
  reads 0x52 with NO pb.memory write, pays +20 there, 0 on a plain tile, and
  0 again on the next grass step (one-shot);
- default mode (pokerl_mode=False) exposes none of the new info keys and the
  reward is unchanged by the new machinery.

Also asserts (by construction / inspection) that no hard-coded route or
waypoint table was added to the repo.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from orion.pokemon.environment import POKERL_REWARD_WEIGHTS, PokemonEnv

ROM = r"D:\roms\Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb"
NEW_INFO_KEYS = tuple(POKERL_REWARD_WEIGHTS.keys())


def in_world_features(map_id=0x0C, x=8, y=8, in_battle=0, party=1, level=6, exp=3000.0):
    """Synthetic state vector that passes the in-world gate (map nameable)."""
    s = np.zeros(20, dtype=np.float32)
    s[4] = map_id / 255.0
    s[1] = party / 6.0
    s[3] = level / 100.0
    s[6] = x / 255.0
    s[7] = y / 255.0
    s[5] = float(in_battle)
    s[9] = exp / 1000000.0
    return s


def cr(e, old, new):
    """Run _compute_reward with dict-wrapped feature vectors."""
    r, i = e._compute_reward({"state": old}, {"state": new})
    return r, i


# ---------------------------------------------------------------------------
# 1) Default mode: byte-identical -- no new keys, new machinery never runs.
# ---------------------------------------------------------------------------
e = PokemonEnv(rom_path=ROM, headless=True, state_path=None, pokerl_mode=False)
obs, _ = e.reset()
old = in_world_features()
new = in_world_features(map_id=0x0C, x=9, y=8)  # new tile + exp rise + heal
new[2] = 0.5
new[9] = 3300 / 1000000.0
r_def, info_def = cr(e, old, new)
leaked = [k for k in NEW_INFO_KEYS if k in info_def]
assert not leaked, f"default mode leaked pokerl keys: {leaked}"
# Same transition in default mode must produce exactly the pre-Task-3 result,
# computed from the env's OWN base terms (the pokerl machinery is fully gated
# off, so this IS the untouched shaping):
#   exp 0.15 * gained, where gained = int(new_exp - old_exp) on the float32
#        state vector = 3299.99998 - 3000.00003 = 299 (float32 rounding!)
#   heal +20, dense tile +0.1, step -0.005
#   map_progress 2000 * progress(ROUTE_1=$0C) = 2000 * 0.06 = 120
#   money 0 (boot RAM unchanged) -> no term
full_exp = int(float(new[9]) * 1000000.0 - float(old[9]) * 1000000.0)
expected = (
    e.reward_config["exp_gained"] * full_exp
    + e.reward_config["hp_healed"]
    + e.reward_config["new_tile_explored"]
    + e.reward_config["step_penalty"]
    + e.reward_config["map_progress"] * e._map_progress_for(new)
)
assert full_exp == 299, full_exp  # documents the float32 rounding behavior
assert abs(r_def - expected) < 1e-6, (r_def, expected)
assert e._ep_tiles == set() and e._ep_maps == set(), "default mode touched ep state"
assert e._pokerl_standing_on_grass(new) is False
print(
    f"default mode: no new keys, reward byte-identical ({r_def:.3f} = "
    f"exp {e.reward_config['exp_gained'] * full_exp} + heal 20 + tile 0.1 "
    f"- step 0.005 + map_progress 120), helpers inert"
)
e.close()

# ---------------------------------------------------------------------------
# 2) pokerl_mode micro: new tile +1.0, first-ever visit +0.5, revisit -> 0
# ---------------------------------------------------------------------------
e = PokemonEnv(rom_path=ROM, headless=True, state_path=None, pokerl_mode=True)
obs, _ = e.reset()

a = in_world_features(map_id=0x0C, x=8, y=8)
b = in_world_features(map_id=0x0C, x=8, y=9)
r, info = cr(e, a, b)
assert abs(info["new_tile"] - 1.0) < 1e-9, info
assert "first_visit" in info and abs(info["first_visit"] - 0.5) < 1e-9, info
r, info = cr(e, b, b)  # same tile again (no movement) -> no new-tile terms
assert "new_tile" not in info and "first_visit" not in info, info
r, info = cr(e, b, a)  # stepping back to (8,8): never touched -> pays +1.5
assert abs(info["new_tile"] - 1.0) < 1e-9, info
assert "first_visit" in info and abs(info["first_visit"] - 0.5) < 1e-9, info
r, info = cr(e, a, a)  # same tile again -> no terms
assert "new_tile" not in info and "first_visit" not in info, info
assert (0x0C, 8, 8) in e._ep_tiles and (0x0C, 8, 9) in e._ep_tiles
assert e._lifetime_tiles == {(0x0C, 8, 8), (0x0C, 8, 9)}, e._lifetime_tiles
print("micro tiles: +1.0 new tile, +0.5 first-ever, 0 on revisits")

# first-ever visit is per env instance: a NEW episode on a fresh tile still
# pays first_visit; a lifetime tile in a new episode pays only new_tile.
e._clear_antiloop()
c = in_world_features(map_id=0x0C, x=10, y=8)
r, info = cr(e, b, c)
assert abs(info["new_tile"] - 1.0) < 1e-9 and abs(info["first_visit"] - 0.5) < 1e-9, (
    info
)
r, info = cr(e, c, b)  # lifetime tile (8,9) revisited in the new episode
assert abs(info["new_tile"] - 1.0) < 1e-9 and "first_visit" not in info, info
print("micro tiles: first-ever persists across episodes; first_visit fires once")

# ---------------------------------------------------------------------------
# 3) pokerl_mode meso maps: +10 transition, +5 first entry (+15 total), +2 region
# ---------------------------------------------------------------------------
e._clear_antiloop()
# First entry: pre-seed the target map's family so the region term is not in
# play -- the strict +15 / +10 numbers from RESEARCH_FASTER_TRAINING.md.
e._ep_map_families.add("CITY")  # VIRIDIAN_CITY family
start = in_world_features(map_id=0x00, x=2, y=2)  # PALLET_TOWN
enter = in_world_features(map_id=0x01, x=2, y=2)  # VIRIDIAN_CITY
r, info = cr(e, start, enter)
assert abs(info["map_transition"] - 10.0) < 1e-9, info
assert abs(info["new_map_entry"] - 5.0) < 1e-9, info
assert "unexplored_region" not in info, info
assert abs(info["map_transition"] + info["new_map_entry"] - 15.0) < 1e-9
# Re-entry (leave and come back): transition only, no first-entry/region.
r, info = cr(e, enter, start)
r, info = cr(e, start, enter)
assert abs(info["map_transition"] - 10.0) < 1e-9, info
assert "new_map_entry" not in info and "unexplored_region" not in info, info
print("meso maps: first entry = 15 total (10 transition + 5 entry); re-entry = 10")

# Full fidelity: a never-seen family pays the +2 unexplored region on entry.
e._clear_antiloop()
start = in_world_features(map_id=0x00, x=2, y=2)
enter = in_world_features(map_id=0x0C, x=2, y=2)  # ROUTE_1 family ROUTE
r, info = cr(e, start, enter)
assert abs(info["unexplored_region"] - 2.0) < 1e-9, info
assert (
    abs(
        info["map_transition"]
        + info["new_map_entry"]
        + info["unexplored_region"]
        - 17.0
    )
    < 1e-9
), info
print("meso maps: +2 unexplored region on first family entry (17 total)")

# ---------------------------------------------------------------------------
# 4) pokerl_mode micro distance: +0.2 * L1-distance decrease toward a goal
# ---------------------------------------------------------------------------
e._clear_antiloop()
e._goal = (0x0C, 10, 10)  # config-style goal on ROUTE_1
p0 = in_world_features(map_id=0x0C, x=12, y=12)  # L1 = 4
p1 = in_world_features(map_id=0x0C, x=11, y=11)  # L1 = 2 -> closer by 2
cr(e, p1, p1)  # pre-register the p1 tile so only the distance term fires below
r, info = cr(e, p0, p1)
assert abs(info["distance_progress"] - 0.4) < 1e-6, info  # 0.2 * 2
total = sum(v for k, v in info.items() if k in NEW_INFO_KEYS)
assert abs(total - 0.4) < 1e-6, (total, info)
r, info = cr(e, p1, p0)  # moving away -> negative progress term
assert "distance_progress" in info and info["distance_progress"] < 0, info
# Without a goal configured the term is 0.
e._goal = None
r, info = cr(e, p0, p1)
assert "distance_progress" not in info, info
print("micro distance: +0.2/L1-tile closer to configured goal; 0 without goal")

# ---------------------------------------------------------------------------
# 5) pokerl_mode meso/macro battle: start +10 rising edge, win +50, no-farm
# ---------------------------------------------------------------------------
e._clear_antiloop()
pre = in_world_features(map_id=0x0C, x=8, y=9, in_battle=0, exp=3000.0)
battle = in_world_features(map_id=0x0C, x=8, y=9, in_battle=1, exp=3000.0)
r, info = cr(e, pre, battle)
assert abs(info["battle_start"] - 10.0) < 1e-9, info
assert abs(e._battle_exp_at_start - 3000.0) < 1e-3, e._battle_exp_at_start
r, info = cr(e, battle, battle)  # steady 1 -> no start, no win
assert "battle_start" not in info and "battle_win" not in info, info
win = in_world_features(map_id=0x0C, x=8, y=9, in_battle=0, exp=3300.0)
r, info = cr(e, battle, win)
assert "battle_win" in info and abs(info["battle_win"] - 50.0) < 1e-9, info
assert e._battle_exp_at_start is None
# Falling edge WITHOUT exp gain since battle start -> no win.
battle2 = in_world_features(map_id=0x0C, x=8, y=9, in_battle=1, exp=3300.0)
lose = in_world_features(map_id=0x0C, x=8, y=9, in_battle=0, exp=3300.0)
r, info = cr(e, win, battle2)
r, info = cr(e, battle2, lose)
assert "battle_win" not in info, info
print("meso/macro battle: start +10 rising edge only, win +50 on EXP-gaining end")

# ---------------------------------------------------------------------------
# 6) pokerl_mode macro catch: party count rises
# ---------------------------------------------------------------------------
e._clear_antiloop()
one = in_world_features(map_id=0x0C, x=8, y=9, party=1)
two = in_world_features(map_id=0x0C, x=8, y=9, party=2)
cr(e, one, one)  # registers the tile (also pays its new_tile 1.0 -- discard)
r0, _ = cr(e, one, one)  # clean no-op baseline: step penalty only
r, info = cr(e, one, two)
assert "pokemon_caught" in info and info["pokemon_caught"] == 1, info
# Catch pays the base new_pokemon_caught AND the pokerl macro +50, which
# reuses the same info["pokemon_caught"] key (no separate info key, per the
# plan). Nothing else fires here (tile pre-registered, same map, no battle).
expected = (
    r0 + e.reward_config["new_pokemon_caught"] + POKERL_REWARD_WEIGHTS["pokemon_caught"]
)
assert abs(r - expected) < 1e-6, (r, expected)
print(
    f"macro catch: party rise pays base {e.reward_config['new_pokemon_caught']:.0f} "
    f"+ pokerl {POKERL_REWARD_WEIGHTS['pokemon_caught']:.0f} "
    f"(info reuses pokemon_caught)"
)

# ---------------------------------------------------------------------------
# 6b) W3: the episode start map is pre-seeded, so leaving and re-entering it
#     pays the +10 transition only (no +5 first-entry / +2 unexplored region).
# ---------------------------------------------------------------------------
e.reset()  # real episode boundary: seeds _ep_maps/_ep_map_families
assert e._ep_maps == {0x00}, e._ep_maps  # boot/start map 0 (PALLET_TOWN)
assert e._ep_map_families == {"TOWN"}, e._ep_map_families
home = in_world_features(map_id=0x00, x=2, y=2)  # PALLET_TOWN (start map)
away = in_world_features(map_id=0x01, x=2, y=2)  # VIRIDIAN_CITY
cr(e, home, home)  # register the home tile so only the map terms can fire
cr(e, home, away)  # leave: transition +10, first-entry +5, region +2
r, info = cr(e, away, home)  # re-enter the START map
assert abs(info["map_transition"] - 10.0) < 1e-9, info
assert "new_map_entry" not in info, info
assert "unexplored_region" not in info, info
print("W3: start-map re-entry pays +10 transition only (no entry/region)")

# ---------------------------------------------------------------------------
# 6c) W4: a malformed goal raises ValueError naming the config key; a reset
#     that omits `goal` restores the config default (no leak across resets).
# ---------------------------------------------------------------------------
try:
    PokemonEnv(ROM, headless=True, reward_config={"goal": (0x0C, 5)})
    raise AssertionError("malformed config goal did not raise")
except ValueError as exc:
    assert "reward_config['goal']" in str(exc), str(exc)
for bad in ((0x0C, 5), (0x0C, "x", 5), "not-a-tuple"):
    try:
        e.reset(options={"goal": bad})
        raise AssertionError(f"malformed goal {bad!r} did not raise")
    except ValueError as exc:
        assert "options['goal']" in str(exc), str(exc)
e.reset(options={"goal": (0x0C, 10, 10)})  # valid per-episode override
assert e._goal == (0x0C, 10, 10), e._goal
e.reset()  # omits `goal` -> config default (None here) restored, no leak
assert e._goal is None, e._goal
print("W4: malformed goal -> ValueError naming the key; reset-without-goal clears it")

# ---------------------------------------------------------------------------
# 6d) W5: boot-screen steps (map 0, x/y = (0,0)) pay 0 new_tile / first_visit
#     and 0 anti-loop. reset() rewinds to the boot snapshot; 6 real steps stay
#     pre-world (frame_skip=8 -> 48 ticks, far short of the intro).
# ---------------------------------------------------------------------------
BOOT_BAD_KEYS = {
    "new_tile",
    "first_visit",
    "visit_penalty",
    "loop_penalty",
    "pattern_penalty",
    "spam_penalty",
}
e.reset()
for i in range(6):
    obs, _r, _term, _trunc, info = e.step(0)  # action 0 = "a" on boot screens
    feats = obs["state"]
    mx = int(round(feats[6] * 255))
    my = int(round(feats[7] * 255))
    assert e._map_key(feats) == 0x00 and (mx, my) == (0, 0), (
        i,
        e._map_key(feats),
        mx,
        my,
    )
    fired = BOOT_BAD_KEYS & set(info)
    assert not fired, (i, fired, info)
print("W5: boot-screen steps pay 0 new_tile/first_visit/anti-loop")

# ---------------------------------------------------------------------------
# 7) tall grass: synthetic RAM feet byte + REAL ROM walk to Pallet's grass
# ---------------------------------------------------------------------------
# The game-authoritative feet byte is wTileMap[9*20+9] = 0xC55D (the same
# byte TryDoWildEncounter compares against wGrassTile for wild encounters;
# `hlcoord 9, 9`). _pokerl_standing_on_grass reads exactly that.
FEET_ADDR = 0xC4A0 + 9 * 20 + 9

# 7a) SYNTHETIC (forced RAM, not a real map read): write a grass tile into the
#     RAM feet byte and assert the one-shot +20; a second grass step in the
#     same episode must report tall_grass 0.0, then a plain tile emits no key.
e._clear_antiloop()
e._pyboy.memory[FEET_ADDR] = 0x52
g = in_world_features(map_id=0x0C, x=8, y=9)
r, info = cr(e, g, g)
assert abs(info["tall_grass"] - 20.0) < 1e-9, info
r, info = cr(e, g, g)  # still grass, same episode -> one-shot 0.0
assert "tall_grass" in info and abs(info["tall_grass"]) < 1e-9, info
e._pyboy.memory[FEET_ADDR] = 0x01  # plain tile -> no tall_grass key at all
r, info = cr(e, g, g)
assert "tall_grass" not in info, info
print("tall grass (synthetic RAM feet): +20 first, 0.0 next, none on plain")

# 7b) REAL ROM (unforced): play the Red intro, walk to Pallet Town's genuine
#     tall grass, and confirm the +20 fires off the real RAM feet byte with NO
#     pb.memory write. From the door landing (5,6): right x4 -> (9,6) plain
#     ground (feet 0x23); up -> (9,5) genuine grass (feet 0x52); up again ->
#     (9,4) still grass, same episode -> one-shot 0.0.
from dbg_probe_pokerl_intro import press, red_intro_to_pallet

pb = e._pyboy
e.reset()  # rewind to the boot snapshot the intro/step recipes are calibrated on
assert red_intro_to_pallet(pb), "intro/door warp to Pallet Town failed"
cur_state = e._get_obs()["state"]
assert e._map_key(cur_state) == 0x00, hex(e._map_key(cur_state))
assert (int(round(cur_state[6] * 255)), int(round(cur_state[7] * 255))) == (5, 6), (
    int(round(cur_state[6] * 255)),
    int(round(cur_state[7] * 255)),
)

for _ in range(4):  # east along the door row to (9,6)
    press(pb, "right", hold=8, settle=28)
cur_state = e._get_obs()["state"]
nx = int(round(cur_state[6] * 255))
ny = int(round(cur_state[7] * 255))
assert (nx, ny) == (9, 6), (nx, ny)
plain_feet = int(pb.memory[FEET_ADDR])
assert plain_feet != 0x52, hex(plain_feet)
r, info = cr(e, cur_state, cur_state)
assert "tall_grass" not in info, info
print(f"tall grass (REAL ROM): ({nx},{ny}) plain feet={plain_feet:#x} -> no reward")

press(pb, "up", hold=8, settle=28)  # onto genuine grass at (9,5)
cur_state = e._get_obs()["state"]
nx = int(round(cur_state[6] * 255))
ny = int(round(cur_state[7] * 255))
assert (nx, ny) == (9, 5), (nx, ny)
real_feet = int(pb.memory[FEET_ADDR])
assert real_feet == 0x52, hex(real_feet)  # UNFORCED real ROM byte
r, info = cr(e, cur_state, cur_state)
assert abs(info["tall_grass"] - 20.0) < 1e-9, info
print(
    f"tall grass (REAL ROM): ({nx},{ny}) genuine feet={real_feet:#x} "
    f"(no RAM write) -> +20"
)

press(pb, "up", hold=8, settle=28)  # still grass at (9,4), same episode
cur_state = e._get_obs()["state"]
nx = int(round(cur_state[6] * 255))
ny = int(round(cur_state[7] * 255))
feet_now = int(pb.memory[FEET_ADDR])
assert feet_now == 0x52, hex(feet_now)
r, info = cr(e, cur_state, cur_state)
assert "tall_grass" in info and abs(info["tall_grass"]) < 1e-9, info
print(f"tall grass (REAL ROM): ({nx},{ny}) still grass -> 0.0 (one-shot)")

e.close()
print("PROBE OK")
