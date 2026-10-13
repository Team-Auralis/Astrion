"""Probe: PokeRL per-map coverage observation (RAM feature, no CNN).

RESEARCH_FASTER_TRAINING.md 4.5: add spatial memory via a per-map visited
mask. This is the cheap RAM-feature version -- the observation stays
image-free; pokerl_mode only grows the state vector 20 -> 30.

Checks:
- pokerl_mode=True yields state_feature_size == 30 and a 30-dim vector;
  pokerl_mode=False is byte-identical at 20 dims.
- walking onto new tiles increases current-map coverage (mark via
  _update_map_coverage, then read from the observation).
- the coverage tail is zeros off-world (unnameable map), matching the
  existing off-world zeroing convention.
- the ring buffer reports the 8 most recently visited maps as
  (map_id/255.0, coverage) pairs, oldest->newest, zero-padded.
- the network builds with state_dim=30 (default 20) and accepts the vector.
- a REAL-ROM walk: reset, play the intro to Pallet Town, then step east onto
  new tiles and confirm coverage rises off the real RAM coords.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from orion.pokemon.environment import PokemonEnv
from orion.pokemon.network import PokemonResNet

ROM = r"D:\roms\Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb"


def feats(map_id=0x00, x=0, y=0):
    """Synthetic 20-dim state vector with a controllable map/x/y."""
    s = np.zeros(20, dtype=np.float32)
    s[4] = map_id / 255.0
    s[1] = 1 / 6.0
    s[3] = 6 / 100.0
    s[6] = x / 255.0
    s[7] = y / 255.0
    return s


# ---------------------------------------------------------------------------
# 1) Dimension: 30 in pokerl_mode, 20 (unchanged) by default.
# ---------------------------------------------------------------------------
e_pk = PokemonEnv(rom_path=ROM, headless=True, pokerl_mode=True)
assert e_pk.state_feature_size == 30, e_pk.state_feature_size
obs_pk, _ = e_pk.reset()
assert obs_pk["state"].shape == (30,), obs_pk["state"].shape

# W1: a fresh reset must NOT seed coverage from the boot/title frame. At boot
# the RAM reads map 0 (PALLET_TOWN, nameable) with x/y = (0,0), which the
# weaker _in_game_world gate would accept; the coverage gate now uses
# _pokerl_controllable_world, so no grid is created and the recent ring stays
# empty (no phantom Pallet entry).
assert e_pk.current_map_coverage == 0.0, e_pk.current_map_coverage
assert e_pk._current_map_key is None, e_pk._current_map_key
assert all(p == (0.0, 0.0) for p in e_pk.recent_map_coverage), e_pk.recent_map_coverage
# Explicitly replay a boot-frame update: still no grid, key or ring entry.
e_pk._update_map_coverage(feats(map_id=0x00, x=0, y=0))
assert e_pk._map_tile_grid == {}, e_pk._map_tile_grid
assert e_pk._current_map_key is None and not e_pk._map_recent
assert e_pk.current_map_coverage == 0.0
print("W1: fresh reset coverage=0.0, no phantom Pallet in grid/ring")

e_def = PokemonEnv(rom_path=ROM, headless=True, pokerl_mode=False)
assert e_def.state_feature_size == 20, e_def.state_feature_size
obs_def, _ = e_def.reset()
assert obs_def["state"].shape == (20,), obs_def["state"].shape
print("dims: pokerl_mode=30 (obs 30), default=20 (obs 20)")
e_def.close()

# ---------------------------------------------------------------------------
# 2) Coverage rises as new tiles are visited (synthetic in-world coords).
# ---------------------------------------------------------------------------
e_pk._map_tile_grid = {}
e_pk._map_recent.clear()
e_pk._current_map_key = None
assert e_pk.current_map_coverage == 0.0

c = []
# Non-origin tiles: the coverage gate rejects (0,0) as a boot bootstrap (W1),
# so real overworld tiles only.
for x, y in ((2, 3), (3, 3), (4, 3), (2, 4)):
    e_pk._update_map_coverage(feats(map_id=0x00, x=x, y=y))
    c.append(e_pk.current_map_coverage)
assert all(b > a for a, b in zip(c, c[1:])), c
# Re-visiting a known tile must NOT change coverage.
e_pk._update_map_coverage(feats(map_id=0x00, x=2, y=3))
assert abs(e_pk.current_map_coverage - c[-1]) < 1e-12
# The observation tail carries the coverage at index 20.
tail = e_pk._coverage_features(feats(map_id=0x00, x=2, y=3))
assert abs(tail[0] - c[-1]) < 1e-9, (tail[0], c[-1])
print(f"coverage: 0 -> {c[-1]:.4f} over {len(c)} new tiles; revisit unchanged")

# ---------------------------------------------------------------------------
# 3) Off-world tail is zeros (unnameable map), and off-world updates are no-ops.
# ---------------------------------------------------------------------------
off = feats(map_id=250, x=5, y=5)  # 250 is not a known Gen 1 map id
assert not e_pk._in_game_world(off), "probe vector must be off-world"
assert np.array_equal(e_pk._coverage_features(off), np.zeros(10, np.float32))
before = {k: v.copy() for k, v in e_pk._map_tile_grid.items()}
e_pk._update_map_coverage(off)
assert set(before) == set(e_pk._map_tile_grid), "off-world polluted the grid"
# Off-world the property agrees with the zeroed observation tail.
assert e_pk.current_map_coverage == 0.0, e_pk.current_map_coverage
assert e_pk._current_map_key is None, e_pk._current_map_key
print("off-world: coverage tail all zeros; grid untouched")

# ---------------------------------------------------------------------------
# 4) Ring buffer reports recent maps, oldest->newest, zero-padded.
# ---------------------------------------------------------------------------
e_pk._map_tile_grid = {}
e_pk._map_recent.clear()
e_pk._current_map_key = None
for mid in (0x00, 0x0C, 0x01):
    e_pk._update_map_coverage(feats(map_id=mid, x=1, y=1))
ring = e_pk.recent_map_coverage
assert len(ring) == 8, len(ring)
assert ring[0] == (0.0, 0.0) and ring[4] == (0.0, 0.0), ring  # padded front
assert abs(ring[-3][0] - 0x00 / 255.0) < 1e-9, ring
assert abs(ring[-2][0] - 0x0C / 255.0) < 1e-9, ring
assert abs(ring[-1][0] - 0x01 / 255.0) < 1e-9, ring  # most recent last
assert all(cov > 0 for _mid, cov in ring[-3:]), ring
print("ring buffer: 8 pairs oldest->newest, front zero-padded, ids/255")
e_pk.close()

# ---------------------------------------------------------------------------
# 5) Network accepts the 30-dim state; default still 20.
# ---------------------------------------------------------------------------
import torch

m30 = PokemonResNet(num_buttons=6, state_dim=30)
p30, v30 = m30(torch.randn(2, 4, 72, 80), torch.randn(2, 30))
assert p30.shape == (2, 6) and v30.shape == (2, 1), (p30.shape, v30.shape)
m20 = PokemonResNet(num_buttons=8)  # default STATE_DIM
p20, v20 = m20(torch.randn(2, 4, 72, 80), torch.randn(2, 20))
assert p20.shape == (2, 8), p20.shape
print("network: state_dim=30 forward OK; default 20 unchanged")

# ---------------------------------------------------------------------------
# 6) REAL ROM: walk east in Pallet Town and watch coverage rise.
# ---------------------------------------------------------------------------
from dbg_probe_pokerl_intro import press, red_intro_to_pallet

e = PokemonEnv(rom_path=ROM, headless=True, pokerl_mode=True)
pb = e._pyboy
e.reset()  # boot snapshot the intro recipe is calibrated on
assert red_intro_to_pallet(pb), "intro/door warp to Pallet Town failed"
e._extract_state_features()  # update the grid at the landing tile
cov0 = e.current_map_coverage
assert e._current_map_key == 0x00 and cov0 > 0.0, (e._current_map_key, cov0)

for _ in range(4):  # east along the door row onto fresh tiles
    press(pb, "right", hold=8, settle=28)
e._extract_state_features()
cov1 = e.current_map_coverage
assert cov1 > cov0, (cov0, cov1)
print(f"REAL ROM: Pallet coverage {cov0:.4f} -> {cov1:.4f} after walking east")
e.close()

print("PROBE OK")
