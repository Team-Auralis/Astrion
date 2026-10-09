"""Consolidated M1 probe: PokeRL env + curriculum + recurrent PPO knobs.

One probe covering the six M1 features (RESEARCH_FASTER_TRAINING.md 7) that
landed in Tasks 1-6, so the whole bundle can be validated with a single
command instead of re-running four probes by hand:

1. Reduced action set: pokerl_mode exposes 6 buttons (a,b,up,down,left,right,
   no start/select); default mode keeps all 8.
2. Anti-loop / anti-spam magnitudes: a short stuck loop fires the visit/loop/
   pattern/spam layers with the exact values asserted in dbg_probe_antiloop.py
   (and default mode stays a no-op).
3. Hierarchical reward tiers: map transition first-entry +15 (10 transition +
   5 entry) vs re-entry +10, new tile +1.0 then 0, one-shot tall grass +20
   then 0, battle start +10, battle win +50.
4. Coverage slots: state_feature_size 30 with pokerl / 20 without; the 10-dim
   coverage tail is nonzero in-world and all zeros off-world.
5. Savestate fast-path: states/red_pallet.state boots controllable at step 0
   in PALLET_TOWN (map 0x00) with party == 1 (SKIPped when the state or ROM
   is absent, not failed).
6. Recurrent build: PokemonRecurrentNet has LSTM params (PokemonResNet does
   not); a cheap hx=None forward returns ((logits, value), hx_new) with the
   right shapes.

Follows the dbg_probe_exp.py "PROBE OK" / exit-0 convention. ROM-dependent
sections (1-5) are SKIPped with a clear line -- not failed -- when the ROM
(or, for the fast-path, the recorded states) is missing.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from orion.pokemon.network import PokemonRecurrentNet, PokemonResNet

ROM = r"D:\roms\Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb"
PALLET_STATE = os.path.join("states", "red_pallet.state")
FEET_ADDR = 0xC4A0 + 9 * 20 + 9  # wTileMap row 9 col 9 (0xC55D), the feet tile
NEW_INFO_KEYS = (
    "visit_penalty",
    "loop_penalty",
    "pattern_penalty",
    "spam_penalty",
    "diversity_bonus",
)

ROM_PRESENT = os.path.exists(ROM)


def in_world(map_id=40, x=5, y=5, in_battle=0, party=1, level=6, exp=3000.0):
    """Synthetic 20-dim state vector that passes the in-world gate.

    map 40 (OAKS_LAB) is nameable (in-world); map 250 is not (off-world).
    """
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
    return e._compute_reward({"state": old}, {"state": new})


def skip(what):
    print(f"SKIP: {what}")


# ===========================================================================
# 6) Recurrent build (no ROM required)
# ===========================================================================
ff = PokemonResNet(num_buttons=6, state_dim=30)
rec = PokemonRecurrentNet(num_buttons=6, state_dim=30)
ff_lstm = [n for n, _ in ff.named_parameters() if "lstm" in n]
rec_lstm = [n for n, _ in rec.named_parameters() if "lstm" in n]
assert not ff_lstm, f"PokemonResNet unexpectedly has LSTM params: {ff_lstm}"
assert rec_lstm, "PokemonRecurrentNet has no LSTM params"
(logits, value), hx_new = rec(torch.randn(1, 4, 72, 80), torch.randn(1, 30), None)
assert isinstance(logits, torch.Tensor) and logits.shape == (1, 6), logits.shape
assert value.shape == (1, 1), value.shape
assert hx_new is not None and len(hx_new) == 2, hx_new
h, c = hx_new
assert h.shape == (1, 1, 256) and c.shape == (1, 1, 256), (h.shape, c.shape)
print(
    f"recurrent: {len(rec_lstm)} LSTM params (FF has 0); hx=None forward -> "
    f"logits {tuple(logits.shape)}, value {tuple(value.shape)}, h {tuple(h.shape)}"
)

if not os.path.exists(ROM):
    skip(f"ROM not found at {ROM!r}; skipping action/anti-loop/reward/coverage")
else:
    from orion.pokemon.environment import PokemonEnv

    e_pk = PokemonEnv(rom_path=ROM, headless=True, state_path=None, pokerl_mode=True)
    e_def = PokemonEnv(rom_path=ROM, headless=True, state_path=None, pokerl_mode=False)

    # -----------------------------------------------------------------------
    # 1) Reduced action set
    # -----------------------------------------------------------------------
    assert e_pk.action_space.n == 6, e_pk.action_space.n
    assert tuple(e_pk._action_buttons) == ("a", "b", "up", "down", "left", "right")
    assert e_def.action_space.n == 8, e_def.action_space.n
    print("action set: pokerl=6 (a,b,up,down,left,right, no start/select); default=8")

    # -----------------------------------------------------------------------
    # 2) Anti-loop / anti-spam magnitudes (dbg_probe_antiloop assertions)
    # -----------------------------------------------------------------------
    # Default mode: helper is a byte-identical no-op and no keys leak.
    pen, pinfo = e_def._pokerl_antiloop(0, in_world())
    assert pen == 0.0 and pinfo == {}, (pen, pinfo)
    e_def.reset()
    _, _, _, _, info_def = e_def.step(0)
    leaked = [k for k in NEW_INFO_KEYS if k in info_def]
    assert not leaked, f"default mode leaked anti-loop keys: {leaked}"

    # Forced 10-step same-position loop: every layer fires with its magnitude.
    e_pk.reset()
    state = in_world()
    totals = {k: 0.0 for k in NEW_INFO_KEYS}
    total_pen = 0.0
    last_info = {}
    for _ in range(10):
        pen, pinfo = e_pk._pokerl_antiloop(0, state)
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
    assert abs(sum(totals.values()) - total_pen) < 1e-6, (totals, total_pen)
    print(
        f"anti-loop 10-step loop: total={total_pen:.3f} "
        f"per-key={ {k: round(v, 3) for k, v in totals.items()} }"
    )

    # Varied rollout: anti-loop penalties ~0, diversity bonus accrues.
    e_pk.reset()
    totals = {k: 0.0 for k in NEW_INFO_KEYS}
    for i in range(12):
        pen, pinfo = e_pk._pokerl_antiloop(i % 6, in_world(x=(i % 6) + 1, y=1))
        for k, v in pinfo.items():
            totals[k] += v
        for k in ("visit_penalty", "loop_penalty", "pattern_penalty", "spam_penalty"):
            assert k not in pinfo, (i, pinfo)
    for k in ("visit_penalty", "loop_penalty", "pattern_penalty", "spam_penalty"):
        assert abs(totals[k]) < 1e-9, (k, totals[k])
    assert 0.10 <= totals["diversity_bonus"] <= 0.12, totals["diversity_bonus"]

    # Per-step cap: 64-step identical streak clamps to -0.5; step 8 uncapped.
    e_pk.reset()
    cap_hit = False
    for i in range(64):
        pen, _ = e_pk._pokerl_antiloop(0, in_world())
        assert pen >= -0.5, (i, pen)
        if i == 7:
            assert abs(pen - (-0.37)) < 1e-6, (i, pen)
        if i == 63:
            assert pen == -0.5, (i, pen)
            cap_hit = True
    assert cap_hit
    print("anti-loop varied rollout ~0 + diversity bonus; cap -0.5 engages")

    # -----------------------------------------------------------------------
    # 3) Hierarchical reward tiers
    # -----------------------------------------------------------------------
    # Map transition: first entry = 10 transition + 5 entry = 15; re-entry = 10.
    e_pk._clear_antiloop()
    e_pk._ep_map_families.add("CITY")  # VIRIDIAN_CITY family pre-seen
    start = in_world(map_id=0x00, x=2, y=2)  # PALLET_TOWN
    enter = in_world(map_id=0x01, x=2, y=2)  # VIRIDIAN_CITY
    _, info = cr(e_pk, start, enter)
    assert abs(info["map_transition"] - 10.0) < 1e-9, info
    assert abs(info["new_map_entry"] - 5.0) < 1e-9, info
    assert abs(info["map_transition"] + info["new_map_entry"] - 15.0) < 1e-9
    cr(e_pk, enter, start)
    _, info = cr(e_pk, start, enter)
    assert abs(info["map_transition"] - 10.0) < 1e-9, info
    assert "new_map_entry" not in info, info
    print("reward tiers: map first-entry +15 (10+5), re-entry +10")

    # New tile: +1.0 first time, 0 on revisit in the episode.
    e_pk._clear_antiloop()
    a = in_world(map_id=0x0C, x=8, y=8)
    b = in_world(map_id=0x0C, x=8, y=9)
    _, info = cr(e_pk, a, b)
    assert abs(info["new_tile"] - 1.0) < 1e-9, info
    _, info = cr(e_pk, b, b)
    assert "new_tile" not in info, info
    print("reward tiers: new tile +1.0 then 0 on revisit")

    # Battle: +10 on the rising edge, +50 on an EXP-gaining falling edge.
    e_pk._clear_antiloop()
    pre = in_world(map_id=0x0C, x=8, y=9, in_battle=0, exp=3000.0)
    battle = in_world(map_id=0x0C, x=8, y=9, in_battle=1, exp=3000.0)
    _, info = cr(e_pk, pre, battle)
    assert abs(info["battle_start"] - 10.0) < 1e-9, info
    win = in_world(map_id=0x0C, x=8, y=9, in_battle=0, exp=3300.0)
    _, info = cr(e_pk, battle, win)
    assert abs(info["battle_win"] - 50.0) < 1e-9, info
    print("reward tiers: battle start +10, win +50")

    # Tall grass: one-shot +20 then 0 (game-authoritative RAM feet byte).
    e_pk._clear_antiloop()
    feet_addr = 0xC4A0 + 9 * 20 + 9
    e_pk._pyboy.memory[FEET_ADDR] = 0x52
    g = in_world(map_id=0x0C, x=8, y=9)
    _, info = cr(e_pk, g, g)
    assert abs(info["tall_grass"] - 20.0) < 1e-9, info
    _, info = cr(e_pk, g, g)
    assert "tall_grass" in info and abs(info["tall_grass"]) < 1e-9, info
    e_pk._pyboy.memory[FEET_ADDR] = 0x01
    print("reward tiers: tall grass +20 then 0 (one-shot)")

    # -----------------------------------------------------------------------
    # 4) Coverage slots
    # -----------------------------------------------------------------------
    assert e_pk.state_feature_size == 30, e_pk.state_feature_size
    assert e_def.state_feature_size == 20, e_def.state_feature_size
    obs_pk, _ = e_pk.reset()
    assert obs_pk["state"].shape == (30,), obs_pk["state"].shape
    obs_def, _ = e_def.reset()
    assert obs_def["state"].shape == (20,), obs_def["state"].shape

    e_pk._map_tile_grid = {}
    e_pk._map_recent.clear()
    e_pk._current_map_key = None
    for x, y in ((2, 3), (3, 3), (4, 3)):
        e_pk._update_map_coverage(in_world(map_id=0x00, x=x, y=y))
    tail = e_pk._coverage_features(in_world(map_id=0x00, x=2, y=3))
    assert tail.shape == (10,) and tail[0] > 0.0, tail
    off = in_world(map_id=250, x=5, y=5)
    assert not e_pk._in_game_world(off), "probe vector must be off-world"
    assert np.array_equal(e_pk._coverage_features(off), np.zeros(10, np.float32))
    print(
        f"coverage: state_dim pokerl=30/default=20; tail in-world[0]={tail[0]:.4f}, "
        "off-world zeros"
    )

    e_pk.close()
    e_def.close()

# ===========================================================================
# 5) Savestate fast-path (curriculum)
# ===========================================================================
if not os.path.exists(ROM):
    skip(f"ROM not found at {ROM!r}; skipping curriculum fast-path")
elif not os.path.isfile(PALLET_STATE):
    skip(f"{PALLET_STATE} not found; run `python record_intro_state.py <rom.gb>` first")
else:
    from orion.pokemon.environment import PokemonEnv

    e_fast = PokemonEnv(
        rom_path=ROM, headless=True, state_path=PALLET_STATE, pokerl_mode=True
    )
    obs, _ = e_fast.reset()
    st = obs["state"]
    assert e_fast._map_key(st) == 0x00, hex(e_fast._map_key(st))
    assert int(st[1] * 6) == 1, st[1]
    assert not bool(st[5] > 0), "must boot out of battle"
    assert e_fast._pokerl_controllable_world(st), "not controllable at step 0"
    print(
        f"curriculum pallet: boots at step 0 in PALLET_TOWN (0x00) "
        f"({int(round(st[6] * 255))},{int(round(st[7] * 255))}) party=1, controllable"
    )
    e_fast.close()

print("PROBE OK")
