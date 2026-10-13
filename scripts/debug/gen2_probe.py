"""Empirical Gen 2 RAM fingerprinting.

Boots a Gen 2 ROM in PyBoy, plays a mostly-A bot to advance the intro, and
diffs full WRAM before/after to locate game-progress variables (party count,
map group/number, money, badges). The pret disassembly computes these at
assemble time, so we derive them from observed behavior instead.

Usage:
    python gen2_probe.py [ticks] [rom]
"""

import sys
import os
import numpy as np
from pyboy import PyBoy

PROG = os.path.join("logs", "gen2_probe_progress.log")
os.makedirs("logs", exist_ok=True)


def note(msg):
    with open(PROG, "a") as f:
        f.write(f"{msg}\n")
    print(msg, flush=True)


ROM = (
    sys.argv[2]
    if len(sys.argv) > 2
    else r"D:\roms\Pokemon - Gold Version (USA, Europe) (SGB Enhanced) (GB Compatible).gbc"
)
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000

pb = PyBoy(ROM, window="null", sound_emulated=False)
pb.set_emulation_speed(0)

try:
    note("cartridge: " + str(pb.cartridge_title))
except Exception:
    pass
note("ticking through boot logos...")
for _ in range(3000):
    pb.tick()
note("boot done")


def dump(svbk=1):
    try:
        pb.memory[0xFF70] = svbk
    except Exception:
        pass
    return {a: pb.memory[a] for a in range(0xC000, 0xE000)}


A0 = dump(1)

btns = ["a", "b", "start", "select", "up", "down", "left", "right"]
rng = np.random.default_rng(7)
frame = 0
log_step = TICKS // 20
while frame < TICKS:
    r = rng.random()
    if r < 0.55:
        b = "a"
    elif r < 0.65:
        b = "start"
    elif r < 0.75:
        b = "down"
    elif r < 0.80:
        b = "right"
    else:
        b = btns[int(rng.integers(8))]
    hold = 12 if b in ("up", "down", "left", "right") else 8
    pb.button_press(b)
    for _ in range(hold):
        pb.tick()
        frame += 1
    pb.button_release(b)
    for _ in range(2):
        pb.tick()
        frame += 1
    if frame % log_step < 20:
        note(f"  tick {frame}")

note("bot done at %d ticks; dumping final WRAM..." % frame)
A1 = dump(1)

diff = [(a, A0[a], A1[a]) for a in range(0xC000, 0xE000) if A0[a] != A1[a]]
note("changed bytes: %d / 8192" % len(diff))

# Page-level activity so we can see which WRAM regions are in play
pages = {}
for a, v0, v1 in diff:
    pg = a // 0x100
    pages[pg] = pages.get(pg, 0) + 1
note("activity per 256B page (page: count):")
for pg in sorted(pages):
    note("  %02X: %d" % (pg, pages[pg]))

# Bytes that became a small nonzero value from 0 (candidates for counters)
small = [(a, v1) for a, v0, v1 in diff if v0 == 0 and 1 <= v1 <= 8]
note("candidates that became 1..8 (addr: value): %s" % (small[:60],))

# Also report the tail of D0xx-D1xx and known wPlayerData-ish regions verbatim
note("D300-D3FF:")
note(" ".join("%02X" % A1[a] for a in range(0xD300, 0xD400)))
note("D400-D4FF:")
note(" ".join("%02X" % A1[a] for a in range(0xD400, 0xD500)))

pb.stop()
