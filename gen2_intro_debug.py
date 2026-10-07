"""Screenshot debugger for the Gen 2 intro.

Boots a Gen 2 ROM, advances through the intro with the same button script
as gen2_intro.py, and dumps a PNG of the screen at each stage so we can see
where the intro flow actually stalls.

Usage:
    python gen2_intro_debug.py [rom] [outdir]
"""

import os
import sys

import numpy as np
from PIL import Image
from pyboy import PyBoy

ROM = (
    sys.argv[1]
    if len(sys.argv) > 1
    else r"D:\roms\Pokemon - Gold Version (USA, Europe) (SGB Enhanced) (GB Compatible).gbc"
)
OUT = sys.argv[2] if len(sys.argv) > 2 else "logs/intro_debug"
os.makedirs(OUT, exist_ok=True)


def main():
    pb = PyBoy(ROM, window="null", sound_emulated=False)
    pb.set_emulation_speed(0)

    def shot(tag):
        arr = np.asarray(pb.screen.ndarray, dtype=np.uint8)
        img = Image.fromarray(arr)
        p = os.path.join(OUT, f"{tag}.png")
        img.save(p)
        print(f"[dbg] {tag}: screen saved -> {p}", flush=True)

    def tap(btn, ticks=10):
        pb.button_press(btn)
        for _ in range(ticks):
            pb.tick()
        pb.button_release(btn)

    def wait(ticks):
        for _ in range(ticks):
            pb.tick()

    def mash(n, gap=80):
        for _ in range(n):
            tap("a", 10)
            wait(gap)

    # boot
    for _ in range(2400):
        pb.tick()
    shot("00_boot")

    tap("start", 10)
    wait(300)
    shot("01_after_start")

    mash(5, 80)
    wait(300)
    shot("02_mash5")

    mash(10, 80)
    wait(600)
    shot("03_mash15")

    tap("a", 10)
    wait(400)
    shot("04_menu_a")

    # wait longer in case a cinematic is playing
    mash(10, 80)
    wait(800)
    shot("05_longer_mash")

    pb.stop()


if __name__ == "__main__":
    main()
