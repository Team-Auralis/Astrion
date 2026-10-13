"""Diagnostic: boot Gold, press through the intro stages, dump PNGs at each
stage so the orchestrator can inspect the actual screen."""

import os
import sys
from pyboy import PyBoy
from PIL import Image


def main():
    rom = (
        sys.argv[1]
        if len(sys.argv) > 1
        else r"D:\roms\Pokemon - Gold Version (USA, Europe) (SGB Enhanced) (GB Compatible).gbc"
    )
    out = sys.argv[2] if len(sys.argv) > 2 else r"D:\orion-game-ai\dbg"
    os.makedirs(out, exist_ok=True)

    pb = PyBoy(rom, window="null", sound_emulated=False)
    pb.set_emulation_speed(0)
    t = 0

    def snap(tag):
        arr = pb.screen.ndarray
        img = Image.fromarray(arr[:, :, :3])
        p = os.path.join(out, f"{tag}_{t}.png")
        img.save(p)
        print(f"[dbg] {tag} t={t} saved {p}", flush=True)

    def wait(n):
        nonlocal t
        for _ in range(n):
            pb.tick()
        t += n

    def tap(btn, n=10):
        pb.button_press(btn)
        wait(n)
        pb.button_release(btn)

    print("[dbg] booting...", flush=True)
    wait(2600)
    snap("boot_after2600")

    tap("a")  # cancel splash if up
    wait(100)
    snap("after_a1")

    wait(2900)
    snap("pre_start")

    for _ in range(4):
        tap("start")
        wait(700)
    wait(900)
    snap("after_start")

    for i in range(12):
        tap("a")
        wait(700)
        tt = pb.memory[0xC765] | (pb.memory[0xC766] << 8)
        print(f"[dbg] A press {i}: tt={tt} tbuf={pb.memory[0xC900]}", flush=True)
        if i in (0, 3, 6, 11):
            snap(f"a_press_{i}")

    for _ in range(6):
        tap("a")
        wait(500)
    wait(500)
    snap("clock_stage")

    for i in range(6):
        tap("down")
        wait(120)
        tap("a")
        wait(350)
        if i in (1, 3, 5):
            snap(f"hammer_{i}")

    wait(600)
    snap("end")
    pb.stop()
    print("[dbg] done", flush=True)


if __name__ == "__main__":
    main()
