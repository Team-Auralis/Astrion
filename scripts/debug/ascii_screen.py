"""Render saved screenshots as ASCII art so the orchestrator can 'see' them."""

import os
import sys
from PIL import Image

CHARS = " .:-=+*#%@"
W, H = 48, 22


def render(path):
    img = Image.open(path).convert("L").resize((W, H))
    px = img.load()
    lines = []
    for y in range(H):
        row = "".join(CHARS[min(9, (px[x, y] * 10) // 256)] for x in range(W))
        lines.append(row)
    print("\n".join(lines))


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else r"D:\orion-game-ai\dbg"
    picks = sys.argv[2:] or [
        "pre_start_5610.png",
        "after_start_9350.png",
        "a_press_3_12190.png",
        "a_press_11_17870.png",
        "clock_stage_21430.png",
        "hammer_5_24370.png",
    ]
    for p in picks:
        path = os.path.join(d, p)
        if not os.path.exists(path):
            print(f"--- {p}: MISSING")
            continue
        print(f"\n=== {p} ===")
        render(path)


if __name__ == "__main__2":
    pass
