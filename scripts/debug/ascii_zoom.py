"""Render a cropped region of a screenshot as fine ASCII (readable text)."""

import os
import sys
from PIL import Image

CHARS = " .:-=+*#%@"


def render(path, x0, y0, x1, y1, cols=110):
    img = Image.open(path).convert("L")
    crop = img.crop((x0, y0, x1, y1))
    w, h = crop.size
    rows = max(10, int(cols * h / w * 0.5))
    crop = crop.resize((cols, rows))
    px = crop.load()
    out = []
    for y in range(rows):
        out.append("".join(CHARS[min(9, (px[x, y] * 10) // 256)] for x in range(cols)))
    print("\n".join(out))


if __name__ == "__main__":
    path = sys.argv[1]
    x0, y0, x1, y1 = map(int, sys.argv[2:6])
    render(path, x0, y0, x1, y1, cols=int(sys.argv[6]) if len(sys.argv) > 6 else 110)
