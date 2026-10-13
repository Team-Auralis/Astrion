"""Diagnostic: where does the boot state sit, and can exploration reach a starter?

Boots the ROM exactly like PokemonEnv does, prints RAM, then runs a random
policy and reports max progress (party, badges, maps, position).
"""

import io
import sys
import numpy as np

from pyboy import PyBoy

ROM = (
    sys.argv[1]
    if len(sys.argv) > 1
    else r"D:\roms\Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb"
)
TICKS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000

pb = PyBoy(ROM, window="null", sound_emulated=False)
pb.set_emulation_speed(0)

# Same boot advance as PokemonEnv
for _ in range(TICKS):
    pb.tick()


def rd(a):
    return pb.memory[a]


def rd16(a):
    return (pb.memory[a] << 8) | pb.memory[a + 1]


def snap(tag):
    print(f"--- {tag} ---")
    print(f"  title        : {pb.cartridge_title}")
    print(f"  party_count  : {rd(0xD163)}")
    print(f"  map_id       : {rd(0xD35E)}")
    print(f"  player x,y   : {rd(0xD362)},{rd(0xD361)}")
    print(f"  badges       : {rd(0xD356):08b}")
    print(f"  in_battle    : {rd(0xD057)}")
    print(f"  hp/max       : {rd16(0xD16C)}/{rd16(0xD18D)}")
    print(f"  level        : {rd(0xD18C)}")
    print(f"  game_state   : {rd(0xD355)}")
    print(f"  dialogue     : {rd(0xD35D)}")


snap("boot @ %d ticks" % TICKS)

# Save a screenshot for the user
try:
    img = pb.screen.ndarray
    from PIL import Image

    Image.fromarray(img[:, :, :3]).save("logs/boot_screen.png")
    print("saved logs/boot_screen.png")
except Exception as e:
    print("screenshot failed:", e)

# Snapshot the boot state so we can replay
buf = io.BytesIO()
pb.save_state(buf)
boot = buf.getvalue()

buttons = ["a", "b", "start", "select", "up", "down", "left", "right"]
rng = np.random.default_rng(0)

max_party = 0
max_badges = 0
maps = set()
max_level = 0
start_party = rd(0xD163)

pb.load_state(io.BytesIO(boot))
for i in range(STEPS):
    b = buttons[rng.integers(0, 8)]
    pb.button_press(b)
    for _ in range(4):
        pb.tick()
    pb.button_release(b)
    for _ in range(4):
        pb.tick()
    p = rd(0xD163)
    maps.add(rd(0xD35E))
    max_party = max(max_party, p)
    max_badges = max(max_badges, bin(rd(0xD356)).count("1"))
    max_level = max(max_level, rd(0xD18C))
    if i % 500 == 0:
        print(
            f"  step {i}: party={p} badges={max_badges} maps={len(maps)} lvl={max_level} map_id={rd(0xD35E)} @({rd(0xD362)},{rd(0xD361)})"
        )

print("=== RANDOM POLICY RESULT ===")
print(f"  start party   : {start_party}")
print(f"  max party     : {max_party}")
print(f"  max badges    : {max_badges}")
print(f"  max level     : {max_level}")
print(f"  maps visited  : {sorted(maps)}")
snap("after random exploration")
pb.stop()
