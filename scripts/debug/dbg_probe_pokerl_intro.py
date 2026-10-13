"""Red intro skipper and verified step walker, calibrated against the real
Pokemon Red ROM (used by dbg_probe_pokerl.py section 7).

Calibration (empirically verified on PyBoy, Pokemon Red USA):
- boot 600 -> A(120)+300 -> START(60)+120 -> A(30)+120 -> 12x [A(40)+300]
  lands the player in-world in REDS_HOUSE_1F at tile (2,3), party=0.
- Single steps down (2,3) -> (2,7); the door tile alone does not warp
  (CheckWarpsNoCollision needs a held direction). Holding DOWN 45 ticks on
  the door tile reliably warps to Pallet Town at (5,6) (map 0x00).
- press(d, hold=10, settle=24): wX/wY update only AFTER release (~ticks
  16-34), so position is read only after the settle window.
"""

ROM = r"D:\roms\Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb"
MAP = 0xD35E
X = 0xD362
Y = 0xD361
TILEMAP = 0xC4A0
FEET = TILEMAP + 9 * 20 + 9  # 0xC55D

_t = 0


def _wait(pb, n):
    global _t
    for _ in range(n):
        pb.tick()
    _t += n


def press(pb, d, hold=10, settle=24):
    """Press a direction for `hold` ticks, release, settle `settle` ticks."""
    global _t
    pb.button_press(d)
    for _ in range(hold):
        pb.tick()
        _t += 1
    pb.button_release(d)
    _wait(pb, settle)


def tap(pb, btn, n=30):
    pb.button_press(btn)
    _wait(pb, n)
    pb.button_release(btn)


def _pos(pb):
    return pb.memory[MAP], pb.memory[X], pb.memory[Y]


def _walk_direction(x, y, tx, ty):
    """One-step direction to move from (x,y) toward (tx,ty)."""
    if y > ty:
        return "up"
    if y < ty:
        return "down"
    if x > tx:
        return "left"
    return "right"


def red_intro_to_pallet(pb, max_door_tries=5):
    """Play the Red intro and walk out of the house into Pallet Town.

    Ends with the player standing in Pallet Town (map 0x00) near the door
    exit (5,6). Returns True on success; retries the door warp (the warp
    occasionally needs more than one held-down pass).
    """
    global _t
    _t = 0
    _wait(pb, 600)
    tap(pb, "a", 120)
    _wait(pb, 300)
    tap(pb, "start", 60)
    _wait(pb, 120)
    tap(pb, "a", 30)
    _wait(pb, 120)
    for _ in range(12):
        tap(pb, "a", 40)
        _wait(pb, 300)

    m, x, y = _pos(pb)
    if m != 0x25 or (x, y) != (2, 3):
        # Tolerate a slightly different resting tile, but stay in the house.
        if m != 0x25:
            raise RuntimeError("intro did not reach REDS_HOUSE_1F (map=%02x)" % m)

    # Walk down to the door tile (2,7).
    for _ in range(30):
        m, x, y = _pos(pb)
        if m != 0x25 or (x, y) in ((2, 7), (3, 7)):
            break
        press(pb, _walk_direction(x, y, 2, 7))

    for _ in range(max_door_tries):
        m, x, y = _pos(pb)
        if m == 0x00:
            return True  # already in Pallet (e.g. a previous try warped)
        # Stand on the door tile, then hold DOWN to fire the warp.
        for _ in range(10):
            m, x, y = _pos(pb)
            if x >= 2 and y >= 7 and m == 0x25:
                break
            if m != 0x25:
                break
            press(pb, _walk_direction(x, y, 2, 7))
        pb.button_press("down")
        _wait(pb, 45)
        pb.button_release("down")
        _wait(pb, 500)  # door exit animation
        m, x, y = _pos(pb)
        if m == 0x00:
            return True
    return False


def red_intro_walk_to(pb, tx, ty, state, max_steps=30):
    """Walk (map 0x00 only) to tile (tx, ty) using the verified step recipe.

    Returns (walked, state): `walked` is False when the player left Pallet
    Town or the tile was unreachable. `state` is the (unchanged) input state
    -- the caller re-reads live state from its own env after a successful
    walk.
    """
    for i in range(max_steps):
        m, x, y = _pos(pb)
        if m != 0x00:
            return False, state
        if (x, y) == (tx, ty):
            return True, state
        before = (x, y)
        press(pb, _walk_direction(x, y, tx, ty))
        m, x, y = _pos(pb)
        if m != 0x00:
            return False, state
        if (x, y) == before:
            # Blocked: try a perpendicular step before giving up.
            alt = "left" if _walk_direction(x, y, tx, ty) in ("up", "down") else "up"
            press(pb, alt)
            m, x, y = _pos(pb)
            if m != 0x00 or (x, y) == before:
                return False, state
    m, x, y = _pos(pb)
    return (m == 0x00 and (x, y) == (tx, ty)), state
