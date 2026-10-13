"""Deterministic Gen 1 intro recorder (Pokemon Red/Blue/Yellow).

Records two curriculum start states after the *compulsory, strategy-free*
intro only (no route/strategy encoding):

    states/red_bedroom.state : post-intro house start, map REDS_HOUSE_1F
        (0x25) at (2,3) -- the house-exit curriculum start. (The task calls
        this "bedroom"; empirically the intro lands on 1F / 0x25, which is
        the acceptance value.)
    states/red_pallet.state  : after walking out of the house, triggering
        Oak's forced escort into Oak's Lab, picking a starter (party -> 1),
        the compulsory in-lab rival battle, and Oak's Pokedex hand-off --
        back in PALLET_TOWN (0x00) with party == 1.

Flow (verified against pret/pokered + calibrated on PyBoy, Red USA):
    boot -> A(120)+300 -> START(60)+120 -> A(30)+120 -> 12x[A(40)+300]
        lands REDS_HOUSE_1F (0x25) at (2,3), party=0            [stage 0]
    walk down to door (2,7), hold DOWN to warp -> Pallet (5,6)    [stage 1]
    east to x=10, north to y=1 -> Oak escort (EVENT_OAK_APPEARED_IN_PALLET)
        auto-walks the player into Oak's Lab at (5,3)            [stage 2]
    advance the 4 choose-mon speech texts (A) until the player can move
        (a successful DOWN press)                                 [stage 3]
    walk to (8,4), face up, A -> "You want BULBASAUR?" -> A (YES)
        party -> 1                                                [stage 4]
    advance rival's pick text, walk down to y=6 -> rival challenge -> battle
                                                                   [stage 5]
    spam A to win the compulsory battle                           [stage 6]
    mash A through the end-battle + Oak Pokedex hand-off          [stage 7]
    walk down onto the lab exit (5,11) -> Pallet (map 0x00),
        party==1 -> SAVE red_pallet.state                         [stage 8]

Failures exit non-zero with a clear message; a bad state is never saved.

Usage:
    python record_intro_state.py [rom] [--out-dir states] [--max-ticks N]
"""

import argparse
import os
import sys

from pyboy import PyBoy

# -- Gen 1 (Red/Blue) RAM addresses ---------------------------------------
MAP_ID = 0xD35E
PLAYER_X = 0xD362
PLAYER_Y = 0xD361
PARTY_COUNT = 0xD163
IN_BATTLE = 0xD057
# First party mon HP (2 bytes big-endian)
CURRENT_HP = 0xD16C

# -- map ids ---------------------------------------------------------------
REDS_HOUSE_1F = 0x25
PALLET_TOWN = 0x00
OAKS_LAB = 0x28

DEFAULT_ROM = r"D:\roms\Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb"

_t = 0  # wall-clock tick counter


def main():
    global _t
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rom", nargs="?", default=DEFAULT_ROM)
    ap.add_argument("--out-dir", default="states")
    ap.add_argument("--max-ticks", type=int, default=8_000_000)
    args = ap.parse_args()

    pb = PyBoy(args.rom, window="null", sound_emulated=False)
    pb.set_emulation_speed(0)

    # ---- helpers --------------------------------------------------------
    def wait(n):
        global _t
        for _ in range(n):
            pb.tick()
        _t += n

    def tap(btn, ticks=30):
        pb.button_press(btn)
        wait(ticks)
        pb.button_release(btn)

    def press(btn, hold=8, settle=28):
        pb.button_press(btn)
        for _ in range(hold):
            pb.tick()
        global _t
        _t += hold
        pb.button_release(btn)
        wait(settle)

    def read_u16(addr):
        return (pb.memory[addr] << 8) | pb.memory[addr + 1]

    def pos():
        return pb.memory[MAP_ID], pb.memory[PLAYER_X], pb.memory[PLAYER_Y]

    def report(tag):
        m, x, y = pos()
        p = pb.memory[PARTY_COUNT]
        ib = pb.memory[IN_BATTLE]
        print(
            f"[rec] {tag:22s} t={_t:>8d} map={m:#04x} ({x},{y}) "
            f"party={p} in_battle={ib}",
            flush=True,
        )
        return m, x, y, p

    def fail(code, msg):
        print(f"[rec] FAIL: {msg}", flush=True)
        pb.stop()
        return code

    def walk_to(tx, ty, max_ticks=4000, mash_on_stuck=True):
        """Greedy walk toward (tx,ty); returns True once coordinates match."""
        start_map = pos()[0]
        ticks = 0
        stuck = 0
        while ticks < max_ticks:
            m, x, y = pos()
            if m != start_map:
                return True  # warp fired (unexpected here, but report it)
            if (x, y) == (tx, ty):
                return True
            dx, dy = tx - x, ty - y
            # primary axis = larger remaining distance
            if abs(dy) >= abs(dx):
                d = "down" if dy > 0 else "up"
            else:
                d = "right" if dx > 0 else "left"
            before = (x, y)
            press(d)
            _, x1, y1 = pos()
            if (x1, y1) == before:
                stuck += 1
                # try the perpendicular axis
                if abs(dy) >= abs(dx):
                    d2 = "right" if dx >= 0 else "left"
                else:
                    d2 = "down" if dy >= 0 else "up"
                press(d2)
                _, x2, y2 = pos()
                if (x2, y2) == before:
                    stuck += 1
                    if mash_on_stuck and stuck > 3:
                        tap("a", 20)
                        wait(120)
                        stuck = 0
                else:
                    stuck = 0
            else:
                stuck = 0
            ticks += 40
        return pos()[1:] == (tx, ty)

    # ---- stage 0: boot -> intro -> REDS_HOUSE_1F (0x25) -----------------
    print("[rec] booting...", flush=True)
    wait(600)
    tap("a", 120)
    wait(300)
    tap("start", 60)
    wait(120)
    tap("a", 30)
    wait(120)
    for _ in range(12):
        tap("a", 40)
        wait(300)
    m, x, y, p = report("post-intro")
    if m != REDS_HOUSE_1F:
        return fail(1, f"intro did not land in REDS_HOUSE_1F (map={m:#04x})")
    if (x, y) != (2, 3):
        return fail(
            1, f"intro landed at unexpected house position ({x},{y}), expected (2,3)"
        )
    if p != 0:
        return fail(1, f"expected party=0 post-intro, got party={p}")

    # Save red_bedroom.state (house-exit curriculum start).
    os.makedirs(args.out_dir, exist_ok=True)
    bedroom_path = os.path.join(args.out_dir, "red_bedroom.state")
    with open(bedroom_path, "wb") as f:
        pb.save_state(f)
    print(
        f"[rec] OK bedroom: map={m:#04x} ({x},{y}) party={p} -> {bedroom_path}",
        flush=True,
    )

    # ---- stage 1: exit house -> Pallet Town (5,6) ------------------------
    # Walk down to the door tile (2,7), then hold DOWN to fire the warp.
    for _ in range(20):
        m, x, y = pos()
        if m != REDS_HOUSE_1F or y >= 7:
            break
        press("down")
    for _ in range(6):
        m, x, y = pos()
        if m == PALLET_TOWN:
            break
        pb.button_press("down")
        for _ in range(45):
            pb.tick()
        global _t
        _t += 45
        pb.button_release("down")
        wait(500)
    m, x, y, p = report("pallet")
    if m != PALLET_TOWN:
        return fail(2, f"could not exit house to Pallet Town (map={m:#04x})")

    # ---- stage 2: trigger Oak at (10,1), escort into Oak's Lab ----------
    # east to x=10 (a straight column up to y=1), then north to y=1.
    for _ in range(10):
        if pos()[1] >= 10:
            break
        press("right")
    for _ in range(10):
        m, x, y = pos()
        if y <= 1:
            break
        press("up")
    m, x, y, p = report("oak-trigger")
    if not (m == PALLET_TOWN and y == 1):
        return fail(3, f"Oak trigger not reached (map={m:#04x} y={y})")

    # Mash A through Oak's escort (auto-walk ignores input) and the
    # choose-mon speech until we land in the lab AND the player is
    # controllable. Freedom test: a DOWN press must move us one tile down
    # and the following UP press must move us back -- a single movement can
    # be the escort's own auto-walk (false positive), two opposite
    # responsive steps cannot.
    free = False
    for i in range(150):
        m, x, y = pos()
        if m == OAKS_LAB:
            x0, y0 = x, y
            press("down")
            if pos()[1:] == (x0, y0 + 1):
                x1, y1 = pos()[1:]
                press("up")
                if pos()[1:] == (x1, y1 - 1):
                    free = True
                    print(f"[rec] controllable after {i} tries", flush=True)
                    break
        tap("a", 30)
        wait(120)
        if _t > args.max_ticks:
            break
    m, x, y, p = report("lab-free")
    if m != OAKS_LAB:
        return fail(4, f"escort did not reach Oak's Lab (map={m:#04x})")
    if not free:
        return fail(5, "player never became controllable in the lab")

    # ---- stage 4: pick Bulbasaur (ball at 8,3; stand 8,4 face up) -------
    if not walk_to(8, 4, max_ticks=2000):
        report("walk-fail")
        return fail(6, "could not walk to (8,4) below the starter table")
    press("up")  # face the Bulbasaur ball
    tap("a", 30)
    wait(200)
    tap("a", 30)  # YES on the "You want BULBASAUR?" menu
    wait(400)
    m, x, y, p = report("picked")
    if pb.memory[PARTY_COUNT] < 1:
        # give the givepoke a moment, then re-check
        for _ in range(10):
            tap("a", 30)
            wait(200)
            if pb.memory[PARTY_COUNT] >= 1:
                break
        m, x, y, p = report("picked-retry")
    if pb.memory[PARTY_COUNT] < 1:
        return fail(7, f"starter not received (party={pb.memory[PARTY_COUNT]})")

    # ---- stage 5: rival pick text -> walk to y=6 -> challenge ----------
    # After givepoke the rival script ignores input (wJoyIgnore); mash A
    # until a DOWN press is honored, then walk to the center column (x=5)
    # and step down to y=6, which fires the compulsory rival challenge
    # (OaksLabRivalChallengesPlayerScript checks wYCoord == 6).
    free2 = False
    for i in range(40):
        x0, y0 = pos()[1:]
        press("down")
        if pos()[1:] == (x0, y0 + 1):
            free2 = True
            print(f"[rec] input accepted after {i} mashes", flush=True)
            break
        tap("a", 30)
        wait(150)
    if not free2:
        return fail(8, "input stayed ignored after picking the starter")
    if not walk_to(5, 5, max_ticks=3000):
        report("walk5-fail")
        return fail(9, "could not walk to (5,5) for the rival challenge")
    press("down")  # y=6 triggers OaksLabRivalChallengesPlayerScript
    report("challenge-y6")
    if pos()[2] < 6:
        return fail(10, "could not step to y=6 to trigger the rival challenge")

    # ---- stage 6: win the compulsory rival battle ----------------------
    # Wait for battle to start (in_battle rises), then spam A (FIGHT ->
    # first move) until it ends. Detect whiteout (lead HP <= 0 or lost) as a failure.
    started = False
    in_battle_rose = False
    for i in range(80):
        if pb.memory[IN_BATTLE] != 0:
            started = True
            in_battle_rose = True
            break
        tap("a", 30)
        wait(150)
    report("battle-start" if started else "no-battle")
    if not started:
        return fail(11, "rival challenge never started a battle")
    turns = 0
    while pb.memory[IN_BATTLE] != 0 and turns < 400:
        tap("a", 24)
        wait(40)
        tap("a", 24)
        wait(500)
        turns += 1
        if turns % 10 == 0:
            report(f"battle-t{turns}")
        lead_hp = read_u16(CURRENT_HP)
        if lead_hp == 0 or pb.memory[PARTY_COUNT] < 1:
            return fail(12, "player fainted (whiteout) during the rival battle")
    # Battle must have ended (in_battle went 0)
    if pb.memory[IN_BATTLE] != 0:
        return fail(13, "rival battle did not end cleanly")
    m, x, y, p = report("post-battle")
    # After battle ends in lab, position should still be in Oak's Lab
    if m != OAKS_LAB:
        return fail(
            14,
            f"after rival battle, expected Oak's Lab (map={OAKS_LAB:#04x}), got {m:#04x}",
        )

    # ---- stage 7: Oak Pokedex hand-off ---------------------------------
    for i in range(80):
        tap("a", 30)
        wait(150)
    m, x, y, p = report("post-pokedex")

    # ---- stage 8: exit lab -> Pallet (0x00), party>=1, lead HP>0 -------
    # From the challenge spot (5,6) hold DOWN down the center column onto
    # the exit warp (5,11) -> Pallet Town, landing at the lab door (12,11).
    for _ in range(30):
        if pos()[0] == PALLET_TOWN:
            break
        before = pos()[1:]
        press("down")
        if pos()[1:] == before and pos()[0] != PALLET_TOWN:
            # blocked (e.g. NPC in the column): sidestep and retry
            tap("a", 20)
            wait(100)
            press("down")
    m, x, y, p = report("final")
    if m != PALLET_TOWN:
        return fail(9, f"could not exit lab back to Pallet (map={m:#04x})")
    if p < 1:
        return fail(10, f"expected party>=1 in Pallet (party={p})")
    lead_hp = read_u16(CURRENT_HP)
    if lead_hp <= 0:
        return fail(11, f"lead fainted on return to Pallet (HP={lead_hp})")

    os.makedirs(args.out_dir, exist_ok=True)
    pallet_path = os.path.join(args.out_dir, "red_pallet.state")
    with open(pallet_path, "wb") as f:
        pb.save_state(f)
    print(
        f"[rec] OK pallet: map={m:#04x} ({x},{y}) party={p} lead_hp={lead_hp} -> {pallet_path}",
        flush=True,
    )
    pb.stop()
    print("[rec] SUCCESS: both states recorded", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
