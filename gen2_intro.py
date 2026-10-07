"""Deterministic Gen 2 intro skipper (Gold/Silver; Crystal uses its own addresses).

Flow verified from the pret pokegold disassembly (not guessed):
    Title (poll wTitleScreenTimer) -> START -> main menu (default NEW GAME)
    A on NEW GAME -> NewGame: ResetWRAM -> OakSpeech -> InitClock (6x A, 10:00)
    -> Oak texts + MARILL pic -> name menu (default NEW NAME)
    -> DOWN+A hammer: DOWN first (no-op during text) so A never hits NEW NAME
    -> "GOLD" stored (poll wPlayerName 01:D002)
    -> OakText7 -> InitializeWorld -> bedroom (24,7)
    -> greedy walk to stairs (7,0) -> 1F (24,6); walk to door (7,7) -> New Bark (24,4)
    -> walk to lab door (6,3) -> Elms Lab (24,5)
    -> scene mash -> walk to the Pokeballs, A -> YES -> givepoke -> party>=1
    -> mash Elm's directions -> SAVE STATE (consistent overworld, party=1)

The walker is coordinate-greedy with wall-slide; when coordinates stop changing
(forced-text scenes like Mom / Elm) it presses A to advance the script. Each
stage reports (map, xy, name, party, markers) so failures are visible in one run.

Usage:
    python gen2_intro.py [rom] [--save out.state] [--max-ticks N]
"""

import argparse
import os
import sys

from pyboy import PyBoy

NAME_LEN = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "rom",
        nargs="?",
        default=r"D:\roms\Pokemon - Gold Version (USA, Europe) (SGB Enhanced) (GB Compatible).gbc",
    )
    ap.add_argument("--save", default=r"D:\orion-game-ai\states\gen2_starter.state")
    ap.add_argument("--max-ticks", type=int, default=6_000_000)
    args = ap.parse_args()

    pb = PyBoy(args.rom, window="null", sound_emulated=False)
    pb.set_emulation_speed(0)

    title = str(getattr(pb, "cartridge_title", "")).upper()
    is_crystal = "CRYSTAL" in title
    if is_crystal:
        MAP_GROUP = (1, 0xDCDA)
        MAP_NUMBER = (1, 0xDCDB)
        Y_COORD = (1, 0xDCDC)
        X_COORD = (1, 0xDCDD)
        PARTY = (6, 0xD44A)
        NAME = (1, 0xD4A2)
        TITLE_TIMER = (1, 0xC765)  # same WRAM as gold
        TITLE_OPTION = (1, 0xC764)
    else:
        MAP_GROUP = (1, 0xD85F)
        MAP_NUMBER = (1, 0xD860)
        Y_COORD = (1, 0xD861)
        X_COORD = (1, 0xD862)
        PARTY = (1, 0xD881)
        NAME = (1, 0xD002)
        TITLE_TIMER = (1, 0xC765)
        TITLE_OPTION = (1, 0xC764)

    # Diagnostic markers (all bank 0).
    M_JUMPTABLE = 0xC763  # wJumptableIndex
    M_MENU_CURSOR = 0xC7E0  # wMenuCursorY
    M_WHICH_INDEX = 0xC7AD  # wWhichIndexSet
    M_NAMING_TYPE = 0xC9CC  # wNamingScreenType (>0 = keyboard open)
    M_TIME_BUF = 0xC900  # wTimeSetBuffer (nonzero while clock screen up)

    def read_bank(bank, addr):
        """Read WRAM with the right SVBK bank switched in (bank-1 safe)."""
        prev = pb.memory[0xFF70]
        if prev != bank:
            pb.memory[0xFF70] = bank
        try:
            return pb.memory[addr]
        finally:
            if prev != bank:
                pb.memory[0xFF70] = prev

    def read_map():
        return read_bank(*MAP_GROUP), read_bank(*MAP_NUMBER)

    def read_xy():
        return read_bank(*Y_COORD), read_bank(*X_COORD)

    def read_party():
        return read_bank(*PARTY)

    def read_name():
        base = NAME[1]
        prev = pb.memory[0xFF70]
        if prev != NAME[0]:
            pb.memory[0xFF70] = NAME[0]
        try:
            raw = bytes(pb.memory[base + i] for i in range(NAME_LEN))
        finally:
            if prev != NAME[0]:
                pb.memory[0xFF70] = prev
        return "".join(chr(c) for c in raw if 0x20 <= c < 0x7F)

    def read_u16(addr):
        return pb.memory[addr] | (pb.memory[addr + 1] << 8)

    def marker_str():
        return (
            f"jt={pb.memory[M_JUMPTABLE]} cur={pb.memory[M_MENU_CURSOR]} "
            f"idx={pb.memory[M_WHICH_INDEX]} nst={pb.memory[M_NAMING_TYPE]} "
            f"tbuf={pb.memory[M_TIME_BUF]}"
        )

    print(f"[intro] {title} crystal={is_crystal}", flush=True)

    # ---- button helpers ------------------------------------------------
    def tap(btn, ticks=10):
        pb.button_press(btn)
        for _ in range(ticks):
            pb.tick()
        pb.button_release(btn)

    def hold(btn, ticks=18):
        pb.button_press(btn)
        for _ in range(ticks):
            pb.tick()
        pb.button_release(btn)

    def wait(ticks):
        for _ in range(ticks):
            pb.tick()

    t = 0  # wall-clock tick counter

    def report(tag):
        g, m = read_map()
        p = read_party()
        nm = read_name()
        y, x = read_xy()
        print(
            f"[intro] {tag:16s} t={t:>8d} map=({g},{m}) xy=({x},{y}) "
            f"party={p} name='{nm}' {marker_str()}",
            flush=True,
        )
        return (g, m), p

    def wait_until(fn, timeout, tag, gap=60):
        """Run frames until fn() is true or timeout ticks pass; report on exit."""
        nonlocal t
        waited = 0
        while waited < timeout:
            if fn():
                print(f"[intro] {tag}: reached after {waited} ticks", flush=True)
                return True
            wait(gap)
            waited += gap
            t += gap
        print(f"[intro] {tag}: TIMEOUT after {waited} ticks", flush=True)
        return False

    def walk_to_tile(tx, ty, max_ticks=20000, mash_on_stuck=True):
        """Greedy walk toward (tx,ty) with wall-slide; returns True on map change.

        Uses read_xy() / read_map() from the enclosing scope. Returns True as
        soon as the map changes (warp hit), False if we never arrive.
        """
        nonlocal t
        start_map = read_map()
        ticks = 0
        stuck = 0
        while ticks < max_ticks:
            if read_map() != start_map:
                return True
            x, y = read_xy()
            dx = tx - x
            dy = ty - y
            if abs(dx) <= 1 and abs(dy) <= 1:
                # Adjacent to the target tile: step onto it (triggers warp).
                d = (
                    "down"
                    if dy > 0
                    else ("up" if dy < 0 else ("right" if dx > 0 else "left"))
                )
                pb.button_press(d)
                for _ in range(14):
                    pb.tick()
                    ticks += 1
                    t += 1
                pb.button_release(d)
                if read_map() != start_map:
                    return True
                # Warp missed (non-warp target): back off slightly and retry.
                pb.button_press("up" if dy < 0 else ("down" if dy > 0 else "left"))
                for _ in range(10):
                    pb.tick()
                    ticks += 1
                    t += 1
                pb.button_release("up" if dy < 0 else ("down" if dy > 0 else "left"))
                continue
            # Primary axis = larger remaining distance.
            if abs(dy) >= abs(dx):
                d = "down" if dy > 0 else "up"
            else:
                d = "right" if dx > 0 else "left"
            y0, x0 = read_xy()
            pb.button_press(d)
            for _ in range(12):
                pb.tick()
                ticks += 1
                t += 1
            pb.button_release(d)
            if read_map() != start_map:
                return True
            y1, x1 = read_xy()
            if (y1, x1) != (y0, x0):
                stuck = 0
                continue
            # Blocked on primary: try the perpendicular.
            if abs(dy) >= abs(dx):
                d = "right" if dx > 0 else "left"
            else:
                d = "down" if dy > 0 else "up"
            y0, x0 = read_xy()
            pb.button_press(d)
            for _ in range(12):
                pb.tick()
                ticks += 1
                t += 1
            pb.button_release(d)
            if read_map() != start_map:
                return True
            y1, x1 = read_xy()
            if (y1, x1) != (y0, x0):
                stuck = 0
                continue
            stuck += 1
            if stuck > 5 and mash_on_stuck:
                # Walled in / forced-text scene (Mom, Elm): advance the script.
                tap("a", 10)
                wait(140)
                ticks += 150
                t += 150
                stuck = 0
        return read_map() != start_map

    # ---- 0. boot -> press through to the main menu ----------------------
    # 0xC763-0xC766 is shared scratch (jumptable/timer/option) reused by the
    # splash and the intro movie, so no RAM marker reliably identifies the
    # title screen (observed: identical-byte garbage ~0x0D0D oscillating).
    # Strategy: press A once during the copyright/logo splash to cancel it
    # (SKIP_SPLASH_F -> StartTitleScreen), wait for the title to load, then
    # START to open the main menu. Presses that land while a screen is
    # loading are simply eaten; the self-healing NEW GAME loop below presses
    # only A (never DOWN pre-clock, so the OPTION menu can never be opened).
    print("[intro] booting...", flush=True)
    wait(2600)
    t += 2600
    tap("a", 10)
    wait(100)
    t += 110
    wait(2900)
    t += 2900
    for _ in range(4):
        tap("start", 10)
        wait(700)
        t += 710
    wait(900)
    t += 900
    report("menu")

    # ---- 2. A on NEW GAME until the clock screen appears -------------
    # Only A is pressed here. A on the title opens the menu, A on the menu
    # selects NEW GAME (default cursor), A on the wake-up text advances into
    # the clock screen (wTimeSetBuffer nonzero). If a press is eaten while a
    # screen is loading, the next A covers it.
    clock_seen = False
    for attempt in range(10):
        tap("a", 10)
        wait(700)
        t += 710
        if pb.memory[M_TIME_BUF] != 0:
            clock_seen = True
            print(f"[intro] clock screen reached (attempt {attempt})", flush=True)
            break
    report("new game")
    if not clock_seen:
        # Extra insurance in case the transition ran long.
        for _ in range(4):
            tap("a", 10)
            wait(500)
            t += 510
        report("clock retry")

    # ---- 3. clock (InitClock): 6 A presses, defaults 10:00 -------------
    if not clock_seen:
        # The menu may have needed a second tap; give a few more A's.
        for _ in range(4):
            tap("a", 10)
            wait(500)
            t += 510
        report("clock retry")
    for _ in range(6):
        tap("a", 10)
        wait(500)
        t += 510
    wait(500)
    t += 500
    report("clock")

    # ---- 4. Oak speech -> name menu: DOWN+A hammer ---------------------
    # DOWN is a no-op during text; the menu appears only right after an A,
    # so the first DOWN to land on it moves the cursor off NEW NAME before
    # any A can select it. Iterate until wPlayerName starts with "GOLD".
    # If the naming KEYBOARD somehow opens (nst>0, e.g. an A landed on NEW
    # NAME), press B to close it and retry with DOWN+A.
    named = False
    for i in range(24):
        nm = read_name()
        if nm.startswith("GOLD"):
            named = True
            print(f"[intro] hammer iter {i}: NAME SET -> '{nm}'", flush=True)
            break
        if pb.memory[M_NAMING_TYPE] > 0:
            print(f"[intro] hammer iter {i}: keyboard open, B to bail", flush=True)
            tap("b", 10)
            wait(300)
            t += 310
            for _ in range(3):
                tap("b", 10)
                wait(150)
                t += 160
        if i % 3 == 0:
            g, m = read_map()
            print(
                f"[intro] hammer iter {i:2d}: map=({g},{m}) xy={read_xy()} "
                f"name='{nm}' {marker_str()}",
                flush=True,
            )
        tap("down", 10)
        wait(120)
        t += 130
        tap("a", 10)
        wait(350)
        t += 360
    wait(600)
    t += 600
    report("named")
    if not named:
        print("[intro] FAIL: name not set to GOLD", flush=True)
        pb.stop()
        return 2

    # ---- 5. OakText7 + world init --------------------------------------
    mash_ok = False
    for _ in range(6):
        if read_map() in ((24, 4), (24, 5), (24, 6), (24, 7)):
            mash_ok = True
            break
        tap("a", 10)
        wait(350)
        t += 360
    if not mash_ok:
        wait(1000)
        t += 1000
    (g, m), p = report("world")
    if (g, m) not in ((24, 4), (24, 5), (24, 6), (24, 7)):
        print("[intro] FAIL: world did not initialize (still pre-world)", flush=True)
        pb.stop()
        return 3

    # ---- 6. bedroom (24,7) -> 1F (24,6) via stairs (7,0) ---------------
    if (g, m) == (24, 7):
        walk_to_tile(7, 0, max_ticks=12000)
        (g, m), p = report("1F")
        if (g, m) != (24, 6):
            print(f"[intro] unexpected map after stairs: ({g},{m})", flush=True)
            pb.stop()
            return 4

    # ---- 7. 1F (24,6) -> New Bark Town (24,4) via door (7,7) -----------
    if (g, m) == (24, 6):
        walk_to_tile(7, 7, max_ticks=16000)  # Mom scene handled by mash-on-stuck
        (g, m), p = report("town")
        if (g, m) != (24, 4):
            print(f"[intro] unexpected map after 1F door: ({g},{m})", flush=True)
            pb.stop()
            return 5

    # ---- 8. town (24,4) -> Elm's lab (24,5) via lab door (6,3) ---------
    if (g, m) == (24, 4):
        walk_to_tile(6, 3, max_ticks=24000)
        if read_map() != (24, 5):
            # Door might be slightly off; try approaching from below.
            walk_to_tile(6, 4, max_ticks=12000)
        (g, m), p = report("lab")
        if (g, m) != (24, 5):
            print("[intro] FAIL: could not reach Elm's lab", flush=True)
            pb.stop()
            return 6

    # ---- 9. lab scene + starter ----------------------------------------
    # Entry text + auto-walk locks the player; mash to clear, then dance to
    # the ball table pressing A until the party has 1 mon.
    mash_ok = False
    for _ in range(10):
        if read_party() >= 1:
            mash_ok = True
            break
        tap("a", 10)
        wait(240)
        t += 250
    wait(700)
    t += 700
    report("lab scene")

    steps = [
        ("down", 50),
        ("down", 50),
        ("a", 10),
        ("a", 10),
        ("right", 25),
        ("a", 10),
        ("left", 50),
        ("a", 10),
        ("down", 40),
        ("a", 10),
        ("up", 40),
        ("a", 10),
        ("left", 30),
        ("down", 30),
        ("a", 10),
        ("a", 10),
        ("right", 30),
        ("down", 50),
        ("a", 10),
        ("a", 10),
        ("up", 30),
        ("a", 10),
        ("down", 60),
        ("a", 10),
        ("a", 10),
    ]
    for btn, tck in steps:
        if read_party() >= 1:
            break
        if btn == "a":
            tap("a", 10)
            wait(260)
            t += 270
        else:
            pb.button_press(btn)
            for _ in range(tck):
                pb.tick()
                t += 1
            pb.button_release(btn)
            wait(60)
            t += 60
            # If a text box interrupted movement, advance it.
            y0, x0 = read_xy()
            if read_party() < 1:
                tap("a", 10)
                wait(200)
                t += 210

    (g, m), p = report("starter")
    if p >= 1:
        # Elm's directions + aide scene: mash generously, then save.
        for _ in range(12):
            tap("a", 10)
            wait(220)
            t += 230
        wait(500)
        t += 500
        (g, m), p = report("post-script")
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, "wb") as f:
            pb.save_state(f)
        print(
            f"[intro] SUCCESS party={p} map=({g},{m}) name='{read_name()}' "
            f"saved -> {args.save}",
            flush=True,
        )
        pb.stop()
        return 0

    print(f"[intro] FAILED party={p} map=({g},{m}) name='{read_name()}'", flush=True)
    pb.stop()
    return 1


if __name__ == "__main__":
    sys.exit(main())
