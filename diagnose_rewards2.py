"""Definitive test: sum each reward component over one episode for Red vs Yellow."""

from orion.pokemon.environment import PokemonEnv, Gen1RAM

ROMS = {
    "RED": r"D:\roms\Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb",
    "YELLOW": r"D:\roms\Pokemon - Yellow Version - Special Pikachu Edition (USA, Europe) (GBC,SGB Enhanced).gb",
}

# Authoritative pokered addresses from PyBoy
PARTY_MONS_ADDRESS = 0xD16B
CORRECT_MAX_HP = PARTY_MONS_ADDRESS + 0x22  # 0xD18D
OUR_MAX_HP = Gen1RAM.MAX_HP  # 0xD16D (suspected wrong)

for name, rom in ROMS.items():
    print(f"\n{'=' * 64}\n{name}\n{'=' * 64}")
    env = PokemonEnv(rom_path=rom, headless=True, max_steps=2048)
    env.reset()

    totals = {
        "badge": 0.0,
        "level": 0.0,
        "map": 0.0,
        "hp_healed": 0.0,
        "party": 0.0,
        "battle": 0.0,
        "step": 0.0,
        "dialogue": 0.0,
    }
    counts = {k: 0 for k in totals}

    # Track raw values that drive hp_healed
    hp_feature_vals = []
    raw_cur, raw_our_max, raw_correct_max = [], [], []

    prev, _ = env.reset()
    for i in range(2048):
        raw_cur.append(env._read_ram_16(Gen1RAM.CURRENT_HP))
        raw_our_max.append(env._read_ram(OUR_MAX_HP))
        raw_correct_max.append(env._read_ram_16(CORRECT_MAX_HP))

        obs, r, term, trunc, info = env.step(i % 8)
        of, nf = prev["state"], obs["state"]

        # Recompute each component exactly as environment does
        if int(nf[0] * 8) > int(of[0] * 8):
            totals["badge"] += 1000.0
            counts["badge"] += 1
        if nf[3] * 100 > of[3] * 100:
            gain = int(nf[3] * 100 - of[3] * 100)
            totals["level"] += 50.0 * gain
            counts["level"] += 1
        if (
            int(nf[4] * 255) != int(of[4] * 255)
            and int(nf[4] * 255) not in env._visited_maps
        ):
            totals["map"] += 100.0
            counts["map"] += 1
        if nf[2] > of[2] + 0.05:
            totals["hp_healed"] += 200.0
            counts["hp_healed"] += 1
        if int(nf[1] * 6) > int(of[1] * 6):
            gain = int(nf[1] * 6 - of[1] * 6)
            totals["party"] += 500.0 * gain
            counts["party"] += 1
        if nf[5] > 0:
            totals["battle"] += -1.0
            counts["battle"] += 1
        if nf[17] > 0 and of[17] == 0:
            totals["dialogue"] += 10.0
            counts["dialogue"] += 1
        totals["step"] += -0.1
        counts["step"] += 1

        hp_feature_vals.append(nf[2])
        prev = obs
        if term or trunc:
            break
    env.close()

    grand = sum(totals.values())
    print(f"{'Component':<14} {'Count':>6} {'Total':>10}")
    print("-" * 34)
    for k in [
        "badge",
        "level",
        "map",
        "hp_healed",
        "party",
        "dialogue",
        "battle",
        "step",
    ]:
        print(f"{k:<14} {counts[k]:>6} {totals[k]:>10.1f}")
    print("-" * 34)
    print(f"{'GRAND TOTAL':<14} {'':>6} {grand:>10.1f}")

    print(f"\nMAX_HP address comparison:")
    print(f"  OUR address    0x{OUR_MAX_HP:04X}: {sorted(set(raw_our_max))[:10]}")
    print(
        f"  CORRECT addr   0x{CORRECT_MAX_HP:04X}: {sorted(set(raw_correct_max))[:10]}"
    )
    print(f"  CURRENT_HP raw: min={min(raw_cur)} max={max(raw_cur)}")
    print(
        f"  hp feature: min={min(hp_feature_vals):.3f} max={max(hp_feature_vals):.3f} "
        f"unique={len(set(round(v, 3) for v in hp_feature_vals))}"
    )
