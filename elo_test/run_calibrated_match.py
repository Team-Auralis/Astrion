#!/usr/bin/env python3
"""
Calibrated Verification Match for ORION v2.3 Engine
Runs paired head-to-head matches against calibrated reference engines:
1. Stockfish-1350 (Calibrated anchor 1350 Elo)
2. Sunfish-2026 (Python benchmark ~1750 Elo)
3. Vice-1.2 (CCRL anchor ~1850 Elo)

Saves PGN to elo_test/v2_3_verification.pgn and runs Ordo rating calculation.
"""

import os
import sys
import time
import subprocess
import argparse
from typing import List, Dict
import chess
import chess.pgn

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("experiments/calibrated_eval"))

from tournament_engine import (
    OrionModelRunner,
    OrionSearchPlayer,
    ExternalEnginePlayer,
    CHECKPOINT_PATH,
    ORDO_EXE,
    EPD_BOOK,
    evaluate_material
)
from run_round_robin_tournament import load_openings, play_game


def run_verification_match(num_pairs: int = 2, time_limit: float = 0.05):
    print("=" * 70)
    print("ORION v2.3 SEARCH-OPTIMIZED ENGINE VERIFICATION MATCH")
    print(f"Paired Openings: {num_pairs} (each played twice, color-swapped)")
    print(f"Time Budget per Move: {time_limit:.2f}s")
    print("=" * 70)

    openings = load_openings(EPD_BOOK, num_pairs)
    runner = OrionModelRunner(CHECKPOINT_PATH)
    orion_v23 = OrionSearchPlayer(runner)
    orion_v23.name = "ORION-Search-v2.3"

    opponents = [
        ExternalEnginePlayer("Stockfish-1350", [os.path.abspath("stockfish.exe")], options={"UCI_LimitStrength": True, "UCI_Elo": 1350}),
        ExternalEnginePlayer("Sunfish-2026", ["python", os.path.abspath("tools/engines/sunfish-master/sunfish.py"), "-u"]),
        ExternalEnginePlayer("Vice-1.2", [os.path.abspath("tools/engines/vice.exe")])
    ]

    for opp in opponents:
        opp.start()

    all_games = []
    round_id = 1
    scores = {orion_v23.name: {"W": 0, "L": 0, "D": 0, "pts": 0.0, "total": 0}}

    for opp in opponents:
        scores[opp.name] = {"W": 0, "L": 0, "D": 0, "pts": 0.0, "total": 0}
        print(f"\n--- Matchup: {orion_v23.name} vs {opp.name} ({len(openings)*2} games) ---")

        for fen in openings:
            # Game 1: ORION White, Opponent Black
            g1 = play_game(orion_v23, opp, fen, round_id, time_limit=time_limit)
            round_id += 1
            all_games.append(g1)
            res1 = g1.headers["Result"]
            print(f"Round {round_id-1:02d}: {orion_v23.name} (W) vs {opp.name} (B) -> {res1}")

            # Record Game 1
            scores[orion_v23.name]["total"] += 1
            scores[opp.name]["total"] += 1
            if res1 == "1-0":
                scores[orion_v23.name]["W"] += 1
                scores[orion_v23.name]["pts"] += 1.0
                scores[opp.name]["L"] += 1
            elif res1 == "0-1":
                scores[opp.name]["W"] += 1
                scores[opp.name]["pts"] += 1.0
                scores[orion_v23.name]["L"] += 1
            else:
                scores[orion_v23.name]["D"] += 1
                scores[orion_v23.name]["pts"] += 0.5
                scores[opp.name]["D"] += 1
                scores[opp.name]["pts"] += 0.5

            # Game 2: Opponent White, ORION Black (Color Swap on same opening FEN)
            g2 = play_game(opp, orion_v23, fen, round_id, time_limit=time_limit)
            round_id += 1
            all_games.append(g2)
            res2 = g2.headers["Result"]
            print(f"Round {round_id-1:02d}: {opp.name} (W) vs {orion_v23.name} (B) -> {res2}")

            # Record Game 2
            scores[orion_v23.name]["total"] += 1
            scores[opp.name]["total"] += 1
            if res2 == "1-0":
                scores[opp.name]["W"] += 1
                scores[opp.name]["pts"] += 1.0
                scores[orion_v23.name]["L"] += 1
            elif res2 == "0-1":
                scores[orion_v23.name]["W"] += 1
                scores[orion_v23.name]["pts"] += 1.0
                scores[opp.name]["L"] += 1
            else:
                scores[orion_v23.name]["D"] += 1
                scores[orion_v23.name]["pts"] += 0.5
                scores[opp.name]["D"] += 1
                scores[opp.name]["pts"] += 0.5

    for opp in opponents:
        opp.close()

    os.makedirs("elo_test", exist_ok=True)
    pgn_path = "elo_test/v2_3_verification.pgn"
    with open(pgn_path, "w", encoding="utf-8") as f:
        for g in all_games:
            print(g, file=f, end="\n\n")

    print("\n" + "=" * 70)
    print("VERIFICATION MATCH RESULTS")
    print("=" * 70)
    for name, s in scores.items():
        pct = (s["pts"] / s["total"] * 100.0) if s["total"] > 0 else 0.0
        print(f"{name:<20}: {s['W']}W - {s['L']}L - {s['D']}D ({s['pts']:.1f}/{s['total']}, {pct:.1f}%)")

    # Ordo Rating Calculation
    ordo_report_path = "elo_test/v2_3_ordo_report.txt"
    cmd = [
        ORDO_EXE,
        "-p", os.path.abspath(pgn_path),
        "-o", os.path.abspath(ordo_report_path),
        "-A", "Vice-1.2",
        "-a", "1850.0",
        "-s", "1000",
        "-F", "95",
        "-G"
    ]
    print(f"\n--- Running Ordo Maximum Likelihood Estimation ---")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Ordo Exit:", res.returncode)
    if os.path.exists(ordo_report_path):
        with open(ordo_report_path, "r") as f:
            content = f.read()
        print(content)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=1, help="Pairs of openings per opponent")
    parser.add_argument("--time", type=float, default=0.05, help="Time limit per move")
    args = parser.parse_args()

    run_verification_match(num_pairs=args.pairs, time_limit=args.time)
