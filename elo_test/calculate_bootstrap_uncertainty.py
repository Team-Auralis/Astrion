"""
Paired-Opening Bootstrap Confidence Interval Calculator
Computes 95% empirical confidence intervals for score % and Elo deltas
using 1,000 paired-opening resamples.
"""
import os
import re
import math
import random
import json
from collections import defaultdict

PGN_FILE = r"D:\orion-game-ai\elo_test\games.pgn"
OUTPUT_JSON = r"D:\orion-game-ai\elo_test\uncertainty.json"

def parse_games_with_openings(pgn_path):
    if not os.path.exists(pgn_path):
        return []
    with open(pgn_path, "r", encoding="utf-8") as f:
        content = f.read()

    games = []
    raw_games = content.split('[Event ')
    for rg in raw_games:
        if not rg.strip():
            continue
        g = '[Event ' + rg
        white = re.search(r'\[White "([^"]+)"\]', g)
        black = re.search(r'\[Black "([^"]+)"\]', g)
        result = re.search(r'\[Result "([^"]+)"\]', g)
        opening = re.search(r'\[Opening "([^"]+)"\]', g)
        round_tag = re.search(r'\[Round "([^"]+)"\]', g)
        
        if white and black and result:
            w = white.group(1)
            b = black.group(1)
            r = result.group(1)
            op = opening.group(1) if opening else "Standard"
            rnd = round_tag.group(1) if round_tag else "1"
            score = 1.0 if r == "1-0" else (0.5 if r == "1/2-1/2" else (0.0 if r == "0-1" else None))
            if score is not None:
                games.append({
                    "white": w, "black": b, "result": r, "white_score": score,
                    "opening": op, "round": rnd
                })
    return games

def elo_diff_from_score(score_pct):
    if score_pct <= 0.001:
        return -800.0
    if score_pct >= 0.999:
        return 800.0
    return -400.0 * math.log10((1.0 - score_pct) / score_pct)

def bootstrap_analysis(games, n_bootstrap=1000):
    # Group games by opponent pairs
    pairs = defaultdict(list)
    for g in games:
        key = tuple(sorted([g["white"], g["black"]]))
        pairs[key].append(g)

    results = {}
    random.seed(42)

    for pair_key, pair_games in pairs.items():
        e1, e2 = pair_key
        n_games = len(pair_games)
        if n_games < 2:
            continue

        # Actual score for e1
        e1_points = sum(g["white_score"] if g["white"] == e1 else (1.0 - g["white_score"]) for g in pair_games)
        actual_pct = (e1_points / n_games) * 100.0

        # Group by paired encounters (consecutive pairs of games with same opening)
        # Resample encounters with replacement
        encounter_units = []
        for i in range(0, n_games, 2):
            encounter_units.append(pair_games[i:i+2])

        boot_pcts = []
        for _ in range(n_bootstrap):
            resampled_units = random.choices(encounter_units, k=len(encounter_units))
            flat = [g for unit in resampled_units for g in unit]
            pts = sum(g["white_score"] if g["white"] == e1 else (1.0 - g["white_score"]) for g in flat)
            boot_pcts.append((pts / len(flat)) * 100.0)

        boot_pcts.sort()
        ci_lower = boot_pcts[int(0.025 * n_bootstrap)]
        ci_upper = boot_pcts[int(0.975 * n_bootstrap)]
        mean_pct = sum(boot_pcts) / len(boot_pcts)

        actual_elo_diff = elo_diff_from_score(actual_pct / 100.0)
        elo_lower = elo_diff_from_score(ci_lower / 100.0)
        elo_upper = elo_diff_from_score(ci_upper / 100.0)

        pair_name = f"{e1} vs {e2}"
        results[pair_name] = {
            "engine_1": e1,
            "engine_2": e2,
            "total_games": n_games,
            "actual_score_pct_e1": round(actual_pct, 2),
            "bootstrap_mean_pct_e1": round(mean_pct, 2),
            "ci_95_pct_lower": round(ci_lower, 2),
            "ci_95_pct_upper": round(ci_upper, 2),
            "implied_elo_diff": round(actual_elo_diff, 1),
            "ci_95_elo_lower": round(elo_lower, 1),
            "ci_95_elo_upper": round(elo_upper, 1)
        }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Computed bootstrap uncertainty for {len(results)} matchups. Saved to {OUTPUT_JSON}")
    return results

if __name__ == "__main__":
    games = parse_games_with_openings(PGN_FILE)
    if games:
        bootstrap_analysis(games)
    else:
        print(f"No games found in {PGN_FILE}")
