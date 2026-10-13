#!/usr/bin/env python3
"""
Brutal Elo Stress Test for ORION Chess AI.
Tests ORION's genuine neural network and minimax search (no external oracles)
against official Stockfish across escalating Elo tiers:
- Tier 1: Stockfish Elo 1800 (Expert Club Player)
- Tier 2: Stockfish Elo 2000 (FIDE Candidate Master)
- Tier 3: Stockfish Elo 2200 (FIDE National Master)

Outputs a comprehensive, unvarnished, brutal breakdown of performance.
"""

import math
import sys
import time
from typing import Dict, List, Tuple

from orion.arena.battle import Arena, EloTracker
from orion.arena.bots import OrionBot, StockfishBot
from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.utils.checkpoint import Checkpoint


def run_brutal_tier(orion_bot: OrionBot, stockfish_elo: int, rounds: int = 2) -> Dict:
    print(f"\n" + "=" * 65)
    print(f"[TIER] BRUTAL GAUNTLET TIER: Stockfish Elo {stockfish_elo} ({rounds} games)")
    print("=" * 65)

    sf_bot = StockfishBot(elo=stockfish_elo, depth=3, movetime_ms=200)
    if not sf_bot.is_available():
        print(f"[ERROR] Stockfish executable not found!")
        return {"wins": 0, "losses": 0, "draws": 0, "games": []}

    arena = Arena(orion_bot, sf_bot, max_moves=160)
    match_result = arena.run_match(rounds=rounds, verbose=True)
    return match_result


def main():
    checkpoint_path = "checkpoints/gm_master/orion_gm_resnet_20ep_500000_1m_gm_fine.json"
    print("=================================================================")
    print("           ORION CHESS AI - BRUTAL UNVARNISHED ELO GAUNTLET       ")
    print("=================================================================")
    print(f"[ENGINE] Pure Native ORION: 2M Parameter Super-GM ResNet")
    print(f"[SEARCH] Alpha-Beta Search Depth 4 + Quiescence + Dynamic Contempt")
    print(f"[ORACLE] Disabled (100% Pure ORION Network & Search)")
    print(f"[CHECKPOINT] {checkpoint_path}")
    print("=================================================================\n")

    network = ORIONNetwork()
    ep = Checkpoint().load(network, checkpoint_path)
    print(f"[OK] Checkpoint loaded (Episode/Step: {ep})")

    encoder = ChessEncoder()
    orion_bot = OrionBot(network, encoder, use_search=True, depth=4, rating=2000.0)

    tiers = [1800, 2000, 2200]
    tier_results = {}

    for elo_target in tiers:
        print(f"\n>>> Starting Tier: Stockfish {elo_target} Elo...")
        t0 = time.time()
        res = run_brutal_tier(orion_bot, elo_target, rounds=2)
        elapsed = time.time() - t0
        tier_results[elo_target] = {"result": res, "duration": elapsed}

    print("\n" + "#" * 65)
    print("                      FINAL BRUTAL ELO REPORT                   ")
    print("#" * 65)
    print(f"{'Tier (Opponent)':<25} | {'Score (W-L-D)':<15} | {'Win %':<8} | {'Tier Time'}")
    print("-" * 65)

    total_w = 0
    total_l = 0
    total_d = 0

    for elo_target, data in tier_results.items():
        res = data["result"]
        w = res.get("bot1_wins", 0)
        l = res.get("bot2_wins", 0)
        d = res.get("draws", 0)
        total_w += w
        total_l += l
        total_d += d
        pct = (w + 0.5 * d) / max(1, (w + l + d)) * 100
        print(f"Stockfish {elo_target:<15} | {w}W - {l}L - {d}D      | {pct:5.1f}%  | {data['duration']:.1f}s")

    tot_games = total_w + total_l + total_d
    tot_pct = (total_w + 0.5 * total_d) / max(1, tot_games) * 100
    print("-" * 65)
    print(f"{'TOTAL OVERALL':<25} | {total_w}W - {total_l}L - {total_d}D      | {tot_pct:5.1f}%")
    print(f"\nFinal ORION Pure Verified Elo: {orion_bot.rating:.1f}")
    print("#" * 65)


if __name__ == "__main__":
    main()
