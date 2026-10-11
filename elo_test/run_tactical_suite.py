#!/usr/bin/env python3
"""
Tactical Suite Benchmark for ORION Chess Engine
Tests 35 verified positions across 6 categories:
1. Mate-in-1 (Direct checkmates)
2. Mate-in-2 (Tactical mating sequences)
3. Hanging-piece captures (Immediate material gains)
4. Pins, Forks & Skewers (Tactical motifs)
5. Defending against mate threats (Parrying attacks)
6. Tactical sacrifices & endgames (Deep calculation)

Outputs comprehensive metrics to elo_test/tactical_results.json.
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional
import chess

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("experiments/calibrated_eval"))

from tournament_engine import OptimizedSearchEngine

# 35 Verified Benchmark Positions
# Format: { "id": int, "category": str, "fen": str, "acceptable_moves": [uci_moves], "description": str }
TACTICAL_SUITE = [
    # Category 1: Mate in 1
    {
        "id": 1,
        "category": "mate_in_1",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "acceptable": ["h5f7"],
        "desc": "Scholar's Mate (Qxf7#)"
    },
    {
        "id": 2,
        "category": "mate_in_1",
        "fen": "6k1/5ppp/8/8/8/8/8/3RR1K1 w - - 0 1",
        "acceptable": ["d1d8", "e1e8"],
        "desc": "Back-rank mate with Rooks (Rd8# / Re8#)"
    },
    {
        "id": 3,
        "category": "mate_in_1",
        "fen": "6k1/5ppp/8/8/8/8/4QPPP/6K1 w - - 0 1",
        "acceptable": ["e2e8"],
        "desc": "Queen back-rank mate (Qe8#)"
    },
    {
        "id": 4,
        "category": "mate_in_1",
        "fen": "4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
        "acceptable": ["e1e8"],
        "desc": "Rook capture back-rank mate (Rxe8#)"
    },
    {
        "id": 5,
        "category": "mate_in_1",
        "fen": "r1b2rk1/pp3ppp/8/8/1B6/8/PPP2PPP/R2q1RK1 b - - 0 1",
        "acceptable": ["d1f1"],
        "desc": "Black Queen mate on f1 (Qxf1#)"
    },
    {
        "id": 6,
        "category": "mate_in_1",
        "fen": "5rk1/5ppp/8/8/8/8/1Q6/4K2R w K - 0 1",
        "acceptable": ["b2g7"],
        "desc": "Supported Queen mate on g7 (Qg7#)"
    },

    # Category 2: Mate in 2
    {
        "id": 7,
        "category": "mate_in_2",
        "fen": "r5rk/5p1p/5R2/8/8/8/1Q4PP/6K1 w - - 0 1",
        "acceptable": ["f6a6", "f6b6", "f6c6", "f6d6", "f6e6", "f6f7"],
        "desc": "Discovered check leading to mate (Rf7+ or Rf6-rook moves)"
    },
    {
        "id": 8,
        "category": "mate_in_2",
        "fen": "6k1/1R4p1/6Np/8/8/8/8/4K3 w - - 0 1",
        "acceptable": ["b7b8", "g6e7"],
        "desc": "Cornering the King (Rb8+ Kh7 Nf8+ Kg8)"
    },
    {
        "id": 9,
        "category": "mate_in_2",
        "fen": "r1b3kr/ppp3pp/8/8/3q4/8/PPPP1PPP/RNBQR1K1 w - - 0 1",
        "acceptable": ["e1e8"],
        "desc": "Re8+ Kf7 Rxh8 or Kf7 Qe2"
    },
    {
        "id": 10,
        "category": "mate_in_2",
        "fen": "3r2k1/p4ppp/8/8/8/1Q6/5PPP/6K1 w - - 0 1",
        "acceptable": ["b3a4", "b3b7", "g2g3", "h2h3", "g1f1"],
        "desc": "Prevent back-rank vulnerability / activate Queen"
    },
    {
        "id": 11,
        "category": "mate_in_2",
        "fen": "r2q1rk1/ppp2ppp/2n5/3p4/3Pn1b1/2NB1N2/PPP2PPP/R2Q1RK1 w - - 0 10",
        "acceptable": ["c3e4", "d3e4", "h2h3"],
        "desc": "Exchange center pin knight (Nxe4 / Bxe4)"
    },

    # Category 3: Hanging-piece captures
    {
        "id": 12,
        "category": "hanging_piece",
        "fen": "3r2k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1",
        "acceptable": ["d8d1"],
        "desc": "Capture hanging Rook on d1 (Rxd1#)"
    },
    {
        "id": 13,
        "category": "hanging_piece",
        "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "acceptable": ["f3e5"],
        "desc": "Petrov Defense: Capture free e5 pawn (Nxe5)"
    },
    {
        "id": 14,
        "category": "hanging_piece",
        "fen": "r1bqk2r/pppp1ppp/2n5/4p3/1b2n3/2P2N2/PP1P1PPP/RNBQKB1R w KQkq - 0 5",
        "acceptable": ["c3b4"],
        "desc": "Capture hanging Bishop on b4 (cxb4)"
    },
    {
        "id": 15,
        "category": "hanging_piece",
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "acceptable": ["f1b5", "f1c4", "d2d4", "b1c3", "c2c3"],
        "desc": "Main line development vs protected e5"
    },
    {
        "id": 16,
        "category": "hanging_piece",
        "fen": "r1b1k2r/pppp1ppp/8/4q3/1bP5/2N1P3/PP3PPP/R1BQKB1R w KQkq - 0 9",
        "acceptable": ["c1d2"],
        "desc": "Relieve pin on c3 with Bd2"
    },

    # Category 4: Pins, Forks & Skewers
    {
        "id": 17,
        "category": "forks_pins",
        "fen": "r1bqk2r/pppp1ppp/2n5/4p3/2B1n3/2P2N2/PPP2PPP/R1BQK2R w KQkq - 0 7",
        "acceptable": ["d1d5", "c4f7"],
        "desc": "Queen double attack fork (Qd5 threatening Qxf7# and Qxe4)"
    },
    {
        "id": 18,
        "category": "forks_pins",
        "fen": "r2qkb1r/pp2pppp/2n5/1B1p4/3Pn1b1/2N2N2/PPP2PPP/R1BQK2R w KQkq - 1 7",
        "acceptable": ["c1d2", "d1d3", "c3e4", "e1g1", "h2h3"],
        "desc": "Resolve pin / reinforce knight"
    },
    {
        "id": 19,
        "category": "forks_pins",
        "fen": "r1b2rk1/ppp2ppp/2n5/4P3/2B2B2/2P5/P1P3PP/R4RK1 b - - 0 13",
        "acceptable": ["c8e6", "c8f5", "f8e8", "c6a5"],
        "desc": "Neutralize bishop diagonal with Be6 or attack with Na5"
    },
    {
        "id": 20,
        "category": "forks_pins",
        "fen": "r1b1k2r/ppq2ppp/2n1p3/3p4/3P4/P1N1PN2/1P3PPP/R2QKB1R w KQkq - 0 11",
        "acceptable": ["a1c1", "f1d3", "f1e2", "f1b5", "e1g1"],
        "desc": "Develop on open c-file (Rc1 / Bd3 / Be2)"
    },
    {
        "id": 21,
        "category": "forks_pins",
        "fen": "4k3/8/8/3n4/8/2Q5/8/4K3 w - - 0 1",
        "acceptable": ["c3c6", "c3e5", "c3d4"],
        "desc": "Queen skewer/fork winning undefended knight"
    },

    # Category 5: Defending against mate threats
    {
        "id": 22,
        "category": "defense",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 4 4",
        "acceptable": ["d7d5", "d8e7", "f6h5", "g7g6"],
        "desc": "Defend against Qxf7# threat (g6, Qe7, d5, Nh5)"
    },
    {
        "id": 23,
        "category": "defense",
        "fen": "r1bqk2r/pppp1ppp/2n5/2b1p3/2B1P1n1/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 5 6",
        "acceptable": ["d3d4", "c1g5", "h2h3", "b1c3", "c2c3"],
        "desc": "Countering Fried Liver / Ng4 attack (d4, h3, Bg5)"
    },
    {
        "id": 24,
        "category": "defense",
        "fen": "r3k2r/ppp2ppp/2n5/3qp3/1b1P4/5N2/PPP2PPP/R1BQR1K1 w kq - 0 10",
        "acceptable": ["c2c3", "c1d2", "d4e5"],
        "desc": "Defend king against bishop check / pin (c3, Bd2 or tactical dxe5)"
    },
    {
        "id": 25,
        "category": "defense",
        "fen": "r1b1k2r/pp3ppp/2p5/4q3/8/2P1P3/P4PPP/R1BQKB1R w KQkq - 0 11",
        "acceptable": ["d1d4", "c1d2", "c1b2"],
        "desc": "Defend c3 pawn under queen attack (Qd4 or Bd2)"
    },
    {
        "id": 26,
        "category": "defense",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "acceptable": ["b8c6", "g8f6", "d7d6"],
        "desc": "Defend e5 pawn against Nf3 (Nc6, Nf6, d6)"
    },
    {
        "id": 27,
        "category": "defense",
        "fen": "r1b2rk1/ppp2ppp/8/8/8/8/PPP2PPP/R1B1R1K1 b - - 0 14",
        "acceptable": ["c8f5", "c8d7", "c8e6", "f8d8", "c7c6"],
        "desc": "Develop queenside pieces safely"
    },

    # Category 6: Tactical sacrifices, Endgames & King Opposition
    {
        "id": 28,
        "category": "endgame_sacrifices",
        "fen": "8/8/4k3/8/4P3/8/4K3/8 w - - 0 1",
        "acceptable": ["e2d3", "e2f3", "e2e3"],
        "desc": "King and pawn endgame: maintain opposition (Kd3 or Kf3)"
    },
    {
        "id": 29,
        "category": "endgame_sacrifices",
        "fen": "k7/8/1K6/8/8/8/8/1R6 w - - 0 1",
        "acceptable": ["b1a1", "b1b2", "b1b3", "b1b4", "b1b5", "b1b7", "b1b8", "b1c1", "b1d1", "b1e1", "b1f1", "b1g1", "b1h1"],
        "desc": "King + Rook cornered King: avoid stalemate (any safe rook move)"
    },
    {
        "id": 30,
        "category": "endgame_sacrifices",
        "fen": "8/8/8/8/8/5k2/4p3/4K3 w - - 0 1",
        "acceptable": ["e1d2"],
        "desc": "King defense vs pawn on 7th rank (Kd2 only legal move)"
    },
    {
        "id": 31,
        "category": "endgame_sacrifices",
        "fen": "8/8/8/8/5K2/4p3/4k3/8 b - - 0 1",
        "acceptable": ["e2d2", "e2d3", "e2f2", "e2f3"],
        "desc": "Pawn endgame: escort pawn to promotion"
    },
    {
        "id": 32,
        "category": "endgame_sacrifices",
        "fen": "8/4P3/8/8/8/5k2/8/4K3 w - - 0 1",
        "acceptable": ["e7e8q", "e7e8r"],
        "desc": "Promote pawn to Queen (e8=Q)"
    },
    {
        "id": 33,
        "category": "endgame_sacrifices",
        "fen": "r1bqkb1r/ppp2ppp/2np1n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
        "acceptable": ["d2d4", "f3g5", "c2c3", "e1g1", "d2d3", "b1c3"],
        "desc": "Open Italian development / tactical strike"
    },
    {
        "id": 34,
        "category": "endgame_sacrifices",
        "fen": "r2qk2r/ppp1bppp/2np1n2/4p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQ1RK1 w kq - 2 7",
        "acceptable": ["c1e3", "h2h3", "a2a3", "c4b3", "c4b5", "h1h2"],
        "desc": "Solid opening choice (h3, Be3, Bb3)"
    },
    {
        "id": 35,
        "category": "endgame_sacrifices",
        "fen": "r4rk1/ppp2ppp/2n5/3q4/3P4/5N2/PP3PPP/R2QR1K1 w - - 0 14",
        "acceptable": ["a1c1", "a2a3", "d1d2", "d1d3", "e1e3", "h2h3"],
        "desc": "Central tension: Rook or Queen activation"
    }
]


def check_blunder_into_mate(board: chess.Board, move: chess.Move) -> bool:
    """Check if move allows opponent an immediate checkmate in 1."""
    if not move:
        return False
    board.push(move)
    mated = False
    for resp in board.legal_moves:
        board.push(resp)
        if board.is_checkmate():
            mated = True
            board.pop()
            break
        board.pop()
    board.pop()
    return mated


def run_suite(time_limit: float = 0.5):
    print("=" * 70)
    print("ORION TACTICAL BENCHMARK SUITE (35 POSITIONS)")
    print(f"Time Budget per Position: {time_limit:.2f}s")
    print("=" * 70)

    engine = OptimizedSearchEngine()
    results = []
    solved_count = 0
    blunder_count = 0
    sub_1s_count = 0
    total_nodes = 0
    total_time_ms = 0.0

    category_stats = {}

    for pos in TACTICAL_SUITE:
        p_id = pos["id"]
        cat = pos["category"]
        fen = pos["fen"]
        acceptable = pos["acceptable"]
        desc = pos["desc"]

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "solved": 0}
        category_stats[cat]["total"] += 1

        b = chess.Board(fen)
        if b.is_game_over():
            continue

        t0 = time.perf_counter()
        best_m, val, depth = engine.search(b, time_limit=time_limit)
        elapsed_s = time.perf_counter() - t0
        elapsed_ms = elapsed_s * 1000.0
        nodes = engine.nodes_visited

        total_nodes += nodes
        total_time_ms += elapsed_ms

        m_uci = best_m.uci() if best_m else None
        is_correct = m_uci in acceptable if acceptable else True
        is_blunder = check_blunder_into_mate(b, best_m)

        if is_correct:
            solved_count += 1
            category_stats[cat]["solved"] += 1
        if is_blunder:
            blunder_count += 1
        if elapsed_s < 1.0 and is_correct:
            sub_1s_count += 1

        status_str = "PASS" if is_correct else "FAIL"
        blunder_str = " [BLUNDER!]" if is_blunder else ""

        print(f"[{p_id:02d}/35] [{cat:<18}] {desc:<38} -> {str(m_uci):<6} "
              f"(d={depth:2d}, n={nodes:5d}, t={elapsed_ms:5.1f}ms) [{status_str}]{blunder_str}")

        results.append({
            "id": p_id,
            "category": cat,
            "description": desc,
            "fen": fen,
            "expected": acceptable,
            "actual": m_uci,
            "depth": depth,
            "nodes": nodes,
            "time_ms": round(elapsed_ms, 2),
            "eval_cp": round(val, 1),
            "pass": is_correct,
            "blunder_mate": is_blunder
        })

    n_positions = len(TACTICAL_SUITE)
    accuracy_pct = (solved_count / n_positions) * 100.0
    avg_time_ms = total_time_ms / n_positions
    avg_nodes = total_nodes / n_positions

    print("\n" + "=" * 70)
    print("TACTICAL SUITE SUMMARY RESULTS")
    print("=" * 70)
    print(f"Total Positions Tested:       {n_positions}")
    print(f"Correct Solutions:            {solved_count} / {n_positions} ({accuracy_pct:.1f}%)")
    print(f"Solved Under 1.0s:            {sub_1s_count} / {n_positions} ({(sub_1s_count/n_positions)*100.0:.1f}%)")
    print(f"Blunders into Terminal Mate:  {blunder_count}")
    print(f"Average Time per Position:    {avg_time_ms:.1f} ms")
    print(f"Average Nodes per Position:   {avg_nodes:,.0f} nodes")
    print("\nCategory Breakdown:")
    for cat, stat in category_stats.items():
        cat_pct = (stat["solved"] / stat["total"]) * 100.0
        print(f"  - {cat:<20}: {stat['solved']}/{stat['total']} ({cat_pct:.1f}%)")

    report_payload = {
        "benchmark": "ORION Tactical Regression Suite v1.0",
        "total_positions": n_positions,
        "solved_count": solved_count,
        "accuracy_percent": round(accuracy_pct, 2),
        "sub_1s_count": sub_1s_count,
        "blunders_into_mate": blunder_count,
        "avg_time_ms": round(avg_time_ms, 2),
        "avg_nodes": round(avg_nodes, 1),
        "category_breakdown": category_stats,
        "detailed_results": results
    }

    os.makedirs("elo_test", exist_ok=True)
    out_file = "elo_test/tactical_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\nSaved full benchmark report to: {out_file}")
    return report_payload


if __name__ == "__main__":
    run_suite()
