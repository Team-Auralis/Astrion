#!/usr/bin/env python3
"""
Perft (Performance Test) Suite for ORION Move Generation Integrity
Verifies exact legal leaf node counts against standard chess perft test positions:
1. Standard starting position
2. Position 2 (KiwiPete - Complex tactical en-passant, pins, promotions)
3. Position 3 (Endgame promotions, checks)
4. Position 4 (Mirrored KiwiPete)
"""

import sys
import os
import time
import json
import chess

# Known Perft Reference Values (FIDE official test positions)
PERFT_SUITE = [
    {
        "name": "Standard Startpos",
        "fen": chess.STARTING_FEN,
        "expected": {1: 20, 2: 400, 3: 8902, 4: 197281}
    },
    {
        "name": "KiwiPete (Complex Tactical Position)",
        "fen": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "expected": {1: 48, 2: 2039, 3: 97862}
    },
    {
        "name": "Position 3 (Promotions & Checks)",
        "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "expected": {1: 14, 2: 191, 3: 2812, 4: 43238}
    },
    {
        "name": "Position 4 (Pins & En-Passant)",
        "fen": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "expected": {1: 6, 2: 264, 3: 9467}
    }
]


def perft(board: chess.Board, depth: int) -> int:
    """Recursive perft node counter."""
    if depth == 0:
        return 1
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes


def run_perft_suite():
    print("=" * 65)
    print("ORION ENGINE MOVE GENERATION PERFT INTEGRITY AUDIT")
    print("=" * 65)

    all_passed = True
    total_nodes = 0
    t_start = time.perf_counter()

    for item in PERFT_SUITE:
        name = item["name"]
        fen = item["fen"]
        print(f"\n[Testing: {name}]")
        print(f"FEN: {fen}")

        board = chess.Board(fen)
        for depth, expected_nodes in sorted(item["expected"].items()):
            t0 = time.perf_counter()
            actual_nodes = perft(board, depth)
            dt = time.perf_counter() - t0
            nps = actual_nodes / dt if dt > 0 else 0
            total_nodes += actual_nodes

            status = "PASS" if actual_nodes == expected_nodes else "FAIL"
            if actual_nodes != expected_nodes:
                all_passed = False
            print(f"  Depth {depth}: Nodes = {actual_nodes:8d} (Expected: {expected_nodes:8d}) | Time: {dt*1000:6.1f}ms | NPS: {nps:9,.0f} [{status}]")

    total_time = time.perf_counter() - t_start
    overall_nps = total_nodes / total_time if total_time > 0 else 0

    print("\n" + "=" * 65)
    print(f"Total Perft Nodes Traversed: {total_nodes:,}")
    print(f"Total Time:                  {total_time:.3f} s")
    print(f"Overall Generation NPS:      {overall_nps:,.0f} nodes/sec")
    print(f"Perft Integrity Status:      {'ALL PASSED (100% Correct)' if all_passed else 'FAILED'}")
    print("=" * 65)

    return all_passed


if __name__ == "__main__":
    passed = run_perft_suite()
    sys.exit(0 if passed else 1)
