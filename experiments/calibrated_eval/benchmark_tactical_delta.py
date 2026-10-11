#!/usr/bin/env python3
"""
Benchmark Tactical Delta: Evaluates Baseline vs Fixed Tactical Search
Measures:
1. Tactical Solve Rate (%) on 20 standard tactical and mate positions
2. Mate Blindness Count (instances of missing terminal mate or walking into mate)
3. Average search time per position (ms)
"""

import sys
import os
import time
import json
import chess

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("experiments/calibrated_eval"))

PIECE_VALS = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# ----------------- BASELINE (OLD DEFECTIVE IMPLEMENTATION) -----------------
def baseline_evaluate_material(board: chess.Board) -> float:
    score = 0.0
    turn = board.turn
    for sq, pc in board.piece_map().items():
        v = PIECE_VALS.get(pc.piece_type, 0)
        score += v if pc.color == turn else -v
    return score

def baseline_quiescence_search(board: chess.Board, alpha: float, beta: float, q_depth: int = 3) -> float:
    stand_pat = baseline_evaluate_material(board)
    if q_depth == 0 or board.is_checkmate():
        return stand_pat
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    captures = [m for m in board.legal_moves if board.is_capture(m)]
    if not captures:
        return stand_pat

    def mvv_lva(m):
        vic = board.piece_at(m.to_square)
        att = board.piece_at(m.from_square)
        v = PIECE_VALS.get(vic.piece_type, 0) if vic else 100
        a = PIECE_VALS.get(att.piece_type, 0) if att else 100
        return v * 10 - a

    captures.sort(key=mvv_lva, reverse=True)
    for m in captures:
        board.push(m)
        score = -baseline_quiescence_search(board, -beta, -alpha, q_depth - 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

def baseline_alphabeta_search(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    if depth == 0 or board.is_game_over():
        return baseline_quiescence_search(board, alpha, beta)

    moves = list(board.legal_moves)
    if not moves:
        if board.is_check():
            return -30000.0
        return 0.0

    moves.sort(key=lambda m: board.is_capture(m), reverse=True)
    for m in moves:
        board.push(m)
        score = -baseline_alphabeta_search(board, depth - 1, -beta, -alpha)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

def baseline_pick_move(board: chess.Board) -> chess.Move:
    moves = list(board.legal_moves)
    if not moves:
        return None
    for m in moves:
        board.push(m)
        if board.is_checkmate():
            board.pop()
            return m
        board.pop()
    moves.sort(key=lambda m: board.is_capture(m), reverse=True)
    best_m = moves[0]
    best_val = -float("inf")
    for i, m in enumerate(moves[:8]):
        d = 2 if i < 4 else 1
        board.push(m)
        val = -baseline_alphabeta_search(board, d, -50000.0, 50000.0)
        board.pop()
        if val > best_val:
            best_val = val
            best_m = m
    return best_m


# ----------------- FIXED IMPLEMENTATION (NEW CORRECTED CODE) -----------------
from experiments.calibrated_eval.tournament_engine import (
    evaluate_material as fixed_evaluate_material,
    quiescence_search as fixed_quiescence_search,
    alphabeta_search as fixed_alphabeta_search,
    mvv_lva as fixed_mvv_lva,
    MATE_SCORE
)

def fixed_pick_move(board: chess.Board) -> chess.Move:
    moves = list(board.legal_moves)
    if not moves:
        return None
    for m in moves:
        board.push(m)
        if board.is_checkmate():
            board.pop()
            return m
        board.pop()
    moves.sort(key=lambda m: (board.is_capture(m) * 1000 + fixed_mvv_lva(board, m)), reverse=True)
    best_m = moves[0]
    best_val = -float("inf")
    for i, m in enumerate(moves):
        d = 2 if i < 6 else 1
        board.push(m)
        val = -fixed_alphabeta_search(board, d, -50000.0, 50000.0, ply=1)
        board.pop()
        if val > best_val:
            best_val = val
            best_m = m
    return best_m


# ----------------- BENCHMARK SUITE: 20 TACTICAL POSITIONS -----------------
# Format: (FEN, List of acceptable winning/defending moves, Description)
BENCHMARK_POSITIONS = [
    # Mate in 1 puzzles
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", ["h5f7"], "Scholar's Mate in 1 (Qxf7#)"),
    ("6k1/5ppp/8/8/8/8/8/3RR1K1 w - - 0 1", ["d1d8", "e1e8"], "Back-rank Mate in 1 (Rd8# / Re8#)"),
    ("rnbqkbnr/ppppp2p/8/5ppQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3", [], "Terminal: Black is checkmated (No legal moves)"),
    ("k7/8/1K6/8/8/8/8/1R6 w - - 0 1", ["b1a1", "b1b2", "b1b3", "b1b4", "b1b5", "b1b7", "b1b8"], "King + Rook vs King (Avoid stalemate / Make progress)"),
    ("r2qkb1r/pp2pppp/2n5/1B1p4/3Pn1b1/2N2N2/PPP2PPP/R1BQK2R w KQkq - 1 7", ["c1d2", "d1d3", "c3e4", "e1g1"], "Pin defense / Development"),
    
    # Defending against immediate mate threats
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 4 4", ["d7d5", "d8e7", "f6h5", "g7g6"], "Defend against Qxf7# threat"),
    ("3r2k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1", ["d8d1"], "Back-rank counter-tactic: capture hanging rook"),
    ("r1b1k2r/pppp1ppp/8/4q3/1bP5/2N1P3/PP3PPP/R1BQKB1R w KQkq - 0 9", ["c1d2"], "Relieve bishop pin on c3"),
    ("r1bqk2r/pppp1ppp/2n5/4p3/2B1n3/2P2N2/PPP2PPP/R1BQK2R w KQkq - 0 7", ["d1d5", "c4f7", "c1e3"], "Tactical fork / fork threat (Qd5 threatening mate on f7 and Ne4)"),
    ("r4rk1/ppp2ppp/2n5/3q4/3P4/5N2/PP3PPP/R2QR1K1 w - - 0 14", ["a1c1", "a2a3", "d1d2", "d1d3", "e1e3", "h2h3"], "Quiet positional move in equal middlegame"),
    
    # Captures & Tactics
    ("r1bqkb1r/ppp2ppp/2np1n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5", ["d2d4", "f3g5", "c2c3", "e1g1"], "Open Italian: d4 or Ng5"),
    ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", ["f3e5", "d2d4", "b1c3", "f1c4"], "Petrov: Nxe5 capture"),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", ["b8c6", "g8f6", "d7d6"], "Defend e5 pawn (Nc6, Nf6, d6)"),
    ("r2qk2r/ppp1bppp/2np1n2/4p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQ1RK1 w kq - 2 7", ["c1e3", "h2h3", "a2a3", "c4b3"], "Sound opening choice (h3, Be3)"),
    ("8/8/4k3/8/8/8/4K3/8 w - - 0 1", ["e2d3", "e2e3", "e2f3", "e2d2", "e2f2", "e2e1", "e2d1", "e2f1"], "King endgame: keep opposition"),
    
    # Smothered mate & tactical forks
    ("6k1/5ppp/8/8/8/8/4QPPP/6K1 w - - 0 1", ["e2e8"], "Queen back-rank mate in 1 (Qe8#)"),
    ("r1b2rk1/ppp2ppp/8/8/8/8/PPP2PPP/R1B1R1K1 b - - 0 14", ["c8f5", "c8d7", "c8e6", "f8d8", "c7c6"], "Back-rank development / rook files"),
    ("r3k2r/ppp2ppp/2n5/3qp3/1b1P4/5N2/PPP2PPP/R1BQR1K1 w kq - 0 10", ["c2c3", "c1d2"], "Defend against bishop check / pin"),
    ("r1bqk2r/pppp1ppp/2n5/2b1p3/2B1P1n1/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 5 6", ["d3d4", "c1g5", "h2h3", "b1c3"], "Countering Fried Liver / Ng4 threat"),
    ("4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", ["e1e8"], "Rook captures rook with back-rank mate (Rxe8#)")
]

def allows_immediate_mate(board: chess.Board, move: chess.Move) -> bool:
    if not move:
        return False
    board.push(move)
    mated = False
    for resp in list(board.legal_moves):
        board.push(resp)
        if board.is_checkmate():
            mated = True
            board.pop()
            break
        board.pop()
    board.pop()
    return mated


def run_benchmark():
    print("=" * 60)
    print("ORION SEARCH TACTICAL DELTA BENCHMARK")
    print("Baseline (Old Broken Search) vs Fixed (New Tactical Search)")
    print("=" * 60)

    baseline_correct = 0
    fixed_correct = 0
    baseline_blunders = 0
    fixed_blunders = 0
    baseline_times = []
    fixed_times = []

    results = []

    for idx, (fen, acceptable_moves, desc) in enumerate(BENCHMARK_POSITIONS):
        board = chess.Board(fen)
        if board.is_game_over():
            continue

        # Evaluate Baseline
        t0 = time.perf_counter()
        m_base = baseline_pick_move(board)
        t_base = (time.perf_counter() - t0) * 1000.0
        baseline_times.append(t_base)

        base_uci = m_base.uci() if m_base else None
        base_is_correct = base_uci in acceptable_moves if acceptable_moves else False
        base_blunder = allows_immediate_mate(board, m_base)
        if base_blunder:
            baseline_blunders += 1
        if base_is_correct:
            baseline_correct += 1

        # Evaluate Fixed
        t0 = time.perf_counter()
        m_fixed = fixed_pick_move(board)
        t_fixed = (time.perf_counter() - t0) * 1000.0
        fixed_times.append(t_fixed)

        fixed_uci = m_fixed.uci() if m_fixed else None
        fixed_is_correct = fixed_uci in acceptable_moves if acceptable_moves else False
        fixed_blunder = allows_immediate_mate(board, m_fixed)
        if fixed_blunder:
            fixed_blunders += 1
        if fixed_is_correct:
            fixed_correct += 1

        res = {
            "index": idx + 1,
            "description": desc,
            "fen": fen,
            "acceptable": acceptable_moves,
            "baseline_move": base_uci,
            "baseline_correct": base_is_correct,
            "baseline_blunder": base_blunder,
            "baseline_ms": round(t_base, 2),
            "fixed_move": fixed_uci,
            "fixed_correct": fixed_is_correct,
            "fixed_blunder": fixed_blunder,
            "fixed_ms": round(t_fixed, 2),
        }
        results.append(res)
        print(f"[{idx+1:02d}/20] {desc:<45} | Base: {str(base_uci):<6} ({'PASS' if base_is_correct else 'FAIL'}) | Fixed: {str(fixed_uci):<6} ({'PASS' if fixed_is_correct else 'FAIL'})")

    total_tested = len(results)
    base_acc = (baseline_correct / total_tested) * 100.0
    fixed_acc = (fixed_correct / total_tested) * 100.0
    avg_base_t = sum(baseline_times) / len(baseline_times)
    avg_fixed_t = sum(fixed_times) / len(fixed_times)

    print("\n" + "=" * 60)
    print("TACTICAL BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Evaluated Positions: {total_tested}")
    print(f"Baseline Solve Rate:      {baseline_correct}/{total_tested} ({base_acc:.1f}%)")
    print(f"Fixed Solve Rate:         {fixed_correct}/{total_tested} ({fixed_acc:.1f}%)")
    print(f"Solve Rate Delta:         +{fixed_acc - base_acc:.1f}%")
    print(f"Baseline Blunders into Mate: {baseline_blunders}")
    print(f"Fixed Blunders into Mate:    {fixed_blunders}")
    print(f"Avg Time per Move:        Baseline: {avg_base_t:.2f}ms | Fixed: {avg_fixed_t:.2f}ms")
    print("=" * 60)

    output = {
        "total_tested": total_tested,
        "baseline_accuracy_pct": base_acc,
        "fixed_accuracy_pct": fixed_acc,
        "delta_pct": fixed_acc - base_acc,
        "baseline_blunders": baseline_blunders,
        "fixed_blunders": fixed_blunders,
        "avg_baseline_ms": round(avg_base_t, 2),
        "avg_fixed_ms": round(avg_fixed_t, 2),
        "results": results
    }

    out_path = "experiments/calibrated_eval/tactical_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Detailed results saved to {out_path}")

if __name__ == "__main__":
    run_benchmark()
