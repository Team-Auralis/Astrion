#!/usr/bin/env python3
"""
Experiment C: Component Speed & Correctness Ablation Study
Measures NPS, Completed Depth, Average Move Time, Nodes Searched, and Move Correctness
across 4 configurations:
1. Config A: Baseline v2.2 (Standard Alpha-Beta, no PVS, no repetition guard, infinite window)
2. Config B: v2.2 + Repetition Guard + Move Ordering (Queen promotion & countermove)
3. Config C: v2.2 + PVS (Zero-window search) + Move Ordering
4. Config D: Full v2.3 Candidate (PVS + Repetition + Move Ordering + Aspiration Windows)

Verifies: "Faster search is useful only if it preserves correctness and helps choose better moves."
Outputs report to elo_test/speed_ablation_report.json and markdown.
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Tuple, Optional
import chess

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("experiments/calibrated_eval"))

from tournament_engine import (
    TranspositionTable,
    evaluate_tapered,
    quiescence_search,
    mvv_lva,
    MATE_SCORE,
    FLAG_EXACT,
    FLAG_LOWERBOUND,
    FLAG_UPPERBOUND
)
from elo_test.run_tactical_suite import TACTICAL_SUITE


class ParameterizedSearchEngine:
    def __init__(self, use_pvs: bool, use_rep_guard: bool, use_aspiration: bool, use_enhanced_ordering: bool):
        self.use_pvs = use_pvs
        self.use_rep_guard = use_rep_guard
        self.use_aspiration = use_aspiration
        self.use_enhanced_ordering = use_enhanced_ordering

        self.tt = TranspositionTable(max_entries=500000)
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(64)]
        self.history_table: Dict[Tuple[int, int], int] = {}
        self.counter_moves: Dict[Tuple[int, int], chess.Move] = {}
        self.nodes_visited = 0
        self.start_time = 0.0
        self.time_limit = 0.1
        self.stop_search = False

    def clear(self):
        self.killer_moves = [[None, None] for _ in range(64)]
        self.history_table.clear()
        self.counter_moves.clear()
        self.nodes_visited = 0
        self.stop_search = False

    def score_move(self, board: chess.Board, m: chess.Move, tt_move: Optional[chess.Move], ply: int, prev_move: Optional[chess.Move] = None) -> int:
        if tt_move and m == tt_move:
            return 1000000
        if self.use_enhanced_ordering and m.promotion == chess.QUEEN:
            return 950000
        if board.is_capture(m):
            return 100000 + mvv_lva(board, m)
        if self.use_enhanced_ordering and m.promotion:
            return 90000
        if ply < 64:
            if m == self.killer_moves[ply][0]:
                return 80000
            elif m == self.killer_moves[ply][1]:
                return 70000
        if self.use_enhanced_ordering and prev_move:
            if self.counter_moves.get((prev_move.from_square, prev_move.to_square)) == m:
                return 60000
        return min(self.history_table.get((m.from_square, m.to_square), 0), 50000)

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0, prev_move: Optional[chess.Move] = None) -> float:
        self.nodes_visited += 1
        if (self.nodes_visited & 1023) == 0:
            if time.perf_counter() - self.start_time >= self.time_limit:
                self.stop_search = True
                return alpha

        if self.use_rep_guard and ply > 0 and (board.is_repetition(2) or board.is_fifty_moves()):
            return 0.0

        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        in_check = board.is_check()
        if in_check:
            depth += 1

        if depth <= 0:
            return quiescence_search(board, alpha, beta, ply=ply)

        pos_key = chess.polyglot.zobrist_hash(board)
        tt_val, tt_move = self.tt.probe(pos_key, depth, alpha, beta, ply)
        if tt_val is not None:
            return tt_val

        # Null-Move Pruning
        if depth >= 3 and not in_check and ply > 0:
            has_pieces = any(
                pc.piece_type not in (chess.PAWN, chess.KING) and pc.color == board.turn
                for pc in board.piece_map().values()
            )
            if has_pieces:
                r = 2 + (depth // 6)
                board.push(chess.Move.null())
                null_score = -self.negamax(board, depth - 1 - r, -beta, -beta + 1, ply + 1, prev_move=None)
                board.pop()
                if self.stop_search:
                    return alpha
                if null_score >= beta:
                    return beta

        moves = list(board.legal_moves)
        if not moves:
            if in_check:
                return -MATE_SCORE + ply
            return 0.0

        moves.sort(key=lambda m: self.score_move(board, m, tt_move, ply, prev_move), reverse=True)
        best_score = -float("inf")
        best_m = None
        orig_alpha = alpha

        for i, m in enumerate(moves):
            is_capture = board.is_capture(m)
            board.push(m)
            gives_check = board.is_check()

            if self.use_pvs:
                if i == 0:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, prev_move=m)
                else:
                    if i >= 4 and depth >= 3 and not in_check and not is_capture and not gives_check and not m.promotion:
                        reduced_depth = max(1, depth - 2)
                        score = -self.negamax(board, reduced_depth, -alpha - 1, -alpha, ply + 1, prev_move=m)
                        if score > alpha and not self.stop_search:
                            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, prev_move=m)
                    else:
                        score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, prev_move=m)
                        if alpha < score < beta and not self.stop_search:
                            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, prev_move=m)
            else:
                # No PVS (standard full window)
                if i >= 4 and depth >= 3 and not in_check and not is_capture and not gives_check and not m.promotion:
                    reduced_depth = max(1, depth - 2)
                    score = -self.negamax(board, reduced_depth, -alpha - 1, -alpha, ply + 1, prev_move=m)
                    if score > alpha and not self.stop_search:
                        score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, prev_move=m)
                else:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, prev_move=m)

            board.pop()
            if self.stop_search:
                return alpha

            if score > best_score:
                best_score = score
                best_m = m
            if score > alpha:
                alpha = score

            if alpha >= beta:
                if not is_capture and ply < 64:
                    if self.killer_moves[ply][0] != m:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = m
                    h_key = (m.from_square, m.to_square)
                    self.history_table[h_key] = self.history_table.get(h_key, 0) + depth * depth
                    if prev_move:
                        self.counter_moves[(prev_move.from_square, prev_move.to_square)] = m
                break

        if best_score <= orig_alpha:
            flag = FLAG_UPPERBOUND
        elif best_score >= beta:
            flag = FLAG_LOWERBOUND
        else:
            flag = FLAG_EXACT

        self.tt.store(pos_key, depth, best_score, flag, best_m, ply)
        return best_score

    def search(self, board: chess.Board, time_limit: float = 0.1, max_depth: int = 15) -> Tuple[chess.Move, float, int]:
        self.start_time = time.perf_counter()
        self.time_limit = time_limit
        self.clear()

        moves = list(board.legal_moves)
        if not moves:
            return None, 0.0, 0
        if len(moves) == 1:
            return moves[0], 0.0, 1

        for m in moves:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return m, MATE_SCORE - 1, 1
            board.pop()

        moves.sort(key=lambda m: (board.is_capture(m) * 1000 + mvv_lva(board, m)), reverse=True)
        best_move = moves[0]
        best_val = 0.0
        completed_depth = 0

        for d in range(1, max_depth + 1):
            iter_best_move = None
            iter_best_val = -float("inf")

            if self.use_aspiration and d >= 4 and abs(best_val) < MATE_SCORE - 100:
                delta = 40.0
                alpha = best_val - delta
                beta = best_val + delta
            else:
                delta = 0.0
                alpha = -float("inf")
                beta = float("inf")

            while True:
                for m in moves:
                    board.push(m)
                    val = -self.negamax(board, d - 1, -beta, -alpha, ply=1, prev_move=m)
                    board.pop()
                    if self.stop_search:
                        break
                    if val > iter_best_val:
                        iter_best_val = val
                        iter_best_move = m
                    if val > alpha:
                        alpha = val

                if self.stop_search or iter_best_move is None:
                    break

                if delta > 0.0:
                    if iter_best_val <= (best_val - delta):
                        alpha = -float("inf")
                        delta *= 2
                        continue
                    elif iter_best_val >= (best_val + delta):
                        beta = float("inf")
                        delta *= 2
                        continue
                break

            if self.stop_search or iter_best_move is None:
                break

            best_move = iter_best_move
            best_val = iter_best_val
            completed_depth = d

            if best_move in moves:
                moves.remove(best_move)
                moves.insert(0, best_move)

            if abs(best_val) > MATE_SCORE - 100:
                break

            elapsed = time.perf_counter() - self.start_time
            if elapsed >= time_limit * 0.6:
                break

        return best_move, best_val, completed_depth


def run_ablation_benchmarks():
    print("=" * 70)
    print("EXPERIMENT C: COMPONENT SPEED & ABLATION STUDY")
    print("Comparing NPS, Depth, Latency, and Accuracy across 4 configurations")
    print("=" * 70)

    configs = [
        {
            "name": "Config A: Baseline v2.2",
            "desc": "Standard Alpha-Beta, No PVS, No Repetition, Infinite Window",
            "engine": ParameterizedSearchEngine(use_pvs=False, use_rep_guard=False, use_aspiration=False, use_enhanced_ordering=False)
        },
        {
            "name": "Config B: v2.2 + Move Ordering & Repetition Guard",
            "desc": "Repetition guard + Queen promotion ordering + Countermoves",
            "engine": ParameterizedSearchEngine(use_pvs=False, use_rep_guard=True, use_aspiration=False, use_enhanced_ordering=True)
        },
        {
            "name": "Config C: v2.2 + PVS & Move Ordering",
            "desc": "Zero-window search (PVS) + Move ordering + Repetition guard",
            "engine": ParameterizedSearchEngine(use_pvs=True, use_rep_guard=True, use_aspiration=False, use_enhanced_ordering=True)
        },
        {
            "name": "Config D: Full v2.3 Candidate",
            "desc": "PVS + Move ordering + Repetition guard + Aspiration Windows",
            "engine": ParameterizedSearchEngine(use_pvs=True, use_rep_guard=True, use_aspiration=True, use_enhanced_ordering=True)
        }
    ]

    test_positions = TACTICAL_SUITE[:25]  # 25 varied positions
    results = []

    for cfg in configs:
        name = cfg["name"]
        engine = cfg["engine"]
        print(f"\n--- Testing {name} ---")
        print(f"Features: {cfg['desc']}")

        total_nodes = 0
        total_time_s = 0.0
        depths_reached = []
        solved_count = 0

        for pos in test_positions:
            b = chess.Board(pos["fen"])
            if b.is_game_over():
                continue

            t0 = time.perf_counter()
            best_m, val, depth = engine.search(b, time_limit=0.1)
            dt = time.perf_counter() - t0

            nodes = engine.nodes_visited
            total_nodes += nodes
            total_time_s += dt
            depths_reached.append(depth)

            m_uci = best_m.uci() if best_m else None
            is_correct = m_uci in pos["acceptable"] if pos["acceptable"] else True
            if is_correct:
                solved_count += 1

        n_tested = len(test_positions)
        avg_nps = total_nodes / total_time_s if total_time_s > 0 else 0
        avg_depth = sum(depths_reached) / len(depths_reached) if depths_reached else 0
        avg_time_ms = (total_time_s / n_tested) * 1000.0
        solve_rate = (solved_count / n_tested) * 100.0

        print(f"Results for {name}:")
        print(f"  Average NPS:         {avg_nps:,.0f} nodes/sec")
        print(f"  Average Depth:       {avg_depth:.2f} plies")
        print(f"  Average Move Time:   {avg_time_ms:.1f} ms")
        print(f"  Tactical Accuracy:   {solved_count}/{n_tested} ({solve_rate:.1f}%)")

        results.append({
            "config": name,
            "description": cfg["desc"],
            "avg_nps": round(avg_nps, 1),
            "avg_depth": round(avg_depth, 2),
            "avg_time_ms": round(avg_time_ms, 1),
            "total_nodes": total_nodes,
            "solved_count": solved_count,
            "total_tested": n_tested,
            "accuracy_pct": round(solve_rate, 1)
        })

    # Fixed Depth 4 Benchmark (Node Tree Pruning Comparison)
    print("\n" + "=" * 70)
    print("FIXED DEPTH 4 NODE PRUNING COMPARISON (Startpos & KiwiPete)")
    print("=" * 70)
    fixed_depth_positions = [
        ("Startpos", chess.STARTING_FEN),
        ("KiwiPete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        ("Tension Middlegame", "r1bqk2r/pppp1ppp/2n5/4p3/2B1n3/2P2N2/PPP2PPP/R1BQK2R w KQkq - 0 7")
    ]
    node_comparison = {}
    for cfg in configs:
        c_name = cfg["name"]
        node_comparison[c_name] = {}
        for pos_name, fen in fixed_depth_positions:
            b = chess.Board(fen)
            cfg["engine"].start_time = time.perf_counter()
            cfg["engine"].time_limit = 100.0
            cfg["engine"].clear()
            t0 = time.perf_counter()
            best_m, val, d = cfg["engine"].search(b, time_limit=100.0, max_depth=4)
            dt = time.perf_counter() - t0
            nodes = cfg["engine"].nodes_visited
            node_comparison[c_name][pos_name] = {"nodes": nodes, "time_ms": round(dt * 1000, 1), "move": best_m.uci()}
            print(f"[{c_name:<30}] {pos_name:<20}: Nodes = {nodes:6d} | Time = {dt*1000:6.1f}ms | Move = {best_m.uci()}")

    # Save to JSON
    os.makedirs("elo_test", exist_ok=True)
    out_json = "elo_test/speed_ablation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"timed_ablation": results, "fixed_depth_pruning": node_comparison}, f, indent=2)

    # Save to Markdown
    out_md = "elo_test/speed_ablation_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Experiment C: Component Speed & Ablation Study Report\n\n")
        f.write("Systematic comparison of throughput (NPS), search depth, move latency, and move accuracy across engine variants.\n\n")
        f.write("## 1. Fixed Time Benchmark (0.10s budget per move)\n\n")
        f.write("| Configuration | Average NPS | Completed Depth | Avg Move Time | Tactical Accuracy | Notes |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :--- |\n")
        for r in results:
            f.write(f"| **{r['config']}** | {r['avg_nps']:,.0f} nps | **{r['avg_depth']} plies** | {r['avg_time_ms']:.1f} ms | **{r['accuracy_pct']}%** ({r['solved_count']}/{r['total_tested']}) | {r['description']} |\n")

        f.write("\n\n## 2. Fixed Depth 4 Tree Pruning Comparison (Nodes Searched to Depth 4)\n\n")
        f.write("| Configuration | Startpos Nodes | KiwiPete Nodes | Tactical Middlegame Nodes | Node Reduction vs Baseline |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        base_nodes = sum(node_comparison["Config A: Baseline v2.2"][p]["nodes"] for p, _ in fixed_depth_positions)
        for c_name in node_comparison:
            c_nodes = sum(node_comparison[c_name][p]["nodes"] for p, _ in fixed_depth_positions)
            reduction = ((base_nodes - c_nodes) / base_nodes * 100.0) if base_nodes > 0 else 0.0
            sp = node_comparison[c_name]["Startpos"]["nodes"]
            kp = node_comparison[c_name]["KiwiPete"]["nodes"]
            tm = node_comparison[c_name]["Tension Middlegame"]["nodes"]
            f.write(f"| **{c_name}** | {sp:,} nodes | {kp:,} nodes | {tm:,} nodes | **{reduction:+.1f}%** |\n")

    print(f"\nSaved speed ablation report to: {out_json} and {out_md}")
    return results


if __name__ == "__main__":
    run_ablation_benchmarks()
