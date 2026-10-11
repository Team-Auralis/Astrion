#!/usr/bin/env python3
"""
Experiment A: Controlled Head-to-Head Benchmark — ORION v2.2 vs ORION v2.3
100 Paired Games (50 openings played twice with swapped colors)
Identical hardware, neural weights, and time controls.
Measures empirical playing-strength delta and Ordo rating gap.
"""

import os
import sys
import time
import math
import subprocess
import argparse
from typing import List, Tuple, Dict, Optional
import chess
import chess.pgn
import chess.polyglot

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
    FLAG_UPPERBOUND,
    POLYGLOT_BOOK,
    CHECKPOINT_PATH,
    ORDO_EXE,
    EPD_BOOK,
    OrionModelRunner,
    fast_encode_action
)


# ==========================================
# 1. ORION v2.2 SEARCH ENGINE (FROZEN BASELINE)
# ==========================================
class SearchEngineV22:
    """Frozen v2.2 Baseline: Alpha-Beta, TT, NMP, LMR, Check Extension (No PVS, No Repetition, No Aspiration)."""
    def __init__(self, book_path: Optional[str] = POLYGLOT_BOOK):
        self.tt = TranspositionTable(max_entries=500000)
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(64)]
        self.history_table: Dict[Tuple[int, int], int] = {}
        self.nodes_visited = 0
        self.start_time = 0.0
        self.time_limit = 0.05
        self.stop_search = False
        self.book_path = book_path

    def clear_heuristics(self):
        self.killer_moves = [[None, None] for _ in range(64)]
        self.history_table.clear()
        self.nodes_visited = 0
        self.stop_search = False

    def probe_book(self, board: chess.Board) -> Optional[chess.Move]:
        if self.book_path and os.path.exists(self.book_path):
            try:
                with chess.polyglot.open_reader(self.book_path) as reader:
                    entry = reader.get(board)
                    if entry is not None and entry.move in board.legal_moves:
                        return entry.move
            except Exception:
                pass
        return None

    def score_move(self, board: chess.Board, m: chess.Move, tt_move: Optional[chess.Move], ply: int) -> int:
        if tt_move and m == tt_move:
            return 1000000
        if board.is_capture(m):
            return 100000 + mvv_lva(board, m)
        if ply < 64:
            if m == self.killer_moves[ply][0]:
                return 90000
            elif m == self.killer_moves[ply][1]:
                return 80000
        return self.history_table.get((m.from_square, m.to_square), 0)

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0) -> float:
        self.nodes_visited += 1
        if (self.nodes_visited & 1023) == 0:
            if time.perf_counter() - self.start_time >= self.time_limit:
                self.stop_search = True
                return alpha

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
                null_score = -self.negamax(board, depth - 1 - r, -beta, -beta + 1, ply + 1)
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

        moves.sort(key=lambda m: self.score_move(board, m, tt_move, ply), reverse=True)

        best_score = -float("inf")
        best_m = None
        orig_alpha = alpha

        for i, m in enumerate(moves):
            is_capture = board.is_capture(m)
            board.push(m)
            gives_check = board.is_check()

            # LMR (Standard full window fallback)
            if i >= 4 and depth >= 3 and not in_check and not is_capture and not gives_check and not m.promotion:
                reduced_depth = max(1, depth - 2)
                score = -self.negamax(board, reduced_depth, -alpha - 1, -alpha, ply + 1)
                if score > alpha and not self.stop_search:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)

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
                break

        if best_score <= orig_alpha:
            flag = FLAG_UPPERBOUND
        elif best_score >= beta:
            flag = FLAG_LOWERBOUND
        else:
            flag = FLAG_EXACT

        self.tt.store(pos_key, depth, best_score, flag, best_m, ply)
        return best_score

    def search(self, board: chess.Board, time_limit: float = 0.05, max_depth: int = 15, root_prior: Optional[Dict[chess.Move, float]] = None) -> Tuple[chess.Move, float, int]:
        book_m = self.probe_book(board)
        if book_m:
            return book_m, 0.0, 99

        self.start_time = time.perf_counter()
        self.time_limit = time_limit
        self.clear_heuristics()

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

        if root_prior:
            moves.sort(key=lambda m: (board.is_capture(m) * 100.0 + root_prior.get(m, 0.0)), reverse=True)
        else:
            moves.sort(key=lambda m: (board.is_capture(m) * 1000 + mvv_lva(board, m)), reverse=True)

        best_move = moves[0]
        best_val = 0.0
        completed_depth = 0

        for d in range(1, max_depth + 1):
            iter_best_move = None
            iter_best_val = -float("inf")
            alpha = -float("inf")
            beta = float("inf")

            for m in moves:
                board.push(m)
                val = -self.negamax(board, d - 1, -beta, -alpha, ply=1)
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


# ==========================================
# 2. ORION v2.3 SEARCH ENGINE (CANDIDATE)
# ==========================================
from tournament_engine import OptimizedSearchEngine as SearchEngineV23


# ==========================================
# 3. PLAYER WRAPPERS
# ==========================================
class PlayerV22:
    def __init__(self, runner: OrionModelRunner):
        self.name = "ORION-v2.2"
        self.runner = runner
        self.engine = SearchEngineV22()

    def play(self, board: chess.Board, time_limit: float = 0.05) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        book_m = self.engine.probe_book(board)
        if book_m:
            return book_m
        p_scores = self.runner.get_policy(board)
        root_prior = {m: float(p_scores[fast_encode_action(m)]) for m in moves}
        best_m, _, _ = self.engine.search(board, time_limit=time_limit, root_prior=root_prior)
        return best_m


class PlayerV23:
    def __init__(self, runner: OrionModelRunner):
        self.name = "ORION-v2.3"
        self.runner = runner
        self.engine = SearchEngineV23()

    def play(self, board: chess.Board, time_limit: float = 0.05) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        book_m = self.engine.probe_book(board)
        if book_m:
            return book_m
        p_scores = self.runner.get_policy(board)
        root_prior = {m: float(p_scores[fast_encode_action(m)]) for m in moves}
        best_m, _, _ = self.engine.search(board, time_limit=time_limit, root_prior=root_prior)
        return best_m


# ==========================================
# 4. EXPERIMENT A MATCH RUNNER
# ==========================================
def load_opening_suite(epd_path: str, count: int) -> List[str]:
    fens = []
    if os.path.exists(epd_path):
        with open(epd_path, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    fen = line.split(";")[0].strip()
                    try:
                        b = chess.Board(fen)
                        fens.append(b.fen())
                    except Exception:
                        pass
                if len(fens) >= count:
                    break
    return fens


def play_h2h_game(white_p, black_p, start_fen: str, round_id: int, time_limit: float = 0.04) -> chess.pgn.Game:
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    game.headers["Event"] = "ORION Experiment A: v2.2 vs v2.3 Head-to-Head"
    game.headers["Site"] = "Local Rigorous Testbed"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["Round"] = str(round_id)
    game.headers["White"] = white_p.name
    game.headers["Black"] = black_p.name
    game.setup(board)

    node = game
    max_plies = 140
    ply_count = 0

    while not board.is_game_over() and ply_count < max_plies:
        p = white_p if board.turn == chess.WHITE else black_p
        try:
            mv = p.play(board, time_limit=time_limit)
        except Exception:
            break

        if mv is None or mv not in board.legal_moves:
            break

        node = node.add_variation(mv)
        board.push(mv)
        ply_count += 1

    if board.is_checkmate():
        res = "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.is_repetition(3):
        res = "1/2-1/2"
    else:
        from tournament_engine import evaluate_material
        mat = evaluate_material(board)
        if board.turn == chess.BLACK:
            mat = -mat
        if mat >= 300:
            res = "1-0"
        elif mat <= -300:
            res = "0-1"
        else:
            res = "1/2-1/2"

    game.headers["Result"] = res
    return game


def run_experiment_a(num_pairs: int = 50, time_limit: float = 0.04):
    print("=" * 70)
    print("EXPERIMENT A: 100 PAIRED GAMES (ORION v2.2 vs ORION v2.3)")
    print(f"Paired Openings: {num_pairs} (total {num_pairs * 2} games, color-swapped)")
    print(f"Time Budget per Move: {time_limit:.2f}s")
    print("=" * 70)

    openings = load_opening_suite(EPD_BOOK, num_pairs)
    print(f"Loaded {len(openings)} distinct balanced opening positions.")

    runner = OrionModelRunner(CHECKPOINT_PATH)
    v22 = PlayerV22(runner)
    v23 = PlayerV23(runner)

    stats = {
        "v2.3": {"W": 0, "L": 0, "D": 0, "pts": 0.0},
        "v2.2": {"W": 0, "L": 0, "D": 0, "pts": 0.0}
    }

    all_games = []
    round_id = 1
    t0 = time.time()

    for pair_idx, fen in enumerate(openings):
        # Leg 1: v2.3 White, v2.2 Black
        g1 = play_h2h_game(v23, v22, fen, round_id, time_limit=time_limit)
        all_games.append(g1)
        r1 = g1.headers["Result"]
        round_id += 1

        if r1 == "1-0":
            stats["v2.3"]["W"] += 1
            stats["v2.3"]["pts"] += 1.0
            stats["v2.2"]["L"] += 1
        elif r1 == "0-1":
            stats["v2.2"]["W"] += 1
            stats["v2.2"]["pts"] += 1.0
            stats["v2.3"]["L"] += 1
        else:
            stats["v2.3"]["D"] += 1
            stats["v2.3"]["pts"] += 0.5
            stats["v2.2"]["D"] += 1
            stats["v2.2"]["pts"] += 0.5

        # Leg 2: v2.2 White, v2.3 Black (Color Swap)
        g2 = play_h2h_game(v22, v23, fen, round_id, time_limit=time_limit)
        all_games.append(g2)
        r2 = g2.headers["Result"]
        round_id += 1

        if r2 == "1-0":
            stats["v2.2"]["W"] += 1
            stats["v2.2"]["pts"] += 1.0
            stats["v2.3"]["L"] += 1
        elif r2 == "0-1":
            stats["v2.3"]["W"] += 1
            stats["v2.3"]["pts"] += 1.0
            stats["v2.2"]["L"] += 1
        else:
            stats["v2.3"]["D"] += 1
            stats["v2.3"]["pts"] += 0.5
            stats["v2.2"]["D"] += 1
            stats["v2.2"]["pts"] += 0.5

        if (pair_idx + 1) % 5 == 0 or (pair_idx + 1) == num_pairs:
            cur_games = (pair_idx + 1) * 2
            v23_p = stats["v2.3"]["pts"]
            v22_p = stats["v2.2"]["pts"]
            print(f"[{cur_games:03d}/100 Games] Pair {pair_idx+1:02d}: "
                  f"v2.3: {stats['v2.3']['W']}W-{stats['v2.3']['L']}L-{stats['v2.3']['D']}D ({v23_p:.1f}/{cur_games}) | "
                  f"v2.2: {stats['v2.2']['W']}W-{stats['v2.2']['L']}L-{stats['v2.2']['D']}D ({v22_p:.1f}/{cur_games})")

    elapsed = time.time() - t0
    total_games = len(all_games)

    os.makedirs("elo_test", exist_ok=True)
    pgn_path = "elo_test/experiment_a_h2h.pgn"
    with open(pgn_path, "w", encoding="utf-8") as f:
        for g in all_games:
            print(g, file=f, end="\n\n")

    print("\n" + "=" * 70)
    print("EXPERIMENT A FINAL RESULTS")
    print("=" * 70)
    print(f"Total Games Played: {total_games} in {elapsed:.1f}s ({elapsed/total_games:.2f}s/game)")
    v23_pts = stats["v2.3"]["pts"]
    v22_pts = stats["v2.2"]["pts"]
    v23_pct = (v23_pts / total_games) * 100.0
    v22_pct = (v22_pts / total_games) * 100.0

    print(f"ORION-v2.3 Score: {stats['v2.3']['W']}W – {stats['v2.3']['L']}L – {stats['v2.3']['D']}D "
          f"({v23_pts:.1f}/{total_games}, {v23_pct:.1f}%)")
    print(f"ORION-v2.2 Score: {stats['v2.2']['W']}W – {stats['v2.2']['L']}L – {stats['v2.2']['D']}D "
          f"({v22_pts:.1f}/{total_games}, {v22_pct:.1f}%)")

    # Run Ordo
    ordo_out = "elo_test/experiment_a_ordo.txt"
    cmd = [
        ORDO_EXE,
        "-p", os.path.abspath(pgn_path),
        "-o", os.path.abspath(ordo_out),
        "-A", "ORION-v2.2",
        "-a", "1800.0",
        "-s", "1000",
        "-F", "95",
        "-G"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("\n--- Ordo Rating Differential ---")
    if os.path.exists(ordo_out):
        with open(ordo_out, "r") as f:
            print(f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=50, help="Number of opening pairs")
    parser.add_argument("--time", type=float, default=0.035, help="Time limit per move")
    args = parser.parse_args()

    run_experiment_a(num_pairs=args.pairs, time_limit=args.time)
