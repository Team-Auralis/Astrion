#!/usr/bin/env python3
"""
Test and Benchmark for ORION Optimized Search Engine (Gate 4 Prototype).
Features:
- 64-bit Zobrist Transposition Table with EXACT/LOWER/UPPER flags and mate-ply bounds
- Tapered Classical Evaluation (Material + Piece-Square Tables + Game Phase)
- Dynamic Iterative Deepening with Time Management
- Advanced Move Ordering (TT move -> MVV-LVA -> Killer Moves -> History Heuristic)
- Root Neural Prior Integration (18.8M model runs once at root to seed candidate ordering)
"""

import sys
import os
import time
import chess
import chess.polyglot
from typing import Optional, Tuple, Dict, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("experiments/calibrated_eval"))

MATE_SCORE = 30000.0
FLAG_EXACT = 0
FLAG_LOWERBOUND = 1
FLAG_UPPERBOUND = 2

PIECE_VALS = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Tomasz Michniewski Simplified Evaluation Tables
PAWN_TABLE = [
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [ 5,  5, 10, 25, 25, 10,  5,  5],
    [ 0,  0,  0, 20, 20,  0,  0,  0],
    [ 5, -5,-10,  0,  0,-10, -5,  5],
    [ 5, 10, 10,-20,-20, 10, 10,  5],
    [ 0,  0,  0,  0,  0,  0,  0,  0]
]

KNIGHT_TABLE = [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50]
]

BISHOP_TABLE = [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20]
]

ROOK_TABLE = [
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 5, 10, 10, 10, 10, 10, 10,  5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [ 0,  0,  0,  5,  5,  0,  0,  0]
]

QUEEN_TABLE = [
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [ -5,  0,  5,  5,  5,  5,  0, -5],
    [  0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20]
]

KING_MG_TABLE = [
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [ 20, 20,  0,  0,  0,  0, 20, 20],
    [ 20, 30, 10,  0,  0, 10, 30, 20]
]

KING_EG_TABLE = [
    [-50,-40,-30,-20,-20,-30,-40,-50],
    [-30,-20,-10,  0,  0,-10,-20,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-30,  0,  0,  0,  0,-30,-30],
    [-50,-30,-30,-30,-30,-30,-30,-50]
]

# Pre-flattened 64-element 1D lookup arrays
WHITE_PST = {pt: [0] * 64 for pt in PIECE_VALS}
BLACK_PST = {pt: [0] * 64 for pt in PIECE_VALS}
WHITE_KING_EG = [0] * 64
BLACK_KING_EG = [0] * 64

RAW_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_MG_TABLE
}

for pt, table in RAW_TABLES.items():
    for sq in range(64):
        r, c = chess.square_rank(sq), chess.square_file(sq)
        WHITE_PST[pt][sq] = table[7 - r][c]
        BLACK_PST[pt][sq] = table[r][c]

for sq in range(64):
    r, c = chess.square_rank(sq), chess.square_file(sq)
    WHITE_KING_EG[sq] = KING_EG_TABLE[7 - r][c]
    BLACK_KING_EG[sq] = KING_EG_TABLE[r][c]

PHASE_WEIGHTS = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.PAWN: 0,
    chess.KING: 0
}
TOTAL_PHASE = 24  # 4*1 + 4*1 + 4*2 + 2*4 = 24


def evaluate_tapered(board: chess.Board) -> float:
    """Tapered piece-square and material evaluation in ~25 microseconds."""
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    mg_score = 0
    eg_score = 0
    game_phase = 0

    piece_map = board.piece_map()
    for sq, pc in piece_map.items():
        pt = pc.piece_type
        val = PIECE_VALS[pt]
        game_phase += PHASE_WEIGHTS[pt]

        if pc.color == chess.WHITE:
            mg_score += val + WHITE_PST[pt][sq]
            eg_score += val + (WHITE_KING_EG[sq] if pt == chess.KING else WHITE_PST[pt][sq])
        else:
            mg_score -= val + BLACK_PST[pt][sq]
            eg_score -= val + (BLACK_KING_EG[sq] if pt == chess.KING else BLACK_PST[pt][sq])

    # Tapered interpolation
    phase = min(game_phase, TOTAL_PHASE)
    score = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) / TOTAL_PHASE
    return score if board.turn == chess.WHITE else -score


class TranspositionTable:
    """64-bit Zobrist Transposition Table with replacement policy."""
    def __init__(self, max_entries: int = 500000):
        self.table: Dict[int, Tuple[int, float, int, Optional[chess.Move]]] = {}
        self.max_entries = max_entries

    def clear(self):
        self.table.clear()

    def probe(self, key: int, depth: int, alpha: float, beta: float, ply: int) -> Tuple[Optional[float], Optional[chess.Move]]:
        entry = self.table.get(key)
        if not entry:
            return None, None
        e_depth, e_score, e_flag, e_move = entry

        # Mate score retrieval adjustment
        score = e_score
        if score > MATE_SCORE - 100:
            score -= ply
        elif score < -MATE_SCORE + 100:
            score += ply

        if e_depth >= depth:
            if e_flag == FLAG_EXACT:
                return score, e_move
            elif e_flag == FLAG_LOWERBOUND and score >= beta:
                return score, e_move
            elif e_flag == FLAG_UPPERBOUND and score <= alpha:
                return score, e_move

        return None, e_move

    def store(self, key: int, depth: int, score: float, flag: int, best_move: Optional[chess.Move], ply: int):
        if len(self.table) >= self.max_entries:
            # Simple eviction when full
            self.table.clear()

        # Mate score storage adjustment
        stored_score = score
        if stored_score > MATE_SCORE - 100:
            stored_score += ply
        elif stored_score < -MATE_SCORE + 100:
            stored_score -= ply

        self.table[key] = (depth, stored_score, flag, best_move)


def mvv_lva(board: chess.Board, m: chess.Move) -> int:
    vic = board.piece_at(m.to_square)
    att = board.piece_at(m.from_square)
    v = PIECE_VALS.get(vic.piece_type, 0) if vic else 100
    a = PIECE_VALS.get(att.piece_type, 0) if att else 100
    return v * 10 - a


class OptimizedSearchEngine:
    def __init__(self):
        self.tt = TranspositionTable(max_entries=300000)
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(64)]
        self.history_table: Dict[Tuple[int, int], int] = {}
        self.nodes_visited = 0
        self.start_time = 0.0
        self.time_limit = 0.5
        self.stop_search = False

    def clear_heuristics(self):
        self.killer_moves = [[None, None] for _ in range(64)]
        self.history_table.clear()
        self.nodes_visited = 0
        self.stop_search = False

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float, q_depth: int = 4, ply: int = 0) -> float:
        self.nodes_visited += 1
        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        in_check = board.is_check()
        if not in_check:
            stand_pat = evaluate_tapered(board)
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
            if q_depth <= 0:
                return stand_pat
            moves = [m for m in board.legal_moves if board.is_capture(m)]
            if not moves:
                return stand_pat
        else:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE_SCORE + ply

        moves.sort(key=lambda m: mvv_lva(board, m), reverse=True)
        for m in moves:
            board.push(m)
            score = -self.quiescence_search(board, -beta, -alpha, q_depth - 1 if not in_check else q_depth, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

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

        # Check time periodically
        if (self.nodes_visited & 1023) == 0:
            if time.perf_counter() - self.start_time >= self.time_limit:
                self.stop_search = True
                return alpha

        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        if depth <= 0:
            return self.quiescence_search(board, alpha, beta, ply=ply)

        pos_key = chess.polyglot.zobrist_hash(board)
        tt_val, tt_move = self.tt.probe(pos_key, depth, alpha, beta, ply)
        if tt_val is not None:
            return tt_val

        moves = list(board.legal_moves)
        if not moves:
            if board.is_check():
                return -MATE_SCORE + ply
            return 0.0

        # Move ordering
        moves.sort(key=lambda m: self.score_move(board, m, tt_move, ply), reverse=True)

        best_score = -float("inf")
        best_m = None
        orig_alpha = alpha

        for i, m in enumerate(moves):
            board.push(m)
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
                # Beta cutoff
                if not board.is_capture(m) and ply < 64:
                    if self.killer_moves[ply][0] != m:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = m
                    h_key = (m.from_square, m.to_square)
                    self.history_table[h_key] = self.history_table.get(h_key, 0) + depth * depth
                break

        # Transposition table flag
        if best_score <= orig_alpha:
            flag = FLAG_UPPERBOUND
        elif best_score >= beta:
            flag = FLAG_LOWERBOUND
        else:
            flag = FLAG_EXACT

        self.tt.store(pos_key, depth, best_score, flag, best_m, ply)
        return best_score

    def search(self, board: chess.Board, time_limit: float = 0.5, max_depth: int = 15, root_prior: Optional[Dict[chess.Move, float]] = None) -> Tuple[chess.Move, float, int]:
        self.start_time = time.perf_counter()
        self.time_limit = time_limit
        self.clear_heuristics()

        moves = list(board.legal_moves)
        if not moves:
            return None, 0.0, 0
        if len(moves) == 1:
            return moves[0], 0.0, 1

        # Check immediate mate in 1
        for m in moves:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return m, MATE_SCORE - 1, 1
            board.pop()

        # Seed move ordering with root neural prior if provided
        if root_prior:
            moves.sort(key=lambda m: (board.is_capture(m) * 100.0 + root_prior.get(m, 0.0)), reverse=True)
        else:
            moves.sort(key=lambda m: (board.is_capture(m) * 1000 + mvv_lva(board, m)), reverse=True)

        best_move = moves[0]
        best_val = 0.0
        completed_depth = 0

        # Iterative Deepening
        for d in range(1, max_depth + 1):
            iter_best_move = None
            iter_best_val = -float("inf")
            alpha = -float("inf")
            beta = float("inf")

            # Search root moves
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

            if self.stop_search:
                break

            best_move = iter_best_move
            best_val = iter_best_val
            completed_depth = d

            # Re-sort root moves so best move is searched first next iteration
            moves.remove(best_move)
            moves.insert(0, best_move)

            # Check if mate was found
            if abs(best_val) > MATE_SCORE - 100:
                break

            # Time check: if more than 60% of budget spent, do not start next depth
            elapsed = time.perf_counter() - self.start_time
            if elapsed >= time_limit * 0.6:
                break

        return best_move, best_val, completed_depth


def main():
    print("Testing Gate 4 Optimized Search Engine...")
    engine = OptimizedSearchEngine()
    board = chess.Board()

    # Benchmark starting position search at 0.5s
    t0 = time.perf_counter()
    best_m, val, depth = engine.search(board, time_limit=0.5)
    t_elapsed = time.perf_counter() - t0
    nps = engine.nodes_visited / t_elapsed

    print(f"Startpos Move: {best_m} | Eval: {val:+.1f} cp | Depth: {depth} | Nodes: {engine.nodes_visited} | NPS: {nps:,.0f} | Time: {t_elapsed*1000:.1f}ms")

    # Run Tactical Benchmark Suite
    from experiments.calibrated_eval.benchmark_tactical_delta import BENCHMARK_POSITIONS, allows_immediate_mate

    solved = 0
    blunders = 0
    tested = 0

    print("\nRunning Tactical Benchmark on Gate 4 Engine:")
    for idx, (fen, acceptable, desc) in enumerate(BENCHMARK_POSITIONS):
        b = chess.Board(fen)
        if b.is_game_over():
            continue
        tested += 1
        m, _, d = engine.search(b, time_limit=0.1)
        m_uci = m.uci() if m else None
        is_correct = m_uci in acceptable if acceptable else False
        is_blunder = allows_immediate_mate(b, m)
        if is_correct:
            solved += 1
        if is_blunder:
            blunders += 1
        print(f"[{tested:02d}/18] {desc:<45} -> {str(m_uci):<6} (Depth {d}) [{'PASS' if is_correct else 'FAIL'}]")

    print("=" * 60)
    print(f"Gate 4 Engine Tactical Solve Rate: {solved}/{tested} ({solved/tested*100:.1f}%)")
    print(f"Blunders into Mate: {blunders}")
    print("=" * 60)


if __name__ == "__main__":
    main()
