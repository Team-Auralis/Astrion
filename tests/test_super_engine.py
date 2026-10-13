#!/usr/bin/env python3
"""
Test bench for ORION Super Engine with:
- Enhanced Tapered Positional Evaluation (Bishop Pair, Open Rook Files, Passed Pawns)
- Null-Move Pruning (NMP)
- Late Move Reductions (LMR)
- Check Extensions
- Delta Pruning in Quiescence
- Polyglot Opening Book Probing
"""

import os
import sys
import time
import chess
import chess.polyglot
from typing import Optional, Tuple, Dict, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("experiments/calibrated_eval"))

from tournament_engine import (
    MATE_SCORE, FLAG_EXACT, FLAG_LOWERBOUND, FLAG_UPPERBOUND,
    PIECE_VALS, WHITE_PST, BLACK_PST, WHITE_KING_EG, BLACK_KING_EG,
    TOTAL_PHASE, PHASE_WEIGHTS, TranspositionTable, mvv_lva
)

POLYGLOT_BOOK = os.path.abspath(r"tools/engines/vice-main/Vice11/src/performance.bin")


def evaluate_enhanced_tapered(board: chess.Board) -> float:
    """Bitboard-accelerated tapered evaluation with Bishop Pair and Open File heuristics."""
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    mg_score = 0
    eg_score = 0
    game_phase = 0

    w_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
    b_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)
    all_pawns = w_pawns | b_pawns

    # Pawns
    bb = w_pawns
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        mg_score += 100 + WHITE_PST[chess.PAWN][sq]
        eg_score += 100 + WHITE_PST[chess.PAWN][sq]

    bb = b_pawns
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        mg_score -= 100 + BLACK_PST[chess.PAWN][sq]
        eg_score -= 100 + BLACK_PST[chess.PAWN][sq]

    # Knights
    bb = board.pieces_mask(chess.KNIGHT, chess.WHITE)
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        mg_score += 320 + WHITE_PST[chess.KNIGHT][sq]
        eg_score += 320 + WHITE_PST[chess.KNIGHT][sq]
        game_phase += 1

    bb = board.pieces_mask(chess.KNIGHT, chess.BLACK)
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        mg_score -= 320 + BLACK_PST[chess.KNIGHT][sq]
        eg_score -= 320 + BLACK_PST[chess.KNIGHT][sq]
        game_phase += 1

    # Bishops & Bishop Pair
    w_bishops = board.pieces_mask(chess.BISHOP, chess.WHITE)
    w_b_count = 0
    bb = w_bishops
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        mg_score += 330 + WHITE_PST[chess.BISHOP][sq]
        eg_score += 330 + WHITE_PST[chess.BISHOP][sq]
        game_phase += 1
        w_b_count += 1
    if w_b_count >= 2:
        mg_score += 35
        eg_score += 50

    b_bishops = board.pieces_mask(chess.BISHOP, chess.BLACK)
    b_b_count = 0
    bb = b_bishops
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        mg_score -= 330 + BLACK_PST[chess.BISHOP][sq]
        eg_score -= 330 + BLACK_PST[chess.BISHOP][sq]
        game_phase += 1
        b_b_count += 1
    if b_b_count >= 2:
        mg_score -= 35
        eg_score -= 50

    # Rooks & Open / Semi-open files
    bb = board.pieces_mask(chess.ROOK, chess.WHITE)
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        mg_score += 500 + WHITE_PST[chess.ROOK][sq]
        eg_score += 500 + WHITE_PST[chess.ROOK][sq]
        game_phase += 2
        f = sq & 7
        if (all_pawns & chess.BB_FILES[f]) == 0:
            mg_score += 25
            eg_score += 25
        elif (w_pawns & chess.BB_FILES[f]) == 0:
            mg_score += 15
            eg_score += 15

    bb = board.pieces_mask(chess.ROOK, chess.BLACK)
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        mg_score -= 500 + BLACK_PST[chess.ROOK][sq]
        eg_score -= 500 + BLACK_PST[chess.ROOK][sq]
        game_phase += 2
        f = sq & 7
        if (all_pawns & chess.BB_FILES[f]) == 0:
            mg_score -= 25
            eg_score -= 25
        elif (b_pawns & chess.BB_FILES[f]) == 0:
            mg_score -= 15
            eg_score -= 15

    # Queens
    bb = board.pieces_mask(chess.QUEEN, chess.WHITE)
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        mg_score += 900 + WHITE_PST[chess.QUEEN][sq]
        eg_score += 900 + WHITE_PST[chess.QUEEN][sq]
        game_phase += 4

    bb = board.pieces_mask(chess.QUEEN, chess.BLACK)
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        mg_score -= 900 + BLACK_PST[chess.QUEEN][sq]
        eg_score -= 900 + BLACK_PST[chess.QUEEN][sq]
        game_phase += 4

    # Kings
    w_k_sq = board.king(chess.WHITE)
    if w_k_sq is not None:
        mg_score += 20000 + WHITE_PST[chess.KING][w_k_sq]
        eg_score += 20000 + WHITE_KING_EG[w_k_sq]

    b_k_sq = board.king(chess.BLACK)
    if b_k_sq is not None:
        mg_score -= 20000 + BLACK_PST[chess.KING][b_k_sq]
        eg_score -= 20000 + BLACK_KING_EG[b_k_sq]

    phase = min(game_phase, TOTAL_PHASE)
    score = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) / TOTAL_PHASE
    return score if board.turn == chess.WHITE else -score


def quiescence_enhanced(board: chess.Board, alpha: float, beta: float, q_depth: int = 4, ply: int = 0) -> float:
    if board.is_checkmate():
        return -MATE_SCORE + ply
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    in_check = board.is_check()
    if not in_check:
        stand_pat = evaluate_enhanced_tapered(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        if q_depth <= 0:
            return stand_pat
        if stand_pat + 950 < alpha:
            return alpha
        moves = [m for m in board.legal_moves if board.is_capture(m)]
        if not moves:
            return stand_pat
    else:
        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply

    moves.sort(key=lambda m: mvv_lva(board, m), reverse=True)
    for m in moves:
        if not in_check and not m.promotion:
            vic = board.piece_at(m.to_square)
            vic_val = PIECE_VALS.get(vic.piece_type, 0) if vic else 100
            if stand_pat + vic_val + 200 < alpha:
                continue

        board.push(m)
        score = -quiescence_enhanced(board, -beta, -alpha, q_depth - 1 if not in_check else q_depth, ply + 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


class SuperSearchEngine:
    def __init__(self, book_path: Optional[str] = POLYGLOT_BOOK):
        self.tt = TranspositionTable(max_entries=500000)
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(64)]
        self.history_table: Dict[Tuple[int, int], int] = {}
        self.nodes_visited = 0
        self.start_time = 0.0
        self.time_limit = 0.5
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
            return quiescence_enhanced(board, alpha, beta, ply=ply)

        pos_key = chess.polyglot.zobrist_hash(board)
        tt_val, tt_move = self.tt.probe(pos_key, depth, alpha, beta, ply)
        if tt_val is not None:
            return tt_val

        # Null-Move Pruning (NMP)
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

            # Late Move Reductions (LMR)
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

    def search(self, board: chess.Board, time_limit: float = 0.5, max_depth: int = 15, root_prior: Optional[Dict[chess.Move, float]] = None) -> Tuple[chess.Move, float, int]:
        # 1. Polyglot book move
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


def main():
    print("Testing ORION Super Engine...")
    engine = SuperSearchEngine()

    # 1. Starting position (should hit book instantly!)
    b0 = chess.Board()
    t0 = time.perf_counter()
    m0, val0, d0 = engine.search(b0, time_limit=0.1)
    t0_elapsed = time.perf_counter() - t0
    print(f"Startpos Book Probe: {m0} (Depth {d0}) in {t0_elapsed*1000:.2f}ms")

    # 2. Out-of-book middlegame position
    b1 = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7")
    t1 = time.perf_counter()
    m1, val1, d1 = engine.search(b1, time_limit=0.5)
    t1_elapsed = time.perf_counter() - t1
    nps1 = engine.nodes_visited / t1_elapsed if t1_elapsed > 0 else 0
    print(f"Middlegame Search: {m1} | Eval: {val1:+.1f} cp | Depth: {d1} | Nodes: {engine.nodes_visited} | NPS: {nps1:,.0f} | Time: {t1_elapsed*1000:.1f}ms")

    # 3. Tactical Benchmark Suite
    from experiments.calibrated_eval.benchmark_tactical_delta import BENCHMARK_POSITIONS, allows_immediate_mate
    solved = 0
    blunders = 0
    tested = 0

    print("\nRunning Tactical Benchmark on Super Engine:")
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
    print(f"Super Engine Tactical Solve Rate: {solved}/{tested} ({solved/tested*100:.1f}%)")
    print(f"Blunders into Mate: {blunders}")
    print("=" * 60)


if __name__ == "__main__":
    main()
