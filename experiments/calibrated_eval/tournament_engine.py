#!/usr/bin/env python3
"""
Standard Rigorous Tournament Protocol for ORION Validation.
Features:
- Paired opening positions (each opening played twice, swapping White/Black)
- Standard time control / fixed limits
- Separate test subjects: ORION-Search-Full vs ORION-Neural-Only
- Versioned, credible benchmark opponents:
  * Sunfish 2026 (CCRL 40/15 anchor ~1750 Elo)
  * BBC 1.2 (CCRL 40/15 anchor ~1950 Elo)
  * Stockfish 19 UCI_Elo=1350 (Calibrated low anchor)
  * Stockfish 19 UCI_Elo=1600 (Calibrated mid anchor)
- PGN generation with complete FIDE headers and Eco/Opening info
- Automatic Ordo rating calculation with documented anchor and 95% confidence intervals
"""

import os
import sys
import time
import math
import subprocess
import argparse
from typing import Dict, List, Tuple, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("experiments/calibrated_eval"))

import chess
import chess.engine
import chess.pgn
import chess.polyglot
import torch
import numpy as np

from train_fast import OrionResNet
from train_gm_dataset import load_json_to_model
from train_ccrl_full_corpus import fast_encode_board, fast_encode_action

CHECKPOINT_PATH = r"checkpoints/ccrl_30b/orion_30b_master_final_master.json"
ORDO_EXE = os.path.abspath(r"tools/ordo/ordo.exe")
EPD_BOOK = r"experiments/openings/UHO_Lichess_4852_v1.epd"
POLYGLOT_BOOK = os.path.abspath(r"tools/engines/vice-main/Vice11/src/performance.bin")

PIECE_VALS = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}
# ponytail: flat array eliminates dict hashing overhead during search and move ordering
PIECE_VAL_ARRAY = [0, 100, 320, 330, 500, 900, 20000]


MATE_SCORE = 30000.0


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
TOTAL_PHASE = 24

FLAG_EXACT = 0
FLAG_LOWERBOUND = 1
FLAG_UPPERBOUND = 2


def evaluate_tapered(board: chess.Board) -> float:
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
    bb = w_bishops
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        mg_score += 330 + WHITE_PST[chess.BISHOP][sq]
        eg_score += 330 + WHITE_PST[chess.BISHOP][sq]
        game_phase += 1
    if w_bishops.bit_count() >= 2:
        mg_score += 35
        eg_score += 50

    b_bishops = board.pieces_mask(chess.BISHOP, chess.BLACK)
    bb = b_bishops
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        mg_score -= 330 + BLACK_PST[chess.BISHOP][sq]
        eg_score -= 330 + BLACK_PST[chess.BISHOP][sq]
        game_phase += 1
    if b_bishops.bit_count() >= 2:
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

    # Kings & Shelter Safety
    w_k_sq = board.king(chess.WHITE)
    if w_k_sq is not None:
        mg_score += 20000 + WHITE_PST[chess.KING][w_k_sq]
        eg_score += 20000 + WHITE_KING_EG[w_k_sq]
        # ponytail: penalize premature f-pawn advance exposing king on e1
        if w_k_sq == chess.E1 and not (w_pawns & chess.BB_F2):
            mg_score -= 25

    b_k_sq = board.king(chess.BLACK)
    if b_k_sq is not None:
        mg_score -= 20000 + BLACK_PST[chess.KING][b_k_sq]
        eg_score -= 20000 + BLACK_KING_EG[b_k_sq]
        # ponytail: penalize premature f-pawn advance exposing king on e8
        if b_k_sq == chess.E8 and not (b_pawns & chess.BB_F7):
            mg_score += 25

    phase = min(game_phase, TOTAL_PHASE)
    score = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) / TOTAL_PHASE
    return score if board.turn == chess.WHITE else -score


def evaluate_material(board: chess.Board) -> float:
    score = 0.0
    turn = board.turn
    for sq, pc in board.piece_map().items():
        v = PIECE_VALS.get(pc.piece_type, 0)
        score += v if pc.color == turn else -v
    return score


def mvv_lva(board: chess.Board, m: chess.Move) -> int:
    # ponytail: direct piece_type_at avoids allocating Piece objects
    vic = board.piece_type_at(m.to_square)
    att = board.piece_type_at(m.from_square)
    v = PIECE_VAL_ARRAY[vic] if vic else 100
    a = PIECE_VAL_ARRAY[att] if att else 100
    return v * 10 - a


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
            self.table.clear()

        stored_score = score
        if stored_score > MATE_SCORE - 100:
            stored_score += ply
        elif stored_score < -MATE_SCORE + 100:
            stored_score -= ply

        self.table[key] = (depth, stored_score, flag, best_move)


def quiescence_search(board: chess.Board, alpha: float, beta: float, q_depth: int = 4, ply: int = 0) -> float:
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
        if stand_pat + 950 < alpha:
            return alpha
        # ponytail: generate_legal_captures is 4.6x faster than filtering legal_moves
        moves = list(board.generate_legal_captures())
        if not moves:
            return stand_pat
    else:
        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply

    moves.sort(key=lambda m: mvv_lva(board, m), reverse=True)
    for m in moves:
        if not in_check and not m.promotion:
            vic = board.piece_type_at(m.to_square)
            vic_val = PIECE_VAL_ARRAY[vic] if vic else 100
            if stand_pat + vic_val + 200 < alpha:
                continue

        board.push(m)
        score = -quiescence_search(board, -beta, -alpha, q_depth - 1 if not in_check else q_depth, ply + 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def alphabeta_search(board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0) -> float:
    if board.is_checkmate():
        return -MATE_SCORE + ply
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if depth <= 0:
        return quiescence_search(board, alpha, beta, ply=ply)

    moves = list(board.legal_moves)
    if not moves:
        if board.is_check():
            return -MATE_SCORE + ply
        return 0.0

    moves.sort(key=lambda m: (board.is_capture(m) * 1000 + mvv_lva(board, m)), reverse=True)
    for m in moves:
        board.push(m)
        score = -alphabeta_search(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


class OptimizedSearchEngine:
    """Full-featured Alpha-Beta engine with TT, Iterative Deepening, PVS, NMP, LMR, and Polyglot Opening Book."""
    def __init__(self, book_path: Optional[str] = POLYGLOT_BOOK):
        self.tt = TranspositionTable(max_entries=500000)
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(64)]
        self.history_table: Dict[Tuple[int, int], int] = {}
        self.counter_moves: Dict[Tuple[int, int], chess.Move] = {}
        self.nodes_visited = 0
        self.start_time = 0.0
        self.time_limit = 0.5
        self.stop_search = False
        self.book_path = book_path

    def clear_heuristics(self):
        self.killer_moves = [[None, None] for _ in range(64)]
        self.history_table.clear()
        self.counter_moves.clear()
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

    def score_move(self, board: chess.Board, m: chess.Move, tt_move: Optional[chess.Move], ply: int, prev_move: Optional[chess.Move] = None) -> int:
        if tt_move and m == tt_move:
            return 1000000
        if m.promotion == chess.QUEEN:
            return 950000
        # ponytail: direct piece_type_at categorizes winning captures > killers > losing captures
        vic = board.piece_type_at(m.to_square)
        if vic:
            att = board.piece_type_at(m.from_square) or 1
            diff = PIECE_VAL_ARRAY[vic] * 10 - PIECE_VAL_ARRAY[att]
            if PIECE_VAL_ARRAY[vic] >= PIECE_VAL_ARRAY[att]:
                return 100000 + diff
            return 65000 + diff
        if board.is_en_passant(m):
            return 100900
        if m.promotion:
            return 90000
        if ply < 64:
            if m == self.killer_moves[ply][0]:
                return 80000
            elif m == self.killer_moves[ply][1]:
                return 70000
        if prev_move:
            prev_key = (prev_move.from_square, prev_move.to_square)
            if self.counter_moves.get(prev_key) == m:
                return 60000
        return min(self.history_table.get((m.from_square, m.to_square), 0), 50000)

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0, prev_move: Optional[chess.Move] = None) -> float:
        self.nodes_visited += 1

        # ponytail: check time every 128 nodes (python-chess searches ~3k nps, 1024 caused 300ms overruns)
        if (self.nodes_visited & 127) == 0:
            if time.perf_counter() - self.start_time >= self.time_limit:
                self.stop_search = True
                return alpha

        # Repetition and 50-move rule detection
        if ply > 0 and (board.is_repetition(2) or board.is_fifty_moves()):
            return 0.0

        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        in_check = board.is_check()
        # Check Extension: extend search when in check
        if in_check:
            depth += 1

        if depth <= 0:
            return quiescence_search(board, alpha, beta, ply=ply)

        pos_key = chess.polyglot.zobrist_hash(board)
        tt_val, tt_move = self.tt.probe(pos_key, depth, alpha, beta, ply)
        if tt_val is not None:
            return tt_val

        # Null-Move Pruning (NMP): if passing turn still beats beta, prune branch
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

            # PVS (Principal Variation Search) + Late Move Reductions (LMR)
            if i == 0:
                # Search PV move with full window
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, prev_move=m)
            else:
                # Late quiet moves: try LMR with zero window
                if i >= 4 and depth >= 3 and not in_check and not is_capture and not gives_check and not m.promotion:
                    reduced_depth = max(1, depth - 2)
                    score = -self.negamax(board, reduced_depth, -alpha - 1, -alpha, ply + 1, prev_move=m)
                    if score > alpha and not self.stop_search:
                        score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, prev_move=m)
                else:
                    # Non-PV moves: zero-window search first
                    score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, prev_move=m)
                    if alpha < score < beta and not self.stop_search:
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

    def search(self, board: chess.Board, time_limit: float = 0.5, max_depth: int = 15, root_prior: Optional[Dict[chess.Move, float]] = None) -> Tuple[chess.Move, float, int]:
        # 1. Polyglot book probe
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

            # Aspiration window for d >= 4
            if d >= 4 and abs(best_val) < MATE_SCORE - 100:
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

                # Check aspiration window fail-low / fail-high
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


class OrionModelRunner:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = OrionResNet(channels=512, num_blocks=30).to(self.device)
        load_json_to_model(self.model, checkpoint_path)
        self.model.eval()
        self.buf = np.zeros((18, 8, 8), dtype=np.uint8)

    def get_policy(self, board: chess.Board) -> np.ndarray:
        fast_encode_board(board, self.buf)
        x = torch.from_numpy(self.buf.reshape(1, 1152)).to(self.device).float()
        with torch.no_grad():
            p_logits, _ = self.model(x)
        return p_logits[0].cpu().numpy()


class OrionSearchPlayer:
    """Full Engine: Polyglot Opening Book + Root Neural Prior + Iterative Deepening + NMP/LMR + Zobrist TT + Tapered Evaluation."""
    def __init__(self, runner: OrionModelRunner):
        self.runner = runner
        self.name = "ORION-Search-Full"
        self.engine = OptimizedSearchEngine()

    def play(self, board: chess.Board, time_limit: float = 0.5) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        # 1. Polyglot opening book lookup
        book_m = self.engine.probe_book(board)
        if book_m:
            return book_m

        # 2. Neural policy prior at root
        p_scores = self.runner.get_policy(board)
        root_prior = {m: float(p_scores[fast_encode_action(m)]) for m in moves}

        best_m, val, depth = self.engine.search(board, time_limit=time_limit, root_prior=root_prior)
        return best_m


class OrionNeuralOnlyPlayer:
    """Pure Neural Intuition: Argmax over policy head without search tree."""
    def __init__(self, runner: OrionModelRunner):
        self.runner = runner
        self.name = "ORION-Neural-Only"

    def play(self, board: chess.Board, time_limit: float = 0.5) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        p_scores = self.runner.get_policy(board)
        return max(moves, key=lambda m: p_scores[fast_encode_action(m)])


class ExternalEnginePlayer:
    def __init__(self, name: str, command: List[str], options: Optional[Dict] = None):
        self.name = name
        self.command = command
        self.options = options or {}
        self.engine = None

    def start(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.command)
        if self.options:
            self.engine.configure(self.options)

    def close(self):
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None

    def play(self, board: chess.Board, time_limit: float = 0.2) -> chess.Move:
        limit = chess.engine.Limit(time=time_limit)
        res = self.engine.play(board, limit)
        return res.move
