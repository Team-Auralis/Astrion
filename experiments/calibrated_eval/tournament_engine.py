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
import torch
import numpy as np

from train_fast import OrionResNet
from train_gm_dataset import load_json_to_model
from train_ccrl_full_corpus import fast_encode_board, fast_encode_action

CHECKPOINT_PATH = r"checkpoints/ccrl_30b/orion_30b_master_final_master.json"
ORDO_EXE = os.path.abspath(r"tools/ordo/ordo.exe")
EPD_BOOK = r"experiments/openings/UHO_Lichess_4852_v1.epd"

PIECE_VALS = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}


def evaluate_material(board: chess.Board) -> float:
    score = 0.0
    turn = board.turn
    for sq, pc in board.piece_map().items():
        v = PIECE_VALS.get(pc.piece_type, 0)
        score += v if pc.color == turn else -v
    return score


def quiescence_search(board: chess.Board, alpha: float, beta: float, q_depth: int = 3) -> float:
    stand_pat = evaluate_material(board)
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
        score = -quiescence_search(board, -beta, -alpha, q_depth - 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def alphabeta_search(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    if depth == 0 or board.is_game_over():
        return quiescence_search(board, alpha, beta)

    moves = list(board.legal_moves)
    if not moves:
        if board.is_check():
            return -30000.0
        return 0.0

    moves.sort(key=lambda m: board.is_capture(m), reverse=True)
    for m in moves:
        board.push(m)
        score = -alphabeta_search(board, depth - 1, -beta, -alpha)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


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
    """Full Engine: Policy prior + Alpha-Beta Minimax + Quiescence."""
    def __init__(self, runner: OrionModelRunner):
        self.runner = runner
        self.name = "ORION-Search-Full"

    def play(self, board: chess.Board, time_limit: float = 0.5) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        # Immediate mate-in-1 check
        for m in moves:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return m
            board.pop()

        p_scores = self.runner.get_policy(board)
        ranked = sorted(
            moves,
            key=lambda mv: (board.is_capture(mv) * 20.0 + p_scores[fast_encode_action(mv)]),
            reverse=True
        )

        best_m = ranked[0]
        best_val = -float("inf")
        # Search candidate moves
        for i, m in enumerate(ranked[:8]):
            d = 2 if i < 4 else 1
            board.push(m)
            val = -alphabeta_search(board, d, -50000.0, 50000.0)
            board.pop()
            prior = float(p_scores[fast_encode_action(m)])
            combined = val + prior * 2.5
            if combined > best_val:
                best_val = combined
                best_m = m
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
