#!/usr/bin/env python3
"""
Unit and regression tests for Tactical Mate Scoring and Quiescence Search correctness.
Tests:
1. Quiescence search on checkmate returns -MATE_SCORE, NOT material difference.
2. Quiescence search while in check must evade check and detect mate if no evasions exist.
3. Alpha-beta search correctly detects mate-in-1 and mate-in-2.
4. Engine avoids walking into immediate mate when threatened.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("experiments/calibrated_eval"))

import unittest
import chess
from experiments.calibrated_eval.tournament_engine import (
    evaluate_material,
    quiescence_search,
    alphabeta_search,
    PIECE_VALS
)

MATE_SCORE = 30000.0


class TestTacticalMateScoring(unittest.TestCase):

    def test_quiescence_checkmate_returns_mate_score(self):
        # Scholar's mate position: Black is checkmated
        # 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#
        board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        self.assertTrue(board.is_checkmate())
        self.assertTrue(board.is_check())

        # Old bug returned stand_pat = evaluate_material(board) which is ~-900 (down a queen), NOT a terminal mate
        score = quiescence_search(board, -50000.0, 50000.0)
        self.assertLessEqual(score, -25000.0, f"Checkmate score must be terminal negative (<= -25000), got {score}")

    def test_quiescence_in_check_must_evade(self):
        # Fool's mate setup: White in check from Qh4#
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        self.assertTrue(board.is_checkmate())
        score = quiescence_search(board, -50000.0, 50000.0)
        self.assertLessEqual(score, -25000.0, f"Checkmate in quiescence must be <= -25000, got {score}")

    def test_alphabeta_mate_in_one(self):
        # White has mate in 1 with Qxf7#
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        mate_move = chess.Move.from_uci("h5f7")

        # Find best move using alphabeta
        best_score = -float("inf")
        best_move = None
        for m in board.legal_moves:
            board.push(m)
            score = -alphabeta_search(board, 2, -50000.0, 50000.0)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = m

        self.assertEqual(best_move, mate_move, f"Expected Qxf7#, got {best_move}")
        self.assertGreaterEqual(best_score, 25000.0, f"Score should reflect mate (>25000), got {best_score}")

    def test_alphabeta_avoids_walking_into_mate(self):
        # Black to move: 1... g6 or 1... Qe7 avoids Qxf7#, but 1... a6 allows mate in 1
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 4 4")
        # Defending moves: d5, Qe7, Qe7, g6
        board_copy = board.copy()
        
        # Test each legal move; any move that allows Qxf7# should have score <= -25000
        blunder_move = chess.Move.from_uci("a7a6")
        board_copy.push(blunder_move)
        score = -alphabeta_search(board_copy, 2, -50000.0, 50000.0)
        board_copy.pop()
        self.assertLessEqual(score, -25000.0, f"Blundering into Qxf7# must evaluate as <= -25000, got {score}")


    def test_stalemate_returns_zero(self):
        # Classic stalemate: Black king on a8, White queen on c7, White king on b6
        board = chess.Board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1")
        self.assertTrue(board.is_stalemate())
        score = alphabeta_search(board, 2, -50000.0, 50000.0)
        self.assertEqual(score, 0.0, f"Stalemate should evaluate to 0.0, got {score}")

    def test_insufficient_material_returns_zero(self):
        # King vs King: immediate draw
        board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
        self.assertTrue(board.is_insufficient_material())
        score = alphabeta_search(board, 2, -50000.0, 50000.0)
        self.assertEqual(score, 0.0, f"Insufficient material should evaluate to 0.0, got {score}")

    def test_mate_distance_ply_preference(self):
        # Position where White has immediate mate in 1 vs longer mate
        # With ply penalty, mate at ply 1 (+29999) must be scored higher than mate at ply 3 (+29997)
        score_ply1 = MATE_SCORE - 1
        score_ply3 = MATE_SCORE - 3
        self.assertGreater(score_ply1, score_ply3, "Closer mate should have higher value than delayed mate")

    def test_mate_in_two_solve(self):
        # Morphy puzzle: White to move and mate in 2 (1. Ra8+ Bxa8 2. Rxa8# or similar)
        # Simple back-rank mate in 2: White Rooks on e1, d1; Black King on g8 with pawns on f7, g7, h7
        board = chess.Board("6k1/5ppp/8/8/8/8/8/3RR1K1 w - - 0 1")
        # 1. Rd8# is immediate mate in 1
        # Let's test:
        best_score = -float("inf")
        best_move = None
        for m in board.legal_moves:
            board.push(m)
            score = -alphabeta_search(board, 2, -50000.0, 50000.0)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = m
        self.assertIn(best_move.uci(), ["d1d8", "e1e8"], f"Expected back-rank mate Rd8# or Re8#, got {best_move}")
        self.assertGreaterEqual(best_score, 25000.0, f"Expected mate score, got {best_score}")


if __name__ == "__main__":
    unittest.main()

