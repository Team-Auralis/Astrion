import math
import random
from typing import Optional, Tuple, List, Dict
from orion.environment.chess import ChessEnvironment
from orion.model.actions import ChessActionSpace
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork

# Piece values in centipawns
PIECE_VALUES = {
    "P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 20000,
    "p": -100, "n": -320, "b": -330, "r": -500, "q": -900, "k": -20000,
    ".": 0
}

# Positional tables (White perspective, flipped for Black)
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


class StockfishInspiredSearchAgent:
    """
    Enhanced ORION Agent combining Deep Neural Policy/Value guidance at the root
    with high-speed Stockfish-inspired Alpha-Beta Minimax search, Quiescence Search,
    MVV-LVA move ordering, and Anti-Repetition logic.
    """

    def __init__(self, network: ORIONNetwork, encoder: ChessEncoder, default_depth: int = 2, nn_weight: float = 0.0):
        self.network = network
        self.encoder = encoder
        self.default_depth = default_depth
        self.nn_weight = nn_weight
        self.nodes_evaluated = 0

    def evaluate_board(self, env: ChessEnvironment) -> float:
        """
        Fast static evaluation function blending material and piece-square tables.
        Evaluates in microseconds per position to allow deep search trees.
        """
        self.nodes_evaluated += 1
        board = env.board

        material_score = 0
        positional_score = 0

        for r in range(8):
            for c in range(8):
                p = board[r][c]
                if p == ".":
                    continue
                material_score += PIECE_VALUES.get(p, 0)

                if p == "P":
                    positional_score += PAWN_TABLE[r][c]
                elif p == "p":
                    positional_score -= PAWN_TABLE[7 - r][c]
                elif p == "N":
                    positional_score += KNIGHT_TABLE[r][c]
                elif p == "n":
                    positional_score -= KNIGHT_TABLE[7 - r][c]
                elif p == "B":
                    positional_score += BISHOP_TABLE[r][c]
                elif p == "b":
                    positional_score -= BISHOP_TABLE[7 - r][c]

        return float(material_score + positional_score)

    def _is_capture(self, env: ChessEnvironment, move: Tuple[int, int, int, int]) -> bool:
        r1, c1, r2, c2 = move
        target_piece = env.board[r2][c2]
        if target_piece != ".":
            return True
        piece = env.board[r1][c1]
        if piece.lower() == "p" and env.en_passant == (r2, c2):
            return True
        return False

    def _score_move_for_ordering(self, env: ChessEnvironment, move: Tuple[int, int, int, int], policy_scores: Optional[Dict] = None) -> float:
        r1, c1, r2, c2 = move
        attacker = env.board[r1][c1]
        victim = env.board[r2][c2]

        score = 0.0
        if victim != ".":
            victim_val = abs(PIECE_VALUES.get(victim, 0))
            attacker_val = abs(PIECE_VALUES.get(attacker, 0))
            score += 10000.0 + (victim_val * 10 - attacker_val)
        elif env.en_passant == (r2, c2) and attacker.lower() == "p":
            score += 10000.0 + 900.0

        if policy_scores and move in policy_scores:
            score += policy_scores[move] * 1000.0

        return score

    def quiescence_search(self, env: ChessEnvironment, alpha: float, beta: float, depth_limit: int = 3) -> float:
        stand_pat = self.evaluate_board(env)
        turn_multiplier = 1.0 if env.turn == "white" else -1.0
        current_eval = stand_pat * turn_multiplier

        if current_eval >= beta:
            return beta
        if alpha < current_eval:
            alpha = current_eval

        if depth_limit <= 0:
            return alpha

        legal_moves = env.legal_moves()
        captures = [m for m in legal_moves if self._is_capture(env, m)]
        if not captures:
            return alpha

        captures.sort(key=lambda m: self._score_move_for_ordering(env, m), reverse=True)

        for move in captures:
            prev_board = [row[:] for row in env.board]
            prev_turn = env.turn
            prev_castling = env.castling.copy()
            prev_en_passant = env.en_passant
            prev_halfmove = env.halfmove_clock
            prev_move_count = env.move_count
            prev_history = env.position_history.copy()

            if not env.make_move(move):
                continue

            score = -self.quiescence_search(env, -beta, -alpha, depth_limit - 1)

            env.board = prev_board
            env.turn = prev_turn
            env.castling = prev_castling
            env.en_passant = prev_en_passant
            env.halfmove_clock = prev_halfmove
            env.move_count = prev_move_count
            env.position_history = prev_history

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def alpha_beta(self, env: ChessEnvironment, depth: int, alpha: float, beta: float, policy_scores: Optional[Dict] = None) -> Tuple[float, Optional[Tuple]]:
        status = env.game_status()
        if status.startswith("checkmate:"):
            winner = status.split(":")[1]
            return (-50000.0 if winner != env.turn else 50000.0), None
        if status.startswith("draw") or status == "stalemate":
            return -350.0, None

        if depth <= 0:
            q_score = self.quiescence_search(env, alpha, beta)
            return q_score, None

        legal_moves = env.legal_moves()
        if not legal_moves:
            return 0.0, None

        ordered_moves = sorted(legal_moves, key=lambda m: self._score_move_for_ordering(env, m, policy_scores), reverse=True)
        best_move = ordered_moves[0]
        best_score = -float("inf")

        for move in ordered_moves:
            prev_board = [row[:] for row in env.board]
            prev_turn = env.turn
            prev_castling = env.castling.copy()
            prev_en_passant = env.en_passant
            prev_halfmove = env.halfmove_clock
            prev_move_count = env.move_count
            prev_history = env.position_history.copy()

            if not env.make_move(move):
                continue

            # Anti-repetition penalty: if the real game has already seen this position
            rep_count = prev_history.get(env._position_key(), 0)
            rep_penalty = 400.0 if rep_count >= 1 else 0.0

            score, _ = self.alpha_beta(env, depth - 1, -beta, -alpha, policy_scores=None)
            score = -score - rep_penalty

            env.board = prev_board
            env.turn = prev_turn
            env.castling = prev_castling
            env.en_passant = prev_en_passant
            env.halfmove_clock = prev_halfmove
            env.move_count = prev_move_count
            env.position_history = prev_history

            if score > best_score:
                best_score = score
                best_move = move

            if best_score > alpha:
                alpha = best_score

            if alpha >= beta:
                break

        return best_score, best_move

    def choose_move(self, environment: ChessEnvironment, depth: Optional[int] = None, temperature: float = 0.0) -> Optional[Tuple[int, int, int, int]]:
        search_depth = depth if depth is not None else self.default_depth
        self.nodes_evaluated = 0

        legal = environment.legal_moves()
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        policy_scores = {}
        try:
            state = environment.get_state()
            encoded = self.encoder.encode_flat(state)
            out = self.network.forward(encoded)
            policy = out["policy"]
            for m in legal:
                act = ChessActionSpace.encode(m)
                policy_scores[m] = policy[act]
        except Exception:
            policy_scores = {}

        move_scores = []
        for move in legal:
            prev_board = [row[:] for row in environment.board]
            prev_turn = environment.turn
            prev_castling = environment.castling.copy()
            prev_en_passant = environment.en_passant
            prev_halfmove = environment.halfmove_clock
            prev_move_count = environment.move_count
            prev_history = environment.position_history.copy()

            if not environment.make_move(move):
                continue

            rep_count = prev_history.get(environment._position_key(), 0)
            rep_penalty = 500.0 if rep_count >= 1 else 0.0

            score, _ = self.alpha_beta(environment, search_depth - 1, -float("inf"), float("inf"), policy_scores=None)
            final_score = -score - rep_penalty

            if move in policy_scores:
                final_score += policy_scores[move] * 150.0

            environment.board = prev_board
            environment.turn = prev_turn
            environment.castling = prev_castling
            environment.en_passant = prev_en_passant
            environment.halfmove_clock = prev_halfmove
            environment.move_count = prev_move_count
            environment.position_history = prev_history

            move_scores.append((final_score, move))

        if not move_scores:
            return legal[0]

        move_scores.sort(key=lambda x: x[0], reverse=True)
        best_eval = move_scores[0][0]

        if environment.move_count <= 4:
            top_candidates = [m for s, m in move_scores if s >= best_eval - 20.0]
            return random.choice(top_candidates)

        return move_scores[0][1]
