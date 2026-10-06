import math
import random
from typing import Optional, Tuple, List, Dict
from orion.environment.chess import ChessEnvironment
from orion.model.actions import ChessActionSpace
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork

PIECE_VALUES = {
    "P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 20000,
    "p": -100, "n": -320, "b": -330, "r": -500, "q": -900, "k": -20000,
    ".": 0
}

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

KING_MID_TABLE = [
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [ 20, 20,  0,  0,  0,  0, 20, 20],
    [ 20, 30, 10,  0,  0, 10, 30, 20]
]

PST = {
    "P": PAWN_TABLE, "p": PAWN_TABLE,
    "N": KNIGHT_TABLE, "n": KNIGHT_TABLE,
    "B": BISHOP_TABLE, "b": BISHOP_TABLE,
    "R": ROOK_TABLE, "r": ROOK_TABLE,
    "Q": QUEEN_TABLE, "q": QUEEN_TABLE,
    "K": KING_MID_TABLE, "k": KING_MID_TABLE,
}


class StockfishInspiredSearchAgent:

    def __init__(self, network: ORIONNetwork, encoder: ChessEncoder,
                 default_depth: int = 3, nn_weight: float = 0.0):
        self.network = network
        self.encoder = encoder
        self.default_depth = default_depth
        self.nn_weight = nn_weight
        self.nodes_evaluated = 0
        self._tt = {}
        self._killer = [[None, None] for _ in range(64)]
        self._history = {}
        self._ply = 0

    def evaluate_board(self, env: ChessEnvironment) -> float:
        self.nodes_evaluated += 1
        board = env.board
        score = 0
        for r in range(8):
            for c in range(8):
                p = board[r][c]
                if p == ".":
                    continue
                score += PIECE_VALUES.get(p, 0)
                table = PST.get(p)
                if table:
                    if p.isupper():
                        score += table[r][c]
                    else:
                        score += table[7 - r][c]
        return float(score)

    def _is_capture(self, env: ChessEnvironment, move: Tuple[int, int, int, int]) -> bool:
        r1, c1, r2, c2 = move
        target = env.board[r2][c2]
        if target != ".":
            return True
        piece = env.board[r1][c1]
        if piece.lower() == "p" and env.en_passant == (r2, c2):
            return True
        return False

    def _mvv_lva_score(self, env: ChessEnvironment, move: Tuple[int, int, int, int]) -> float:
        r1, c1, r2, c2 = move
        attacker = env.board[r1][c1]
        victim = env.board[r2][c2]
        victim_val = abs(PIECE_VALUES.get(victim, 0))
        attacker_val = abs(PIECE_VALUES.get(attacker, 0))
        return victim_val * 10 - attacker_val

    def _score_move(self, env: ChessEnvironment, move: Tuple[int, int, int, int],
                    tt_move: Optional[Tuple] = None) -> float:
        if move == tt_move:
            return 20000000.0

        score = 0.0

        if self._is_capture(env, move):
            score += 10000.0 + self._mvv_lva_score(env, move)

        r1, c1, r2, c2 = move
        ply = self._ply
        if ply < 64:
            killers = self._killer[ply]
            if move == killers[0]:
                score += 9000.0
            elif move == killers[1]:
                score += 8000.0

        hk = (r1, c1, r2, c2)
        score += self._history.get(hk, 0.0)

        piece = env.board[r1][c1]
        if piece and piece.isupper() and piece != "K":
            if piece == "P":
                score += PAWN_TABLE[r2][c2] * 0.5
            elif piece == "N":
                score += KNIGHT_TABLE[r2][c2] * 0.3

        if piece and piece == "P":
            r2_norm = r2 if env.turn == "white" else 7 - r2
            if r2_norm >= 5:
                score += 100.0

        return score

    def _is_in_check(self, env: ChessEnvironment) -> bool:
        status = env.game_status()
        return status == "check"

    def quiescence_search(self, env: ChessEnvironment, alpha: float, beta: float,
                          depth_limit: int = 6) -> float:
        stand_pat = self.evaluate_board(env)
        turn_mult = 1.0 if env.turn == "white" else -1.0
        eval_std = stand_pat * turn_mult

        if eval_std >= beta:
            return beta
        if eval_std > alpha:
            alpha = eval_std

        if depth_limit <= 0:
            return alpha

        in_check = self._is_in_check(env)

        legal_moves = env.legal_moves()
        if not legal_moves:
            if in_check:
                return -49000.0 + self._ply
            return 0.0

        if in_check:
            captures = legal_moves
        else:
            captures = [m for m in legal_moves if self._is_capture(env, m)]

        if not captures and not in_check:
            return alpha

        captures.sort(key=lambda m: self._mvv_lva_score(env, m), reverse=True)

        for move in captures:
            prev = self._save_env(env)

            if not env.make_move(move):
                self._restore_env(env, prev)
                continue

            self._ply += 1
            score = -self.quiescence_search(env, -beta, -alpha, depth_limit - 1)
            self._ply -= 1

            self._restore_env(env, prev)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def alpha_beta(self, env: ChessEnvironment, depth: int, alpha: float, beta: float,
                   tt_move: Optional[Tuple] = None) -> Tuple[float, Optional[Tuple]]:
        in_check = self._is_in_check(env)

        if in_check:
            depth += 1

        if depth <= 0:
            return self.quiescence_search(env, alpha, beta), None

        status = env.game_status()
        if status.startswith("checkmate:"):
            winner = status.split(":")[1]
            if winner != env.turn:
                return -49000.0 + self._ply, None
            return 49000.0 - self._ply, None
        if status.startswith("draw") or status == "stalemate":
            return 0, None

        pos_key = env._position_key()
        tt_entry = self._tt.get(pos_key)
        stored_tt_move = None

        if tt_entry and tt_entry[0] >= depth:
            stored_score, stored_flag, stored_move = tt_entry[1], tt_entry[2], tt_entry[3]
            stored_tt_move = stored_move
            if stored_flag == 0:
                return stored_score, stored_move
            elif stored_flag == 1:
                alpha = max(alpha, stored_score)
            elif stored_flag == 2:
                beta = min(beta, stored_score)
            if alpha >= beta:
                return stored_score, stored_move

        local_tt_move = stored_tt_move or tt_move

        legal_moves = env.legal_moves()
        if not legal_moves:
            if in_check:
                return -49000.0 + self._ply, None
            return 0, None

        ordered = sorted(
            legal_moves,
            key=lambda m: self._score_move(env, m, local_tt_move),
            reverse=True
        )

        best_move = ordered[0]
        best_score = -float("inf")
        flag = 2

        for i, move in enumerate(ordered):
            prev = self._save_env(env)

            if not env.make_move(move):
                self._restore_env(env, prev)
                continue

            self._ply += 1

            if i == 0:
                score, _ = self.alpha_beta(env, depth - 1, -beta, -alpha)
            else:
                score, _ = self.alpha_beta(env, depth - 1, -alpha - 1, -alpha)
                if alpha < -score < beta:
                    score, _ = self.alpha_beta(env, depth - 1, -beta, -alpha)

            self._ply -= 1
            score = -score

            self._restore_env(env, prev)

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score
                flag = 0

            if alpha >= beta:
                flag = 1
                if not self._is_capture(env, move):
                    ply = self._ply
                    if ply < 64:
                        if move != self._killer[ply][0]:
                            self._killer[ply][1] = self._killer[ply][0]
                            self._killer[ply][0] = move
                    hk = (move[0], move[1], move[2], move[3])
                    self._history[hk] = self._history.get(hk, 0.0) + depth * depth
                break

        self._tt[pos_key] = (depth, best_score, flag, best_move)
        return best_score, best_move

    def _save_env(self, env):
        return (
            [row[:] for row in env.board],
            env.turn,
            env.castling.copy(),
            env.en_passant,
            env.halfmove_clock,
            env.move_count,
            env.position_history.copy(),
        )

    def _restore_env(self, env, state):
        env.board, env.turn, env.castling = state[0], state[1], state[2]
        env.en_passant, env.halfmove_clock = state[3], state[4]
        env.move_count, env.position_history = state[5], state[6]

    def choose_move(self, environment: ChessEnvironment, depth: Optional[int] = None,
                    temperature: float = 0.0) -> Optional[Tuple[int, int, int, int]]:
        search_depth = depth if depth is not None else self.default_depth
        self.nodes_evaluated = 0
        self._tt.clear()
        self._killer = [[None, None] for _ in range(64)]
        self._history.clear()
        self._ply = 0

        legal = environment.legal_moves()
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        best_move = legal[0]
        tt_move = None

        for current_depth in range(1, search_depth + 1):
            saved = self._save_env(environment)

            score, move = self.alpha_beta(
                environment, current_depth,
                -float("inf"), float("inf"),
                tt_move=tt_move
            )

            self._restore_env(environment, saved)

            if move is not None:
                best_move = move
                tt_move = move

        return best_move
