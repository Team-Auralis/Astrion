import os
import random
import shutil
import subprocess
import time
from typing import Optional, Tuple, List, Dict
from orion.environment.chess import ChessEnvironment
from orion.model.actions import ChessActionSpace
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.model.agent import ORIONAgent
from orion.search.search_agent import StockfishInspiredSearchAgent, PIECE_VALUES, PAWN_TABLE, KNIGHT_TABLE


class ChessBot:
    """Base class for all chess bots."""
    name: str = "BaseBot"
    rating: float = 1200.0

    def choose_move(self, env: ChessEnvironment) -> Optional[Tuple[int, int, int, int]]:
        raise NotImplementedError


class RandomBot(ChessBot):
    """Plays random legal moves."""
    name = "RandomBot (Elo ~500)"
    rating = 500.0

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def choose_move(self, env: ChessEnvironment) -> Optional[Tuple[int, int, int, int]]:
        moves = env.legal_moves()
        return self.rng.choice(moves) if moves else None


class GreedyMaterialBot(ChessBot):
    """Greedily prioritizes highest-value captures and checks; otherwise plays safe moves."""
    name = "GreedyMaterialBot (Elo ~1000)"
    rating = 1000.0

    def choose_move(self, env: ChessEnvironment) -> Optional[Tuple[int, int, int, int]]:
        moves = env.legal_moves()
        if not moves:
            return None

        best_score = -float("inf")
        best_moves = []

        for move in moves:
            r1, c1, r2, c2 = move
            target_piece = env.board[r2][c2]
            attacker = env.board[r1][c1]

            score = 0.0
            if target_piece != ".":
                victim_val = abs(PIECE_VALUES.get(target_piece, 0))
                attacker_val = abs(PIECE_VALUES.get(attacker, 0))
                score += victim_val * 10 - attacker_val
            elif env.en_passant == (r2, c2) and attacker.lower() == "p":
                score += 1000

            # Favor moving toward center
            score += (3.5 - abs(3.5 - r2)) + (3.5 - abs(3.5 - c2))

            # Anti-repetition check
            prev_board = [row[:] for row in env.board]
            prev_turn = env.turn
            prev_castling = env.castling.copy()
            prev_en_passant = env.en_passant
            prev_halfmove = env.halfmove_clock
            prev_move_count = env.move_count
            prev_history = env.position_history.copy()

            if env.make_move(move):
                if prev_history.get(env._position_key(), 0) >= 1:
                    score -= 400.0
                env.board = prev_board
                env.turn = prev_turn
                env.castling = prev_castling
                env.en_passant = prev_en_passant
                env.halfmove_clock = prev_halfmove
                env.move_count = prev_move_count
                env.position_history = prev_history

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score >= best_score - 1.0:
                best_moves.append(move)

        return random.choice(best_moves) if best_moves else moves[0]


class MinimaxBot(ChessBot):
    """Classical Alpha-Beta Minimax search bot with material and positional evaluation."""
    name = "MinimaxBot (Elo ~1350)"
    rating = 1350.0

    def __init__(self, depth: int = 2):
        self.depth = depth

    def _eval(self, env: ChessEnvironment) -> float:
        score = 0.0
        for r in range(8):
            for c in range(8):
                p = env.board[r][c]
                if p == ".":
                    continue
                score += PIECE_VALUES.get(p, 0)
                if p == "P":
                    score += PAWN_TABLE[r][c]
                elif p == "p":
                    score -= PAWN_TABLE[7 - r][c]
                elif p == "N":
                    score += KNIGHT_TABLE[r][c]
                elif p == "n":
                    score -= KNIGHT_TABLE[7 - r][c]
        return score if env.turn == "white" else -score

    def _minimax(self, env: ChessEnvironment, depth: int, alpha: float, beta: float) -> Tuple[float, Optional[Tuple]]:
        if depth == 0:
            return self._eval(env), None

        moves = env.legal_moves()
        if not moves:
            status = env.game_status()
            if status.startswith("checkmate:"):
                return -30000.0, None
            return 0.0, None

        best_score = -float("inf")
        best_move = moves[0]

        for move in moves:
            prev_board = [row[:] for row in env.board]
            prev_turn = env.turn
            prev_castling = env.castling.copy()
            prev_en_passant = env.en_passant
            prev_halfmove = env.halfmove_clock
            prev_move_count = env.move_count
            prev_history = env.position_history.copy()

            if not env.make_move(move):
                continue

            score, _ = self._minimax(env, depth - 1, -beta, -alpha)
            score = -score

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

            alpha = max(alpha, best_score)
            if alpha >= beta:
                break

        return best_score, best_move

    def choose_move(self, env: ChessEnvironment) -> Optional[Tuple[int, int, int, int]]:
        moves = env.legal_moves()
        if not moves:
            return None
        _, best_move = self._minimax(env, self.depth, -float("inf"), float("inf"))
        return best_move if best_move is not None else moves[0]


class StockfishBot(ChessBot):
    """
    Subprocess-based UCI wrapper connecting to official Stockfish binary.
    Supports real Elo ratings (1320 - 3190+) and depth control.
    """
    name = "Stockfish"
    rating = 1500.0

    def __init__(self, stockfish_path: Optional[str] = None, elo: int = 1500, depth: int = 3, movetime_ms: int = 200):
        self.stockfish_path = stockfish_path or self._find_stockfish()
        self.elo = elo
        self.depth = depth
        self.movetime_ms = movetime_ms
        self.rating = float(elo)
        self.name = f"Stockfish (Elo {elo})"

    @staticmethod
    def _find_stockfish() -> Optional[str]:
        """Auto-detect Stockfish binary from PATH or standard directories."""
        candidates = [
            shutil.which("stockfish"),
            shutil.which("stockfish.exe"),
            r"C:\Program Files\Stockfish\stockfish.exe",
            r"C:\stockfish\stockfish.exe",
            r"D:\stockfish\stockfish.exe",
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c
        return None

    def is_available(self) -> bool:
        return bool(self.stockfish_path and os.path.exists(self.stockfish_path))

    def _env_to_fen(self, env: ChessEnvironment) -> str:
        """Convert ChessEnvironment to FEN string for UCI."""
        rows = []
        for r in range(8):
            empty = 0
            row_str = ""
            for c in range(8):
                piece = env.board[r][c]
                if piece == ".":
                    empty += 1
                else:
                    if empty > 0:
                        row_str += str(empty)
                        empty = 0
                    row_str += piece
            if empty > 0:
                row_str += str(empty)
            rows.append(row_str)

        board_fen = "/".join(rows)
        turn_fen = "w" if env.turn == "white" else "b"

        castling_chars = ""
        if env.castling.get("K"): castling_chars += "K"
        if env.castling.get("Q"): castling_chars += "Q"
        if env.castling.get("k"): castling_chars += "k"
        if env.castling.get("q"): castling_chars += "q"
        castling_fen = castling_chars if castling_chars else "-"

        if env.en_passant:
            files = "abcdefgh"
            er, ec = env.en_passant
            ep_fen = f"{files[ec]}{8 - er}"
        else:
            ep_fen = "-"

        return f"{board_fen} {turn_fen} {castling_fen} {ep_fen} {env.halfmove_clock} {max(1, env.move_count // 2 + 1)}"

    def _uci_to_move(self, uci: str) -> Tuple[int, int, int, int]:
        files = "abcdefgh"
        c1 = files.index(uci[0])
        r1 = 8 - int(uci[1])
        c2 = files.index(uci[2])
        r2 = 8 - int(uci[3])
        return (r1, c1, r2, c2)

    def choose_move(self, env: ChessEnvironment) -> Optional[Tuple[int, int, int, int]]:
        if not self.is_available():
            raise RuntimeError(
                "Stockfish binary not found. Please install Stockfish and add to PATH or pass --stockfish-path."
            )

        fen = self._env_to_fen(env)

        proc = subprocess.Popen(
            [self.stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        commands = [
            "uci",
            "setoption name UCI_LimitStrength value true",
            f"setoption name UCI_Elo value {self.elo}",
            "isready",
            "ucinewgame",
            f"position fen {fen}",
            f"go depth {self.depth} movetime {self.movetime_ms}",
        ]

        proc.stdin.write("\n".join(commands) + "\n")
        proc.stdin.flush()

        best_uci = None
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    best_uci = parts[1]
                break

        proc.terminate()

        if best_uci and len(best_uci) >= 4:
            move = self._uci_to_move(best_uci[:4])
            legal = env.legal_moves()
            if move in legal:
                return move

        # Fallback to random legal move if parse mismatch
        moves = env.legal_moves()
        return moves[0] if moves else None


class OrionBot(ChessBot):
    """ORION Chess Agent playing with either pure Neural Policy or Stockfish-Inspired Search."""
    name = "ORION AI"

    def __init__(self, network: ORIONNetwork, encoder: ChessEncoder, use_search: bool = True, depth: int = 2, rating: float = 1200.0):
        self.network = network
        self.encoder = encoder
        self.use_search = use_search
        self.depth = depth
        self.rating = rating
        self.agent = StockfishInspiredSearchAgent(network, encoder, default_depth=depth) if use_search else ORIONAgent(network, encoder)
        mode = f"Search Depth {depth}" if use_search else "Policy-Only"
        self.name = f"ORION AI ({mode})"

    def choose_move(self, env: ChessEnvironment) -> Optional[Tuple[int, int, int, int]]:
        return self.agent.choose_move(env)
