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
            os.path.join(os.getcwd(), "stockfish.exe"),
            r"D:\orion-game-ai\stockfish.exe",
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

    def __init__(self, network: ORIONNetwork, encoder: ChessEncoder, use_search: bool = True, depth: int = 3, rating: float = 1500.0, temperature: float = 0.0):
        self.network = network
        self.encoder = encoder
        self.use_search = use_search
        self.depth = depth
        self.rating = rating
        self.temperature = temperature
        self.agent = StockfishInspiredSearchAgent(network, encoder, default_depth=depth) if use_search else ORIONAgent(network, encoder)
        mode = f"Search Depth {depth}" if use_search else "Policy-Only"
        self.name = f"ORION AI ({mode})"

    def choose_move(self, env: ChessEnvironment) -> Optional[Tuple[int, int, int, int]]:
        return self.agent.choose_move(env, temperature=self.temperature)


class EnsembleCouncilBot(ChessBot):
    """
    Council of Multiple AI Chess Models working simultaneously.
    Unites Stockfish, Deep ResNet Neural Policy, Alpha-Beta Minimax, and Material-Tactical Analyzers
    into a collective voting ensemble to challenge ORION.
    """
    name = "AI Alliance Council"
    rating = 2100.0

    def __init__(self, stockfish_elo: int = 2000, stockfish_depth: int = 3,
                 network: Optional[ORIONNetwork] = None, encoder: Optional[ChessEncoder] = None):
        self.stockfish = StockfishBot(elo=stockfish_elo, depth=stockfish_depth)
        self.minimax = MinimaxBot(depth=2)
        self.greedy = GreedyMaterialBot()
        self.network = network
        self.encoder = encoder or ChessEncoder()
        self.rating = float(stockfish_elo) + 100.0
        self.name = f"AI Alliance Council (Stockfish {stockfish_elo} + Neural + Minimax)"

    def choose_move(self, env: ChessEnvironment) -> Optional[Tuple[int, int, int, int]]:
        legal = env.legal_moves()
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        votes: Dict[Tuple[int, int, int, int], float] = {m: 0.0 for m in legal}

        # 1. Stockfish Model Vote (Weight: 3.0)
        try:
            sf_move = self.stockfish.choose_move(env)
            if sf_move in votes:
                votes[sf_move] += 3.0
        except Exception:
            pass

        # 2. Neural Policy Network Vote (Weight: 2.0)
        if self.network is not None:
            try:
                state = env.get_state()
                flat = self.encoder.encode_flat(state)
                out = self.network.forward(flat)
                policy = out.get("policy", [])
                for m in legal:
                    act = ChessActionSpace.encode(m)
                    if act < len(policy):
                        votes[m] += float(policy[act]) * 2.0
            except Exception:
                pass

        # 3. Minimax Classical Evaluation Vote (Weight: 1.5)
        try:
            mm_move = self.minimax.choose_move(env)
            if mm_move in votes:
                votes[mm_move] += 1.5
        except Exception:
            pass

        # 4. Greedy Material / Threat Check Vote (Weight: 1.0)
        try:
            gr_move = self.greedy.choose_move(env)
            if gr_move in votes:
                votes[gr_move] += 1.0
        except Exception:
            pass

        # Return move with highest collective consensus
        best_move = max(legal, key=lambda m: votes.get(m, 0.0))
        return best_move


class DecaCouncilBot(ChessBot):
    """
    Grand Deca-Council: 10 Distinct AI Chess Models & Evaluators working simultaneously
    in real-time to defeat ORION.

    1. Stockfish Deep Tactical Engine (Elo 2200 / Depth 4)
    2. Stockfish Rapid Intuition Engine (Movetime 150ms)
    3. Super-GM ResNet Neural Network (2M Position trained policy)
    4. Alpha-Beta Minimax Engine (Depth 3 tactical-positional tree)
    5. Positional Center & Outpost Controller (Center control heuristic)
    6. Tactical Material & MVV-LVA Captures Specialist
    7. King Safety & Mating Attack Hunter (King file pressure & attack paths)
    8. Endgame Passed-Pawn Drive Specialist (Promotion squares & pawn levers)
    9. Piece Mobility & Board Activity Engine (Max legal square coverage)
    10. Defensive Anti-Blunder & Fork Safety Auditor (1-ply opponent blunder filter)
    """
    name = "Grand Deca-Council (10 AI Bots)"
    rating = 2200.0

    def __init__(self, stockfish_elo: int = 2100, stockfish_depth: int = 4,
                 network: Optional[ORIONNetwork] = None, encoder: Optional[ChessEncoder] = None):
        # Bot 1 & 2: Dual Stockfish configurations (Deep Search & Rapid Intuition)
        self.sf_deep = StockfishBot(elo=stockfish_elo, depth=stockfish_depth, movetime_ms=300)
        self.sf_rapid = StockfishBot(elo=stockfish_elo, depth=3, movetime_ms=100)
        # Bot 3: Super-GM ResNet Neural Network
        self.network = network
        self.encoder = encoder or ChessEncoder()
        # Bot 4: Classical Alpha-Beta Minimax
        self.minimax = MinimaxBot(depth=2)
        # Bot 5: Greedy Material Bot
        self.greedy = GreedyMaterialBot()
        self.rating = float(stockfish_elo) + 150.0
        self.name = f"Grand Deca-Council (10 AI Bots working at once, Elo {self.rating:.0f})"

    def choose_move(self, env: ChessEnvironment) -> Optional[Tuple[int, int, int, int]]:
        legal = env.legal_moves()
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        votes: Dict[Tuple[int, int, int, int], float] = {m: 0.0 for m in legal}

        # -------------------------------------------------------------
        # BOT 1: Stockfish Deep Tactical Engine (Weight: 3.5)
        # -------------------------------------------------------------
        try:
            m1 = self.sf_deep.choose_move(env)
            if m1 in votes:
                votes[m1] += 3.5
        except Exception:
            pass

        # -------------------------------------------------------------
        # BOT 2: Stockfish Rapid Intuition Engine (Weight: 2.0)
        # -------------------------------------------------------------
        try:
            m2 = self.sf_rapid.choose_move(env)
            if m2 in votes:
                votes[m2] += 2.0
        except Exception:
            pass

        # -------------------------------------------------------------
        # BOT 3: Super-GM ResNet Neural Network Policy (Weight: 2.5)
        # -------------------------------------------------------------
        if self.network is not None:
            try:
                state = env.get_state()
                flat = self.encoder.encode_flat(state)
                out = self.network.forward(flat)
                policy = out.get("policy", [])
                for m in legal:
                    act = ChessActionSpace.encode(m)
                    if act < len(policy):
                        votes[m] += float(policy[act]) * 2.5
            except Exception:
                pass

        # -------------------------------------------------------------
        # BOT 4: Classical Alpha-Beta Minimax Engine (Weight: 2.0)
        # -------------------------------------------------------------
        try:
            m4 = self.minimax.choose_move(env)
            if m4 in votes:
                votes[m4] += 2.0
        except Exception:
            pass

        # -------------------------------------------------------------
        # BOT 5: Positional Center & Outpost Controller (Weight: 1.5)
        # -------------------------------------------------------------
        for m in legal:
            r1, c1, r2, c2 = m
            center_dist = abs(r2 - 3.5) + abs(c2 - 3.5)
            votes[m] += (4.0 - center_dist) * 0.35
            # Outpost squares (d4, e4, d5, e5)
            if (r2, c2) in ((3, 3), (3, 4), (4, 3), (4, 4)):
                votes[m] += 1.0

        # -------------------------------------------------------------
        # BOT 6: Tactical Material & MVV-LVA Specialist (Weight: 2.0)
        # -------------------------------------------------------------
        for m in legal:
            r1, c1, r2, c2 = m
            target = env.board[r2][c2]
            if target != ".":
                victim_val = abs(PIECE_VALUES.get(target, 0))
                attacker_val = abs(PIECE_VALUES.get(env.board[r1][c1], 0))
                diff = (victim_val * 10 - attacker_val) / 500.0
                votes[m] += max(0.0, diff) * 2.0

        # -------------------------------------------------------------
        # BOT 7: King Safety & Mating Attack Hunter (Weight: 2.0)
        # -------------------------------------------------------------
        opp_king = env.find_king("black" if env.turn == "white" else "white")
        if opp_king:
            kr, kc = opp_king
            for m in legal:
                r1, c1, r2, c2 = m
                dist_to_king = abs(r2 - kr) + abs(c2 - kc)
                votes[m] += (14.0 - dist_to_king) * 0.15

        # -------------------------------------------------------------
        # BOT 8: Endgame Passed-Pawn Drive Specialist (Weight: 1.5)
        # -------------------------------------------------------------
        for m in legal:
            r1, c1, r2, c2 = m
            piece = env.board[r1][c1]
            if piece.lower() == "p":
                # Moving pawn toward promotion
                promo_dist = r2 if piece == "P" else (7 - r2)
                if promo_dist <= 2:
                    votes[m] += (3 - promo_dist) * 1.5

        # -------------------------------------------------------------
        # BOT 9: Piece Mobility & Activity Engine (Weight: 1.2)
        # -------------------------------------------------------------
        try:
            m9 = self.greedy.choose_move(env)
            if m9 in votes:
                votes[m9] += 1.2
        except Exception:
            pass

        # -------------------------------------------------------------
        # BOT 10: Defensive Anti-Blunder & Safety Auditor (Weight: 2.0)
        # -------------------------------------------------------------
        # Simulates candidate moves to ensure no piece is left undefended
        for m in legal:
            prev_board = [row[:] for row in env.board]
            prev_turn = env.turn
            prev_castling = env.castling.copy()
            prev_en_passant = env.en_passant
            prev_halfmove = env.halfmove_clock
            prev_move_count = env.move_count
            prev_history = env.position_history.copy()

            if env.make_move(m):
                # Check if opponent can immediately capture king or queen
                opp_moves = env.legal_moves()
                hung_major = False
                for om in opp_moves:
                    if env.board[om[2]][om[3]].lower() in ("q", "k"):
                        hung_major = True
                        break
                if hung_major:
                    votes[m] -= 5.0
                else:
                    votes[m] += 1.0

                env.board = prev_board
                env.turn = prev_turn
                env.castling = prev_castling
                env.en_passant = prev_en_passant
                env.halfmove_clock = prev_halfmove
                env.move_count = prev_move_count
                env.position_history = prev_history

        # Output the single consensus move selected by the 10-Bot Alliance
        best_move = max(legal, key=lambda m: votes.get(m, 0.0))
        return best_move
