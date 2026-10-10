#!/usr/bin/env python3
"""
ORION Complete Engine UCI Wrapper (Test B)
Evaluates the integrated engine:
- ResNet dual-head neural network (policy + value)
- Iterative deepening alpha-beta minimax search
- Zobrist 64-bit transposition hashing
- Reverse Futility Pruning (RFP) & Late Move Pruning (LMP)
- Static Exchange Evaluation (SEE) in quiescence search
- Tapered 24-point game-phase positional evaluation
Compliant with standard UCI protocol for Cute Chess / Arena / GUI.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
from typing import Optional

from orion.environment.chess import ChessEnvironment
from orion.model.actions import ChessActionSpace
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.search.search_agent import StockfishInspiredSearchAgent
from orion.utils.checkpoint import Checkpoint

DEFAULT_CHECKPOINT = r"D:\orion-game-ai\checkpoints\distilled\orion_distill_5ep.json"


def uci_to_env_move(uci_str: str):
    files = "abcdefgh"
    c1 = files.index(uci_str[0])
    r1 = 8 - int(uci_str[1])
    c2 = files.index(uci_str[2])
    r2 = 8 - int(uci_str[3])
    return (r1, c1, r2, c2)


def env_move_to_uci(env: ChessEnvironment, move):
    r1, c1, r2, c2 = move
    files = "abcdefgh"
    base = files[c1] + str(8 - r1) + files[c2] + str(8 - r2)
    piece = env.board[r1][c1]
    if piece.lower() == "p" and (r2 == 0 or r2 == 7):
        return base + "q"
    return base


def main():
    network = ORIONNetwork()
    encoder = ChessEncoder()

    ckpt = DEFAULT_CHECKPOINT
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        ckpt = sys.argv[1]

    if os.path.exists(ckpt):
        try:
            Checkpoint().load(network, ckpt)
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to load checkpoint {ckpt}: {e}\n")

    default_depth = 3
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        default_depth = int(sys.argv[2])

    agent = StockfishInspiredSearchAgent(network, encoder, default_depth=default_depth)
    env = ChessEnvironment()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            cmd = parts[0]

            if cmd == "uci":
                print("id name ORION Complete Engine v0.6")
                print("id author Team-Auralis")
                print(f"option name Depth type spin default {default_depth} min 1 max 8")
                print(f"option name Checkpoint type string default {ckpt}")
                print("uciok")
                sys.stdout.flush()

            elif cmd == "isready":
                print("readyok")
                sys.stdout.flush()

            elif cmd == "ucinewgame":
                env = ChessEnvironment()
                agent._tt.clear()
                agent._history.clear()

            elif cmd == "setoption":
                if len(parts) >= 5 and parts[1].lower() == "name" and parts[3].lower() == "value":
                    opt_name = parts[2].lower()
                    opt_val = parts[4]
                    if opt_name == "depth":
                        default_depth = int(opt_val)
                        agent.default_depth = default_depth
                    elif opt_name == "checkpoint":
                        if os.path.exists(opt_val):
                            Checkpoint().load(network, opt_val)

            elif cmd == "position":
                env = ChessEnvironment()
                moves_idx = -1
                if parts[1] == "startpos":
                    if "moves" in parts:
                        moves_idx = parts.index("moves")
                elif parts[1] == "fen":
                    fen_tokens = []
                    idx = 2
                    while idx < len(parts) and parts[idx] != "moves":
                        fen_tokens.append(parts[idx])
                        idx += 1
                    try:
                        env.load_fen(" ".join(fen_tokens))
                    except Exception:
                        env = ChessEnvironment()
                    if idx < len(parts) and parts[idx] == "moves":
                        moves_idx = idx

                if moves_idx != -1:
                    for m in parts[moves_idx + 1:]:
                        if len(m) >= 4:
                            env.make_move(uci_to_env_move(m[:4]))

            elif cmd == "go":
                # Determine search depth based on depth or time controls
                search_d = default_depth
                if "depth" in parts:
                    d_idx = parts.index("depth")
                    if d_idx + 1 < len(parts):
                        search_d = int(parts[d_idx + 1])
                elif "movetime" in parts:
                    mt_idx = parts.index("movetime")
                    if mt_idx + 1 < len(parts):
                        movetime_ms = int(parts[mt_idx + 1])
                        # Adaptive depth: depth 4 for >= 1000ms, depth 3 for >= 300ms, else depth 2
                        search_d = 4 if movetime_ms >= 1000 else (3 if movetime_ms >= 300 else 2)
                elif "wtime" in parts and "btime" in parts:
                    # Clock game time allocation
                    my_time_str = parts[parts.index("wtime") + 1] if env.turn == "white" else parts[parts.index("btime") + 1]
                    my_time = int(my_time_str)
                    if my_time < 1500:
                        search_d = 1
                    elif my_time < 5000:
                        search_d = 2
                    elif my_time < 20000:
                        search_d = 3
                    else:
                        search_d = min(default_depth, 3)

                legal = env.legal_moves()
                if not legal:
                    print("bestmove 0000")
                    sys.stdout.flush()
                    continue

                best_move = agent.choose_move(env, depth=search_d)
                if best_move is None or best_move not in legal:
                    best_move = legal[0]

                uci_mv = env_move_to_uci(env, best_move)
                print(f"bestmove {uci_mv}")
                sys.stdout.flush()

            elif cmd == "stop":
                pass

            elif cmd == "quit":
                break

        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            sys.stderr.write(f"[ORION COMPLETE UCI ERROR] {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
