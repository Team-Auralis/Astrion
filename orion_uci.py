#!/usr/bin/env python3
"""
ORION Universal Chess Interface (UCI) Engine
Enables ORION to connect to any standard chess GUI (Arena, Cutechess, Banksia)
and Lichess Bot API.
"""

import sys
import os
from typing import Optional

from orion.environment.chess import ChessEnvironment
from orion.model.actions import ChessActionSpace
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.search.search_agent import StockfishInspiredSearchAgent
from orion.utils.checkpoint import Checkpoint
from cli import find_best_checkpoint


def uci_to_move(uci_str: str):
    files = "abcdefgh"
    c1 = files.index(uci_str[0])
    r1 = 8 - int(uci_str[1])
    c2 = files.index(uci_str[2])
    r2 = 8 - int(uci_str[3])
    return (r1, c1, r2, c2)


def main():
    network = ORIONNetwork()
    encoder = ChessEncoder()

    best_ckpt = find_best_checkpoint()
    if best_ckpt and os.path.exists(best_ckpt):
        try:
            Checkpoint().load(network, best_ckpt)
        except Exception:
            pass

    depth = 3
    agent = StockfishInspiredSearchAgent(network, encoder, default_depth=depth)
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
            command = parts[0]

            if command == "uci":
                print("id name ORION Gaming Mind v0.6")
                print("id author Team-Auralis")
                print(f"option name Depth type spin default {depth} min 1 max 6")
                print(f"option name Checkpoint type string default {best_ckpt or 'none'}")
                print("uciok")
                sys.stdout.flush()

            elif command == "isready":
                print("readyok")
                sys.stdout.flush()

            elif command == "setoption":
                if len(parts) >= 5 and parts[1].lower() == "name" and parts[3].lower() == "value":
                    opt_name = parts[2].lower()
                    opt_val = parts[4]
                    if opt_name == "depth":
                        depth = int(opt_val)
                        agent.default_depth = depth
                    elif opt_name == "checkpoint":
                        if os.path.exists(opt_val):
                            Checkpoint().load(network, opt_val)

            elif command == "ucinewgame":
                env = ChessEnvironment()

            elif command == "position":
                env = ChessEnvironment()
                moves_index = -1
                if parts[1] == "startpos":
                    if "moves" in parts:
                        moves_index = parts.index("moves")
                elif parts[1] == "fen":
                    fen_parts = []
                    idx = 2
                    while idx < len(parts) and parts[idx] != "moves":
                        fen_parts.append(parts[idx])
                        idx += 1
                    fen_str = " ".join(fen_parts)
                    try:
                        env.load_fen(fen_str)
                    except Exception:
                        env = ChessEnvironment()
                    if idx < len(parts) and parts[idx] == "moves":
                        moves_index = idx

                if moves_index != -1:
                    for m_uci in parts[moves_index + 1:]:
                        if len(m_uci) >= 4:
                            move = uci_to_move(m_uci[:4])
                            env.make_move(move)

            elif command == "go":
                search_d = depth
                if "depth" in parts:
                    d_idx = parts.index("depth")
                    if d_idx + 1 < len(parts):
                        search_d = int(parts[d_idx + 1])

                best_move = agent.choose_move(env, depth=search_d)
                if best_move:
                    best_uci = ChessActionSpace.move_to_uci(best_move)
                    print(f"bestmove {best_uci}")
                else:
                    print("bestmove 0000")
                sys.stdout.flush()

            elif command == "quit":
                break

        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            sys.stderr.write(f"[ORION UCI ERROR] {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
