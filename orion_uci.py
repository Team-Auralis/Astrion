#!/usr/bin/env python3
"""
ORION Universal Chess Interface (UCI) Engine v2.2
Enables ORION to connect to any standard chess GUI (Arena, Cutechess, Banksia)
and Lichess Bot API with full 1900+ Elo search engine capabilities.
"""

import sys
import os
from typing import Optional, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("experiments/calibrated_eval"))

import chess
from tournament_engine import (
    OrionModelRunner,
    OrionSearchPlayer,
    OptimizedSearchEngine,
    CHECKPOINT_PATH,
    POLYGLOT_BOOK
)


def main():
    # Attempt to load master neural model runner
    player = None
    if os.path.exists(CHECKPOINT_PATH):
        try:
            # Suppress non-UCI prints from loading
            old_stdout = sys.stdout
            sys.stdout = sys.stderr
            runner = OrionModelRunner(CHECKPOINT_PATH)
            player = OrionSearchPlayer(runner)
            sys.stdout = old_stdout
        except Exception as e:
            sys.stderr.write(f"[WARN] Neural runner load failed: {e}\n")

    # Fallback to pure optimized search engine if weights not loaded
    if player is None:
        engine = OptimizedSearchEngine(book_path=POLYGLOT_BOOK)
    else:
        engine = player.engine

    board = chess.Board()

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
                print("id name Astrion v2.3")
                print("id author Team-Auralis")
                print("option name Book type string default tools/engines/vice-main/Vice11/src/performance.bin")
                print("uciok")
                sys.stdout.flush()

            elif command == "isready":
                print("readyok")
                sys.stdout.flush()

            elif command == "ucinewgame":
                engine.clear_heuristics()
                engine.tt.clear()
                board = chess.Board()

            elif command == "position":
                board = chess.Board()
                moves_index = -1
                if len(parts) > 1 and parts[1] == "startpos":
                    if "moves" in parts:
                        moves_index = parts.index("moves")
                elif len(parts) > 1 and parts[1] == "fen":
                    fen_parts = []
                    idx = 2
                    while idx < len(parts) and parts[idx] != "moves":
                        fen_parts.append(parts[idx])
                        idx += 1
                    fen_str = " ".join(fen_parts)
                    try:
                        board = chess.Board(fen_str)
                    except Exception:
                        board = chess.Board()
                    if idx < len(parts) and parts[idx] == "moves":
                        moves_index = idx

                if moves_index != -1:
                    for m_uci in parts[moves_index + 1:]:
                        try:
                            m = chess.Move.from_uci(m_uci)
                            if m in board.legal_moves:
                                board.push(m)
                        except Exception:
                            pass

            elif command == "go":
                # Compute calculation time budget
                time_limit = 0.5
                max_d = 15

                if "movetime" in parts:
                    idx = parts.index("movetime")
                    if idx + 1 < len(parts):
                        time_limit = max(0.01, float(parts[idx + 1]) / 1000.0)

                elif "depth" in parts:
                    idx = parts.index("depth")
                    if idx + 1 < len(parts):
                        max_d = int(parts[idx + 1])

                elif "wtime" in parts and "btime" in parts:
                    w_idx = parts.index("wtime")
                    b_idx = parts.index("btime")
                    w_time = float(parts[w_idx + 1]) / 1000.0
                    b_time = float(parts[b_idx + 1]) / 1000.0
                    my_time = w_time if board.turn == chess.WHITE else b_time
                    time_limit = max(0.02, min(my_time / 30.0, 3.0))

                if player is not None:
                    best_m = player.play(board, time_limit=time_limit)
                else:
                    best_m, _, _ = engine.search(board, time_limit=time_limit, max_depth=max_d)

                if best_m:
                    print(f"bestmove {best_m.uci()}")
                else:
                    legal = list(board.legal_moves)
                    if legal:
                        print(f"bestmove {legal[0].uci()}")
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
