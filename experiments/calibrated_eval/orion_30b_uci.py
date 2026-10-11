#!/usr/bin/env python3
"""
ORION-30B Universal Chess Interface (UCI) Executable
Connects the 30-Block Master Checkpoint (21.27M params) to any standard
GUI or tournament arbiter like cutechess-cli.
"""

import sys
import os
import argparse
from typing import Optional, List

# Ensure project root is in path
sys.path.insert(0, os.path.abspath("."))

import chess
import torch
import numpy as np

from train_fast import OrionResNet
from train_gm_dataset import load_json_to_model
from train_ccrl_full_corpus import fast_encode_board, fast_encode_action
from experiments.calibrated_eval.tournament_engine import (
    OrionModelRunner,
    OrionSearchPlayer,
    OrionNeuralOnlyPlayer,
    CHECKPOINT_PATH
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-only", action="store_true", help="Run pure neural policy intuition without search")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    args, unknown = parser.parse_known_args()

    engine_name = "ORION-Neural-Only" if args.neural_only else "ORION-Search-Full"

    # Suppress non-UCI prints from loading routines to stderr
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    runner = OrionModelRunner(args.checkpoint)
    if args.neural_only:
        player = OrionNeuralOnlyPlayer(runner)
    else:
        player = OrionSearchPlayer(runner)
    sys.stdout = old_stdout

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
            cmd = parts[0].lower()

            if cmd == "uci":
                print(f"id name {engine_name}")
                print("id author Team-Auralis")
                print("uciok")
                sys.stdout.flush()

            elif cmd == "isready":
                print("readyok")
                sys.stdout.flush()

            elif cmd == "ucinewgame":
                board = chess.Board()

            elif cmd == "position":
                if parts[1] == "startpos":
                    board = chess.Board()
                    if "moves" in parts:
                        idx = parts.index("moves")
                        for m_str in parts[idx + 1:]:
                            board.push(chess.Move.from_uci(m_str))
                elif parts[1] == "fen":
                    moves_idx = parts.index("moves") if "moves" in parts else len(parts)
                    fen_str = " ".join(parts[2:moves_idx])
                    board = chess.Board(fen_str)
                    if "moves" in parts:
                        for m_str in parts[moves_idx + 1:]:
                            board.push(chess.Move.from_uci(m_str))

            elif cmd == "go":
                m = player.play(board)
                if m:
                    print(f"bestmove {m.uci()}")
                else:
                    print("bestmove 0000")
                sys.stdout.flush()

            elif cmd == "quit":
                break

        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            sys.stderr.write(f"[{engine_name} ERROR] {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
