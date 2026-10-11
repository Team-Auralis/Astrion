#!/usr/bin/env python3
"""
Official Calibrated Round-Robin & Gauntlet Tournament Runner.
Controls:
- Paired Openings from standard EPD book
- Accurate time control / move limits
- Strict PGN recording
- Ordo execution and statistical report generation
"""

import os
import sys
import time
import subprocess
import argparse
from typing import List, Tuple, Dict

import chess
import chess.pgn

from tournament_engine import (
    OrionModelRunner,
    OrionSearchPlayer,
    OrionNeuralOnlyPlayer,
    ExternalEnginePlayer,
    CHECKPOINT_PATH,
    ORDO_EXE,
    EPD_BOOK
)


def load_openings(epd_path: str, count: int) -> List[str]:
    fens = []
    if os.path.exists(epd_path):
        with open(epd_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    # Strip any EPD opcodes if present
                    fen = line.split(";")[0].strip()
                    try:
                        b = chess.Board(fen)
                        fens.append(b.fen())
                    except Exception:
                        pass
                if len(fens) >= count:
                    break
    if not fens:
        # Fallback to standard symmetrical open/semi-open opening FENs
        fens = [
            chess.STARTING_FEN,
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", # 1. e4 e5
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", # 1. e4 c5
            "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",   # 1. e4 e6
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2", # 1. d4 d5
            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2", # 1. d4 d5 2. c4
            "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",  # 1. d4 Nf6
            "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1",    # 1. c4
        ]
    return fens[:count]


def play_game(white_player, black_player, start_fen: str, round_id: int, time_limit: float = 0.2) -> chess.pgn.Game:
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    game.headers["Event"] = "ORION Calibrated Benchmark Arena"
    game.headers["Site"] = "Local Rigorous Testbed"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["Round"] = str(round_id)
    game.headers["White"] = white_player.name
    game.headers["Black"] = black_player.name
    game.setup(board)

    node = game
    max_plies = 150
    ply_count = 0

    while not board.is_game_over() and ply_count < max_plies:
        current_player = white_player if board.turn == chess.WHITE else black_player
        try:
            mv = current_player.play(board, time_limit=time_limit)
        except Exception as e:
            print(f"[ERROR] Engine {current_player.name} failed: {e}")
            break

        if mv is None or mv not in board.legal_moves:
            break

        node = node.add_variation(mv)
        board.push(mv)
        ply_count += 1

    # Adjudication
    if board.is_checkmate():
        res = "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves():
        res = "1/2-1/2"
    else:
        # Move limit adjudication based on piece values
        from tournament_engine import evaluate_material
        mat = evaluate_material(board)
        # pov of White
        if board.turn == chess.BLACK:
            mat = -mat
        if mat >= 300:
            res = "1-0"
        elif mat <= -300:
            res = "0-1"
        else:
            res = "1/2-1/2"

    game.headers["Result"] = res
    return game


def run_tournament(num_pairings: int, time_limit: float = 0.2, pgn_path: str = "experiments/calibrated_eval/matches.pgn"):
    os.makedirs(os.path.dirname(pgn_path), exist_ok=True)
    openings = load_openings(EPD_BOOK, num_pairings)

    runner = OrionModelRunner(CHECKPOINT_PATH)
    orion_search = OrionSearchPlayer(runner)
    orion_neural = OrionNeuralOnlyPlayer(runner)

    opponents = [
        ExternalEnginePlayer("Sunfish-2026", ["python", "tools/engines/sunfish-master/sunfish.py", "-u"]),
        ExternalEnginePlayer("BBC-1.2", [os.path.abspath("tools/engines/bbc-master/bin/windows/bbc_1.2_64bit_windows.exe")]),
        ExternalEnginePlayer("Stockfish-1350", [os.path.abspath("stockfish.exe")], options={"UCI_LimitStrength": True, "UCI_Elo": 1350}),
        ExternalEnginePlayer("Stockfish-1600", [os.path.abspath("stockfish.exe")], options={"UCI_LimitStrength": True, "UCI_Elo": 1600}),
    ]

    for opp in opponents:
        opp.start()

    all_games = []
    round_counter = 1

    test_subjects = [orion_search, orion_neural]

    print(f"=== Starting Calibrated Tournament ({len(openings)} paired openings x 2 colors) ===")
    print(f"Subjects: {[s.name for s in test_subjects]}")
    print(f"Anchors : {[o.name for o in opponents]}")

    for subj in test_subjects:
        for opp in opponents:
            print(f"\n--- Pairing: {subj.name} vs {opp.name} ({len(openings)*2} games) ---")
            for fen_idx, fen in enumerate(openings, 1):
                # Game 1: Subject as White
                g1 = play_game(subj, opp, fen, round_counter, time_limit=time_limit)
                round_counter += 1
                all_games.append(g1)
                print(f"Round {round_counter-1:03d}: {subj.name} (White) vs {opp.name} -> {g1.headers['Result']}")

                # Game 2: Subject as Black (Swapped Colors, exact same opening)
                g2 = play_game(opp, subj, fen, round_counter, time_limit=time_limit)
                round_counter += 1
                all_games.append(g2)
                print(f"Round {round_counter-1:03d}: {opp.name} (White) vs {subj.name} -> {g2.headers['Result']}")

    for opp in opponents:
        opp.close()

    # Save PGN
    with open(pgn_path, "w") as f:
        for g in all_games:
            exporter = chess.pgn.FileExporter(f)
            g.accept(exporter)

    print(f"\nSaved {len(all_games)} tournament games to {pgn_path}")
    return pgn_path


def run_ordo_analysis(pgn_path: str, anchor_name: str = "Sunfish-2026", anchor_rating: float = 1750.0) -> str:
    out_txt = os.path.splitext(pgn_path)[0] + "_ordo.txt"
    cmd = [
        ORDO_EXE,
        "-p", os.path.abspath(pgn_path),
        "-o", os.path.abspath(out_txt),
        "-A", anchor_name,
        "-a", str(anchor_rating),
        "-s", "1000",
        "-F", "95"
    ]
    print(f"\nRunning Ordo rating calculation:")
    print("Command:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Ordo Exit Code:", res.returncode)
    print(res.stdout)
    if os.path.exists(out_txt):
        with open(out_txt, "r") as f:
            content = f.read()
        print("\n=== Ordo Rating Table ===")
        print(content)
        return content
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=1, help="Number of paired openings")
    parser.add_argument("--time", type=float, default=0.1, help="Time limit per move")
    parser.add_argument("--pilot", action="store_true", help="Run pilot check only")
    args = parser.parse_args()

    pgn_file = "experiments/calibrated_eval/pilot.pgn" if args.pilot else "experiments/calibrated_eval/calibrated_tournament.pgn"
    pgn_out = run_tournament(num_pairings=args.pairs, time_limit=args.time, pgn_path=pgn_file)
    run_ordo_analysis(pgn_out)
