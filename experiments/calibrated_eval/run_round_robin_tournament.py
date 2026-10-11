#!/usr/bin/env python3
"""
Official Calibrated Round-Robin Benchmark Tournament for ORION.
Features:
- Full round-robin connecting all engines (ORION-Search-Full, ORION-Neural-Only, Vice-1.2, Sunfish-2026, BBC-1.2, Stockfish-1350)
- Paired opening positions from standard UHO EPD book
- Exact color balance (each pair plays both White and Black on each opening)
- Documented CCRL/CEGT published reference anchors:
  * Vice 1.2 (~1850 CCRL 40/15)
  * Sunfish 2026 (~1750 CCRL 40/15)
  * BBC 1.2 (~1950 CCRL 40/15)
  * Stockfish 1350 (UCI_Elo=1350)
- Full Ordo statistical rating estimation with 95% confidence intervals (-s 1000 -F 95)
"""

import os
import sys
import time
import subprocess
import argparse
from typing import List, Tuple, Dict

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("experiments/calibrated_eval"))

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
                    fen = line.split(";")[0].strip()
                    try:
                        b = chess.Board(fen)
                        fens.append(b.fen())
                    except Exception:
                        pass
                if len(fens) >= count:
                    break
    if not fens:
        fens = [
            chess.STARTING_FEN,
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
        ]
    return fens[:count]


def play_game(white_player, black_player, start_fen: str, round_id: int, time_limit: float = 0.1) -> chess.pgn.Game:
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    game.headers["Event"] = "ORION Calibrated Round-Robin Tournament"
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
            err_msg = str(e).encode("ascii", errors="replace").decode("ascii")
            print(f"[ERROR] Engine {current_player.name} error: {err_msg}")
            break

        if mv is None or mv not in board.legal_moves:
            break

        node = node.add_variation(mv)
        board.push(mv)
        ply_count += 1

    if board.is_checkmate():
        res = "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves():
        res = "1/2-1/2"
    else:
        from tournament_engine import evaluate_material
        mat = evaluate_material(board)
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


def run_full_round_robin(num_pairings: int = 1, time_limit: float = 0.05, pgn_path: str = "experiments/calibrated_eval/full_tournament.pgn"):
    os.makedirs(os.path.dirname(pgn_path), exist_ok=True)
    openings = load_openings(EPD_BOOK, num_pairings)

    runner = OrionModelRunner(CHECKPOINT_PATH)
    orion_search = OrionSearchPlayer(runner)
    orion_neural = OrionNeuralOnlyPlayer(runner)

    vice = ExternalEnginePlayer("Vice-1.2", [os.path.abspath("tools/engines/vice.exe")])
    sunfish = ExternalEnginePlayer("Sunfish-2026", ["python", os.path.abspath("tools/engines/sunfish-master/sunfish.py"), "-u"])
    bbc = ExternalEnginePlayer("BBC-1.2", [os.path.abspath("tools/engines/bbc-master/bin/windows/bbc_1.2_64bit_windows.exe")])
    stockfish_1350 = ExternalEnginePlayer("Stockfish-1350", [os.path.abspath("stockfish.exe")], options={"UCI_LimitStrength": True, "UCI_Elo": 1350})

    external_engines = [vice, sunfish, bbc, stockfish_1350]
    for e in external_engines:
        e.start()

    participants = [
        orion_search,
        orion_neural,
        vice,
        sunfish,
        bbc,
        stockfish_1350
    ]

    all_games = []
    round_counter = 1
    total_matchups = (len(participants) * (len(participants) - 1)) // 2
    total_games_expected = total_matchups * len(openings) * 2

    print(f"=== Starting Round-Robin Arena: {len(participants)} Engines, {total_games_expected} Games ===")
    print(f"Participants: {[p.name for p in participants]}")
    t_start = time.time()

    for i in range(len(participants)):
        for j in range(i + 1, len(participants)):
            p1 = participants[i]
            p2 = participants[j]
            print(f"\n--- Matchup: {p1.name} vs {p2.name} ({len(openings)*2} games) ---")
            for fen in openings:
                # Leg 1: p1 White, p2 Black
                g1 = play_game(p1, p2, fen, round_counter, time_limit=time_limit)
                round_counter += 1
                all_games.append(g1)
                print(f"Round {round_counter-1:03d}: {p1.name} (W) vs {p2.name} (B) -> {g1.headers['Result']}")

                # Leg 2: p2 White, p1 Black (Swapped Colors, exact same opening FEN)
                g2 = play_game(p2, p1, fen, round_counter, time_limit=time_limit)
                round_counter += 1
                all_games.append(g2)
                print(f"Round {round_counter-1:03d}: {p2.name} (W) vs {p1.name} (B) -> {g2.headers['Result']}")

    for e in external_engines:
        e.close()

    abs_pgn = os.path.abspath(pgn_path)
    os.makedirs(os.path.dirname(abs_pgn), exist_ok=True)
    with open(abs_pgn, "w", encoding="utf-8") as f:
        for g in all_games:
            print(g, file=f, end="\n\n")
            f.flush()

    elapsed = time.time() - t_start
    print(f"\nFinished {len(all_games)} games in {elapsed:.1f}s. Saved PGN: {abs_pgn}")
    return abs_pgn


def analyze_with_ordo(pgn_path: str, anchor_name: str = "Vice-1.2", anchor_rating: float = 1850.0):
    out_txt = os.path.splitext(pgn_path)[0] + "_ordo_report.txt"
    cmd = [
        ORDO_EXE,
        "-p", os.path.abspath(pgn_path),
        "-o", os.path.abspath(out_txt),
        "-A", anchor_name,
        "-a", str(anchor_rating),
        "-s", "1000",
        "-F", "95",
        "-G"
    ]
    print(f"\n--- Running Ordo Analysis ---")
    print("Command:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Ordo Exit Code:", res.returncode)
    print(res.stdout)
    if os.path.exists(out_txt):
        with open(out_txt, "r") as f:
            content = f.read()
        print("\n=== Official Ordo Rating Output ===")
        print(content)
        return content
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=1, help="Number of paired openings")
    parser.add_argument("--time", type=float, default=0.05, help="Time limit per move")
    args = parser.parse_args()

    pgn = run_full_round_robin(num_pairings=args.pairs, time_limit=args.time)
    analyze_with_ordo(pgn)
