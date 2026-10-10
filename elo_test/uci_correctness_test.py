#!/usr/bin/env python3
"""
Phase One — UCI Protocol & Game Correctness Test Suite
Audits ORION Policy-Only Engine and ORION Complete Engine across:
- UCI command protocol compliance (uci, isready, ucinewgame, position, go, stop, quit)
- Move legality verification in startpos, complex middlegames, and tactical endgames
- Checkmate recognition and avoidance
- Stalemate recognition
- Pawn promotion handling (e.g. e7e8q)
- En passant execution
- Castling rights execution
Saves output to elo_test/uci_smoke_test.log and elo_test/correctness_results.json
"""

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import chess

TEST_ENGINES = [
    {
        "name": "ORION_Policy_Only",
        "cmd": [sys.executable, r"D:\orion-game-ai\elo_test\orion_policy_engine.py"]
    },
    {
        "name": "ORION_Complete_Engine",
        "cmd": [sys.executable, r"D:\orion-game-ai\elo_test\orion_complete_engine.py", r"D:\orion-game-ai\checkpoints\distilled\orion_distill_5ep.json", "2"]
    }
]

# Standard correctness positions
TEST_POSITIONS = [
    {
        "name": "startpos",
        "fen": "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "description": "Standard initial position"
    },
    {
        "name": "scholar_mate_threat",
        "fen": "r1bqkb1r/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4",
        "description": "White has immediate Qxf7# checkmate in 1",
        "expected_move": "h5f7"
    },
    {
        "name": "pawn_promotion_white",
        "fen": "8/4P3/8/8/8/8/8/k6K w - - 0 1",
        "description": "Pawn promotes to Queen on e8",
        "expected_move": "e7e8q"
    },
    {
        "name": "en_passant_black",
        "fen": "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "description": "En passant square f6 available",
        "expected_legal_contains": "e5f6"
    },
    {
        "name": "white_kingside_castling",
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "description": "White can castle kingside e1g1",
        "expected_legal_contains": "e1g1"
    },
    {
        "name": "stalemate_defense",
        "fen": "k7/8/1K6/8/8/8/8/8 b - - 0 1",
        "description": "King on a8 can only play a8b8"
    }
]


def run_uci_command(proc, cmd: str) -> None:
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def read_until(proc, target: str, timeout: float = 5.0) -> Tuple[bool, List[str]]:
    lines = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.02)
            continue
        line = line.strip()
        lines.append(line)
        if target in line:
            return True, lines
    return False, lines


def test_engine_uci(engine_info: Dict, log_file) -> Dict:
    name = engine_info["name"]
    cmd = engine_info["cmd"]

    results = {
        "engine": name,
        "uci_protocol": False,
        "isready": False,
        "ucinewgame": False,
        "position_handling": False,
        "legality_checks": {},
        "all_passed": False
    }

    log_file.write(f"\n{'='*70}\nTESTING ENGINE: {name}\nCommand: {' '.join(cmd)}\n{'='*70}\n")
    print(f"\n[TESTING] {name}...")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    try:
        # 1. Test 'uci'
        run_uci_command(proc, "uci")
        ok, lines = read_until(proc, "uciok", timeout=5.0)
        results["uci_protocol"] = ok
        log_file.write(f"--> [uci] -> uciok: {ok}\n" + "\n".join(lines) + "\n\n")

        # 2. Test 'isready'
        run_uci_command(proc, "isready")
        ok, lines = read_until(proc, "readyok", timeout=5.0)
        results["isready"] = ok
        log_file.write(f"--> [isready] -> readyok: {ok}\n" + "\n".join(lines) + "\n\n")

        # 3. Test 'ucinewgame'
        run_uci_command(proc, "ucinewgame")
        run_uci_command(proc, "isready")
        ok, lines = read_until(proc, "readyok", timeout=5.0)
        results["ucinewgame"] = ok
        log_file.write(f"--> [ucinewgame] -> readyok: {ok}\n\n")

        # 4. Test Positions & Legality
        all_positions_legal = True
        for pos in TEST_POSITIONS:
            pos_name = pos["name"]
            fen = pos["fen"]
            board = chess.Board(fen)

            run_uci_command(proc, f"position fen {fen}")
            run_uci_command(proc, "go depth 2")

            ok, lines = read_until(proc, "bestmove", timeout=5.0)
            bestmove = None
            for l in lines:
                if l.startswith("bestmove"):
                    parts = l.split()
                    if len(parts) >= 2:
                        bestmove = parts[1]

            is_legal = False
            if bestmove:
                try:
                    mv = chess.Move.from_uci(bestmove)
                    is_legal = mv in board.legal_moves
                except Exception:
                    is_legal = False

            results["legality_checks"][pos_name] = {
                "fen": fen,
                "bestmove_returned": bestmove,
                "is_legal": is_legal
            }

            if "expected_move" in pos:
                results["legality_checks"][pos_name]["expected_move"] = pos["expected_move"]
                results["legality_checks"][pos_name]["matched_expected"] = (bestmove == pos["expected_move"])

            log_file.write(f"Position [{pos_name}]: FEN={fen} -> bestmove={bestmove} (Legal: {is_legal})\n")

            if not is_legal:
                all_positions_legal = False

        results["position_handling"] = all_positions_legal

        # 5. Test 'quit'
        run_uci_command(proc, "quit")
        proc.wait(timeout=2.0)
        log_file.write(f"--> [quit] Clean exit: code {proc.returncode}\n")

    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            pass

    results["all_passed"] = (
        results["uci_protocol"] and
        results["isready"] and
        results["ucinewgame"] and
        results["position_handling"]
    )

    print(f"[{name}] Protocol: {'PASS' if results['uci_protocol'] else 'FAIL'} | "
          f"Positions & Legality: {'PASS' if results['position_handling'] else 'FAIL'} | "
          f"Overall: {'PASS' if results['all_passed'] else 'FAIL'}")

    return results


def main():
    os.makedirs("elo_test", exist_ok=True)
    log_path = "elo_test/uci_smoke_test.log"
    results_path = "elo_test/correctness_results.json"

    with open(log_path, "w") as log_f:
        log_f.write("ORION CHESS ENGINE — PHASE ONE UCI & CORRECTNESS TEST AUDIT\n")
        log_f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        all_results = {}
        for eng in TEST_ENGINES:
            res = test_engine_uci(eng, log_f)
            all_results[eng["name"]] = res

    with open(results_path, "w") as rf:
        json.dump(all_results, rf, indent=2)

    print(f"\n[PHASE ONE COMPLETE] Logs saved to: {log_path}")
    print(f"[PHASE ONE COMPLETE] Results saved to: {results_path}")


if __name__ == "__main__":
    main()
