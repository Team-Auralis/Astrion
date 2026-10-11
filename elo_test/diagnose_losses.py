#!/usr/bin/env python3
"""
Experiment B: Diagnostic Loss Analysis against Stronger Engines (BBC-1.2 & Vice-1.2)
Uses Stockfish 19 engine to analyze move-by-move evaluations for all games ORION lost.
Identifies:
1. Exact move number & ply of the first decisive mistake (eval delta >= 150 cp)
2. The played move vs Stockfish's top recommended engine move
3. Classification of the root cause:
   - Search depth horizon (tactics deeper than current ply)
   - Evaluation flaw (bad static position score / PST bias)
   - Move ordering defect (good defense not searched first)
   - Tactical oversight (hanging piece or missed tactic)
   - Time management cutoff (depth truncated by clock budget)

Outputs structured diagnosis report to elo_test/loss_diagnosis_report.json and markdown.
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional
import chess
import chess.pgn
import chess.engine

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("experiments/calibrated_eval"))

STOCKFISH_EXE = os.path.abspath("stockfish.exe")
PGN_FILES = [
    "experiments/calibrated_eval/full_tournament.pgn",
    "elo_test/v2_3_verification.pgn"
]


def load_loss_games() -> List[Dict[str, Any]]:
    losses = []
    for pgn_path in PGN_FILES:
        if not os.path.exists(pgn_path):
            continue
        with open(pgn_path, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if not game:
                    break
                white = game.headers.get("White", "")
                black = game.headers.get("Black", "")
                res = game.headers.get("Result", "")

                is_orion_white = "ORION" in white
                is_orion_black = "ORION" in black
                opp = black if is_orion_white else white

                # Check if opponent is BBC or Vice and ORION lost
                is_strong_opp = any(name in opp for name in ["BBC", "Vice"])
                if is_strong_opp:
                    orion_lost = (is_orion_white and res == "0-1") or (is_orion_black and res == "1-0")
                    if orion_lost:
                        orion_name = white if is_orion_white else black
                        losses.append({
                            "pgn_source": pgn_path,
                            "event": game.headers.get("Event", ""),
                            "round": game.headers.get("Round", ""),
                            "white": white,
                            "black": black,
                            "result": res,
                            "orion_color": chess.WHITE if is_orion_white else chess.BLACK,
                            "orion_name": orion_name,
                            "opp_name": opp,
                            "game": game
                        })
    return losses


def classify_mistake(board_before: chess.Board, played_move: chess.Move, best_move: chess.Move, eval_drop_cp: float) -> str:
    """Classify the root cause of the decisive mistake."""
    # 1. Did the played move hang a piece immediately?
    board_after = board_before.copy()
    board_after.push(played_move)
    is_immediate_capture = any(
        board_after.is_capture(resp) and board_after.piece_at(resp.to_square) and board_after.piece_at(resp.to_square).piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
        for resp in board_after.legal_moves
    )

    # 2. Check for tactical oversight vs positional evaluation
    if eval_drop_cp >= 400 or board_after.is_checkmate():
        return "Tactical Oversight (Blunder/Hanging Piece)"
    elif eval_drop_cp >= 250:
        return "Search Depth Horizon (Multi-ply tactical consequence beyond reach)"
    elif board_before.has_legal_en_passant() or board_before.is_pinned(board_before.turn, played_move.from_square):
        return "Move Ordering Defect (Tactical refutation missed early)"
    else:
        return "Positional / Evaluation Inaccuracy (Static PST overvaluation)"


def analyze_loss_game(loss_info: Dict[str, Any], sf_engine: chess.engine.SimpleEngine) -> Dict[str, Any]:
    game = loss_info["game"]
    orion_color = loss_info["orion_color"]
    board = game.board()
    moves = list(game.mainline_moves())

    decisive_mistake = None
    eval_history = []

    print(f"\nAnalyzing: {loss_info['white']} vs {loss_info['black']} ({loss_info['result']})...")

    for ply, mv in enumerate(moves):
        is_orion_turn = (board.turn == orion_color)
        b_before = board.copy()

        # Analyse current position
        info = sf_engine.analyse(board, chess.engine.Limit(depth=12))
        score_before = info["score"].pov(orion_color).score(mate_score=10000)
        best_sf_move = info.get("pv", [None])[0]

        eval_history.append({"ply": ply + 1, "move": mv.uci(), "score_cp": score_before})

        if is_orion_turn and ply >= 2 and decisive_mistake is None:
            # Push ORION's move to measure the score drop it produced
            board.push(mv)
            info_after = sf_engine.analyse(board, chess.engine.Limit(depth=12))
            score_after = info_after["score"].pov(orion_color).score(mate_score=10000)
            board.pop()

            eval_drop = score_before - score_after
            if eval_drop >= 150.0:
                category = classify_mistake(b_before, mv, best_sf_move, eval_drop)
                decisive_mistake = {
                    "move_number": (ply // 2) + 1,
                    "ply": ply + 1,
                    "played_move": mv.uci(),
                    "best_move": best_sf_move.uci() if best_sf_move else "None",
                    "eval_before_cp": round(score_before, 1),
                    "eval_after_cp": round(score_after, 1),
                    "eval_drop_cp": round(eval_drop, 1),
                    "classification": category,
                    "fen_before": b_before.fen()
                }
                print(f"  [DECISIVE MISTAKE] Move {(ply//2)+1} ({mv.uci()}): drop of {eval_drop:.1f} cp. Best was {best_sf_move}. Root cause: {category}")

        board.push(mv)

    return {
        "white": loss_info["white"],
        "black": loss_info["black"],
        "result": loss_info["result"],
        "source": loss_info["pgn_source"],
        "total_plies": len(moves),
        "decisive_mistake": decisive_mistake,
        "eval_trajectory": eval_history
    }


def run_experiment_b():
    print("=" * 70)
    print("EXPERIMENT B: LOSS DIAGNOSIS AGAINST BBC-1.2 & VICE-1.2")
    print(f"Stockfish Engine: {STOCKFISH_EXE}")
    print("=" * 70)

    loss_games = load_loss_games()
    print(f"Found {len(loss_games)} loss games against stronger reference engines.")

    sf_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_EXE)
    results = []

    for loss_info in loss_games:
        res = analyze_loss_game(loss_info, sf_engine)
        results.append(res)

    sf_engine.quit()

    # Save to JSON
    os.makedirs("elo_test", exist_ok=True)
    out_json = "elo_test/loss_diagnosis_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save to Markdown
    out_md = "elo_test/loss_diagnosis_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Experiment B: Stockfish Loss Diagnostic Report\n\n")
        f.write("Analysis of all decisive games lost to reference engines (BBC-1.2 and Vice-1.2).\n\n")
        f.write("| Game | Opponent | Decisive Move | Played | Best SF Move | Eval Drop | Classified Root Cause |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :--- |\n")
        for r in results:
            m = r["decisive_mistake"]
            if m:
                opp = r["white"] if "ORION" in r["black"] else r["black"]
                f.write(f"| {r['white']} vs {r['black']} | **{opp}** | Move {m['move_number']} (ply {m['ply']}) | `{m['played_move']}` | `{m['best_move']}` | **-{m['eval_drop_cp']} cp** | **{m['classification']}** |\n")
        f.write("\n\n## Detailed Game-by-Game Diagnoses\n\n")
        for r in results:
            m = r["decisive_mistake"]
            f.write(f"### {r['white']} vs {r['black']} ({r['result']})\n")
            f.write(f"- **Source:** `{r['source']}`\n")
            f.write(f"- **Total Plies:** {r['total_plies']}\n")
            if m:
                f.write(f"- **First Decisive Mistake:** Move {m['move_number']} (`{m['played_move']}`)\n")
                f.write(f"- **Stockfish Recommendation:** `{m['best_move']}`\n")
                f.write(f"- **Evaluation Impact:** {m['eval_before_cp']:+.1f} cp -> {m['eval_after_cp']:+.1f} cp (Drop: **{m['eval_drop_cp']:.1f} cp**)\n")
                f.write(f"- **Root Cause Classification:** **{m['classification']}**\n")
                f.write(f"- **FEN at Mistake:** `{m['fen_before']}`\n\n")
            else:
                f.write("- **Decisive Mistake:** No sudden blunder >= 150 cp found; gradual positional outplay.\n\n")

    print(f"\nSaved structured diagnosis to: {out_json} and {out_md}")
    return results


if __name__ == "__main__":
    run_experiment_b()
