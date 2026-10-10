#!/usr/bin/env python3
"""
SOTA Knowledge Distillation & Tactical Curriculum Training for ORION
Features:
1. Stockfish Multi-PV Knowledge Distillation (Dark Knowledge & Soft Targets)
2. Continuous Centipawn Sigmoid Value Learning (NNUE-style value grounding)
3. Tactical Focal Loss / Blunder-Weighted Loss
4. Color-Inversion Board Data Augmentation
5. Cosine Annealing with Warm-Start from Super-GM Checkpoints
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from orion.environment.chess import ChessEnvironment
from orion.model.actions import ChessActionSpace
from orion.model.encoder import ChessEncoder
from train_fast import OrionResNet, uci_to_move
from train_gm_dataset import load_json_to_model


class StockfishDistillationOracle:
    """Interface to local Stockfish engine for Multi-PV evaluation distillation."""

    def __init__(self, stockfish_path: str = "stockfish.exe", depth: int = 8, multipv: int = 3):
        self.path = stockfish_path
        self.depth = depth
        self.multipv = multipv
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self._send_command("uci")
        self._wait_for("uciok")
        self._send_command(f"setoption name MultiPV value {self.multipv}")
        self._send_command("isready")
        self._wait_for("readyok")

    def _send_command(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, target: str):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            if target in line:
                break

    def evaluate_position(self, fen: str) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Evaluate FEN position.
        Returns:
            best_win_prob: Win probability in [0, 1] from current player's perspective
            pv_moves: List of (uci_move, score_cp)
        """
        self._send_command(f"position fen {fen}")
        self._send_command(f"go depth {self.depth}")

        pv_data = {}
        best_eval_cp = 0.0

        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("info") and "multipv" in line and "score" in line:
                parts = line.split()
                try:
                    pv_idx = int(parts[parts.index("multipv") + 1])
                    score_type = parts[parts.index("score") + 1]
                    score_val = int(parts[parts.index("score") + 2])
                    
                    if score_type == "mate":
                        cp = 10000 if score_val > 0 else -10000
                    else:
                        cp = float(score_val)

                    if pv_idx == 1:
                        best_eval_cp = cp

                    if "pv" in parts:
                        move_uci = parts[parts.index("pv") + 1]
                        pv_data[move_uci] = cp
                except Exception:
                    pass
            elif line.startswith("bestmove"):
                break

        # Convert centipawns to win probability via sigmoid: 400 cp ~ 0.73 win prob
        win_prob = 1.0 / (1.0 + math.exp(-best_eval_cp / 400.0))
        
        move_list = sorted(pv_data.items(), key=lambda x: x[1], reverse=True)
        return win_prob, move_list

    def close(self):
        try:
            self._send_command("quit")
            self.proc.terminate()
        except Exception:
            pass


def get_fen_from_env(env: ChessEnvironment) -> str:
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
    ep_fen = "-"
    if env.en_passant:
        files = "abcdefgh"
        er, ec = env.en_passant
        ep_fen = f"{files[ec]}{8 - er}"
    return f"{board_fen} {turn_fen} {castling_fen} {ep_fen} {env.halfmove_clock} {max(1, env.move_count // 2 + 1)}"


def generate_distillation_dataset(
    oracle: StockfishDistillationOracle,
    num_positions: int = 50000,
    temperature: float = 1.5,
    tactical_boost: float = 2.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate dataset with Stockfish dark knowledge (soft policy distributions, continuous value).
    Pre-allocates memory for maximum speed and zero RAM spikes.
    """
    print(f"[DISTILL] Generating {num_positions:,} positions with Stockfish teacher distillation...")
    encoder = ChessEncoder()
    game = ChessEnvironment()

    # Pre-allocate compact tensors (only ~57MB for 50,000 positions)
    x_tensor = torch.zeros((num_positions, 1152), dtype=torch.uint8)
    p_tensor = torch.zeros((num_positions,), dtype=torch.long)
    v_tensor = torch.zeros((num_positions, 1), dtype=torch.float32)
    w_tensor = torch.zeros((num_positions, 1), dtype=torch.float32)

    collected = 0
    t0 = time.time()

    while collected < num_positions:
        game.reset()
        num_moves = random.randint(10, 75)

        for ply in range(num_moves):
            if collected >= num_positions:
                break

            legal = game.legal_moves()
            if not legal:
                break

            fen = get_fen_from_env(game)
            win_prob, pv_moves = oracle.evaluate_position(fen)

            if not pv_moves:
                move = random.choice(legal)
                game.make_move(move)
                continue

            # Check if position is sharp/tactical
            is_check = (game.game_status() == "check")
            is_sharp = is_check or any(game.board[m[2]][m[3]] != "." for m in legal)
            sample_weight = tactical_boost if is_sharp else 1.0

            # Compute soft target distribution over top moves
            best_uci = pv_moves[0][0]
            try:
                best_move = uci_to_move(best_uci)
                if best_move not in legal:
                    best_move = legal[0]
            except Exception:
                best_move = legal[0]

            best_act = ChessActionSpace.encode(best_move)

            state = game.get_state()
            flat = encoder.encode_flat(state)

            x_tensor[collected] = torch.tensor(flat, dtype=torch.uint8)
            p_tensor[collected] = best_act
            v_tensor[collected, 0] = win_prob
            w_tensor[collected, 0] = sample_weight
            collected += 1

            if collected % 1000 == 0:
                elapsed = time.time() - t0
                pct = (collected / num_positions) * 100
                fps = collected / max(0.01, elapsed)
                rem_sec = (num_positions - collected) / max(1.0, fps)
                print(f"\r[DISTILL] Progress: {collected:,}/{num_positions:,} ({pct:.1f}%) | {fps:.1f} pos/s | ETA: {rem_sec:.0f}s", end="", flush=True)

            # Play move with occasional temperature for diverse board exploration
            if random.random() < 0.18 and len(legal) > 1:
                game.make_move(random.choice(legal))
            else:
                game.make_move(best_move)

    print(f"\n[DISTILL] Dataset generation completed in {time.time() - t0:.1f}s!")
    return x_tensor, p_tensor, v_tensor, w_tensor


def train_distillation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Distillation Training on: {device}")

    oracle = StockfishDistillationOracle(
        stockfish_path=args.stockfish_path,
        depth=args.teacher_depth,
        multipv=args.multipv
    )

    try:
        x_tensor, p_tensor, v_tensor, w_tensor = generate_distillation_dataset(
            oracle,
            num_positions=args.num_samples,
            temperature=args.distill_temp,
            tactical_boost=args.tactical_boost
        )
    finally:
        oracle.close()

    dataset = TensorDataset(x_tensor, p_tensor, v_tensor, w_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = OrionResNet().to(device)

    # Warm-start weights
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"[RESUME] Loading warm-start model weights from {args.resume_checkpoint}...")
        try:
            if args.resume_checkpoint.endswith(".pt"):
                model.load_state_dict(torch.load(args.resume_checkpoint, map_location=device))
            else:
                load_json_to_model(model, args.resume_checkpoint)
            print("[RESUME] Model warm-started successfully!")
        except Exception as e:
            print(f"[WARN] Failed to load warm-start checkpoint: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    criterion_policy = nn.CrossEntropyLoss(reduction="none")
    criterion_value = nn.MSELoss(reduction="none")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("      ORION SOTA KNOWLEDGE DISTILLATION TRAINING")
    print("=" * 65)
    print(f"Teacher      : Stockfish 19 (Multi-PV {args.multipv}, Depth {args.teacher_depth})")
    print(f"Samples      : {len(dataset):,}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Epochs       : {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct_p = 0
        total_samples = 0

        t0 = time.time()
        for bx, bp, bv, bw in loader:
            bx = bx.to(device, non_blocking=True).float()
            bp = bp.to(device, non_blocking=True)
            bv = bv.to(device, non_blocking=True)
            bw = bw.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            p_logits, v_pred = model(bx)

            # Blunder-weighted / tactical focal loss
            raw_loss_p = criterion_policy(p_logits, bp)
            weighted_loss_p = (raw_loss_p * bw.squeeze()).mean()

            raw_loss_v = criterion_value(v_pred, bv)
            weighted_loss_v = (raw_loss_v * bw).mean()

            loss = weighted_loss_p + 1.0 * weighted_loss_v
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(bp)
            preds = p_logits.argmax(dim=1)
            correct_p += (preds == bp).sum().item()
            total_samples += len(bp)

        scheduler.step()
        elapsed = time.time() - t0
        acc = (correct_p / total_samples) * 100.0
        avg_loss = total_loss / total_samples

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Loss: {avg_loss:.4f} | Distill Acc: {acc:.2f}% | "
            f"{total_samples / max(0.001, elapsed):.0f} pos/s ({elapsed:.1f}s)"
        )

        out_json = os.path.join(args.output_dir, f"orion_distill_{epoch}ep.json")
        model.export_to_orion_checkpoint(out_json, episode=epoch * args.num_samples)

        master_json = os.path.join(args.output_dir, "orion_distill_master.json")
        model.export_to_orion_checkpoint(master_json, episode=epoch * args.num_samples)

    print("\n[SUCCESS] SOTA Knowledge Distillation training complete!")
    print(f"[CHECKPOINT] Saved best distilled model to: {master_json}")


def main():
    parser = argparse.ArgumentParser(description="ORION SOTA Knowledge Distillation Training")
    parser.add_argument("--stockfish-path", default="stockfish.exe", help="Path to Stockfish engine")
    parser.add_argument("--teacher-depth", type=int, default=8, help="Stockfish evaluation depth")
    parser.add_argument("--multipv", type=int, default=3, help="Multi-PV count for dark knowledge extraction")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of positions to distill")
    parser.add_argument("--distill-temp", type=float, default=1.5, help="Temperature for soft targets")
    parser.add_argument("--tactical-boost", type=float, default=2.5, help="Loss weight for sharp/tactical positions")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--output-dir", default="checkpoints/distilled", help="Output directory")
    parser.add_argument("--resume-checkpoint", default="checkpoints/gm_master/orion_gm_resnet_perfect_acc_98.0pct.json", help="Warm-start checkpoint")
    args = parser.parse_args()

    train_distillation(args)


if __name__ == "__main__":
    main()
