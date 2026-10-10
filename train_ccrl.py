#!/usr/bin/env python3
"""
================================================================================
          ORION Deep Neural Super-GM Training: CCRL 40/40 Engine Corpus
================================================================================
Streams 2,463,682 tournament games played by 4,753 top computer engines directly
from CCRL-4040.[2463682].pgn.7z with ZERO disk footprint (tar -xOf streaming).

Filters for Super-GM / High-Elo engine play (Elo >= 2600+), extracting rich
opening, middlegame, tactical, and endgame positions.

Trains the ResNet dual-head architecture:
- Policy head: CrossEntropy loss on master engine moves
- Value head: BCE loss on verified tournament game outcomes
Exports fused weights for native ORION inference.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn
import numpy as np
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

PIECE_CHANNELS = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
}

DEFAULT_ARCHIVE = r"D:\orion-game-ai\CCRL-4040.[2463682].pgn.7z"


def fast_encode_board(board: chess.Board, out_buf: np.ndarray):
    """Encodes a chess.Board directly into a 18x8x8 uint8 buffer matching ChessEncoder."""
    out_buf.fill(0)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            ch = PIECE_CHANNELS[piece.symbol()]
            r = 7 - (sq >> 3)
            c = sq & 7
            out_buf[ch, r, c] = 1

    if board.turn == chess.WHITE:
        out_buf[12, :, :] = 1

    if board.ep_square is not None:
        out_buf[13, 7 - (board.ep_square >> 3), board.ep_square & 7] = 1

    if board.has_kingside_castling_rights(chess.WHITE):
        out_buf[14, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        out_buf[15, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        out_buf[16, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        out_buf[17, :, :] = 1


def fast_encode_action(move: chess.Move) -> int:
    """Encodes a chess.Move into [0, 4095] action space matching ChessActionSpace."""
    r1 = 7 - (move.from_square >> 3)
    c1 = move.from_square & 7
    r2 = 7 - (move.to_square >> 3)
    c2 = move.to_square & 7
    source = (r1 << 3) + c1
    target = (r2 << 3) + c2
    return (source << 6) + target


def fast_material_value(board: chess.Board) -> float:
    """Computes material balance in [0, 1] from current player's perspective."""
    piece_vals = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    white_score = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in piece_vals.items())
    black_score = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in piece_vals.items())
    diff = white_score - black_score
    if board.turn == chess.BLACK:
        diff = -diff
    norm = max(-9.0, min(9.0, float(diff))) / 9.0
    return (norm + 1.0) / 2.0


def stream_ccrl_dataset(
    archive_path: str,
    max_positions: int = 150000,
    min_elo: int = 2600,
    stride: int = 2,
    min_ply: int = 6,
    max_ply: int = 90
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Streams games from the CCRL .7z archive without decompressing to disk.
    Pre-allocates compact tensors in RAM.
    """
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"CCRL archive not found at: {archive_path}")

    print(f"[STREAM] Opening stream pipeline: tar -xOf {archive_path}")
    print(f"[STREAM] Filter: min_elo >= {min_elo}, ply window: [{min_ply}, {max_ply}], stride: {stride}")

    cmd = ["tar", "-xOf", archive_path]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="latin1",
        bufsize=4194304
    )

    # Pre-allocate compact memory (1152 bytes uint8 per position)
    x_tensor = torch.zeros((max_positions, 1152), dtype=torch.uint8)
    p_tensor = torch.zeros((max_positions,), dtype=torch.long)
    v_tensor = torch.zeros((max_positions, 1), dtype=torch.float32)

    buf_board = np.zeros((18, 8, 8), dtype=np.uint8)

    positions_collected = 0
    games_parsed = 0
    games_used = 0
    t0 = time.time()

    try:
        while positions_collected < max_positions:
            game = chess.pgn.read_game(proc.stdout)
            if game is None:
                print("\n[STREAM] Reached end of CCRL archive.")
                break

            games_parsed += 1

            # Parse headers
            result = game.headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                continue

            try:
                w_elo = int(game.headers.get("WhiteElo", "0"))
            except ValueError:
                w_elo = 0

            try:
                b_elo = int(game.headers.get("BlackElo", "0"))
            except ValueError:
                b_elo = 0

            if max(w_elo, b_elo) < min_elo:
                continue

            # Outcome from White's perspective
            if result == "1-0":
                white_outcome = 1.0
            elif result == "0-1":
                white_outcome = 0.0
            else:
                white_outcome = 0.5

            board = game.board()
            ply = 0
            game_samples = 0

            for move in game.mainline_moves():
                ply += 1
                if ply < min_ply:
                    board.push(move)
                    continue
                if ply > max_ply or positions_collected >= max_positions:
                    break

                # Stride sampling to maximize variety
                if (ply % stride) == 0:
                    fast_encode_board(board, buf_board)
                    action = fast_encode_action(move)

                    # Value target: outcome + small material blend
                    turn_outcome = white_outcome if board.turn == chess.WHITE else (1.0 - white_outcome)
                    mat_val = fast_material_value(board)
                    target_val = 0.85 * turn_outcome + 0.15 * mat_val

                    x_tensor[positions_collected] = torch.from_numpy(buf_board.flatten())
                    p_tensor[positions_collected] = action
                    v_tensor[positions_collected, 0] = target_val
                    positions_collected += 1
                    game_samples += 1

                board.push(move)

            if game_samples > 0:
                games_used += 1

            if games_parsed % 250 == 0 or positions_collected >= max_positions:
                elapsed = time.time() - t0
                pct = (positions_collected / max_positions) * 100.0
                rate = positions_collected / elapsed if elapsed > 0 else 0
                print(
                    f"\r[STREAM] Samples: {positions_collected:,}/{max_positions:,} ({pct:.1f}%) | "
                    f"Games used: {games_used:,} | Speed: {rate:.0f} pos/s | Time: {elapsed:.1f}s",
                    end="",
                    flush=True
                )
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            pass

    print()
    if positions_collected < max_positions:
        x_tensor = x_tensor[:positions_collected]
        p_tensor = p_tensor[:positions_collected]
        v_tensor = v_tensor[:positions_collected]

    total_time = time.time() - t0
    print(f"[STREAM] Stream complete: {len(p_tensor):,} Super-GM positions extracted in {total_time:.1f}s!")
    return x_tensor, p_tensor, v_tensor


def train_ccrl(
    archive_path: str = DEFAULT_ARCHIVE,
    max_positions: int = 150000,
    min_elo: int = 2600,
    stride: int = 2,
    batch_size: int = 512,
    epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    checkpoint_in: Optional[str] = None,
    checkpoint_out: str = "checkpoints/ccrl_master/orion_ccrl_deep_master.json"
):
    print("=" * 72)
    print("      ORION DEEP SUPER-GM NEURAL TRAINING (CCRL 40/40 CORPUS)")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Hardware accelerator: {device}")
    if device.type == "cuda":
        print(f"[DEVICE] GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Stream positions directly from CCRL
    x_tensor, p_tensor, v_tensor = stream_ccrl_dataset(
        archive_path=archive_path,
        max_positions=max_positions,
        min_elo=min_elo,
        stride=stride
    )

    total_samples = len(p_tensor)
    if total_samples == 0:
        print("[ERROR] No positions collected. Exiting.")
        return

    # Train / Val Split (90% / 10%)
    val_size = max(1, int(total_samples * 0.10))
    train_size = total_samples - val_size

    indices = torch.randperm(total_samples)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_ds = TensorDataset(x_tensor[train_idx], p_tensor[train_idx], v_tensor[train_idx])
    val_ds = TensorDataset(x_tensor[val_idx], p_tensor[val_idx], v_tensor[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    print(f"[DATASET] Split: {train_size:,} train samples, {val_size:,} validation samples")

    # 2. Initialize ResNet Model
    model = OrionResNet().to(device)

    # 3. Warm-Start from previous best weights if available
    init_ckpt = checkpoint_in
    if not init_ckpt:
        for candidate in [
            "checkpoints/distilled/orion_distill_master.json",
            "checkpoints/distilled/orion_distill_best.json",
            "checkpoints/gm_master/orion_gm_1m_best.json"
        ]:
            if os.path.exists(candidate):
                init_ckpt = candidate
                break

    if init_ckpt and os.path.exists(init_ckpt):
        print(f"[WARM-START] Loading foundation weights from: {init_ckpt}")
        try:
            load_json_to_model(model, init_ckpt)
            print("[WARM-START] Successfully loaded foundation weights!")
        except Exception as e:
            print(f"[WARN] Failed to load foundation weights: {e}")

    # 4. Losses and Optimizers
    policy_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    value_criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print("\n[TRAINING] Starting deep optimization loop...")
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_p_loss = 0.0
        total_v_loss = 0.0
        total_combined_loss = 0.0
        train_correct = 0
        train_count = 0
        t_epoch = time.time()

        for batch_x, batch_p, batch_v in train_loader:
            batch_x = batch_x.to(device, non_blocking=True).float()
            batch_p = batch_p.to(device, non_blocking=True)
            batch_v = batch_v.to(device, non_blocking=True)

            optimizer.zero_grad()
            policy_logits, value_pred = model(batch_x)

            p_loss = policy_criterion(policy_logits, batch_p)
            v_loss = value_criterion(value_pred, batch_v)
            loss = p_loss + 0.5 * v_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = len(batch_p)
            total_p_loss += p_loss.item() * bs
            total_v_loss += v_loss.item() * bs
            total_combined_loss += loss.item() * bs

            preds = torch.argmax(policy_logits, dim=1)
            train_correct += (preds == batch_p).sum().item()
            train_count += bs

        scheduler.step()
        elapsed = time.time() - t_epoch

        train_acc = (train_correct / train_count) * 100.0
        avg_loss = total_combined_loss / train_count
        cur_lr = scheduler.get_last_lr()[0]

        # Validation pass
        model.eval()
        val_correct = 0
        val_count = 0
        val_p_loss = 0.0
        with torch.no_grad():
            for vx, vp, vv in val_loader:
                vx = vx.to(device, non_blocking=True).float()
                vp = vp.to(device, non_blocking=True)
                vv = vv.to(device, non_blocking=True)
                p_logits, v_pred = model(vx)
                val_p_loss += policy_criterion(p_logits, vp).item() * len(vp)
                preds = torch.argmax(p_logits, dim=1)
                val_correct += (preds == vp).sum().item()
                val_count += len(vp)

        val_acc = (val_correct / val_count) * 100.0 if val_count > 0 else 0.0
        avg_val_loss = val_p_loss / val_count if val_count > 0 else 0.0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {cur_lr:.2e} | Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(checkpoint_out) or ".", exist_ok=True)
            model.export_to_orion_checkpoint(checkpoint_out, episode=total_samples * epoch)
            print(f"  --> Saved new best CCRL master checkpoint to {checkpoint_out} (Val Acc: {val_acc:.2f}%)")

    print("\n" + "=" * 72)
    print(f"[COMPLETE] Deep CCRL Training Finished! Peak Val Acc: {best_val_acc:.2f}%")
    print(f"[MASTER CHECKPOINT] {checkpoint_out}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="ORION Deep Super-GM Training on CCRL 40/40 Archive")
    parser.add_argument("--archive", type=str, default=DEFAULT_ARCHIVE, help="Path to CCRL-4040 .7z archive")
    parser.add_argument("--max-positions", type=int, default=150000, help="Number of positions to sample and train on")
    parser.add_argument("--min-elo", type=int, default=2600, help="Minimum engine Elo filter")
    parser.add_argument("--stride", type=int, default=2, help="Move sampling stride per game")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--warm-start", type=str, default=None, help="Path to checkpoint to warm-start from")
    parser.add_argument("--output", type=str, default="checkpoints/ccrl_master/orion_ccrl_deep_master.json", help="Output checkpoint path")
    args = parser.parse_args()

    train_ccrl(
        archive_path=args.archive,
        max_positions=args.max_positions,
        min_elo=args.min_elo,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_in=args.warm_start,
        checkpoint_out=args.output
    )


if __name__ == "__main__":
    main()
