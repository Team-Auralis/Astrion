#!/usr/bin/env python3
"""
High-Precision Grandmaster Training for ORION.
Deduplicates positions by FEN so every position has exactly one authoritative master move.
Trains the neural network to convergence, driving accuracy to near 100%.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import fsspec
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from orion.environment.chess import ChessEnvironment
from orion.model.actions import ChessActionSpace
from orion.model.encoder import ChessEncoder
from orion.search.search_agent import StockfishInspiredSearchAgent
from train_fast import OrionResNet, uci_to_move
from train_gm_dataset import load_json_to_model

DATASET_URL = "https://huggingface.co/datasets/angeluriot/chess_games/resolve/main/dataset.parquet"


def stream_deduplicated_gm_positions(max_unique_positions: int = 50000, min_elo: int = 2400,
                                      max_moves_per_game: int = 40, start_group: int = 150):
    print(f"[STREAM] Collecting {max_unique_positions:,} UNIQUE Grandmaster positions (Elo >= {min_elo})...")
    print(f"[STREAM] Deduplicating by FEN so each position has a single authoritative master move...")

    encoder = ChessEncoder()
    game = ChessEnvironment()

    fen_to_sample = {}

    with fsspec.open(DATASET_URL, mode="rb") as f:
        pfile = pq.ParquetFile(f)
        total_groups = pfile.num_row_groups
        group_idx = start_group
        t0 = time.time()

        while len(fen_to_sample) < max_unique_positions and group_idx < total_groups:
            table = pfile.read_row_group(
                group_idx,
                columns=["white_elo", "black_elo", "winner", "moves_uci"]
            )
            group_idx += 1

            white_elos = table["white_elo"].to_pylist()
            black_elos = table["black_elo"].to_pylist()
            winners = table["winner"].to_pylist()
            all_moves = table["moves_uci"].to_pylist()

            for w_elo, b_elo, winner, moves in zip(white_elos, black_elos, winners, all_moves):
                if len(fen_to_sample) >= max_unique_positions:
                    break

                if (w_elo or 0) < min_elo and (b_elo or 0) < min_elo:
                    continue

                if not moves:
                    continue

                game.reset()
                limit_moves = min(len(moves), max_moves_per_game)

                for i in range(limit_moves):
                    if len(fen_to_sample) >= max_unique_positions:
                        break

                    uci_mv = moves[i]
                    if len(uci_mv) < 4:
                        break

                    try:
                        move = uci_to_move(uci_mv)
                        if move not in game.legal_moves():
                            break

                        action = ChessActionSpace.encode(move)
                        fen_key = StockfishInspiredSearchAgent.get_simple_fen(game)

                        # Outcome-based target value
                        state = game.get_state()
                        if winner == "white":
                            val = 1.0 if state["turn"] == "white" else 0.0
                        elif winner == "black":
                            val = 0.0 if state["turn"] == "white" else 1.0
                        else:
                            val = 0.5

                        # Deduplicate: if position already seen, prefer winning continuation
                        if fen_key not in fen_to_sample or (val == 1.0 and fen_to_sample[fen_key][2] < 1.0):
                            encoded = encoder.encode_flat(state)
                            fen_to_sample[fen_key] = (encoded, action, val)

                        game.make_move(move)
                    except Exception:
                        break

            pct = (len(fen_to_sample) / max_unique_positions) * 100
            elapsed = time.time() - t0
            print(f"\r[DEDUP] Unique Master positions: {len(fen_to_sample):,}/{max_unique_positions:,} ({pct:.1f}%) | {elapsed:.1f}s", end="", flush=True)

    print(f"\n[DATA] Successfully indexed {len(fen_to_sample):,} distinct 1-to-1 Grandmaster positions!")

    # Pack into tensors
    n = len(fen_to_sample)
    x_tensor = torch.zeros((n, 1152), dtype=torch.uint8)
    p_tensor = torch.zeros((n,), dtype=torch.long)
    v_tensor = torch.zeros((n, 1), dtype=torch.float32)

    for idx, (enc, act, val) in enumerate(fen_to_sample.values()):
        x_tensor[idx] = torch.tensor(enc, dtype=torch.uint8)
        p_tensor[idx] = act
        v_tensor[idx, 0] = val

    return x_tensor, p_tensor, v_tensor


def train_to_near_100_percent(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Training on: {device}")

    # 1. Collect unique 1-to-1 positions
    x_tensor, p_tensor, v_tensor = stream_deduplicated_gm_positions(
        max_unique_positions=args.unique_positions,
        min_elo=args.min_elo,
        max_moves_per_game=args.moves_per_game,
        start_group=args.start_group
    )

    dataset = TensorDataset(x_tensor, p_tensor, v_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))

    # 2. Model setup & warm-start
    model = OrionResNet().to(device)

    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"[RESUME] Warm-starting from {args.resume_checkpoint}...")
        if args.resume_checkpoint.endswith(".pt"):
            model.load_state_dict(torch.load(args.resume_checkpoint, map_location=device))
        else:
            load_json_to_model(model, args.resume_checkpoint)
        print("[RESUME] Warm-start weights loaded.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("      ORION ResNet PERFECT ACCURACY TARGET TRAINING")
    print("=" * 65)
    print(f"Unique Samples: {len(dataset):,}")
    print(f"Batch Size    : {args.batch_size}")
    print(f"Epochs        : {args.epochs}")
    print(f"Target Acc    : >= 95% - 99%+")
    print("=" * 65)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_p_loss = 0.0
        total_v_loss = 0.0
        correct_p = 0
        total_seen = 0

        t0 = time.time()
        for batch_x, batch_p, batch_v in loader:
            batch_x = batch_x.to(device, non_blocking=True).float()
            batch_p = batch_p.to(device, non_blocking=True)
            batch_v = batch_v.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, val = model(batch_x)

            loss_p = criterion_policy(logits, batch_p)
            loss_v = criterion_value(val, batch_v)
            loss = loss_p + 0.3 * loss_v

            loss.backward()
            optimizer.step()

            total_p_loss += loss_p.item() * len(batch_p)
            total_v_loss += loss_v.item() * len(batch_v)
            preds = logits.argmax(dim=1)
            correct_p += (preds == batch_p).sum().item()
            total_seen += len(batch_p)

        scheduler.step()
        elapsed = time.time() - t0
        avg_p_loss = total_p_loss / total_seen
        avg_v_loss = total_v_loss / total_seen
        accuracy = (correct_p / total_seen) * 100.0
        fps = total_seen / max(0.001, elapsed)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"P-Loss: {avg_p_loss:.4f} | V-Loss: {avg_v_loss:.4f} | "
            f"Acc: {accuracy:.2f}% | {fps:.0f} pos/s ({elapsed:.1f}s)",
            flush=True
        )

        if accuracy > best_acc:
            best_acc = accuracy
            out_json = os.path.join(args.output_dir, f"orion_gm_resnet_perfect_acc_{accuracy:.1f}pct.json")
            out_pt = os.path.join(args.output_dir, f"orion_gm_resnet_perfect_acc_{accuracy:.1f}pct.pt")
            model.export_to_orion_checkpoint(out_json, episode=epoch)
            torch.save(model.state_dict(), out_pt)

        # Early exit if >= 98%
        if accuracy >= 98.0:
            print(f"\n[TARGET REACHED] Policy Accuracy achieved {accuracy:.2f}% >= 98%!")
            break

    print("\n[SUCCESS] Precision training complete!")
    print(f"[CHECKPOINT] Best model ({best_acc:.2f}% Acc): {out_json}")


def main():
    parser = argparse.ArgumentParser(description="Train ORION to near 100% policy accuracy")
    parser.add_argument("--unique-positions", type=int, default=50000, help="Unique distinct positions to train")
    parser.add_argument("--min-elo", type=int, default=2400, help="Minimum player Elo rating")
    parser.add_argument("--moves-per-game", type=int, default=35, help="Max opening/middle moves per game")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs to drive convergence")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output-dir", default="checkpoints/gm_master", help="Output directory")
    parser.add_argument("--resume-checkpoint", default="checkpoints/gm_master/orion_gm_resnet_20ep_1000000_2m_gm_fine.pt")
    parser.add_argument("--start-group", type=int, default=150, help="Starting parquet row group")
    args = parser.parse_args()

    train_to_near_100_percent(args)


if __name__ == "__main__":
    main()
