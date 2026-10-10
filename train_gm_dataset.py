#!/usr/bin/env python3
"""
Stream and train ORION ResNet on high-level master games from angeluriot/chess_games.
Filter for players with Elo >= 2300 (Grandmasters/Masters).
Trains the Policy and Value networks on real tournament games directly on GPU.
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

import fsspec
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from orion.environment.chess import ChessEnvironment
from orion.model.actions import ChessActionSpace
from orion.model.encoder import ChessEncoder
from orion.utils.checkpoint import Checkpoint
from train_fast import OrionResNet, uci_to_move, estimate_material_value

DATASET_URL = "https://huggingface.co/datasets/angeluriot/chess_games/resolve/main/dataset.parquet"


def stream_gm_positions(max_positions: int = 1000000, min_elo: int = 2200, max_moves_per_game: int = 40, start_group: int = 0):
    print(f"[STREAM] Connecting to angeluriot/chess_games over HTTP...")
    print(f"[STREAM] Filtering: min_elo >= {min_elo}, sampling first {max_moves_per_game} moves/game, starting at group {start_group}...")

    encoder = ChessEncoder()
    game = ChessEnvironment()

    # Pre-allocate compact memory tensors (1.07 GB per 1M positions)
    x_tensor = torch.zeros((max_positions, 1152), dtype=torch.uint8)
    p_tensor = torch.zeros((max_positions,), dtype=torch.long)
    v_tensor = torch.zeros((max_positions, 1), dtype=torch.float32)

    with fsspec.open(DATASET_URL, mode="rb") as f:
        pfile = pq.ParquetFile(f)
        total_groups = pfile.num_row_groups
        print(f"[STREAM] Connected! Total row groups in dataset: {total_groups} (~14.2M games)")

        positions_collected = 0
        group_idx = start_group

        t0 = time.time()
        while positions_collected < max_positions and group_idx < total_groups:
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
                if positions_collected >= max_positions:
                    break

                # Filter by master level rating
                if (w_elo or 0) < min_elo and (b_elo or 0) < min_elo:
                    continue

                if not moves:
                    continue

                game.reset()
                # Play through the game moves and collect training samples
                limit_moves = min(len(moves), max_moves_per_game)
                for i in range(limit_moves):
                    if positions_collected >= max_positions:
                        break

                    uci_mv = moves[i]
                    if len(uci_mv) < 4:
                        break

                    try:
                        move = uci_to_move(uci_mv)
                        if move not in game.legal_moves():
                            break

                        # Target action
                        action = ChessActionSpace.encode(move)
                        state = game.get_state()
                        encoded = encoder.encode_flat(state)

                        # Outcome-based target value
                        if winner == "white":
                            val = 1.0 if state["turn"] == "white" else 0.0
                        elif winner == "black":
                            val = 0.0 if state["turn"] == "white" else 1.0
                        else:
                            val = 0.5

                        x_tensor[positions_collected] = torch.tensor(encoded, dtype=torch.uint8)
                        p_tensor[positions_collected] = action
                        v_tensor[positions_collected, 0] = val
                        positions_collected += 1

                        game.make_move(move)
                    except Exception:
                        break

            pct = (positions_collected / max_positions) * 100
            elapsed = time.time() - t0
            print(f"\r[DATA] Pre-encoding GM positions: {positions_collected:,}/{max_positions:,} ({pct:.1f}%) | {elapsed:.1f}s", end="", flush=True)

    if positions_collected < max_positions:
        x_tensor = x_tensor[:positions_collected]
        p_tensor = p_tensor[:positions_collected]
        v_tensor = v_tensor[:positions_collected]

    print(f"\n[DATA] Successfully collected {len(p_tensor):,} high-Elo tournament positions!")
    return x_tensor, p_tensor, v_tensor


def load_json_to_model(model: OrionResNet, json_path: str):
    import json
    with open(json_path, "r") as f:
        data = json.load(f)

    with torch.no_grad():
        model.input_fc.weight.copy_(torch.tensor(data["input_w"], dtype=torch.float32))
        model.input_fc.bias.copy_(torch.tensor(data["input_b"], dtype=torch.float32))
        model.input_bn.weight.fill_(1.0)
        model.input_bn.bias.fill_(0.0)
        model.input_bn.running_mean.fill_(0.0)
        model.input_bn.running_var.fill_(1.0)

        for i, block in enumerate(model.blocks):
            block.fc1.weight.copy_(torch.tensor(data["block_wa"][i], dtype=torch.float32))
            block.fc1.bias.copy_(torch.tensor(data["block_ba"][i], dtype=torch.float32))
            block.bn1.weight.fill_(1.0)
            block.bn1.bias.fill_(0.0)
            block.bn1.running_mean.fill_(0.0)
            block.bn1.running_var.fill_(1.0)

            block.fc2.weight.copy_(torch.tensor(data["block_wb"][i], dtype=torch.float32))
            block.fc2.bias.copy_(torch.tensor(data["block_bb"][i], dtype=torch.float32))
            block.bn2.weight.fill_(1.0)
            block.bn2.bias.fill_(0.0)
            block.bn2.running_mean.fill_(0.0)
            block.bn2.running_var.fill_(1.0)

        model.policy_fc1.weight.copy_(torch.tensor(data["policy_w1"], dtype=torch.float32))
        model.policy_fc1.bias.copy_(torch.tensor(data["policy_b1"], dtype=torch.float32))
        model.policy_bn.weight.fill_(1.0)
        model.policy_bn.bias.fill_(0.0)
        model.policy_bn.running_mean.fill_(0.0)
        model.policy_bn.running_var.fill_(1.0)

        model.policy_fc2.weight.copy_(torch.tensor(data["policy_w2"], dtype=torch.float32))
        model.policy_fc2.bias.copy_(torch.tensor(data["policy_b2"], dtype=torch.float32))

        model.value_fc1.weight.copy_(torch.tensor(data["value_w1"], dtype=torch.float32))
        model.value_fc1.bias.copy_(torch.tensor(data["value_b1"], dtype=torch.float32))
        model.value_bn.weight.fill_(1.0)
        model.value_bn.bias.fill_(0.0)
        model.value_bn.running_mean.fill_(0.0)
        model.value_bn.running_var.fill_(1.0)

        model.value_fc2.weight.copy_(torch.tensor([data["value_w2"]], dtype=torch.float32))
        model.value_fc2.bias.copy_(torch.tensor([data["value_b2"]], dtype=torch.float32))
    print(f"[RESUME] Successfully loaded warm-start weights from {json_path}")


def train_on_gm_data(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Training on: {device}")

    # 1. Stream positions
    x_tensor, p_tensor, v_tensor = stream_gm_positions(
        max_positions=args.max_positions,
        min_elo=args.min_elo,
        max_moves_per_game=args.moves_per_game,
        start_group=args.start_group
    )

    dataset = TensorDataset(x_tensor, p_tensor, v_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))

    # 3. Model setup - either warm start from existing checkpoint or new
    model = OrionResNet().to(device)

    # Warm-start if requested
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"[RESUME] Loading previous weights from {args.resume_checkpoint}...")
        try:
            if args.resume_checkpoint.endswith(".pt"):
                model.load_state_dict(torch.load(args.resume_checkpoint, map_location=device))
                print(f"[RESUME] Checkpoint {args.resume_checkpoint} loaded via PyTorch state dict.")
            else:
                load_json_to_model(model, args.resume_checkpoint)
        except Exception as e:
            print(f"[WARN] Could not warm-start: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("      ORION ResNet MASTER GM TOURNAMENT TRAINING")
    print("=" * 65)
    print(f"Dataset      : angeluriot/chess_games (Elo >= {args.min_elo})")
    print(f"Total Samples: {len(dataset):,}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Epochs       : {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 65)

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
            loss = loss_p + 0.5 * loss_v

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
            f"Acc: {accuracy:.2f}% | {fps:.0f} pos/s ({elapsed:.1f}s)"
        )

        tag = f"_{args.tag}" if args.tag else ""
        out_path = os.path.join(args.output_dir, f"orion_gm_resnet_{epoch}ep_{args.max_positions}{tag}.json")
        model.export_to_orion_checkpoint(out_path, episode=epoch * args.max_positions)
        pt_path = os.path.join(args.output_dir, f"orion_gm_resnet_{epoch}ep_{args.max_positions}{tag}.pt")
        torch.save(model.state_dict(), pt_path)

    print("\n[SUCCESS] Master GM training finished!")
    print(f"[CHECKPOINT] Best model saved at: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ORION on Grandmaster dataset")
    parser.add_argument("--max-positions", type=int, default=100000, help="Max positions to extract")
    parser.add_argument("--min-elo", type=int, default=2200, help="Minimum player Elo rating")
    parser.add_argument("--moves-per-game", type=int, default=35, help="Max opening/middle moves per game")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--output-dir", default="checkpoints/gm_master", help="Output directory")
    parser.add_argument("--resume-checkpoint", default=None, help="Resume from previous checkpoint")
    parser.add_argument("--start-group", type=int, default=0, help="Starting parquet row group")
    parser.add_argument("--tag", default=None, help="Tag for checkpoint filename")
    args = parser.parse_args()

    train_on_gm_data(args)


if __name__ == "__main__":
    main()
