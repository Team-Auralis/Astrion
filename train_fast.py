#!/usr/bin/env python3
"""
High-Speed Vectorized PyTorch Supervised Training for ORION
ResNet architecture with dual-head policy+value training, LR scheduling,
and full Lichess dataset support. Exports to native ORION JSON format
with batch-norm fusion for pure-Python inference.
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from orion.environment.chess import ChessEnvironment
from orion.model.actions import ChessActionSpace
from orion.model.encoder import ChessEncoder
from orion.utils.checkpoint import Checkpoint


def uci_to_move(uci: str) -> Tuple[int, int, int, int]:
    if len(uci) < 4:
        raise ValueError(f"Invalid UCI move: {uci}")
    return (
        8 - int(uci[1]),
        ord(uci[0]) - ord("a"),
        8 - int(uci[3]),
        ord(uci[2]) - ord("a"),
    )


CHANNELS = 256
NUM_BLOCKS = 6
INPUT_SIZE = 18 * 8 * 8
POLICY_SIZE = 4096


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.fc2 = nn.Linear(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out = F.relu(out + identity)
        return out


class OrionResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_fc = nn.Linear(INPUT_SIZE, CHANNELS)
        self.input_bn = nn.BatchNorm1d(CHANNELS)
        self.blocks = nn.ModuleList([ResBlock(CHANNELS) for _ in range(NUM_BLOCKS)])

        self.policy_fc1 = nn.Linear(CHANNELS, 512)
        self.policy_bn = nn.BatchNorm1d(512)
        self.policy_fc2 = nn.Linear(512, POLICY_SIZE)

        self.value_fc1 = nn.Linear(CHANNELS, 64)
        self.value_bn = nn.BatchNorm1d(64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = F.relu(self.input_bn(self.input_fc(x)))
        for block in self.blocks:
            out = block(out)

        policy_logits = self.policy_fc2(F.relu(self.policy_bn(self.policy_fc1(out))))
        value = torch.sigmoid(self.value_fc2(F.relu(self.value_bn(self.value_fc1(out)))))
        return policy_logits, value

    def export_to_orion_checkpoint(self, path: str, episode: int):
        """Export PyTorch weights with BN fusion into native ORION JSON format."""
        with torch.no_grad():
            w_in, b_in = self._fuse_bn(self.input_fc, self.input_bn)
            block_wa, block_ba, block_wb, block_bb = [], [], [], []
            for block in self.blocks:
                wa, ba = self._fuse_bn(block.fc1, block.bn1)
                wb, bb = self._fuse_bn(block.fc2, block.bn2)
                block_wa.append(wa)
                block_ba.append(ba)
                block_wb.append(wb)
                block_bb.append(bb)

            pw1, pb1 = self._fuse_bn(self.policy_fc1, self.policy_bn)
            pw2 = self.policy_fc2.weight.cpu().numpy().tolist()
            pb2 = self.policy_fc2.bias.cpu().numpy().tolist()

            vw1, vb1 = self._fuse_bn(self.value_fc1, self.value_bn)
            vw2 = self.value_fc2.weight.squeeze(0).cpu().numpy().tolist()
            vb2 = float(self.value_fc2.bias.squeeze().item())

        data = {
            "episode": episode,
            "arch": "resnet",
            "channels": CHANNELS,
            "num_blocks": NUM_BLOCKS,
            "input_size": INPUT_SIZE,
            "input_w": w_in,
            "input_b": b_in,
            "block_wa": block_wa,
            "block_ba": block_ba,
            "block_wb": block_wb,
            "block_bb": block_bb,
            "policy_w1": pw1,
            "policy_b1": pb1,
            "policy_w2": pw2,
            "policy_b2": pb2,
            "value_w1": vw1,
            "value_b1": vb1,
            "value_w2": vw2,
            "value_b2": vb2,
        }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[CHECKPOINT] Exported to: {path}")

    @staticmethod
    def _fuse_bn(fc, bn):
        """Fuse BatchNorm into preceding Linear layer for inference."""
        w = fc.weight.data
        b = fc.bias.data
        gamma = bn.weight.data
        beta = bn.bias.data
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        scale = gamma / torch.sqrt(var + eps)
        fused_w = w * scale.unsqueeze(1)
        fused_b = (b - mean) * scale + beta

        return fused_w.cpu().numpy().tolist(), fused_b.cpu().numpy().tolist()


class ChessPuzzleDataset(Dataset):
    def __init__(self, inputs: List[List[float]], policy_targets: List[int],
                 value_targets: List[float]):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.policy_targets = torch.tensor(policy_targets, dtype=torch.long)
        self.value_targets = torch.tensor(value_targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.policy_targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.policy_targets[idx], self.value_targets[idx]


def estimate_material_value(state):
    """Estimate position value from material balance for value head training."""
    values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0,
              "p": -1, "n": -3, "b": -3, "r": -5, "q": -9, "k": 0}
    score = 0
    board = state["board"]
    for row in board:
        for piece in row:
            score += values.get(piece, 0)
    normalized = max(-9.0, min(9.0, score)) / 9.0
    return (normalized + 1.0) / 2.0


def load_puzzles_from_dir(data_dir: str, max_puzzles: int = 50000):
    encoder = ChessEncoder()
    game = ChessEnvironment()

    inputs, policy_targets, value_targets = [], [], []

    files = glob.glob(os.path.join(data_dir, "lichess_*.json"))
    files.sort(key=lambda f: os.path.getsize(f), reverse=True)
    print(f"[DATA] Found {len(files)} JSON dataset files in {data_dir}...")

    total_loaded = 0
    for fpath in files:
        if total_loaded >= max_puzzles:
            break
        try:
            with open(fpath, "r") as f:
                content = json.load(f)
            rows = content.get("rows", [])
            for item in rows:
                if total_loaded >= max_puzzles:
                    break
                row = item.get("row", item)
                fen = row.get("FEN")
                moves_str = row.get("Moves", "")
                if not fen or not moves_str:
                    continue
                moves = moves_str.split()
                if not moves:
                    continue
                target_uci = moves[0]
                if len(target_uci) != 4:
                    continue

                try:
                    game.load_fen(fen)
                    target_move = uci_to_move(target_uci)
                    if target_move not in game.legal_moves():
                        continue
                    action = ChessActionSpace.encode(target_move)
                    state = game.get_state()
                    encoded = encoder.encode_flat(state)
                    val = estimate_material_value(state)
                    inputs.append(encoded)
                    policy_targets.append(action)
                    value_targets.append(val)
                    total_loaded += 1
                except Exception:
                    continue
        except Exception as e:
            print(f"[WARN] Error reading {fpath}: {e}")

    print(f"[DATA] Successfully pre-encoded {len(policy_targets)} valid chess puzzles.")
    return inputs, policy_targets, value_targets


def load_puzzles_from_hf(max_puzzles: int = 50000):
    from datasets import load_dataset
    encoder = ChessEncoder()
    game = ChessEnvironment()

    inputs, policy_targets, value_targets = [], [], []
    print(f"[DATASET] Loading from full Lichess dataset (6,100,960 puzzles available)...")
    ds = load_dataset("Lichess/chess-puzzles", split="train")

    total_to_load = min(len(ds), max_puzzles)
    print(f"[DATASET] Pre-encoding {total_to_load} puzzles into spatial tensors...")

    for idx in range(total_to_load):
        item = ds[idx]
        row = item.get("row", item)
        fen = row.get("FEN")
        moves_str = row.get("Moves", "")
        if not fen or not moves_str:
            continue
        moves = moves_str.split()
        if not moves:
            continue
        target_uci = moves[0]
        if len(target_uci) < 4:
            continue
        try:
            game.load_fen(fen)
            target_move = uci_to_move(target_uci)
            if target_move not in game.legal_moves():
                continue
            action = ChessActionSpace.encode(target_move)
            state = game.get_state()
            encoded = encoder.encode_flat(state)
            val = estimate_material_value(state)
            inputs.append(encoded)
            policy_targets.append(action)
            value_targets.append(val)
            if len(policy_targets) % 10000 == 0:
                print(f"[DATASET] Pre-encoded {len(policy_targets)}/{total_to_load} puzzles...")
        except Exception:
            continue

    print(f"[DATASET] Successfully loaded {len(policy_targets)} valid puzzles.")
    return inputs, policy_targets, value_targets


def main():
    parser = argparse.ArgumentParser(description="Fast ResNet Training for ORION.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-puzzles", type=int, default=500000)
    parser.add_argument("--checkpoint-dir", default="checkpoints/resnet_v1")
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    if args.use_hf:
        inputs, policy_targets, value_targets = load_puzzles_from_hf(args.max_puzzles)
    else:
        inputs, policy_targets, value_targets = load_puzzles_from_dir(args.data_dir, args.max_puzzles)

    if not policy_targets:
        print("[ERROR] No training samples available.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Training on: {device}")

    dataset = ChessPuzzleDataset(inputs, policy_targets, value_targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = OrionResNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] ResNet: {total_params:,} parameters")

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print("================================================================")
    print("           ORION ResNet DUAL-HEAD TORCH TRAINING                ")
    print("================================================================")
    print(f"Total Samples : {len(dataset):,}")
    print(f"Batch Size    : {args.batch_size}")
    print(f"Epochs        : {args.epochs}")
    print(f"Learning Rate : {args.lr}")
    print(f"Weight Decay  : {args.weight_decay}")
    print(f"Policy Weight : {args.policy_weight}")
    print(f"Value Weight  : {args.value_weight}")
    print(f"Parameters    : {total_params:,}")
    print("================================================================\n")

    best_policy_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_combined_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for batch_x, batch_policy_y, batch_value_y in loader:
            batch_x = batch_x.to(device)
            batch_policy_y = batch_policy_y.to(device)
            batch_value_y = batch_value_y.to(device)

            optimizer.zero_grad()
            policy_logits, value_pred = model(batch_x)

            p_loss = policy_criterion(policy_logits, batch_policy_y)
            v_loss = value_criterion(value_pred, batch_value_y)
            loss = args.policy_weight * p_loss + args.value_weight * v_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_policy_loss += p_loss.item() * len(batch_policy_y)
            total_value_loss += v_loss.item() * len(batch_policy_y)
            total_combined_loss += loss.item() * len(batch_policy_y)

            preds = torch.argmax(policy_logits, dim=1)
            correct += (preds == batch_policy_y).sum().item()
            total += len(batch_policy_y)

        scheduler.step()

        elapsed = time.time() - t0
        avg_p_loss = total_policy_loss / total
        avg_v_loss = total_value_loss / total
        avg_loss = total_combined_loss / total
        accuracy = (correct / total) * 100.0
        speed = total / elapsed if elapsed > 0 else 0
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"P-Loss: {avg_p_loss:.4f} | V-Loss: {avg_v_loss:.4f} | "
              f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | "
              f"LR: {lr:.2e} | {speed:.0f} samples/s ({elapsed:.1f}s)")

        if accuracy > best_policy_acc:
            best_policy_acc = accuracy

        if epoch % args.save_every == 0 or epoch == args.epochs:
            out_path = os.path.join(args.checkpoint_dir, f"orion_resnet_{epoch}ep_{len(dataset)}.json")
            model.export_to_orion_checkpoint(out_path, episode=len(dataset) * epoch)

    print(f"\n[SUCCESS] Training complete! Best accuracy: {best_policy_acc:.2f}%")
    print(f"[HINT] Run: python cli.py battle --opponent minimax --rounds 6 --checkpoint {args.checkpoint_dir}/orion_resnet_{args.epochs}ep_{len(dataset)}.json")


if __name__ == "__main__":
    main()
