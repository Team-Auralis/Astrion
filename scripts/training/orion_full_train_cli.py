import argparse
import os
import sys
from typing import Dict, Iterable

import torch
from datasets import load_dataset
from tqdm import tqdm

from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.training.reinforcement import ReinforcementTrainer
from orion.utils.checkpoint import Checkpoint


def uci_to_move(uci: str):
    """Convert a UCI string like 'e2e4' or 'h7h8q' to a move tuple (r1, c1, r2, c2)."""
    if len(uci) < 4:
        raise ValueError(f"Invalid UCI move: {uci}")
    return (
        8 - int(uci[1]),
        ord(uci[0]) - ord("a"),
        8 - int(uci[3]),
        ord(uci[2]) - ord("a"),
    )


def move_to_action(move):
    r1, c1, r2, c2 = move
    from_sq = r1 * 8 + c1
    to_sq = r2 * 8 + c2
    return from_sq * 64 + to_sq


def validate_puzzle(row: Dict) -> tuple:
    """Validate a single puzzle row and return environment and target info."""
    game = ChessEnvironment()
    game.load_fen(row["FEN"])
    moves = row["Moves"].split()
    if not moves:
        raise ValueError("Puzzle has no moves")
    target_uci = moves[0]
    target_move = uci_to_move(target_uci)
    legal = game.legal_moves()
    if target_move not in legal:
        raise ValueError(f"Target move {target_uci} is not legal for the given FEN")
    target_action = move_to_action(target_move)
    return game, target_move, target_action, legal


def stream_hf_rows(dataset_name: str) -> Iterable[Dict]:
    """Yield rows from a HuggingFace Lichess dataset using streaming mode."""
    ds = load_dataset(dataset_name, split="train", streaming=True)
    for example in ds:
        # Some datasets wrap under "row"
        if "row" in example:
            example = example["row"]
        yield {"FEN": example["FEN"], "Moves": example["Moves"]}


def aggregate_rows(selected: list) -> Iterable[Dict]:
    mapping = {
        "puzzles": "Lichess/chess-puzzles",
        "openings": "Lichess/chess-openings",
        "antichess": "Lichess/antichess-chess-openings",
    }
    for name in selected:
        hf_name = mapping[name]
        print(f"Streaming {hf_name} ...")
        for row in stream_hf_rows(hf_name):
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ORION on full Lichess HF datasets with streaming.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=["puzzles", "openings", "antichess"],
        default=["puzzles", "openings", "antichess"],
        help="Which HuggingFace Lichess datasets to include",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/resnet_v1",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of puzzles to train on (optional)",
    )
    args = parser.parse_args()

    rows_iter = aggregate_rows(args.datasets)
    if args.max_examples is not None:
        rows_iter = (row for i, row in enumerate(rows_iter) if i < args.max_examples)

    print("========================================")
    print("ORION FULL HF SUPERVISED TRAINING (streaming)")
    print("========================================")
    print(f"Datasets   : {', '.join(args.datasets)}")
    if args.max_examples:
        print(f"Training on first {args.max_examples} puzzles")
    print()

    encoder = ChessEncoder()
    network = ORIONNetwork()
    trainer = ReinforcementTrainer(network)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    total_loss = 0.0
    valid = 0
    skipped = 0

    # tqdm for a nice progress bar (no total length when streaming)
    for idx, row in enumerate(tqdm(rows_iter, desc="Training puzzles"), start=1):
        try:
            game, target_move, target_action, legal_moves = validate_puzzle(row)
            state = game.get_state()
            encoded = encoder.encode_flat(state)
            result = network.forward(encoded, return_cache=True)
            cache = result["cache"]
            loss = trainer.train_supervised_sample(
                cache=cache,
                target_action=target_action,
                legal_actions=[move_to_action(m) for m in legal_moves],
            )
            total_loss += loss["policy_loss"]
            valid += 1
        except Exception as exc:
            skipped += 1
            # minimal output for skips to keep speed
            if idx % 1000 == 0:
                print(f"[SKIP] Puzzle {idx}: {exc}")

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================")
    print(f"Valid puzzles : {valid}")
    print(f"Skipped       : {skipped}")
    if valid:
        print(f"Average policy loss : {total_loss / valid:.6f}")
    else:
        print("[FAIL] No puzzles were successfully trained.")
        sys.exit(1)

    checkpoint = Checkpoint(args.checkpoint_dir)
    checkpoint_path = checkpoint.save(network, episode=valid)
    print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    main()
