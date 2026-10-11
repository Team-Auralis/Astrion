#!/usr/bin/env python3
"""Audit baseline model architecture, checkpoint hash, and tournament PGN results."""

import os
import sys
import hashlib
import json
import torch
import chess.pgn

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("experiments/calibrated_eval"))

# 1. Hash Checkpoint
ckpt_path = "checkpoints/ccrl_30b/orion_30b_master_final_master.json"
with open(ckpt_path, "rb") as f:
    ckpt_hash = hashlib.sha256(f.read()).hexdigest()
size_bytes = os.path.getsize(ckpt_path)
print(f"Checkpoint: {ckpt_path}")
print(f"Size: {size_bytes:,} bytes (~{size_bytes / (1024 * 1024):.2f} MB)")
print(f"SHA-256: {ckpt_hash}")

# 2. Inspect Architecture
from train_fast import OrionResNet
from train_gm_dataset import load_json_to_model

model = OrionResNet(channels=512, num_blocks=30)
load_json_to_model(model, ckpt_path)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal Model Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

# Layer breakdown
blocks_count = len(model.blocks)
print(f"Residual Blocks Count: {blocks_count}")
print(f"Input Linear: {model.input_fc}")
print(f"Policy Head: {model.policy_fc1} -> {model.policy_fc2}")
print(f"Value Head: {model.value_fc1} -> {model.value_fc2}")

# 3. Audit PGN Game Records
pgn_path = "experiments/calibrated_eval/full_tournament.pgn"
records = {}
games_count = 0
games_list = []

with open(pgn_path, "r", encoding="utf-8") as f:
    while True:
        g = chess.pgn.read_game(f)
        if not g:
            break
        games_count += 1
        w = g.headers["White"]
        b = g.headers["Black"]
        res = g.headers["Result"]
        games_list.append({"white": w, "black": b, "result": res})
        for p in (w, b):
            if p not in records:
                records[p] = {"wins": 0, "losses": 0, "draws": 0, "pts": 0.0, "games": 0}
            records[p]["games"] += 1
        if res == "1-0":
            records[w]["wins"] += 1
            records[w]["pts"] += 1.0
            records[b]["losses"] += 1
        elif res == "0-1":
            records[b]["wins"] += 1
            records[b]["pts"] += 1.0
            records[w]["losses"] += 1
        elif res == "1/2-1/2":
            records[w]["draws"] += 1
            records[w]["pts"] += 0.5
            records[b]["draws"] += 1
            records[b]["pts"] += 0.5

print(f"\nParsed {games_count} games from {pgn_path}:")
for p, s in sorted(records.items(), key=lambda x: x[1]["pts"], reverse=True):
    pct = (s["pts"] / s["games"]) * 100 if s["games"] > 0 else 0
    w_cnt = s["wins"]
    l_cnt = s["losses"]
    d_cnt = s["draws"]
    pts = s["pts"]
    g_cnt = s["games"]
    print(f"{p:<22}: {w_cnt}W - {l_cnt}L - {d_cnt}D ({pts:.1f}/{g_cnt}, {pct:.1f}%)")

# Save baseline_manifest.json
manifest = {
    "checkpoint": {
        "path": ckpt_path,
        "sha256": ckpt_hash,
        "size_bytes": size_bytes,
        "parameters": total_params,
        "residual_blocks": blocks_count,
        "channels": 512,
        "input_features": 1152,
        "policy_outputs": 4096,
        "value_outputs": 1
    },
    "tournament_audit": {
        "pgn_path": pgn_path,
        "total_games": games_count,
        "records": records
    }
}

os.makedirs("elo_test", exist_ok=True)
with open("elo_test/baseline_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\nSaved baseline manifest to elo_test/baseline_manifest.json")
