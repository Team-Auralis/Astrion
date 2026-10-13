#!/usr/bin/env python3
import os
import chess
import chess.pgn
import subprocess
import torch
import numpy as np
from train_ccrl_full_corpus import fast_encode_board, fast_encode_action, fast_material_value, compute_metrics
from train_fast import OrionResNet
from train_gm_dataset import load_json_to_model
from cli import find_best_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OrionResNet().to(device)
best_ckpt = find_best_checkpoint()
print(f"Loading checkpoint: {best_ckpt}")
load_json_to_model(model, best_ckpt)
model.eval()

# Sample 1000 fresh test positions from CCRL
proc = subprocess.Popen(
    ['tar', '-xOf', 'D:/orion-game-ai/CCRL-4040.[2463682].pgn.7z'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding='latin1',
    bufsize=2097152
)
buf_board = np.zeros((18, 8, 8), dtype=np.uint8)

x_list, p_list, v_list = [], [], []
count = 0
games_seen = 0
while count < 1000:
    g = chess.pgn.read_game(proc.stdout)
    if not g:
        break
    games_seen += 1
    res = g.headers.get('Result', '*')
    if res not in ('1-0', '0-1', '1/2-1/2'):
        continue
    w_out = 1.0 if res == '1-0' else (0.0 if res == '0-1' else 0.5)
    b = g.board()
    ply = 0
    for m in g.mainline_moves():
        ply += 1
        if ply > 12 and (ply % 3 == 0) and count < 1000:
            fast_encode_board(b, buf_board)
            act = fast_encode_action(m)
            turn_out = w_out if b.turn == chess.WHITE else (1.0 - w_out)
            t_val = 0.85 * turn_out + 0.15 * fast_material_value(b)
            x_list.append(torch.from_numpy(buf_board.flatten()))
            p_list.append(act)
            v_list.append(t_val)
            count += 1
        b.push(m)
proc.terminate()

x_t = torch.stack(x_list).to(device).float()
p_t = torch.tensor(p_list, dtype=torch.long).to(device)
v_t = torch.tensor(v_list, dtype=torch.float32).unsqueeze(1).to(device)

checkpoints = [
    "checkpoints/ccrl_master/orion_ccrl_deep_master.json",
    "checkpoints/ccrl_master/orion_ccrl_full_corpus_best.json",
    "checkpoints/ccrl_master/orion_ccrl_full_corpus_latest.json",
    "checkpoints/distilled/orion_distill_master.json"
]

print("=" * 75)
print(f"BENCHMARK COMPARISON ON {len(p_t):,} FRESH CCRL TEST POSITIONS")
print("=" * 75)
for ckpt in checkpoints:
    if not os.path.exists(ckpt):
        continue
    m_test = OrionResNet().to(device)
    load_json_to_model(m_test, ckpt)
    m_test.eval()
    with torch.no_grad():
        p_logits, v_pred = m_test(x_t)
        res = compute_metrics(p_logits, p_t, v_pred, v_t)
    name = os.path.basename(ckpt)
    print(f"{name:38s} | Top-1: {res['top1']:5.2f}% | Top-3: {res['top3']:5.2f}% | Top-5: {res['top5']:5.2f}% | Val: {res['value']:5.2f}% | Consensus: {res['consensus']:5.2f}%")
print("=" * 75)
