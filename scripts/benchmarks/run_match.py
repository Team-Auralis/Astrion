#!/usr/bin/env python3
import subprocess
import sys
import os

cutechess_path = r"C:\Users\shaur\AppData\Local\Programs\Cute Chess\cutechess-cli.exe"

if not os.path.exists(cutechess_path):
    print(f"[ERROR] cutechess-cli not found at {cutechess_path}")
    sys.exit(1)

# Default options
rounds = "10"
extra_args = []
skip_next = False
for i, arg in enumerate(sys.argv[1:]):
    if skip_next:
        skip_next = False
        continue
    if arg in ("-rounds", "--rounds") and i + 1 < len(sys.argv[1:]):
        rounds = sys.argv[1:][i + 1]
        skip_next = True
    else:
        extra_args.append(arg)

cmd = [
    cutechess_path,
    "-engine", "name=ORION", "cmd=D:/orion-game-ai/orion_engine.bat", "dir=D:/orion-game-ai", "proto=uci",
    "-engine", "name=Stockfish", "cmd=D:/orion-game-ai/stockfish.exe", "proto=uci",
    "-each", "tc=40/60+1",
    "-rounds", str(rounds),
    "-pgnout", "results.pgn"
] + extra_args

print("Starting Cute Chess match: ORION vs Stockfish...")
print("Command:", " ".join(cmd))
print("-" * 65)

try:
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)
except KeyboardInterrupt:
    print("\nMatch aborted by user.")
