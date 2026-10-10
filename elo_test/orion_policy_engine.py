#!/usr/bin/env python3
"""
ORION Standalone Policy-Only UCI Engine (Test A)
Strictly evaluates the neural network policy head:
- Zero tree search
- Zero alpha-beta / minimax
- Zero opening books
- Selects the legal move with the highest policy probability.
Compliant with UCI standard protocol.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from orion.environment.chess import ChessEnvironment
from orion.model.actions import ChessActionSpace
from orion.model.encoder import ChessEncoder
from train_fast import OrionResNet
from train_gm_dataset import load_json_to_model

CHECKPOINT_PATH = r"D:\orion-game-ai\checkpoints\distilled\orion_distill_5ep.json"


def uci_to_env_move(uci_str: str):
    """Convert UCI string (e.g. e2e4 or e7e8q) to (r1, c1, r2, c2)."""
    files = "abcdefgh"
    c1 = files.index(uci_str[0])
    r1 = 8 - int(uci_str[1])
    c2 = files.index(uci_str[2])
    r2 = 8 - int(uci_str[3])
    return (r1, c1, r2, c2)


def env_move_to_uci(env: ChessEnvironment, move):
    """Convert (r1, c1, r2, c2) to proper UCI with promotion suffix if applicable."""
    r1, c1, r2, c2 = move
    files = "abcdefgh"
    base = files[c1] + str(8 - r1) + files[c2] + str(8 - r2)
    piece = env.board[r1][c1]
    if piece.lower() == "p" and (r2 == 0 or r2 == 7):
        return base + "q"
    return base


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrionResNet().to(device)

    ckpt = CHECKPOINT_PATH
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        ckpt = sys.argv[1]

    if not os.path.exists(ckpt):
        sys.stderr.write(f"[ERROR] Checkpoint not found: {ckpt}\n")
        sys.exit(1)

    load_json_to_model(model, ckpt)
    model.eval()

    encoder = ChessEncoder()
    env = ChessEnvironment()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            cmd = parts[0]

            if cmd == "uci":
                print("id name ORION Policy-Only (No Search)")
                print("id author Team-Auralis")
                print(f"option name Checkpoint type string default {ckpt}")
                print("uciok")
                sys.stdout.flush()

            elif cmd == "isready":
                print("readyok")
                sys.stdout.flush()

            elif cmd == "ucinewgame":
                env = ChessEnvironment()

            elif cmd == "position":
                env = ChessEnvironment()
                moves_idx = -1
                if parts[1] == "startpos":
                    if "moves" in parts:
                        moves_idx = parts.index("moves")
                elif parts[1] == "fen":
                    fen_tokens = []
                    idx = 2
                    while idx < len(parts) and parts[idx] != "moves":
                        fen_tokens.append(parts[idx])
                        idx += 1
                    try:
                        env.load_fen(" ".join(fen_tokens))
                    except Exception:
                        env = ChessEnvironment()
                    if idx < len(parts) and parts[idx] == "moves":
                        moves_idx = idx

                if moves_idx != -1:
                    for m in parts[moves_idx + 1:]:
                        if len(m) >= 4:
                            env.make_move(uci_to_env_move(m[:4]))

            elif cmd == "go":
                legal = env.legal_moves()
                if not legal:
                    print("bestmove 0000")
                    sys.stdout.flush()
                    continue

                state = env.get_state()
                flat = encoder.encode_flat(state)
                x = torch.tensor([flat], dtype=torch.float32, device=device)

                with torch.no_grad():
                    logits, _ = model(x)
                    policy = logits[0].cpu().numpy()

                # Rank legal moves strictly by policy score
                ranked = []
                for m in legal:
                    act = ChessActionSpace.encode(m)
                    score = float(policy[act])
                    ranked.append((score, m))

                ranked.sort(key=lambda item: item[0], reverse=True)
                best_m = ranked[0][1]
                uci_mv = env_move_to_uci(env, best_m)

                print(f"bestmove {uci_mv}")
                sys.stdout.flush()

            elif cmd == "stop":
                pass

            elif cmd == "quit":
                break

        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            sys.stderr.write(f"[POLICY UCI ERROR] {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
