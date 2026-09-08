#!/usr/bin/env python3
"""
================================================================================
                    ORION Gaming Mind - Unified Master CLI
                       Developed by Team-Auralis
================================================================================
Single command entry point for training, evaluating, testing against real bots,
playing interactive chess, and launching UCI mode.
"""

import argparse
import os
import sys

from orion.environment.chess import ChessEnvironment
from orion.model.actions import ChessActionSpace
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.search.search_agent import StockfishInspiredSearchAgent
from orion.utils.checkpoint import Checkpoint
from orion.arena import OrionBot, RandomBot, GreedyMaterialBot, MinimaxBot, StockfishBot, Arena


def load_network_checkpoint(network: ORIONNetwork, checkpoint_path: str = "checkpoints/v0.5_supervised/orion_1000.json"):
    if os.path.exists(checkpoint_path):
        try:
            ep = Checkpoint().load(network, checkpoint_path)
            print(f"[CHECKPOINT] Successfully loaded model from {checkpoint_path} (ep: {ep})")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint {checkpoint_path}: {e}")
    else:
        print(f"[INFO] No checkpoint at {checkpoint_path}, using initialized weights.")


def cmd_train(args):
    """Run high-speed batched training."""
    from train_fast import main as train_main
    sys.argv = [
        "train_fast.py",
        "--data-dir", args.data_dir,
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--max-puzzles", str(args.max_puzzles),
        "--checkpoint-dir", args.checkpoint_dir
    ]
    if getattr(args, "use_hf", False):
        sys.argv.append("--use-hf")
    train_main()


def cmd_battle(args):
    """Run bot arena battle against Random, Greedy, Minimax, or real Stockfish."""
    network = ORIONNetwork()
    load_network_checkpoint(network, args.checkpoint)
    encoder = ChessEncoder()

    use_search = not args.policy_only
    orion = OrionBot(network, encoder, use_search=use_search, depth=args.search_depth, rating=args.orion_elo)

    opp_name = args.opponent.lower()
    if opp_name == "random":
        opponent = RandomBot(seed=args.seed)
    elif opp_name == "greedy":
        opponent = GreedyMaterialBot()
    elif opp_name == "minimax":
        opponent = MinimaxBot(depth=args.minimax_depth)
    elif opp_name == "stockfish":
        sf = StockfishBot(stockfish_path=args.stockfish_path, elo=args.stockfish_elo, depth=args.stockfish_depth)
        if not sf.is_available():
            print("\n[ERROR] Stockfish executable was not found on your system PATH.")
            print("To battle real Stockfish:")
            print("1. Download Stockfish from: https://stockfishchess.org/download/")
            print("2. Place stockfish.exe in your PATH or provide --stockfish-path <path/to/stockfish.exe>")
            print("Falling back to MinimaxBot for now...\n")
            opponent = MinimaxBot(depth=2)
        else:
            opponent = sf
    else:
        print(f"[ERROR] Unknown opponent '{args.opponent}'. Choose: random, greedy, minimax, stockfish")
        sys.exit(1)

    arena = Arena(orion, opponent, max_moves=args.max_moves)
    arena.run_match(rounds=args.rounds)


def cmd_play(args):
    """Interactive human vs ORION chess game in the terminal."""
    network = ORIONNetwork()
    load_network_checkpoint(network, args.checkpoint)
    encoder = ChessEncoder()

    use_search = not args.policy_only
    agent = StockfishInspiredSearchAgent(network, encoder, default_depth=args.depth) if use_search else None

    env = ChessEnvironment()
    user_color = args.color.lower()
    if user_color not in ("white", "black"):
        user_color = "white"

    print("================================================================")
    print("                 PLAY CHESS AGAINST ORION                       ")
    print(f"Human: {user_color.capitalize()} | ORION: {'Black' if user_color == 'white' else 'White'} (Depth {args.depth})")
    print("Type your moves in UCI format (e.g. e2e4, g1f3, e7e8q). Type 'quit' to exit.")
    print("================================================================\n")

    env.display()

    while True:
        status = env.game_status()
        if status != "playing" and status != "check":
            print(f"\n[GAME OVER] Result: {status}")
            break

        if env.turn == user_color:
            try:
                user_input = input(f"\nYour move ({env.turn}) > ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if user_input.lower() in ("quit", "exit"):
                print("Game abandoned.")
                break

            if len(user_input) < 4:
                print("Invalid format. Use 4 characters like e2e4.")
                continue

            try:
                files = "abcdefgh"
                c1 = files.index(user_input[0])
                r1 = 8 - int(user_input[1])
                c2 = files.index(user_input[2])
                r2 = 8 - int(user_input[3])
                move = (r1, c1, r2, c2)
            except Exception:
                print("Could not parse move coordinates.")
                continue

            if move not in env.legal_moves():
                print(f"Illegal move {user_input}! Legal moves: {[ChessActionSpace.move_to_uci(m) for m in env.legal_moves()]}")
                continue

            env.make_move(move)
            env.display()
        else:
            print(f"\nORION ({env.turn}) is thinking...")
            if use_search:
                best_move = agent.choose_move(env, depth=args.depth)
            else:
                from orion.model.agent import ORIONAgent
                best_move = ORIONAgent(network, encoder).choose_move(env)

            if best_move is None:
                print("ORION has no legal moves!")
                break

            uci_move = ChessActionSpace.move_to_uci(best_move)
            print(f"ORION played: {uci_move}")
            env.make_move(best_move)
            env.display()


def cmd_uci(args):
    """Launch standard UCI engine loop for chess GUIs."""
    from orion_uci import main as uci_main
    uci_main()


def main():
    parser = argparse.ArgumentParser(
        description="ORION Gaming Mind - Chess AI Platform by Team-Auralis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py train --epochs 5 --max-puzzles 10000
  python cli.py battle --opponent random --rounds 6
  python cli.py battle --opponent greedy --rounds 4
  python cli.py battle --opponent minimax --rounds 4
  python cli.py battle --opponent stockfish --stockfish-elo 1400 --rounds 2
  python cli.py play --color white --depth 2
  python cli.py uci
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # TRAIN
    train_parser = subparsers.add_parser("train", help="High-speed supervised training on Lichess data")
    train_parser.add_argument("--data-dir", default="data", help="Directory containing lichess_*.json")
    train_parser.add_argument("--use-hf", action="store_true", help="Stream directly from HuggingFace Lichess dataset")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--max-puzzles", type=int, default=50000, help="Max puzzles to load")
    train_parser.add_argument("--checkpoint-dir", default="checkpoints/fast_torch", help="Output checkpoint directory")

    # BATTLE / ARENA
    battle_parser = subparsers.add_parser("battle", aliases=["arena"], help="Arena match against real bots with Elo tracking")
    battle_parser.add_argument("--opponent", default="random", choices=["random", "greedy", "minimax", "stockfish"], help="Opponent bot")
    battle_parser.add_argument("--rounds", type=int, default=6, help="Number of games to play")
    battle_parser.add_argument("--search-depth", type=int, default=2, help="ORION search depth")
    battle_parser.add_argument("--policy-only", action="store_true", help="Disable search and use pure policy network")
    battle_parser.add_argument("--checkpoint", default="checkpoints/v0.5_supervised/orion_1000.json", help="Checkpoint file")
    battle_parser.add_argument("--orion-elo", type=float, default=1200.0, help="Starting Elo for ORION")
    battle_parser.add_argument("--minimax-depth", type=int, default=2, help="Depth for MinimaxBot")
    battle_parser.add_argument("--stockfish-path", default=None, help="Path to stockfish executable")
    battle_parser.add_argument("--stockfish-elo", type=int, default=1400, help="Stockfish UCI Elo limit (1320-3190)")
    battle_parser.add_argument("--stockfish-depth", type=int, default=3, help="Stockfish depth")
    battle_parser.add_argument("--max-moves", type=int, default=100, help="Max moves before draw")
    battle_parser.add_argument("--seed", type=int, default=42, help="RNG seed")

    # PLAY
    play_parser = subparsers.add_parser("play", help="Play interactively against ORION in the terminal")
    play_parser.add_argument("--color", default="white", choices=["white", "black"], help="Your piece color")
    play_parser.add_argument("--depth", type=int, default=2, help="ORION search depth")
    play_parser.add_argument("--policy-only", action="store_true", help="Disable search")
    play_parser.add_argument("--checkpoint", default="checkpoints/v0.5_supervised/orion_1000.json", help="Checkpoint file")

    # UCI
    subparsers.add_parser("uci", help="Launch UCI engine protocol for Arena/Cutechess/Lichess")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        cmd_train(args)
    elif args.command in ("battle", "arena"):
        cmd_battle(args)
    elif args.command == "play":
        cmd_play(args)
    elif args.command == "uci":
        cmd_uci(args)


if __name__ == "__main__":
    main()
