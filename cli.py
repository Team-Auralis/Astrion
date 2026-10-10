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
from orion.arena import OrionBot, RandomBot, GreedyMaterialBot, MinimaxBot, StockfishBot, EnsembleCouncilBot, DecaCouncilBot, Arena


import glob


def find_best_checkpoint(requested_path: str = None) -> str:
    if requested_path and os.path.exists(requested_path) and requested_path != "checkpoints/v0.5_supervised/orion_1000.json":
        return requested_path

    # Auto-discover newest checkpoint with validation
    candidates = []
    for pattern in ["checkpoints/ccrl_master/*best*.json", "checkpoints/ccrl_master/*master*.json", "checkpoints/ccrl_master/*.json", "checkpoints/distilled/*.json", "checkpoints/gm_master/*.json", "checkpoints/fast_torch/*.json", "checkpoints/resnet_v1/*.json", "checkpoints/full_hf/*.json", "checkpoints/v0.5_supervised/*.json"]:
        candidates.extend(glob.glob(pattern))

    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    for cand in candidates:
        if os.path.getsize(cand) > 1000000:
            try:
                with open(cand, "r") as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    f.seek(max(0, size - 100))
                    tail = f.read().strip()
                    if tail.endswith("}"):
                        return cand
            except Exception:
                continue

    return requested_path or "checkpoints/v0.5_supervised/orion_1000.json"


def load_network_checkpoint(network: ORIONNetwork, checkpoint_path: str = None):
    actual_path = find_best_checkpoint(checkpoint_path)
    if os.path.exists(actual_path):
        try:
            ep = Checkpoint().load(network, actual_path)
            print(f"[CHECKPOINT] Successfully loaded model from {actual_path} (ep: {ep})")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint {actual_path}: {e}")
    else:
        print(f"[INFO] No checkpoint found, using initialized weights.")


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


def cmd_train_gm(args):
    """Train ORION directly on Grandmaster dataset from angeluriot/chess_games."""
    from train_gm_dataset import train_on_gm_data
    train_on_gm_data(args)


def cmd_train_distill(args):
    """Train ORION using SOTA Knowledge Distillation & Tactical Curriculum."""
    from train_sota_distillation import train_distillation
    train_distillation(args)


def cmd_train_ccrl(args):
    """Train ORION deeply on CCRL 40/40 Super-GM engine tournament archive."""
    from train_ccrl import train_ccrl
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


def cmd_train_full_corpus(args):
    """Train ORION continuously across all 2.46M CCRL games with chunk streaming."""
    from train_ccrl_full_corpus import stream_train_full_corpus
    stream_train_full_corpus(
        archive_path=args.archive,
        chunk_positions=args.chunk_positions,
        min_elo=args.min_elo,
        stride=args.stride,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_in=args.warm_start,
        output_prefix=args.output_prefix,
        target_accuracy=args.target_acc,
        max_total_positions=args.max_positions
    )


def cmd_battle(args):
    """Run bot arena battle against Random, Greedy, Minimax, or real Stockfish."""
    network = ORIONNetwork()
    load_network_checkpoint(network, args.checkpoint)
    encoder = ChessEncoder()

    use_search = not args.policy_only
    temp = getattr(args, "temperature", 0.0)
    orion = OrionBot(network, encoder, use_search=use_search, depth=args.search_depth, rating=args.orion_elo, temperature=temp)

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
    elif opp_name in ("deca", "10bots", "deca-council"):
        opponent = DecaCouncilBot(
            stockfish_elo=args.stockfish_elo,
            stockfish_depth=args.stockfish_depth,
            network=network,
            encoder=encoder
        )
    elif opp_name in ("ensemble", "alliance"):
        opponent = EnsembleCouncilBot(
            stockfish_elo=args.stockfish_elo,
            stockfish_depth=args.stockfish_depth,
            network=network,
            encoder=encoder
        )
    else:
        print(f"[ERROR] Unknown opponent '{args.opponent}'. Choose: random, greedy, minimax, stockfish, ensemble, alliance, deca, 10bots")
        sys.exit(1)

    arena = Arena(orion, opponent, max_moves=args.max_moves)
    arena.run_match(rounds=args.rounds)


def cmd_ordo(args):
    """Run official Ordo maximum-likelihood rating calculation on a PGN file."""
    import subprocess
    ordo_path = "ordo.exe" if os.path.exists("ordo.exe") else ("tools/ordo/ordo.exe" if os.path.exists("tools/ordo/ordo.exe") else None)
    if not ordo_path:
        print("[ERROR] ordo.exe binary not found. Build it with: mingw32-make -C tools/ordo")
        sys.exit(1)

    if not os.path.exists(args.pgn):
        print(f"[ERROR] PGN file not found: {args.pgn}")
        sys.exit(1)

    cmd = [ordo_path, "-p", args.pgn, "-a", str(args.anchor_rating)]
    if args.anchor_player:
        cmd.extend(["-A", args.anchor_player])
    if args.simulations > 0:
        cmd.extend(["-s", str(args.simulations)])

    print("=" * 65)
    print("      OFFICIAL ORDO MAXIMUM LIKELIHOOD RATING CALCULATION       ")
    print("=" * 65)
    print(f"PGN File     : {args.pgn}")
    print(f"Anchor Rating: {args.anchor_rating} ({args.anchor_player or 'Pool Average'})")
    print(f"Simulations  : {args.simulations}")
    print("=" * 65 + "\n")

    subprocess.run(cmd)


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


def cmd_app(args):
    """Launch ORION mobile app API server & PWA."""
    from orion_app_server import run_server
    run_server(host=args.host, port=args.port, checkpoint=args.checkpoint, depth=args.depth)


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
  python cli.py app --port 8000
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

    # TRAIN GM
    gm_parser = subparsers.add_parser("train-gm", help="Stream and train on 2300+ Elo Grandmaster dataset (angeluriot/chess_games)")
    gm_parser.add_argument("--max-positions", type=int, default=100000, help="Max positions to extract")
    gm_parser.add_argument("--min-elo", type=int, default=2200, help="Minimum player Elo (default: 2200)")
    gm_parser.add_argument("--moves-per-game", type=int, default=35, help="Max opening/middle moves per game")
    gm_parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    gm_parser.add_argument("--epochs", type=int, default=10, help="Epochs (default: 10)")
    gm_parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate (default: 0.0005)")
    gm_parser.add_argument("--output-dir", default="checkpoints/gm_master", help="Output directory")
    gm_parser.add_argument("--resume-checkpoint", default=None, help="Resume from previous checkpoint")

    # TRAIN DISTILL (SOTA Knowledge Distillation)
    distill_parser = subparsers.add_parser("train-distill", help="SOTA Knowledge Distillation & Tactical Focal Training from Stockfish")
    distill_parser.add_argument("--num-samples", type=int, default=10000, help="Number of positions to distill")
    distill_parser.add_argument("--multipv", type=int, default=3, help="Multi-PV count for dark knowledge extraction")
    distill_parser.add_argument("--teacher-depth", type=int, default=8, help="Stockfish evaluation depth")
    distill_parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    distill_parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    distill_parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    distill_parser.add_argument("--tactical-boost", type=float, default=2.5, help="Loss weight for tactical positions")
    distill_parser.add_argument("--distill-temp", type=float, default=1.5, help="Temperature for soft targets")
    distill_parser.add_argument("--output-dir", default="checkpoints/distilled", help="Output directory")
    distill_parser.add_argument("--resume-checkpoint", default="checkpoints/gm_master/orion_gm_resnet_perfect_acc_98.0pct.json", help="Warm-start checkpoint")
    distill_parser.add_argument("--stockfish-path", default="stockfish.exe", help="Path to Stockfish engine")

    # TRAIN CCRL (Super-GM Engine Corpus 2.46M games)
    ccrl_parser = subparsers.add_parser("train-ccrl", help="Deep Super-GM Training on CCRL 40/40 2.46M Game Corpus")
    ccrl_parser.add_argument("--archive", default=r"D:\orion-game-ai\CCRL-4040.[2463682].pgn.7z", help="Path to CCRL .7z archive")
    ccrl_parser.add_argument("--max-positions", type=int, default=150000, help="Number of positions to stream and train on")
    ccrl_parser.add_argument("--min-elo", type=int, default=2600, help="Minimum engine Elo filter")
    ccrl_parser.add_argument("--stride", type=int, default=2, help="Move sampling stride per game")
    ccrl_parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    ccrl_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    ccrl_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ccrl_parser.add_argument("--warm-start", default=None, help="Checkpoint to warm start from")
    ccrl_parser.add_argument("--output", default="checkpoints/ccrl_master/orion_ccrl_deep_master.json", help="Output checkpoint path")

    # TRAIN FULL CORPUS (2.46M games chunk-streaming)
    corpus_parser = subparsers.add_parser("train-full-corpus", help="Continuous Chunk-Streaming Trainer across all 2.46M CCRL Games")
    corpus_parser.add_argument("--archive", default=r"D:\orion-game-ai\CCRL-4040.[2463682].pgn.7z", help="Path to CCRL .7z archive")
    corpus_parser.add_argument("--chunk-positions", type=int, default=50000, help="Positions per in-memory training chunk")
    corpus_parser.add_argument("--min-elo", type=int, default=2500, help="Minimum engine Elo filter")
    corpus_parser.add_argument("--stride", type=int, default=2, help="Move sampling stride per game")
    corpus_parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    corpus_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    corpus_parser.add_argument("--target-acc", type=float, default=90.0, help="Target consensus accuracy (default: 90.0 percent)")
    corpus_parser.add_argument("--max-positions", type=int, default=None, help="Optional max positions cap")
    corpus_parser.add_argument("--warm-start", default=None, help="Checkpoint to warm start from")
    corpus_parser.add_argument("--output-prefix", default="checkpoints/ccrl_master/orion_ccrl_full_corpus", help="Output prefix")

    # BATTLE / ARENA
    battle_parser = subparsers.add_parser("battle", aliases=["arena"], help="Arena match against real bots with Elo tracking")
    battle_parser.add_argument("--opponent", default="random", choices=["random", "greedy", "minimax", "stockfish", "ensemble", "alliance", "deca", "10bots"], help="Opponent bot")
    battle_parser.add_argument("--rounds", type=int, default=6, help="Number of games to play")
    battle_parser.add_argument("--search-depth", type=int, default=3, help="ORION search depth")
    battle_parser.add_argument("--policy-only", action="store_true", help="Disable search and use pure policy network")
    battle_parser.add_argument("--checkpoint", default=None, help="Checkpoint file (auto-discovered if not set)")
    battle_parser.add_argument("--orion-elo", type=float, default=1500.0, help="Starting Elo for ORION")
    battle_parser.add_argument("--minimax-depth", type=int, default=2, help="Depth for MinimaxBot")
    battle_parser.add_argument("--stockfish-path", default=None, help="Path to stockfish executable")
    battle_parser.add_argument("--stockfish-elo", type=int, default=1400, help="Stockfish UCI Elo limit (1320-3190)")
    battle_parser.add_argument("--stockfish-depth", type=int, default=3, help="Stockfish depth")
    battle_parser.add_argument("--max-moves", type=int, default=200, help="Max moves before draw")
    battle_parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    battle_parser.add_argument("--temperature", type=float, default=0.0, help="Move randomization temperature (0.0=deterministic, >0=randomize among top moves)")

    # PLAY
    play_parser = subparsers.add_parser("play", help="Play interactively against ORION in the terminal")
    play_parser.add_argument("--color", default="white", choices=["white", "black"], help="Your piece color")
    play_parser.add_argument("--depth", type=int, default=3, help="ORION search depth")
    play_parser.add_argument("--policy-only", action="store_true", help="Disable search")
    play_parser.add_argument("--checkpoint", default=None, help="Checkpoint file (auto-discovered if not set)")

    # UCI
    subparsers.add_parser("uci", help="Launch UCI engine protocol for Arena/Cutechess/Lichess")

    # ORDO
    ordo_parser = subparsers.add_parser("ordo", help="Run official Ordo maximum-likelihood Elo rating calculation on PGN")
    ordo_parser.add_argument("--pgn", default="matches/arena_match.pgn", help="PGN games file")
    ordo_parser.add_argument("--anchor-rating", type=int, default=2200, help="Anchor reference rating")
    ordo_parser.add_argument("--anchor-player", default=None, help="Name of anchor player (e.g. 'Stockfish (Elo 2200)')")
    ordo_parser.add_argument("--simulations", type=int, default=1000, help="Monte Carlo simulations for error margin")

    # APP / SERVER
    app_parser = subparsers.add_parser("app", aliases=["server"], help="Launch ORION mobile app API server & PWA interface")
    app_parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    app_parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    app_parser.add_argument("--depth", type=int, default=3, help="ORION search depth")
    app_parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "train-gm":
        cmd_train_gm(args)
    elif args.command == "train-distill":
        cmd_train_distill(args)
    elif args.command == "train-ccrl":
        cmd_train_ccrl(args)
    elif args.command == "train-full-corpus":
        cmd_train_full_corpus(args)
    elif args.command in ("battle", "arena"):
        cmd_battle(args)
    elif args.command == "ordo":
        cmd_ordo(args)
    elif args.command == "play":
        cmd_play(args)
    elif args.command == "uci":
        cmd_uci(args)
    elif args.command in ("app", "server"):
        cmd_app(args)


if __name__ == "__main__":
    main()
