import math
import time
from typing import Dict, List, Optional, Tuple
from orion.environment.chess import ChessEnvironment
from orion.arena.bots import ChessBot


class EloTracker:
    """Official FIDE-style Elo calculation and rating tracker."""

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

    @staticmethod
    def update_ratings(rating_a: float, rating_b: float, score_a: float, k: float = 32.0) -> Tuple[float, float]:
        """
        score_a: 1.0 for A win, 0.5 for draw, 0.0 for B win.
        Returns: (new_rating_a, new_rating_b)
        """
        expected_a = EloTracker.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        score_b = 1.0 - score_a

        new_a = rating_a + k * (score_a - expected_a)
        new_b = rating_b + k * (score_b - expected_b)
        return round(new_a, 1), round(new_b, 1)

    @staticmethod
    def performance_rating(opponent_ratings: List[float], total_score: float) -> float:
        """Computes tournament performance rating."""
        if not opponent_ratings:
            return 1200.0
        n = len(opponent_ratings)
        avg_opp = sum(opponent_ratings) / n
        pct = total_score / n
        if pct <= 0.01:
            return avg_opp - 400.0
        if pct >= 0.99:
            return avg_opp + 400.0
        dp = -400.0 * math.log10((1.0 - pct) / pct)
        return round(avg_opp + dp, 1)


class Arena:
    """Arena battle runner to pit ORION against real chess bots."""

    def __init__(self, bot1: ChessBot, bot2: ChessBot, max_moves: int = 200):
        self.bot1 = bot1
        self.bot2 = bot2
        self.max_moves = max_moves

    def play_single_game(self, white_bot: ChessBot, black_bot: ChessBot, game_num: int = 1, verbose: bool = False) -> Dict:
        env = ChessEnvironment()
        start_time = time.time()
        moves_list = []

        while env.move_count < self.max_moves:
            status = env.game_status()
            if status != "playing" and status != "check":
                break

            current_bot = white_bot if env.turn == "white" else black_bot
            move = current_bot.choose_move(env)

            if move is None:
                break

            move_txt = env.move_to_text(move)
            moves_list.append(move_txt)

            if env.move_count % 10 == 0 and env.move_count > 0:
                print(f"[Match {game_num}] Ply {env.move_count}... ({current_bot.name})", flush=True)

            if not env.make_move(move):
                # Illegal move forfeit
                winner = "black" if env.turn == "white" else "white"
                return {
                    "winner": winner,
                    "status": f"forfeit_illegal_move_by_{env.turn}",
                    "moves": env.move_count,
                    "moves_list": moves_list,
                    "duration": time.time() - start_time
                }

        final_status = env.game_status()
        winner = None
        if final_status.startswith("checkmate:"):
            winner = final_status.split(":")[1]
        elif env.move_count >= self.max_moves:
            final_status = "max_moves_draw"

        return {
            "winner": winner,
            "status": final_status,
            "moves": env.move_count,
            "moves_list": moves_list,
            "duration": time.time() - start_time
        }

    def run_match(self, rounds: int = 10, verbose: bool = True, pgn_file: Optional[str] = "matches/arena_match.pgn", use_ordo: bool = True) -> Dict:
        """
        Runs an even-game match, alternating colors each round.
        Logs games to PGN and calculates official Ordo ratings if available.
        """
        import os
        import subprocess

        print("================================================================")
        print("                 ORION BATTLE ARENA MATCH                       ")
        print("================================================================")
        print(f"White/Black : [1] {self.bot1.name} (Elo {self.bot1.rating:.0f})")
        print(f"Opponent    : [2] {self.bot2.name} (Elo {self.bot2.rating:.0f})")
        print(f"Rounds      : {rounds} games (alternating colors)")
        print("================================================================\n")

        bot1_wins = 0
        bot2_wins = 0
        draws = 0
        total_moves = 0
        pgn_games = []

        initial_r1 = self.bot1.rating
        initial_r2 = self.bot2.rating

        for r in range(1, rounds + 1):
            if r % 2 == 1:
                white, black = self.bot1, self.bot2
                b1_is_white = True
            else:
                white, black = self.bot2, self.bot1
                b1_is_white = False

            res = self.play_single_game(white, black, game_num=r)
            total_moves += res["moves"]
            winner = res["winner"]

            if winner is None:
                draws += 1
                score1 = 0.5
                pgn_res = "1/2-1/2"
                outcome_str = "1/2-1/2 (Draw)"
            elif (winner == "white" and b1_is_white) or (winner == "black" and not b1_is_white):
                bot1_wins += 1
                score1 = 1.0
                pgn_res = "1-0" if b1_is_white else "0-1"
                outcome_str = f"1-0 ({self.bot1.name} won)"
            else:
                bot2_wins += 1
                score1 = 0.0
                pgn_res = "0-1" if b1_is_white else "1-0"
                outcome_str = f"0-1 ({self.bot2.name} won)"

            # Build PGN record
            moves_str = " ".join(f"{i//2 + 1}. {m}" if i % 2 == 0 else m for i, m in enumerate(res.get("moves_list", [])))
            pgn_entry = (
                f'[Event "ORION Arena Match"]\n'
                f'[Site "Localhost"]\n'
                f'[Round "{r}"]\n'
                f'[White "{white.name}"]\n'
                f'[Black "{black.name}"]\n'
                f'[Result "{pgn_res}"]\n\n'
                f'{moves_str} {pgn_res}\n\n'
            )
            pgn_games.append(pgn_entry)

            # Update live Elo
            self.bot1.rating, self.bot2.rating = EloTracker.update_ratings(
                self.bot1.rating, self.bot2.rating, score1, k=32.0 / math.sqrt(rounds)
            )

            color_tag = "White" if b1_is_white else "Black"
            print(f"Game {r:02d}/{rounds:02d} | {self.bot1.name} ({color_tag}): {outcome_str:<25} | {res['moves']:3d} plies ({res['status']})")

        total_games = bot1_wins + bot2_wins + draws
        b1_score = bot1_wins + 0.5 * draws
        b1_pct = (b1_score / total_games) * 100 if total_games else 0

        # Save PGN
        if pgn_file:
            os.makedirs(os.path.dirname(pgn_file) or ".", exist_ok=True)
            with open(pgn_file, "w") as f:
                f.writelines(pgn_games)
            print(f"\n[PGN] Saved match PGN log to: {pgn_file}")

        print("\n================================================================")
        print("                     MATCH FINAL RESULTS                        ")
        print("================================================================")
        print(f"{self.bot1.name:<30} Score: {bot1_wins}W - {bot2_wins}L - {draws}D  ({b1_score:.1f}/{total_games}, {b1_pct:.1f}%)")
        print(f"{self.bot2.name:<30} Score: {bot2_wins}W - {bot1_wins}L - {draws}D")
        print(f"Average game length: {total_moves / total_games:.1f} plies")
        print("----------------------------------------------------------------")
        delta1 = self.bot1.rating - initial_r1
        delta2 = self.bot2.rating - initial_r2
        print(f"Elo Adjustment for {self.bot1.name:<25}: {initial_r1:.1f} -> {self.bot1.rating:.1f} ({delta1:+.1f})")
        print(f"Elo Adjustment for {self.bot2.name:<25}: {initial_r2:.1f} -> {self.bot2.rating:.1f} ({delta2:+.1f})")
        print("================================================================\n")

        # Run Ordo if available
        ordo_path = "ordo.exe" if os.path.exists("ordo.exe") else ("tools/ordo/ordo.exe" if os.path.exists("tools/ordo/ordo.exe") else None)
        if use_ordo and ordo_path and pgn_file and os.path.exists(pgn_file):
            print("================================================================")
            print("         OFFICIAL ORDO MAXIMUM LIKELIHOOD RATING                ")
            print("================================================================")
            try:
                anchor_rating = str(int(initial_r2))
                anchor_name = self.bot2.name
                cmd = [ordo_path, "-p", pgn_file, "-a", anchor_rating, "-A", anchor_name]
                ordo_proc = subprocess.run(cmd, capture_output=True, text=True)
                if ordo_proc.stdout:
                    # Print relevant Ordo table
                    lines = [ln for ln in ordo_proc.stdout.splitlines() if "PLAYER" in ln or self.bot1.name in ln or self.bot2.name in ln or "advantage" in ln.lower() or "draw rate" in ln.lower()]
                    for ln in lines:
                        print("  " + ln)
                print("================================================================\n")
            except Exception as e:
                print(f"[WARN] Failed to run Ordo: {e}")

        return {
            "bot1_wins": bot1_wins,
            "bot2_wins": bot2_wins,
            "draws": draws,
            "bot1_final_elo": self.bot1.rating,
            "bot2_final_elo": self.bot2.rating,
            "total_games": total_games
        }
