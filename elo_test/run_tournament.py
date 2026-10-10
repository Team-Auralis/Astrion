"""
ORION Chess Engine Empirical Elo Tournament Runner
Executes balanced matches between ORION engines (Test A, Test B) and calibrated reference engines.
Logs PGNs, CSV results, engine stderr, and runs Ordo calculation.
"""
import os
import sys
import subprocess
import csv
import json
import re

CUTECHESS_CLI = r"C:\Users\shaur\AppData\Local\Programs\Cute Chess\cutechess-cli.exe"
STOCKFISH_EXE = r"D:\orion-game-ai\stockfish.exe"
BBC_EXE = r"D:\orion-game-ai\elo_test\bbc_repo\bin\windows\bbc_1.1_64bit_windows.exe"
ORION_POLICY_BAT = r"D:\orion-game-ai\elo_test\orion_policy.bat"
ORION_COMPLETE_BAT = r"D:\orion-game-ai\elo_test\orion_complete.bat"
OPENINGS_PGN = r"D:\orion-game-ai\elo_test\openings.pgn"
OUTPUT_PGN = r"D:\orion-game-ai\elo_test\games.pgn"
RESULTS_CSV = r"D:\orion-game-ai\elo_test\game_results.csv"
LOGS_DIR = r"D:\orion-game-ai\elo_test\engine_logs"
ORDO_EXE = r"D:\orion-game-ai\ordo.exe"

os.makedirs(LOGS_DIR, exist_ok=True)

# Define tournament matchups: (Engine1, Engine2, NumGames, Description)
# Each matchup uses paired openings (-repeat with swapped colors)
MATCHUPS = [
    # Test A: Neural Policy Only vs Reference Engines
    {
        "name": "Policy vs SF 1320",
        "e1_cmd": ORION_POLICY_BAT, "e1_name": "ORION_Policy_Only", "e1_opts": [],
        "e2_cmd": STOCKFISH_EXE, "e2_name": "Stockfish_1320", "e2_opts": ["option.UCI_LimitStrength=true", "option.UCI_Elo=1320"],
        "games": 16
    },
    {
        "name": "Policy vs SF 1500",
        "e1_cmd": ORION_POLICY_BAT, "e1_name": "ORION_Policy_Only", "e1_opts": [],
        "e2_cmd": STOCKFISH_EXE, "e2_name": "Stockfish_1500", "e2_opts": ["option.UCI_LimitStrength=true", "option.UCI_Elo=1500"],
        "games": 16
    },
    # Head-to-Head: Complete Engine vs Policy Only
    {
        "name": "Complete vs Policy",
        "e1_cmd": ORION_COMPLETE_BAT, "e1_name": "ORION_Complete_Engine", "e1_opts": [],
        "e2_cmd": ORION_POLICY_BAT, "e2_name": "ORION_Policy_Only", "e2_opts": [],
        "games": 16
    },
    # Test B: Complete Engine vs Calibrated Reference Engines
    {
        "name": "Complete vs SF 1320",
        "e1_cmd": ORION_COMPLETE_BAT, "e1_name": "ORION_Complete_Engine", "e1_opts": [],
        "e2_cmd": STOCKFISH_EXE, "e2_name": "Stockfish_1320", "e2_opts": ["option.UCI_LimitStrength=true", "option.UCI_Elo=1320"],
        "games": 24
    },
    {
        "name": "Complete vs SF 1500",
        "e1_cmd": ORION_COMPLETE_BAT, "e1_name": "ORION_Complete_Engine", "e1_opts": [],
        "e2_cmd": STOCKFISH_EXE, "e2_name": "Stockfish_1500", "e2_opts": ["option.UCI_LimitStrength=true", "option.UCI_Elo=1500"],
        "games": 24
    },
    {
        "name": "Complete vs SF 1700",
        "e1_cmd": ORION_COMPLETE_BAT, "e1_name": "ORION_Complete_Engine", "e1_opts": [],
        "e2_cmd": STOCKFISH_EXE, "e2_name": "Stockfish_1700", "e2_opts": ["option.UCI_LimitStrength=true", "option.UCI_Elo=1700"],
        "games": 20
    },
    {
        "name": "Complete vs BBC 1.1",
        "e1_cmd": ORION_COMPLETE_BAT, "e1_name": "ORION_Complete_Engine", "e1_opts": [],
        "e2_cmd": BBC_EXE, "e2_name": "BBC_1.1", "e2_opts": [],
        "games": 16
    },
    {
        "name": "Complete vs SF 1900",
        "e1_cmd": ORION_COMPLETE_BAT, "e1_name": "ORION_Complete_Engine", "e1_opts": [],
        "e2_cmd": STOCKFISH_EXE, "e2_name": "Stockfish_1900", "e2_opts": ["option.UCI_LimitStrength=true", "option.UCI_Elo=1900"],
        "games": 16
    },
    # Reference Calibration Match: SF 1320 vs SF 1500
    {
        "name": "SF 1320 vs SF 1500",
        "e1_cmd": STOCKFISH_EXE, "e1_name": "Stockfish_1320", "e1_opts": ["option.UCI_LimitStrength=true", "option.UCI_Elo=1320"],
        "e2_cmd": STOCKFISH_EXE, "e2_name": "Stockfish_1500", "e2_opts": ["option.UCI_LimitStrength=true", "option.UCI_Elo=1500"],
        "games": 12
    },
    # Reference Calibration Match: SF 1500 vs BBC 1.1
    {
        "name": "SF 1500 vs BBC 1.1",
        "e1_cmd": STOCKFISH_EXE, "e1_name": "Stockfish_1500", "e1_opts": ["option.UCI_LimitStrength=true", "option.UCI_Elo=1500"],
        "e2_cmd": BBC_EXE, "e2_name": "BBC_1.1", "e2_opts": [],
        "games": 12
    },
]

def run_matchup(m, match_index):
    temp_pgn = os.path.join(LOGS_DIR, f"match_{match_index}.pgn")
    if os.path.exists(temp_pgn):
        try: os.remove(temp_pgn)
        except: pass

    cmd = [
        CUTECHESS_CLI,
        "-engine", f"cmd={m['e1_cmd']}", f"name={m['e1_name']}", "proto=uci",
    ]
    for opt in m['e1_opts']:
        cmd.append(opt)

    cmd.extend([
        "-engine", f"cmd={m['e2_cmd']}", f"name={m['e2_name']}", "proto=uci",
    ])
    for opt in m['e2_opts']:
        cmd.append(opt)

    rounds = m['games'] // 2
    cmd.extend([
        "-each", "tc=inf", "depth=3",
        "-rounds", str(rounds),
        "-games", "2",
        "-repeat",
        "-openings", f"file={OPENINGS_PGN}", "format=pgn", "order=sequential",
        "-pgnout", temp_pgn,
        "-recover",
        "-concurrency", "2",
        "-draw", "movenumber=45", "movecount=8", "score=15",
        "-resign", "movecount=5", "score=700",
        "-maxmoves", "120"
    ])

    print(f"\n[{match_index}/{len(MATCHUPS)}] Running Match: {m['name']} ({m['games']} games)...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    with open(os.path.join(LOGS_DIR, f"match_{match_index}.log"), "w", encoding="utf-8") as lf:
        lf.write(f"STDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}\n")

    # Append games to master PGN
    if os.path.exists(temp_pgn):
        with open(temp_pgn, "r", encoding="utf-8") as src, open(OUTPUT_PGN, "a", encoding="utf-8") as dst:
            dst.write(src.read() + "\n\n")

    # Parse game counts from stdout
    print(res.stdout.strip().split("\n")[-6:])

def parse_all_games_to_csv():
    """Extract game metadata from games.pgn into game_results.csv"""
    if not os.path.exists(OUTPUT_PGN):
        return 0

    with open(OUTPUT_PGN, "r", encoding="utf-8") as f:
        content = f.read()

    games = content.split('[Event ')
    parsed_rows = []
    
    for g in games:
        if not g.strip():
            continue
        full_game = '[Event ' + g
        
        white = re.search(r'\[White "([^"]+)"\]', full_game)
        black = re.search(r'\[Black "([^"]+)"\]', full_game)
        result = re.search(r'\[Result "([^"]+)"\]', full_game)
        date = re.search(r'\[Date "([^"]+)"\]', full_game)
        termination = re.search(r'\[Termination "([^"]+)"\]', full_game)
        plycount = re.search(r'\[PlyCount "([^"]+)"\]', full_game)
        
        w_val = white.group(1) if white else "Unknown"
        b_val = black.group(1) if black else "Unknown"
        r_val = result.group(1) if result else "*"
        t_val = termination.group(1) if termination else "normal"
        p_val = plycount.group(1) if plycount else "0"
        
        parsed_rows.append({
            "game_number": len(parsed_rows) + 1,
            "white": w_val,
            "black": b_val,
            "result": r_val,
            "termination": t_val,
            "ply_count": p_val
        })

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["game_number", "white", "black", "result", "termination", "ply_count"])
        writer.writeheader()
        writer.writerows(parsed_rows)

    print(f"\nParsed {len(parsed_rows)} games into {RESULTS_CSV}")
    return len(parsed_rows)

def run_ordo_and_stats():
    """Execute Ordo with 95% confidence intervals and multi-anchors"""
    print("\n--- Running Ordo Rating Calculation ---")
    ordo_txt = r"D:\orion-game-ai\elo_test\rating_results.txt"
    ordo_csv = r"D:\orion-game-ai\elo_test\rating_results.csv"
    h2h_txt = r"D:\orion-game-ai\elo_test\head2head.txt"
    cfs_csv = r"D:\orion-game-ai\elo_test\cfs_matrix.csv"

    cmd = [
        ORDO_EXE,
        "-p", OUTPUT_PGN,
        "-a", "1500",
        "-A", "Stockfish_1500",
        "-s", "1000",
        "-F", "95.0",
        "-o", ordo_txt,
        "-c", ordo_csv,
        "-j", h2h_txt,
        "-C", cfs_csv,
        "-W", "-D"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if os.path.exists(ordo_txt):
        with open(ordo_txt, "r", encoding="utf-8") as f:
            print(f.read())

if __name__ == "__main__":
    # If starting fresh, reset master PGN
    if "--resume" not in sys.argv and os.path.exists(OUTPUT_PGN):
        os.remove(OUTPUT_PGN)

    total_planned = sum(m['games'] for m in MATCHUPS)
    print(f"=== Starting Empirical Elo Validation Tournament ({total_planned} planned games) ===")

    for i, m in enumerate(MATCHUPS, 1):
        run_matchup(m, i)

    total_played = parse_all_games_to_csv()
    run_ordo_and_stats()
    print("\nTournament complete! All logs and PGNs stored in elo_test/")
