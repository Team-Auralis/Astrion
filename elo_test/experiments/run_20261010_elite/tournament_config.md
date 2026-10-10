# ORION Tournament Configuration & Protocols

## 1. Hardware & System Configuration
- **Platform**: Windows 11 Home 25H2 (x86_64)
- **CPU**: 11th Gen Intel(R) Core(TM) i5-11260H (6 Cores, 12 Threads) @ 4.40 GHz
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (3.87 GB VRAM, Compute Capability 8.6)
- **Tournament Concurrency**: 2 parallel game workers
- **Power Profile**: High Performance plugged-in AC power

## 2. Tournament Engine Protocols
- **Protocol**: Universal Chess Interface (UCI)
- **Tournament Runner**: Cute Chess CLI v1.5.1
- **Opening Suite**: `elo_test/openings.pgn` (22 balanced grandmaster opening positions)
- **Opening Policy**: Sequential pairs with `-repeat` (each opening played twice with alternating White/Black colors)
- **Color Equality**: Guaranteed 50% White, 50% Black games per pair

## 3. Adjudication & Termination Rules
- **Draw Adjudication**: Automatically awarded if move count >= 45 full moves, score within +/- 15 centipawns for 8 consecutive moves.
- **Resignation Adjudication**: Automatically awarded if score is below -700 centipawns for 5 consecutive moves.
- **Maximum Length**: 120 full moves (prevents non-progress infinite loops).
- **Tablebases**: None utilized during tournament play.
- **Crash Recovery**: Enabled (`-recover`).

## 4. Output Data Streams
- **Master PGN**: [`elo_test/games.pgn`](file:///D:/orion-game-ai/elo_test/games.pgn)
- **Game Results CSV**: [`elo_test/game_results.csv`](file:///D:/orion-game-ai/elo_test/game_results.csv)
- **Engine Process Logs**: [`elo_test/engine_logs/`](file:///D:/orion-game-ai/elo_test/engine_logs/)
- **Ordo Rating Reports**: [`elo_test/rating_results.txt`](file:///D:/orion-game-ai/elo_test/rating_results.txt) and [`elo_test/rating_results.csv`](file:///D:/orion-game-ai/elo_test/rating_results.csv)
- **Bootstrap Uncertainty**: [`elo_test/uncertainty.json`](file:///D:/orion-game-ai/elo_test/uncertainty.json)
