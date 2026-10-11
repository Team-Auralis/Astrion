# Experiment B: Stockfish Loss Diagnostic Report

Analysis of all decisive games lost to reference engines (BBC-1.2 and Vice-1.2).

| Game | Opponent | Decisive Move | Played | Best SF Move | Eval Drop | Classified Root Cause |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| ORION-Search-Full vs BBC-1.2 | **BBC-1.2** | Move 3 (ply 5) | `f2f4` | `c1f4` | **-241 cp** | **Positional / Evaluation Inaccuracy (Static PST overvaluation)** |
| BBC-1.2 vs ORION-Search-Full | **BBC-1.2** | Move 7 (ply 14) | `e4f3` | `f6d5` | **-255 cp** | **Search Depth Horizon (Multi-ply tactical consequence beyond reach)** |
| ORION-Neural-Only vs Vice-1.2 | **Vice-1.2** | Move 3 (ply 5) | `f3e5` | `e2g3` | **-534 cp** | **Tactical Oversight (Blunder/Hanging Piece)** |
| Vice-1.2 vs ORION-Neural-Only | **Vice-1.2** | Move 2 (ply 4) | `g8f6` | `e5a5` | **-826 cp** | **Tactical Oversight (Blunder/Hanging Piece)** |
| ORION-Neural-Only vs BBC-1.2 | **BBC-1.2** | Move 2 (ply 3) | `b2b4` | `g1f3` | **-633 cp** | **Tactical Oversight (Blunder/Hanging Piece)** |
| BBC-1.2 vs ORION-Neural-Only | **BBC-1.2** | Move 2 (ply 4) | `g8f6` | `e5c7` | **-947 cp** | **Tactical Oversight (Blunder/Hanging Piece)** |
| Vice-1.2 vs ORION-Search-v2.3 | **Vice-1.2** | Move 8 (ply 16) | `e8g8` | `c6d8` | **-261 cp** | **Search Depth Horizon (Multi-ply tactical consequence beyond reach)** |


## Detailed Game-by-Game Diagnoses

### ORION-Search-Full vs BBC-1.2 (0-1)
- **Source:** `experiments/calibrated_eval/full_tournament.pgn`
- **Total Plies:** 150
- **First Decisive Mistake:** Move 3 (`f2f4`)
- **Stockfish Recommendation:** `c1f4`
- **Evaluation Impact:** +287.0 cp -> +46.0 cp (Drop: **241.0 cp**)
- **Root Cause Classification:** **Positional / Evaluation Inaccuracy (Static PST overvaluation)**
- **FEN at Mistake:** `rnb1kbnr/pp2pppp/8/4q3/3p4/2N5/PPP1NPPP/R1BQKB1R w KQkq - 0 6`

### BBC-1.2 vs ORION-Search-Full (1-0)
- **Source:** `experiments/calibrated_eval/full_tournament.pgn`
- **Total Plies:** 59
- **First Decisive Mistake:** Move 7 (`e4f3`)
- **Stockfish Recommendation:** `f6d5`
- **Evaluation Impact:** -311.0 cp -> -566.0 cp (Drop: **255.0 cp**)
- **Root Cause Classification:** **Search Depth Horizon (Multi-ply tactical consequence beyond reach)**
- **FEN at Mistake:** `r1b1kb1r/ppp1qppp/2n2n2/1N6/3PpB2/5N2/PPP1BPPP/R2Q1RK1 b kq - 1 10`

### ORION-Neural-Only vs Vice-1.2 (0-1)
- **Source:** `experiments/calibrated_eval/full_tournament.pgn`
- **Total Plies:** 32
- **First Decisive Mistake:** Move 3 (`f3e5`)
- **Stockfish Recommendation:** `e2g3`
- **Evaluation Impact:** +62.0 cp -> -472.0 cp (Drop: **534.0 cp**)
- **Root Cause Classification:** **Tactical Oversight (Blunder/Hanging Piece)**
- **FEN at Mistake:** `rnb1kbnr/ppp2ppp/8/q3p3/8/5N2/PPPPNPPP/R1BQKB1R w KQkq - 0 6`

### Vice-1.2 vs ORION-Neural-Only (1-0)
- **Source:** `experiments/calibrated_eval/full_tournament.pgn`
- **Total Plies:** 19
- **First Decisive Mistake:** Move 2 (`g8f6`)
- **Stockfish Recommendation:** `e5a5`
- **Evaluation Impact:** -101.0 cp -> -927.0 cp (Drop: **826.0 cp**)
- **Root Cause Classification:** **Tactical Oversight (Blunder/Hanging Piece)**
- **FEN at Mistake:** `rnb1kbnr/pp2pppp/2p5/4q3/3P4/2N5/PPP1NPPP/R1BQKB1R b KQkq - 0 5`

### ORION-Neural-Only vs BBC-1.2 (0-1)
- **Source:** `experiments/calibrated_eval/full_tournament.pgn`
- **Total Plies:** 38
- **First Decisive Mistake:** Move 2 (`b2b4`)
- **Stockfish Recommendation:** `g1f3`
- **Evaluation Impact:** +71.0 cp -> -562.0 cp (Drop: **633.0 cp**)
- **Root Cause Classification:** **Tactical Oversight (Blunder/Hanging Piece)**
- **FEN at Mistake:** `rn2kbnr/ppp1pppp/8/4qb2/8/8/PPPPNPPP/R1BQKBNR w KQkq - 4 5`

### BBC-1.2 vs ORION-Neural-Only (1-0)
- **Source:** `experiments/calibrated_eval/full_tournament.pgn`
- **Total Plies:** 37
- **First Decisive Mistake:** Move 2 (`g8f6`)
- **Stockfish Recommendation:** `e5c7`
- **Evaluation Impact:** -107.0 cp -> -1054.0 cp (Drop: **947.0 cp**)
- **Root Cause Classification:** **Tactical Oversight (Blunder/Hanging Piece)**
- **FEN at Mistake:** `rnb1kbnr/pp2pppp/2p5/4q3/3P4/2N5/PPP1BPPP/R1BQK1NR b KQkq - 0 5`

### Vice-1.2 vs ORION-Search-v2.3 (1-0)
- **Source:** `elo_test/v2_3_verification.pgn`
- **Total Plies:** 44
- **First Decisive Mistake:** Move 8 (`e8g8`)
- **Stockfish Recommendation:** `c6d8`
- **Evaluation Impact:** -168.0 cp -> -429.0 cp (Drop: **261.0 cp**)
- **Root Cause Classification:** **Search Depth Horizon (Multi-ply tactical consequence beyond reach)**
- **FEN at Mistake:** `r1b1k2r/ppp1bppp/1qn2p2/8/1P1P1N2/2P5/P3QPPP/R1B1KB1R b KQkq - 0 11`

