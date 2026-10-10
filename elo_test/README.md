# ORION Chess Engine — Empirical Elo Validation Suite

## 1. Overview
This directory contains the complete, independent empirical evaluation suite for the **ORION Chess Engine**, executing the formal audit plan across both:
- **Test A**: Pure Neural Policy Model (`ORION_Policy_Only`) — zero tree search, zero opening book, raw ResNet-6 policy logits.
- **Test B**: Integrated Chess Engine (`ORION_Complete_Engine`) — ResNet neural evaluation (0.35 weight) combined with tapered piece-square mobility (0.65 weight) and full alpha-beta tree search (Zobrist TT, quiescence, SEE, RFP, NMP, LMP, history/killer heuristics).

Evaluations are performed against an independently calibrated reference pool:
1. `Stockfish 19` across calibrated UCI Elo levels: 1320, 1500, 1700, 1900, 2100.
2. `BBC 1.1` (Maksim Korzh), published CCRL 40/40 rating 1888 Elo.

All games are managed by **Cute Chess CLI 1.5.1** under paired opening conditions, with ratings computed by **Ordo** (compiled from source).

---

## 2. Directory Structure & Key Files

| File / Folder | Description |
| :--- | :--- |
| [`manifest.json`](file:///D:/orion-game-ai/elo_test/manifest.json) | Cryptographic SHA-256 hashes of all checkpoints, binaries, and hardware specs. |
| [`engine_inventory.md`](file:///D:/orion-game-ai/elo_test/engine_inventory.md) | Full technical breakdown of search parameters, neural weights, and reference engines. |
| [`tournament_config.md`](file:///D:/orion-game-ai/elo_test/tournament_config.md) | Exact rules, time controls, opening suite, adjudication, and concurrency settings. |
| [`openings.pgn`](file:///D:/orion-game-ai/elo_test/openings.pgn) | 22 balanced master openings 2–4 plies deep used for paired color testing. |
| [`orion_policy_engine.py`](file:///D:/orion-game-ai/elo_test/orion_policy_engine.py) | Standalone UCI engine wrapper for Test A (Policy-only). |
| [`orion_complete_engine.py`](file:///D:/orion-game-ai/elo_test/orion_complete_engine.py) | Full integrated UCI engine wrapper for Test B. |
| [`orion_policy.bat`](file:///D:/orion-game-ai/elo_test/orion_policy.bat) | Windows batch execution wrapper for Cute Chess CLI. |
| [`orion_complete.bat`](file:///D:/orion-game-ai/elo_test/orion_complete.bat) | Windows batch execution wrapper for Cute Chess CLI. |
| [`run_tournament.py`](file:///D:/orion-game-ai/elo_test/run_tournament.py) | Automated master tournament runner executing all matchups. |
| [`calculate_bootstrap_uncertainty.py`](file:///D:/orion-game-ai/elo_test/calculate_bootstrap_uncertainty.py) | 1,000 paired-opening Monte Carlo bootstrap resampling engine. |
| [`uci_correctness_test.py`](file:///D:/orion-game-ai/elo_test/uci_correctness_test.py) | Phase 1 UCI protocol and chess rule verification suite. |
| [`runtime_benchmark.py`](file:///D:/orion-game-ai/elo_test/runtime_benchmark.py) | Phase 3 startup latency, move time, RAM, and GPU memory benchmark. |
| [`reproduce.ps1`](file:///D:/orion-game-ai/elo_test/reproduce.ps1) | Single-command automated reproduction PowerShell script. |
| [`games.pgn`](file:///D:/orion-game-ai/elo_test/games.pgn) | Master PGN containing all played tournament games. |
| [`game_results.csv`](file:///D:/orion-game-ai/elo_test/game_results.csv) | CSV export of every game result, players, plies, and termination reason. |
| [`rating_results.txt`](file:///D:/orion-game-ai/elo_test/rating_results.txt) | Ordo official rating table with anchor calibrations. |
| [`uncertainty.json`](file:///D:/orion-game-ai/elo_test/uncertainty.json) | 95% confidence intervals derived from paired bootstrap resampling. |
| [`report.md`](file:///D:/orion-game-ai/elo_test/report.md) | Comprehensive empirical rating report with executive summary. |

---

## 3. Quick Reproduction
To execute the complete audit pipeline end-to-end:
```powershell
powershell -ExecutionPolicy Bypass -File elo_test/reproduce.ps1
```
