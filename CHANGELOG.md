# Changelog

All notable changes to the ORION chess engine project will be documented in this file.

## [2.2.0] - 2026-10-11
### Added
- **Polyglot Master Opening Book Probing**:
  - Integrated 1.48 MB Polyglot master opening book (`performance.bin`) into root search, executing book moves in <2ms with 0 wasted search time.
- **Bitboard-Accelerated Positional Evaluation**:
  - Native 64-bit bitboard loops replacing dictionary iterations (>500,000 evaluations/sec).
  - Positional terms: Bishop pair bonus (+35/+50 cp), Rook open (+25 cp) & semi-open (+15 cp) file heuristics.
- **Advanced Tree Search Pruning & Extensions**:
  - Null-Move Pruning (NMP) with zugzwang safety guards.
  - Late Move Reductions (LMR) for quiet candidate moves.
  - Check Extensions (+1 ply depth when in check).
  - Delta Pruning & futile capture skipping in Quiescence search.
- **UCI Modernization (`orion_uci.py`)**:
  - Modernized `orion_uci.py` to UCI v2.2 exposing the full tournament search engine to GUIs (Arena, Cutechess) and Lichess bot bridges.

### Changed
- Tournament score climbed to an all-time record **70.0% (7.0 / 10)** in the calibrated 6-engine arena.
- Swept `Sunfish-2026` 2–0 and `Stockfish-1350` 2–0, drew `Vice-1.2` 1–1.
- Maximum-likelihood rating with Vice anchor: **1777.1 Elo**.

## [2.1.0] - 2026-10-11
### Added
- **Gate 4 Search Engine Optimizations (`tournament_engine.py`)**:
  - 64-bit Zobrist Hash Table Transposition Table (`TranspositionTable`) with `EXACT`, `LOWERBOUND`, and `UPPERBOUND` flags.
  - Pre-flattened 64-square midgame/endgame Tapered Piece-Square Tables (`evaluate_tapered`) with linear phase interpolation.
  - Iterative Deepening with aspiration-friendly move persistence.
  - Move ordering enhancement: TT best move $\rightarrow$ MVV-LVA capture sorting $\rightarrow$ 2 Killer move slots per ply $\rightarrow$ History heuristic $\rightarrow$ Neural policy priors.
- **Full Round-Robin Arena Benchmark**:
  - Full round-robin tournament script (`run_round_robin_tournament.py`) connecting all 6 participants (`BBC-1.2`, `ORION-Search-Full`, `Vice-1.2`, `Sunfish-2026`, `Stockfish-1350`, `ORION-Neural-Only`).
  - Dual Ordo maximum-likelihood evaluation (`full_tournament_ordo_report.txt`, `full_tournament_ordo_anchor_sf.txt`).
  - Saved complete 30-game tournament PGN (`full_tournament.pgn`) and machine-readable metadata (`manifest.json`, `game_results.csv`).
- **Research Documentation**:
  - Deep architectural comparison between Stockfish 19 HalfKAv2 NNUE and ORION 30-Block ResNet (`docs/STOCKFISH_VS_ORION_RESEARCH.md`).
  - Phase 1 Baseline Architecture and Performance Audit (`ORION_PHASE1_BASELINE_AUDIT.md`).
  - Formal Anti-Cheating and Evaluation Protocol (`ORION_BENCHMARK_PROTOCOL.md`).

### Changed
- Verified engine speedup from ~115 NPS to **7,500 – 13,000 NPS (~65x – 110x speedup)**.
- Tactical puzzle solve rate improved from 44.4% (8/18) to **100% (18/18)**.
- Official calibrated tournament score improved from 25.0% (2.5/10) to **65.0% (6.5/10)**, placing #2 overall.
- Empirical Ordo rating improved by **+535.6 to +698.4 Elo**:
  - Vice Anchor (~1850 CCRL): **1965.6 ± 262.1 Elo** (up from 1267.2 Elo).
  - Stockfish-1350 Anchor (1350 UCI): **1541.6 ± 277.4 Elo** (up from 1006.0 Elo).

## [2.0.0] - 2026-10-11
### Added
- Independent empirical rating pipeline replacing uncalibrated homemade scripts with versioned engines (`BBC-1.2`, `Vice-1.2`, `Sunfish-2026`, `Stockfish-1350`).
- UHO EPD opening book integration with color-swapped paired games.
- Formal documentation of baseline uncertainty and dual-anchor intervals (`docs/CALIBRATED_BENCHMARK_REPORT.md`).
