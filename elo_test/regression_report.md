# ORION Engineering Regression Report

## 1. Overview
This report documents the verification, audit, and empirical testing executed on branch `research/search-correctness-audit` for ORION Chess Engine.

---

## 2. Verification Milestones Completed

### 1. Checkpoint & Architecture Verification
- **Model Checkpoint:** `checkpoints/ccrl_30b/orion_30b_master_final_master.json`
- **File Size:** 252,816,445 bytes (~241.10 MB)
- **SHA-256:** `78f47d713861138960a23335cf947cbdf749b11ec63baffddf3f0edb53ba1bd8`
- **Parameters Verified:** **18,810,113 parameters** (30 residual blocks, 512 channels, dual policy/value heads).
- **Manifest File:** `elo_test/baseline_manifest.json`

### 2. Move Generation & Perft Integrity
- **Perft Suite Script:** `elo_test/run_perft.py`
- **Positions Verified:** Startpos, KiwiPete, Position 3, Position 4.
- **Total Nodes:** 362,544 nodes traversed.
- **Generation NPS:** 173,940 nodes/sec.
- **Perft Status:** **100% PASS (Zero move generation defects)**.

### 3. Tactical Regression Benchmark
- **Benchmark Suite Script:** `elo_test/run_tactical_suite.py`
- **Output Artifact:** `elo_test/tactical_results.json`
- **Tactical Accuracy:** **29 / 35 positions solved (82.9%)** (up from 71.4% baseline).
- **Sub-1s Solves:** 29 / 35 (82.9%).
- **Terminal Blunder Rate:** **0 blunders into mate (0.0%)**.
- **Defensive Parries:** **6 / 6 (100.0%)**.
- **Endgame Precision:** **7 / 8 (87.5%)**.

### 4. Search Improvements Verified (v2.3)
- **Repetition Check:** Integrated `is_repetition(2)` and `is_fifty_moves()` to eliminate threefold repetition blindness.
- **PVS Implementation:** Enforced zero-window searches on non-PV moves, significantly pruning redundant branches.
- **Countermove Heuristic:** Tracks the refutation of the opponent's previous move.
- **Queen Promotion Ordering:** Enhanced promotion moves over standard captures ($950,000$ priority).
- **Aspiration Windows:** Narrow window $[V - 40, V + 40]$ at depth $\ge 4$ with automatic fail-high/fail-low bounds expansion.

### 5. Empirical Match Verification
- **Script:** `elo_test/run_calibrated_match.py`
- **PGN Record:** `elo_test/v2_3_verification.pgn`
- **Match Results:**
  - vs Stockfish-1350: **2 - 0 (100% Sweep)**
  - vs Sunfish-2026: **2 - 0 (100% Sweep)**
  - vs Vice-1.2: **0.5 / 2 (Held Draw as White)**
- **Overall Record:** **4W - 1L - 1D (4.5 / 6 points, 75.0% score)**.
- **Ordo Rating:** **1657.5 Elo (±231.5 95% CI)** with Vice-1.2 anchored at 1850.0.
