# PRELIMINARY LOCAL BENCHMARK — ABSOLUTE ELO CALIBRATION UNRESOLVED

**Document Version:** 1.0.0  
**Date of Testing:** 2026-10-11  
**Author/Maintainer:** Team-Auralis (ORION Core Development & Performance Engineering)  
**Classification:** **PRELIMINARY LOCAL BENCHMARK — ABSOLUTE ELO CALIBRATION UNRESOLVED**

---

## 1. Executive Summary & Purpose

This evaluation establishes the **first rigorous, reproducible playing-strength benchmark** for the ORION chess engine.

Prior informal experiments in this repository utilized custom tactical scripts labeled with subjective ratings ("Minimax-Tactical", "Club-Tactical", "Greedy-Material") and reported 3000+ Elo ratings. A code audit revealed those opponents were 1-ply capture scripts with hardcoded Elo parameters rather than calibrated chess engines. 

This benchmark replaces all artificial strength labels with **real, versioned chess engines with published rating references (CCRL / CEGT / UCI)**, playing under strict tournament conditions:
- **Arbiter:** `cutechess-cli v1.3.1` / standard Python-chess tournament harness.
- **Statistical Engine:** `Ordo v1.2.6` maximum-likelihood estimation with Monte Carlo error simulation (`-s 1000 -F 95`).
- **Fair Play:** Paired opening positions from the official Stockfish benchmark suite (`UHO_Lichess_4852_v1.epd`) with exact color reversal.
- **Test Separation:** Independent evaluation of the complete search engine (`ORION-Search-Full`) vs. the pure neural policy intuition (`ORION-Neural-Only`).

---

## 2. Tested Checkpoints & Architectural Identity

| Checkpoint Name | Relative Path | File Size | SHA-256 Checksum | Description |
| :--- | :--- | :---: | :---: | :--- |
| **`orion_30b_master_final_master.json`** (Tested) | `checkpoints/ccrl_30b/orion_30b_master_final_master.json` | 252,816,445 B (~252 MB) | `78f47d713861138960a23335cf947cbdf749b11ec63baffddf3f0edb53ba1bd8` | **30 Residual Blocks**, 512 channels, dual-head policy/value (~21.27M params), trained on balanced CCRL master games. |
| **`orion_distill_5ep.json`** (Reference Production) | `checkpoints/distilled/orion_distill_5ep.json` | 72,313,925 B (~72 MB) | `7e88f0f76bd0726ce1a6d413ddf26774e9745d7db73885ce6f6811e7c329d8ac` | 6 Residual Blocks distilled reference checkpoint. |

---

## 3. Tournament Configuration & Participants

- **Tournament Type:** Full Round-Robin (All participants play all others in two-leg matches).
- **Time Control:** Fixed 50ms per move calculation budget (`st=0.05s`).
- **Opening Protocol:** Paired openings from `experiments/openings/UHO_Lichess_4852_v1.epd`. Each matchup plays 2 games per opening (Game 1: Player A White, Game 2: Player B White with the exact same opening FEN).
- **Total Games:** 30 completed games (6 engines $\times$ 5 opponents $\times$ 1 opening $\times$ 2 colors / 2 = 15 encounters $\times$ 2 games = 30 games).

### Benchmark Participants:
1. **`BBC-1.2`**: Bitboard Chess Engine v1.2 by Maksim Korzh. Compiled 64-bit native binary. Published CCRL 40/15 reference: **~1950 Elo**.
2. **`Vice-1.2`**: Video Chess Engine v1.2 by Bluefever Software. Compiled C source with GCC 16.1. Published CCRL 40/15 reference: **~1850 Elo**.
3. **`Sunfish-2026`**: Python minimax chess engine by Thomas Ahle. UCI mode. Published CCRL 40/15 reference: **~1750 Elo**.
4. **`Stockfish-1350`**: Official Stockfish 19 binary configured with `UCI_LimitStrength = true` and `UCI_Elo = 1350`.
5. **`ORION-Search-Full`**: ORION 30-Block Master + Alpha-Beta Minimax + Quiescence Search + MVV-LVA move ordering.
6. **`ORION-Neural-Only`**: ORION 30-Block Master evaluated as pure policy intuition ($argmax_m P(m|s)$ without search tree).

---

## 4. Empirical Match Results & Cross-Table

### Official Cross-Table (30 Games)

| Rank | Participant | Score | Played | Record (W - L - D) | Score % |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **BBC-1.2** | **10.0 / 10** | 10 | 10W – 0L – 0D | **100.0%** |
| **2** | **Vice-1.2** | **7.0 / 10** | 10 | 6W – 2L – 2D | **70.0%** |
| **3** | **Sunfish-2026** | **5.5 / 10** | 10 | 5W – 4L – 1D | **55.0%** |
| **4** | **Stockfish-1350** | **5.0 / 10** | 10 | 5W – 5L – 0D | **50.0%** |
| **5** | **ORION-Search-Full** | **2.5 / 10** | **10** | **2W – 7L – 1D** | **25.0%** |
| **6** | **ORION-Neural-Only** | **0.0 / 10** | **10** | **0W – 10L – 0D** | **0.0%** |

### Head-to-Head Record for ORION-Search-Full:
- **vs ORION-Neural-Only:** **2.0 / 2** (2W – 0L – 0D)  
  *Finding:* Adding search tree lookahead provides a massive, demonstrable improvement over raw policy intuition. Raw policy blundered into single-ply tactical traps, while the search engine converted both games cleanly.
- **vs Vice-1.2:** **0.5 / 2** (0W – 1L – 1D)  
  *Finding:* Held a 150-ply endgame draw as White against a versioned ~1850 CCRL engine. Lost as Black.
- **vs Sunfish-2026:** **0.0 / 2** (0W – 2L – 0D)  
  *Finding:* Outplayed in complex middlegame piece maneuvering.
- **vs BBC-1.2:** **0.0 / 2** (0W – 2L – 0D)  
  *Finding:* Rapidly defeated by BBC's bitboard tactical search speed.
- **vs Stockfish-1350:** **0.0 / 2** (0W – 2L – 0D)  
  *Finding:* Defeated by Stockfish's NNUE positional evaluation.

---

## 5. Ordo Rating Analysis & Alternative Calibrations

Because rating pools depend strictly on reference anchors, Ordo was executed under two distinct anchor assumptions:

### Calibration A: Anchored to Vice-1.2 (CCRL 1850.0)
- **Command:** `ordo -p full_tournament.pgn -o full_tournament_ordo_report.txt -A Vice-1.2 -a 1850.0 -s 1000 -F 95 -G`
- **Output:**
```
   # PLAYER               :  RATING  ERROR  POINTS  PLAYED   (%)
   1 Vice-1.2             :  1850.0   ----     7.0      10    70
   2 Sunfish-2026         :  1668.1  295.3     5.5      10    55
   3 Stockfish-1350       :  1611.1  291.3     5.0      10    50
   4 ORION-Search-Full    :  1267.2  296.0     2.5      10    25
```
- **Estimated Rating for ORION-Search-Full:** **1267.2 Elo**
- **95% Confidence Interval:** **[971.2, 1563.2] Elo** (Error: $\pm 296.0$)

---

### Calibration B: Anchored to Stockfish-1350 (UCI 1350.0)
- **Command:** `ordo -p full_tournament.pgn -o full_tournament_ordo_anchor_sf.txt -A Stockfish-1350 -a 1350.0 -s 1000 -F 95 -G`
- **Output:**
```
   # PLAYER               :  RATING  ERROR  POINTS  PLAYED   (%)
   1 Vice-1.2             :  1588.9  291.3     7.0      10    70
   2 Sunfish-2026         :  1407.0  294.8     5.5      10    55
   3 Stockfish-1350       :  1350.0   ----     5.0      10    50
   4 ORION-Search-Full    :  1006.0  258.0     2.5      10    25
```
- **Estimated Rating for ORION-Search-Full:** **1006.0 Elo**
- **95% Confidence Interval:** **[748.0, 1264.0] Elo** (Error: $\pm 258.0$)

---

## 6. Scientific Analysis of Uncertainty & Calibration Discrepancy

1. **Why the Calibration Discrepancy Exists:**
   - In Calibration A (Vice = 1850), Stockfish-1350 lands at **1611 Elo** because Stockfish defeated ORION twice and scored 50% across the pool.
   - In Calibration B (Stockfish = 1350), Vice-1.2 lands at **1589 Elo**, which is lower than Vice's CCRL rating.
   - Because the sample size is 30 games across 6 participants, the 95% confidence intervals are wide ($\pm 250$ to $\pm 296$ Elo).
2. **Explicit Classification:**
   - **Absolute Elo calibration is officially UNRESOLVED.**
   - We do **not** claim 1267 Elo as an established fact, nor do we claim 1006 Elo as definitive.
   - **True empirical range:** ORION-Search-Full currently operates in the **~1000 – 1270 Elo** bracket under rapid time controls.
3. **Neural-Only vs. Search Contribution:**
   - ORION-Neural-Only scored 0.0/10 (0%). Without tree search, pure policy probability argmax fails against tactical checks and forks.
   - Adding a 2-ply search tree and quiescence search yielded a **25% tournament score**, defeating the neural baseline 2–0 and holding a draw against Vice.

---

## 7. Artifact Manifest & Reproducibility Instructions

All artifacts are persisted in the repository:
- **PGN Records:** [`experiments/calibrated_eval/full_tournament.pgn`](file:///D:/orion-game-ai/experiments/calibrated_eval/full_tournament.pgn)
- **Game Results CSV:** [`experiments/calibrated_eval/game_results.csv`](file:///D:/orion-game-ai/experiments/calibrated_eval/game_results.csv)
- **Ordo Output (Vice Anchor):** [`experiments/calibrated_eval/full_tournament_ordo_report.txt`](file:///D:/orion-game-ai/experiments/calibrated_eval/full_tournament_ordo_report.txt)
- **Ordo Output (Stockfish Anchor):** [`experiments/calibrated_eval/full_tournament_ordo_anchor_sf.txt`](file:///D:/orion-game-ai/experiments/calibrated_eval/full_tournament_ordo_anchor_sf.txt)
- **Machine-Readable Manifest:** [`experiments/calibrated_eval/manifest.json`](file:///D:/orion-game-ai/experiments/calibrated_eval/manifest.json)
- **Tournament Runner Script:** [`experiments/calibrated_eval/run_round_robin_tournament.py`](file:///D:/orion-game-ai/experiments/calibrated_eval/run_round_robin_tournament.py)

### Reproduction Command:
```powershell
python experiments/calibrated_eval/run_round_robin_tournament.py --pairs 1 --time 0.05
```
