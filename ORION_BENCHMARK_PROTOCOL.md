# ORION Chess Engine — Empirical Benchmark & Rating Protocol

**Status:** ACTIVE GOVERNANCE SPECIFICATION  
**Author:** Principal Chess Engine Architect & Independent Benchmark Auditor  
**Effective Date:** October 2026  
**Tools:** Cutechess CLI (v1.3.1), Ordo (v1.2.6), Python-chess (v1.11.2)  

---

## 1. Principles of Empirical Rating Validation

1. **No Fabricated Strength:** Never advertise, publish, or optimize for arbitrary Elo labels. Ratings must be derived exclusively from completed games against versioned, publicly verified reference engines.
2. **Strict Scientific Controls:** Every match must enforce identical time controls, paired opening suites, balanced color assignments, and deterministic hardware allocations.
3. **Statistical Rigor:** All reported ratings must include sample sizes, win/draw/loss counts, score percentages, anchor definitions, and statistically defensible 95% confidence intervals.
4. **Permanent Artifact Preservation:** Every tournament game must be recorded in standard PGN format with FIDE headers, move times, and results preserved.

---

## 2. Standardized Opponent Pool & Anchors

To eliminate rating inflation, ORION is benchmarked against engines with documented source code and established ratings on published rating lists (CCRL, CEGT):

| Engine Identifier | Source / Release | Target Rating Anchor | Anchor Source |
|---|---|---|---|
| **Stockfish 19 (UCI_Elo=1350)** | Stockfish dev team | **1350.0 Elo** | Fixed UCI limit calibration |
| **Sunfish 2026** | Peter Österlund / Thomas Dybdahl Ahle | **~1750.0 Elo** | CCRL 40/15 blitz anchor |
| **Vice 1.2** | Bluefever Software (Richard Allbert) | **1850.0 Elo** | CCRL 40/15 anchor (1850 ± 25) |
| **BBC 1.2** | Maksim Korzh | **~1950.0 Elo** | CCRL 40/15 anchor (1950 ± 20) |
| **Stockfish 19 (UCI_Elo=1600)** | Stockfish dev team | **1600.0 Elo** | Fixed UCI limit calibration |

---

## 3. Tournament Execution Protocol

### A. Paired Openings & Color Equality
- Matches use balanced opening positions (e.g. `experiments/openings/UHO_Lichess_4852_v1.epd` or standard 4-ply opening suites).
- Every opening position is played **twice** per opponent pair:
  - Game 1: Engine A plays White, Engine B plays Black.
  - Game 2: Engine A plays Black, Engine B plays White.
- This neutralizes first-move White advantage and guarantees unbiased evaluation.

### B. Time Control Standards
- **Sprint / Development Verification:** 5.0 seconds + 0.05 seconds increment per move (or fixed 0.5s/move).
- **Tournament / Calibration Benchmark:** 10.0 seconds + 0.1 seconds increment per move.
- **Draw Adjudication:** 
  - Standard 3-fold repetition and 50-move rule enforced by engine/cutechess.
  - Early draw adjudication: Move 40+ if material and evaluation differ by < 5 cp for 10 consecutive plies.
- **Resignation Adjudication:** Allowed if evaluation is $\le -900$ cp for 5 consecutive plies after move 30.

---

## 4. Ordo Rating Calculation Specification

Rating calculations must be performed using the verified Ordo binary:
```powershell
tools/ordo/ordo.exe -p <tournament.pgn> -a <Anchor_Rating> -A "<Anchor_Engine_Name>" -V -s 1000 -o <output_report.txt>
```

### Parameters
- `-p <pgn>`: Path to complete multi-game PGN file.
- `-a <rating>`: Fixed reference anchor numerical value.
- `-A "<name>"`: Exact string matching the anchor participant in PGN headers.
- `-V`: Enable variance calculation for confidence intervals.
- `-s 1000`: 1,000 bootstrap simulations for 95% confidence bounds ($[\text{rating} - 1.96\sigma, \text{rating} + 1.96\sigma]$).
- `-o <file>`: Text file output for permanent audit trail.

---

## 5. Acceptance Gate for Engine Changes (Progression Criteria)

Before any optimization or feature branch is merged into `main`:

1. **Gate A: Unit & Tactical Correctness:**
   - 100% pass rate on `tests/test_tactical_mate_scoring.py`.
   - $\ge 65\%$ solve rate on `experiments/calibrated_eval/benchmark_tactical_delta.py`.
2. **Gate B: No Speed Regression:**
   - Effective NPS or search depth must not regress for equivalent search parameters.
3. **Gate C: Statistical Non-Regression:**
   - Minimum 20 paired games (40 games total) against the benchmark pool.
   - The lower bound of the 95% confidence interval must not indicate a statistically significant rating decline ($p < 0.05$).
4. **Gate D: Artifact Completeness:**
   - PGN file, game results CSV, and Ordo log must be generated and archived in `experiments/` before commit.
