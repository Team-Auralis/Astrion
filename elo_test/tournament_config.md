# ORION Tournament Protocol & Benchmark Specification

## 1. Overview & Testbed Environment
This document defines the rigorous, reproducible benchmarking standard for evaluating ORION Chess Engine against external reference engines.

### Host Hardware
- **Processor:** 11th Gen Intel(R) Core(TM) i5-11260H (12 logical cores @ 4.40 GHz)
- **Discrete GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6 VRAM)
- **RAM:** 8 GB DDR4 System Memory
- **Operating System:** Windows 11 Home 64-bit (Build 26100)
- **Python Runtime:** Python 3.14 / 3.12 64-bit with PyTorch 2.6 CUDA acceleration
- **Board Library:** `python-chess` 1.11.x

---

## 2. Benchmark Opponents & Published Anchors

| Participant | Version / Config | Engine Type | CCRL 40/15 Anchor | CEGT 40/20 Anchor | Notes |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Vice-1.2** | v1.2 (Bluefever) | C Bitboard / 0x88 | **1850.0 Elo** | ~1840 Elo | Primary CCRL Anchor; stable, classical alpha-beta |
| **Sunfish-2026** | 2026 Release | Python PST engine | **1750.0 Elo** | ~1720 Elo | Fast Python baseline opponent |
| **Stockfish-1350** | Stockfish 19 UCI_Elo=1350 | C++ NNUE / classical | **1350.0 Elo** | 1350.0 Elo | Calibrated lower anchor (`UCI_LimitStrength=true`) |
| **BBC-1.2** | v1.2 (Code Monkey King) | Native C Bitboard | **1950.0 Elo** | ~1920 Elo | Upper anchor; deep tactical bitboard search |
| **ORION-Neural-Only**| 30-block ResNet (18.8M) | Policy Head Argmax | Experimental | Experimental | Pure intuition baseline (no search tree) |
| **ORION-Search-Full**| v2.2 / v2.3 Hybrid | ResNet + PVS Search | Target Anchor | Target Anchor | Full production engine |

---

## 3. Opening Suite & Color Pairing Rules
To completely eliminate first-move advantage bias and book memorization variance:
1. **Paired Games:** Every single opening line is played exactly twice between any two opponents.
2. **Color Swap:** Game 1 features Engine A as White, Engine B as Black. Game 2 swaps colors: Engine B plays White, Engine A plays Black from the identical opening FEN.
3. **Opening Book:** Standard opening lines drawn from `experiments/openings/UHO_Lichess_4852_v1.epd` and master performance book (`performance.bin`).

---

## 4. Time Controls & Adjudication Protocol
- **Time Control:** Fixed budget per move (0.05s to 0.5s per move) or standard blitz increment.
- **Terminal Adjudication:**
  - Checkmate: 1-0 or 0-1
  - Stalemate: 1/2-1/2
  - Insufficient Material: 1/2-1/2 (FIDE rules)
  - 50-move rule: 1/2-1/2
  - Threefold repetition: 1/2-1/2
  - Material cutoff: $\ge 300$ cp advantage without terminal moves past ply 150 adjudicated as win/loss.
- **Illegal Moves / Crashes:** Engine that forfeits by illegal move or unhandled exception receives 0 points; opponent receives 1 point.

---

## 5. Statistical Estimation with Ordo
All rating estimations are generated using `ordo.exe` using standard Maximum Likelihood Estimation:
```bash
ordo.exe -p <tournament.pgn> -o <report.txt> -A Vice-1.2 -a 1850.0 -s 1000 -F 95 -G
```
Or with Stockfish-1350 as dual anchor:
```bash
ordo.exe -p <tournament.pgn> -o <report.txt> -A Stockfish-1350 -a 1350.0 -s 1000 -F 95 -G
```
Both 95% confidence intervals and standard errors must be documented.
