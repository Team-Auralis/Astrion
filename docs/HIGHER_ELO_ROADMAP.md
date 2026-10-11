# ORION Engineering Roadmap to 2000+ Elo

## 1. Executive Vision & Objectives
ORION currently plays at **1777.1 Elo** (70.0% win rate against CCRL anchors) and has proven tactical accuracy of **82.9%**.
To scale ORION into the 2000+ Elo tier and compete with intermediate-to-advanced open-source engines, this roadmap outlines high-impact, statistically measurable milestones.

---

## 2. Ranked Milestones & Engineering Targets

```
                    ┌──────────────────────────────────────────────┐
                    │ Current Status: ORION v2.3 (~1777-1850 Elo)   │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Milestone 1: Static Exchange Evaluation (SEE)│
                    │ Expected Gain: +40 to +60 Elo                │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Milestone 2: Native C/C++ / Cython Search    │
                    │ Expected Gain: +100 to +150 Elo              │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Milestone 3: NNUE Quantized Evaluation (AVX2)│
                    │ Expected Gain: +120 to +180 Elo              │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Milestone 4: Syzygy Tablebase Integration    │
                    │ Expected Gain: +30 to +50 Elo                │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Target Strength: ORION v3.0 (>2100 Elo)      │
                    └──────────────────────────────────────────────┘
```

---

## 3. Detailed Milestone Specifications

### Milestone 1: Static Exchange Evaluation (SEE) Pruning (+50 Elo)
- **Problem:** Currently, all captures are sorted using MVV-LVA. A losing capture (e.g. Queen takes defended Pawn) is still evaluated before killer moves, wasting hundreds of search nodes per position.
- **Solution:** Implement a recursive 64-bit bitboard `SEE(move)` algorithm.
  - Good captures ($\text{SEE} \ge 0$) ordered before killers.
  - Bad captures ($\text{SEE} < 0$) pruned in quiescence search and delayed in main search.
- **Engineering Effort:** 2 days.

### Milestone 2: Native C++/Cython Bitboard Engine Compilation (+120 Elo)
- **Problem:** Python interpreter overhead caps node throughput at ~15,000 nodes/sec during search. BBC-1.2 and Vice-1.2 achieve >500,000 to 2,000,000 nodes/sec in native C, giving them 2–4 extra plies of search depth in the same time limit.
- **Solution:** Compile the board make/unmake and alpha-beta negamax recursion in native C++ with 64-bit magic bitboards.
- **Expected Result:** 30x–50x throughput gain (>500,000 NPS), increasing effective search depth from 4–5 plies to 8–10 plies.
- **Engineering Effort:** 1–2 weeks.

### Milestone 3: Efficiently Updatable Neural Network (NNUE) (+150 Elo)
- **Problem:** The 18.8M ResNet takes 12.5 ms per inference, making it only usable at the root node. Internal tree nodes rely on classical tapered PST evaluation.
- **Solution:** Train and integrate an int8 quantized NNUE (e.g., $(768 \to 256) \times 2 \to 32 \to 1$) with incremental feature accumulation on board push/pop.
- **Expected Result:** Sub-microsecond neural inference at every node with AVX2 SIMD acceleration, combining deep neural evaluation with full search depth.
- **Engineering Effort:** 2–3 weeks.

### Milestone 4: Syzygy Endgame Tablebases (+40 Elo)
- **Problem:** In 3-to-5 piece endgames, the engine currently calculates moves dynamically, occasionally missing optimal winning conversions or drawing theoretical wins.
- **Solution:** Connect Syzygy 3-4-5 piece tablebases via `chess.syzygy`. Instant probe of Win/Draw/Loss (WDL) and Distance-to-Zero (DTZ).
- **Engineering Effort:** 2 days.

---

## 4. Verification & Testing Discipline (SPRT)
Every future modification must pass a Sequential Probability Ratio Test (SPRT) before merging:
- **Null Hypothesis ($H_0$):** $\Delta \text{Elo} \le 0$
- **Alternative Hypothesis ($H_1$):** $\Delta \text{Elo} \ge +20$
- **Confidence Gates:** $\alpha = 0.05, \beta = 0.05$
- **Opponent Pool:** Vice-1.2, Stockfish-1350, Sunfish-2026 with paired openings.
