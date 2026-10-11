# ORION — Measurable Engineering Roadmap for Higher Elo

**Status:** ACTIVE SPECIFICATION & ROADMAP  
**Baseline Performance:** 25.0% score (2.5/10) in 30-game preliminary round-robin against calibrated engines (Vice 1.2, Sunfish 2026, BBC 1.2, Stockfish 1350).  
**Calibrated Elo Estimate:** 1006.0 to 1267.2 Elo (Uncertainty ±258 to ±296 Elo; Absolute calibration unresolved).  
**Target Milestone 1:** Solid 1600–1800 Elo against calibrated engines (Vice 1.2, Sunfish).  
**Target Milestone 2:** Competitive 2000+ Elo against BBC 1.2 and calibrated intermediate engines.  

---

## 1. Executive Summary & Root-Cause Diagnosis

In the calibrated round-robin tournament (`docs/CALIBRATED_BENCHMARK_REPORT.md`), `ORION-Search-Full` scored **2 wins, 7 losses, and 1 draw (25.0%)**, while `ORION-Neural-Only` scored **0.0% (0/10)**. 

Rigorous code audit of `experiments/calibrated_eval/tournament_engine.py` identified the following critical failure modes that explain why ORION underperformed:

### Defect 1: Quiescence Search Checkmate Blindness (Critical Correctness Defect)
In `quiescence_search()`:
```python
def quiescence_search(board: chess.Board, alpha: float, beta: float, q_depth: int = 3) -> float:
    stand_pat = evaluate_material(board)
    if q_depth == 0 or board.is_checkmate():
        return stand_pat
```
When a player is delivered checkmate, `board.is_checkmate()` is true, but the function returns `stand_pat = evaluate_material(board)`. If the losing side has a slight material surplus (e.g., +100 cp for a pawn), the engine evaluates being checkmated as **+100 centipawns** instead of **-30,000 (terminal loss)**! The engine was structurally blind to checkmate in tactical capture sequences.

### Defect 2: Missing Check Evasions & In-Check Stand-Pat in Quiescence
Standard quiescence search assumes a player can choose "not to move" (stand-pat). However, when the king is in check, standing pat is illegal under the rules of chess: the player **must** evade check. The current quiescence search evaluated `stand_pat` even while in check and only generated captures, meaning quiet check evasions were pruned, causing fatal tactical blunders.

### Defect 3: Extremely Truncated Candidate Search (Shallow Horizon)
The search function in `OrionSearchPlayer` evaluated only the top 8 moves according to the policy head, searching:
- Moves 0–3 to depth 2
- Moves 4–7 to depth 1
- Moves 8+ were pruned entirely without search

If a critical defensive move was ranked 9th by the neural network's policy head, the engine **never searched it**, resulting in unforced tactical blunders.

### Defect 4: Absence of Iterative Deepening & Transposition Table
The tournament search evaluated at fixed shallow depth with no Transposition Table (TT). Without a TT:
- Identical positions reached through transposition were re-searched from scratch.
- No hash move was available to seed move ordering at the root or interior nodes.
- Time management was static rather than dynamic.

### Defect 5: Pure Material-Only Leaf Evaluation
The evaluation function only counted piece values (`evaluate_material`) without Piece-Square Tables (PST), king shelter, open files, or pawn structure. All positional understanding was left to the neural policy prior.

---

## 2. Six-Stage Engineering Roadmap

```
Stage A: Correctness & Tactical Foundations (Fix Mate Scoring & Q-Search)
   │
   ▼
Stage B: Search Performance & Depth (Iterative Deepening, TT, Move Ordering)
   │
   ▼
Stage C: Classical Positional Evaluation (Tapered PSTs, King Safety, Pawn Structure)
   │
   ▼
Stage D: Hybrid Neural Integration (Batched GPU Inference, Value Head Blending)
   │
   ▼
Stage E: Training Data & Knowledge Distillation (High-Elo Data Curation, Policy Calibration)
   │
   ▼
Stage F: Continuous Calibrated Validation (Cutechess Regressions, Ordo Tracking)
```

---

### Stage A: Correctness & Tactical Foundations (Priority 1)
**Goal:** Eliminate all tactical blindness and illegal evaluation states.

1. **Ply-Relative Checkmate Scoring:**
   - Define terminal checkmate score as `MATE_SCORE = 30000.0`.
   - In search, return `-MATE_SCORE + ply` so the engine prefers faster mates when winning and delays mate when losing.
   - For stalemates, return `0.0`.
2. **In-Check Quiescence Search:**
   - If `board.is_check()`, do not evaluate `stand_pat`. Generate all legal evasions.
   - If no legal evasions exist, return `-MATE_SCORE + ply`.
3. **Move Legality & Termination Verification:**
   - Ensure terminal conditions (`is_stalemate()`, `is_insufficient_material()`, `can_claim_threefold_repetition()`) return `0.0`.
4. **Verification Metric:**
   - 100% solve rate on standard Mate-in-1 and Mate-in-2 tactical puzzle suites.

---

### Stage B: Search Performance & Depth Expansion (Priority 2)
**Goal:** Reach depth 5–7 within 500ms per move.

1. **Iterative Deepening:**
   - Search depth 1, 2, 3, ... iteratively until time limit is reached.
   - Always return the best move from the last fully completed iteration.
2. **Transposition Table (TT):**
   - Implement 64-bit Zobrist hashing for board states.
   - Store: `(key, depth, score, flag, best_move)` where flag is `EXACT`, `LOWERBOUND` (beta cutoff), or `UPPERBOUND` (alpha cutoff).
   - Adjust mate scores when storing/retrieving from TT relative to root ply.
3. **Move Ordering Pipeline:**
   - 1. TT / Hash move (if available)
   - 2. Good captures ordered by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
   - 3. Killer moves (quiet moves that caused beta cutoffs at the same ply)
   - 4. History heuristic scores
   - 5. Neural policy prior score
4. **Pruning Heuristics:**
   - Null Move Pruning (NMP) for depth >= 3 when not in check.
   - Reverse Futility Pruning (RFP) for depth <= 2.
   - Late Move Reductions (LMR) for quiet moves late in the move list.

---

### Stage C: Classical Positional Evaluation (Priority 3)
**Goal:** Guide the search with solid chess fundamentals even when neural inference is shallow.

1. **Tapered Piece-Square Tables (PST):**
   - Middlegame and endgame tables for all piece types.
   - Interpolate evaluation based on non-pawn material phase (game phase 0 to 24).
2. **King Safety:**
   - Pawn shield bonuses around castled kings.
   - Penalty for missing pawns on king files or open enemy files.
3. **Pawn Structure:**
   - Penalties for doubled, isolated, and backward pawns.
   - Bonuses for passed pawns increasing by rank.
4. **Piece Mechanics:**
   - Bishop pair bonus (+30 cp).
   - Rooks on open and semi-open files (+15 to +25 cp).
   - Rooks on the 7th rank (+35 cp).

---

### Stage D: Hybrid Neural Integration (Priority 4)
**Goal:** Maximize synergy between the 21.27M parameter neural network and the alpha-beta search.

1. **Batched GPU Inference:**
   - Evaluate root candidate moves in single batch tensors rather than serial forward passes.
2. **Value Head Blending:**
   - Incorporate the value head output $V \in [-1, 1]$ scaled to centipawns: $\text{eval}_{\text{leaf}} = (1 - w) \cdot \text{eval}_{\text{classical}} + w \cdot (V \times 1000)$.
3. **Policy Prior as Soft Move Ordering:**
   - Use policy log-probabilities strictly to sort moves in the move generator, avoiding arbitrary additions to minimax minimax values that could corrupt tactical thresholds.

---

### Stage E: Training Data & Knowledge Distillation (Priority 5)
**Goal:** Eliminate noise in the neural policy/value weights.

1. **CCRL & Lichess Elite Dataset Filtering:**
   - Restrict training games to players rated >2500 Elo.
   - Filter out rapid/blitz time-trouble blunders.
2. **Distillation from Modern Engines:**
   - Train value and policy heads against evaluations from Stockfish 16+ depth 18+ evaluations.
3. **Controlled Checkpoint Verification:**
   - Every candidate checkpoint must pass a standardized validation loss and tactical test battery before deployment.

---

### Stage F: Continuous Calibrated Validation (Priority 6)
**Goal:** Ensure all claimed Elo gains are empirically verified.

1. **Automated Cutechess Regression Suite:**
   - Automated gauntlet against the reference pool:
     * Sunfish 2026 (Anchor ~1750 Elo)
     * Vice 1.2 (Anchor ~1850 Elo)
     * BBC 1.2 (Anchor ~1950 Elo)
     * Stockfish 19 UCI_Elo=1350
2. **Paired Openings & Color Swaps:**
   - All tests use balanced opening suites (e.g. UHO / 4-ply opening suite) with identical openings played from both sides.
3. **Ordo Rating Updates:**
   - Compute Bayes Elo / Maximum Likelihood rating with explicit 95% confidence intervals and fixed anchors.
   - Never merge search changes into `main` without a statistically significant non-negative Elo delta.

---

## 3. Implementation Phasing & Governance

| Stage | Focus Area | Success Metric | Gate to Next Stage |
|---|---|---|---|
| **A** | Correctness & Mate Scoring | 100% mate-in-1 & mate-in-2 solve rate | Zero illegal/blind mate failures |
| **B** | Iterative Deepening & TT | Search depth 5+ at 0.5s/move; NPS > 10,000 | >50% score against Sunfish 2026 |
| **C** | Tapered Positional Evaluation | Material + PST balance; improved draw retention | >60% score against Sunfish 2026 |
| **D** | Hybrid Neural Integration | Root policy ordering without tactical corruption | >40% score against Vice 1.2 |
| **E** | Checkpoint Distillation | Top-1 policy accuracy > 48% on GM test set | Outperforms previous checkpoint |
| **F** | Calibrated Gauntlet | Ordo rating > 1600 Elo with 95% CI bounds | Stable tournament release candidate |

*Maintained under ORION Open Engineering Standards. All Elo claims must be accompanied by full PGNs and reproducible Ordo logs.*
