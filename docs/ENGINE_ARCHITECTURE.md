# ORION Chess Engine Architecture & Specification (v2.3)

## 1. Executive Summary & Verification Truth
ORION is a hybrid deep neural network and high-performance alpha-beta chess engine developed by Team-Auralis. 

### Core Architectural Reality
Contrary to theoretical or legacy claims (e.g. 21.27M parameters, 3000 Elo), the empirical programmatic audit verifies:
- **Total Parameters:** **18,810,113 parameters** (100% trainable).
- **Production Neural Checkpoint:** `checkpoints/ccrl_30b/orion_30b_master_final_master.json`
- **Checkpoint File Size:** 252,816,445 bytes (~241.10 MB)
- **Checkpoint SHA-256:** `78f47d713861138960a23335cf947cbdf749b11ec63baffddf3f0edb53ba1bd8`
- **Production Playing Strength:** **1777.1 Elo** (30-game tournament baseline, 70.0% win rate against CCRL anchors) and **1657.5 Elo (±231.5 95% CI)** under blitz verification controls.

---

## 2. Neural Network Topology & Layer Breakdown

The neural architecture (`OrionResNet` in `train_fast.py`) is structured as a deep residual feed-forward network with dual policy and value heads:

```
Input Tensor: [Batch, 1152] (18 x 8 x 8 flattened bitboard planes)
        │
        ▼
Linear Layer: Linear(1152 -> 512, bias=True)
BatchNorm1d(512) + ReLU
        │
        ▼
30 x Residual Blocks (ResBlock):
   ┌────────────────────────────────────────┐
   │ x_identity = x                         │
   │ x = Linear(512 -> 512)                 │
   │ x = BatchNorm1d(512) + ReLU            │
   │ x = Linear(512 -> 512)                 │
   │ x = BatchNorm1d(512)                   │
   │ out = ReLU(x + x_identity)             │
   └────────────────────────────────────────┘
        │
   ┌────┴────────────────────────┐
   ▼                             ▼
[Policy Head]              [Value Head]
Linear(512 -> 512)         Linear(512 -> 64)
BatchNorm1d(512) + ReLU    BatchNorm1d(64) + ReLU
Linear(512 -> 4096)        Linear(64 -> 1)
Logits: [Batch, 4096]      Sigmoid: [Batch, 1] ∈ [0.0, 1.0]
```

### Exact Parameter Count Derivation
1. **Input Projection:**
   - Linear(1152, 512): $1152 \times 512 + 512 = 590,336$
   - BatchNorm1d(512): $512 \times 2 = 1,024$
   - Subtotal: **591,360 parameters**
2. **30 Residual Blocks:**
   - Each block has 2 Linear(512, 512) and 2 BatchNorm1d(512):
     $2 \times (512 \times 512 + 512) + 2 \times (512 \times 2) = 2 \times 262,656 + 2,048 = 527,360$ parameters per block.
   - 30 Blocks: $30 \times 527,360 = \mathbf{15,820,800\text{ parameters}}$.
3. **Policy Head:**
   - Linear(512, 512): $512 \times 512 + 512 = 262,656$
   - BatchNorm1d(512): $1,024$
   - Linear(512, 4096): $512 \times 4096 + 4096 = 2,101,248$
   - Subtotal: **2,364,928 parameters**.
4. **Value Head:**
   - Linear(512, 64): $512 \times 64 + 64 = 32,832$
   - BatchNorm1d(64): $128$
   - Linear(64, 1): $64 \times 1 + 1 = 65$
   - Subtotal: **33,025 parameters**.
5. **Grand Total:**
   $$591,360 + 15,820,800 + 2,364,928 + 33,025 = \mathbf{18,810,113\text{ parameters}}.$$

---

## 3. Board Encoding & Action Space

### Input Representation (18 planes of $8 \times 8 = 1152$ features)
- **Planes 0–5:** White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
- **Planes 6–11:** Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
- **Plane 12:** Active side to move (all 1s for White, all 0s for Black)
- **Planes 13–16:** Castling rights (White Kingside, White Queenside, Black Kingside, Black Queenside)
- **Plane 17:** En-passant target square mask

### Action Mapping (4,096 discrete move channels)
Each move is mapped into an integer index:
$$\text{Action Index} = \text{from\_square} \times 64 + \text{to\_square}$$
where squares are indexed $0 \dots 63$.
Promotions are resolved via standard promotion flag offsets in the move generator.

---

## 4. Evaluation Engine (Bitboard Tapered Positional Evaluation)

To achieve over 500,000 evaluations per second without requiring GPU inference on every internal search tree node, ORION uses a bitboard-accelerated tapered evaluation function:

### Evaluation Factors
1. **Material Values:**
   - Pawn: 100 cp
   - Knight: 320 cp
   - Bishop: 330 cp
   - Rook: 500 cp
   - Queen: 900 cp
   - King: 20,000 cp
2. **Piece-Square Tables (PST):**
   - Tomasz Michniewski simplified evaluation tables tailored for opening/middlegame and endgame transitions.
3. **Positional Factors:**
   - **Bishop Pair Bonus:** +35 cp (Middlegame), +50 cp (Endgame)
   - **Rook Open File Bonus:** +25 cp
   - **Rook Semi-Open File Bonus:** +15 cp
4. **Game Phase Tapering:**
   Game phase is computed dynamically based on remaining non-pawn material ($0 \dots 24$). The final evaluation score is linearly interpolated:
   $$\text{Score} = \frac{\text{MG\_Score} \times \text{Phase} + \text{EG\_Score} \times (24 - \text{Phase})}{24}$$

---

## 5. Search Engine Pipeline (v2.3)

1. **Polyglot Opening Book:** Probes 1.48 MB `performance.bin` for instant $<2\text{ms}$ opening book moves.
2. **Root Neural Prior:** At root, the 18.8M model executes once to seed and sort candidate moves with grandmaster intuition.
3. **Iterative Deepening:** Searches depths $1, 2, 3 \dots d$ with responsive time management stopping at $60\%$ of move allocation.
4. **Aspiration Windows:** At depth $\ge 4$, searches within a narrow window $[V - 40, V + 40]$. On fail-low or fail-high, automatically expands bounds.
5. **Principal Variation Search (PVS):** Searches the PV move with full window; all subsequent moves are searched with a zero-window $[- \alpha - 1, -\alpha]$ to trigger maximal cutoffs.
6. **Null-Move Pruning (NMP):** If passing the turn at depth $\ge 3$ still exceeds $\beta$, prunes the entire subtree with reduction $R = 2 + \lfloor \text{depth} / 6 \rfloor$.
7. **Late Move Reductions (LMR):** Late quiet non-tactical moves (index $\ge 4$, depth $\ge 3$) are searched at reduced depth $\max(1, \text{depth} - 2)$.
8. **Check Extensions:** Automatically extends depth by $+1$ ply when the side to move is in check.
9. **Transposition Table (TT):** 64-bit Zobrist hashing with exact, lower-bound, and upper-bound flags, with mate distance normalized by ply.
10. **Move Ordering Heuristics:**
    - TT Hash Move ($1,000,000$)
    - Queen Promotions ($950,000$)
    - MVV-LVA Captures ($100,000 + 10 \times V_{\text{victim}} - V_{\text{attacker}}$)
    - Underpromotions ($90,000$)
    - Killer Moves ($80,000 / 70,000$)
    - Countermove Heuristic ($60,000$)
    - History Heuristic ($\le 50,000$)
11. **Quiescence Search:** Resolves tactical sequences, captures, and checks to prevent horizon effect, with delta pruning.
