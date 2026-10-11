# Comprehensive Research Report: Stockfish 19 NNUE vs. ORION

**Date:** October 2026  
**Investigator:** ORION Engineering & Computer Chess Research Team  
**Methodology:** Agent-Reach Technical Survey & Local Codebase Reverse Engineering  

---

## 1. Executive Summary

Stockfish and ORION represent two fundamentally distinct paradigms in computer chess artificial intelligence:

| Dimension | Stockfish 19 (World Champion, ~3600+ CCRL Elo) | ORION (Current Architecture, ~1000–1270 Calibrated Elo) |
|---|---|---|
| **Core Paradigm** | Ultra-Fast Alpha-Beta Search + Incremental NNUE Value Evaluation | Deep Residual Network (AlphaZero-style Policy+Value) + Shallow Alpha-Beta |
| **Neural Architecture** | **NNUE** (`HalfKAv2_hm` sparse feature transformer + compact quantized MLP) | **ResNet-30** (30 residual blocks, 512 hidden channels, fully connected linear layers) |
| **Parameter Count** | ~5 to 20 Million (quantized **int8 / int16**) | 21.27 Million (**float32**) |
| **Inference Mechanism** | Incremental Accumulator updates via CPU SIMD (AVX2/AVX-512) | Full forward pass through 30 blocks on GPU/CPU for every position |
| **Search Speed (NPS)** | **10,000,000 – 50,000,000+ Nodes Per Second** | **100 – 500 Nodes Per Second** (~100,000x difference) |
| **Search Depth** | Depth **30 – 50+ ply** within standard time controls | Depth **2 – 3 ply** (+ quiescence search) |
| **Output Type** | Single scalar evaluation $V \in [-30000, +30000]$ (centipawns / win probability) | Dual heads: Policy prior (4096 move logits) + Value scalar $V \in [0, 1]$ |
| **Move Selection** | Alpha-Beta minimax, Aspiration windows, PVS, LMR, Null-Move, Singular Extensions | Policy-guided shallow search / Argmax over top policy logits |

---

## 2. In-Depth Architectural Comparison

### A. The Neural Network: NNUE vs. ResNet-30

#### 1. Stockfish's NNUE (Efficiently Updatable Neural Network)
- **Origin & Design:** Introduced in Stockfish 12 and iteratively refined up to Stockfish 19 (`nn-1a298aa575a0.nnue`).
- **Feature Transformer (`HalfKAv2_hm`):**
  - Encodes the board relative to both kings (`HalfKP` / `HalfKA`).
  - While the input feature space is huge (~45,000 to 90,000 potential piece-square features), **only ~32 features are active** at any time.
- **The Secret Sauce — The Accumulator:**
  - When a chess move occurs (e.g. `e2e4`), only 1 piece leaves a square and lands on another.
  - Stockfish **does not recompute the neural network from scratch**. It merely updates the accumulator vector by subtracting the weights of the old square and adding the weights of the new square:
    $$\text{Accumulator}_{\text{new}} = \text{Accumulator}_{\text{old}} - W_{\text{from}} + W_{\text{to}}$$
  - This update executes in **nanoseconds** using vectorized CPU integer SIMD instructions (`_mm256_add_epi16`, `_mm512_add_epi16`).
- **Compact Quantized Layers:**
  - After the accumulator, Stockfish passes the vector through tiny dense layers (e.g. `2x1024 -> 16 -> 32 -> 1`) with Clipped ReLU ($[0, 127]$).
  - All arithmetic is fixed-point integer (`int8` / `int16`). It never touches floating-point units or GPU pipelines.

#### 2. ORION's Model (`train_fast.py` / `orion_30b_master_final_master.json`)
- **Origin & Design:** Inspired by DeepMind's AlphaZero and Leela Chess Zero (LCZero).
- **Structure:**
  - Input: $18 \times 8 \times 8 = 1152$ floating-point channels representing piece placements, side to move, castling rights, and en-passant.
  - Architecture: 30 Residual Blocks. Crucially, in ORION's PyTorch implementation (`train_fast.py` lines 47–83), each residual block is composed of **Linear (Dense) layers** (`nn.Linear(512, 512)`), rather than 2D convolutional filters or attention layers.
  - Output:
    - **Policy Head:** 4096 logits corresponding to all possible chess move origin-destination coordinate pairs.
    - **Value Head:** 1 scalar representing game outcome probability.
- **Evaluation Cost:**
  - Every single board evaluation requires passing a 1152-element vector through 60 separate matrix multiplications ($512 \times 512 = 262,144$ operations per block $\times 30 = \sim 15.7$ million float32 multiply-accumulates).
  - Even on an RTX 3050 GPU, one forward pass takes ~2–5 ms. On CPU, it takes ~20–50 ms.
  - There are **no incremental updates**; transposition or sibling nodes must be computed in full.

---

### B. The Search Engine: Depth 40 vs. Depth 2

In computer chess, tactical strength is exponential in search depth ($O(b^d)$). 

#### 1. Why Stockfish Reaches Depth 40+
Stockfish can search 10 to 50 million positions per second because of:
1. **PVS (Principal Variation Search) & Alpha-Beta:** Narrows search windows aggressively.
2. **Aggressive Pruning:**
   - **Null Move Pruning (NMP):** If skipping a turn still results in a winning position, prune the sub-tree.
   - **Late Move Reductions (LMR):** Moves ranked low in the ordering are searched to significantly reduced depths.
   - **Reverse Futility Pruning (RFP):** Positions overwhelmingly ahead are cut off early.
3. **Move Ordering Mastery:**
   - Transposition Table (TT) hash move.
   - Good captures sorted by MVV-LVA and SEE (Static Exchange Evaluation).
   - Killer moves and history heuristic tables (Butterfly, Capture History, Countermove History).
4. **Deep Extensions:**
   - **Singular Extensions (SE):** If one move is demonstrably superior to all alternatives by a wide margin, extend search depth along that line.

#### 2. Why ORION Operates at Depth 2
- Because ORION's model takes ~2 ms per evaluation, searching 1,000 nodes would take 2 full seconds.
- In `experiments/calibrated_eval/tournament_engine.py`, ORION evaluated candidate moves to depth 2 (with depth 1 for lower candidates).
- At depth 2, the engine cannot see tactical traps that materialize 3 or 4 moves ahead (e.g. piece sacrifices, clearance tactics, skewers, or quiet defensive resources).

---

## 3. Structural Comparison Matrix

| Architectural Feature | Stockfish 19 | ORION | Impact on Playing Strength |
|---|---|---|---|
| **Board Geometry Understanding** | Feature sets (`HalfKAv2`, `FullThreats`, `PP_3Wide`) hardcode spatial relationships | Flattened 1152 1D vector into Linear layers (lacks 2D spatial inductive bias) | Stockfish immediately knows spatial piece-king relationships; ORION must learn rank/file proximity from scratch. |
| **Search Speed** | 20,000,000+ NPS | ~200 NPS | **100,000x gap**; Stockfish sees 20-30 moves deeper into the future. |
| **Tactical Horizon** | 30 to 50 plies | 2 to 4 plies | ORION falls victim to deep tactical combinations. |
| **Evaluation Type** | Centipawn value score | Move probabilities (Policy) + Win probability (Value) | ORION has strong opening "intuition" but lacks tactical calculation verification. |
| **Hardware Requirement** | Highly optimized for single-thread / multi-core standard CPUs | Requires discrete GPU tensor cores for reasonable speed | Stockfish runs at full strength anywhere; ORION is constrained by PyTorch/GPU overhead. |
| **Pruning Sophistication** | 15+ specialized heuristics (NMP, LMR, RFP, SE, ProbCut, History) | Basic alpha-beta with captures-first quiescence | Stockfish prunes 99.9% of irrelevant game tree branches safely. |

---

## 4. Why AlphaZero / LCZero Beat Stockfish in the Past (And Why ORION is Different)

A common question is: *"If AlphaZero and Leela Chess Zero (LCZero) also use deep neural networks and search only 1,000 to 50,000 NPS, why could they defeat Stockfish?"*

Understanding the difference between LCZero and ORION reveals the exact engineering gap:

1. **2D/3D Convolutions vs. Linear Layers:**
   - LCZero uses deep **ResNet Conv2D / Attention Transformer (BT4)** networks that preserve the $8 \times 8$ 2D spatial board topology.
   - ORION uses flattened `nn.Linear` layers, discarding spatial locality (a pawn at a2 has no structural connection to a3 other than learned weights).
2. **Search Architecture (MCTS vs. Alpha-Beta):**
   - AlphaZero and LCZero use **MCTS (Monte Carlo Tree Search) with PUCT**.
   - MCTS is designed for neural networks: it batches tree evaluations into GPU tensors (evaluating 64 or 256 positions simultaneously in one forward pass).
   - Alpha-Beta search is inherently serial (depth-first), which leaves GPU tensor cores 99% idle while waiting for serial Python calls.
3. **Training Scale:**
   - AlphaZero trained on hundreds of millions of self-play games on Google TPU clusters.
   - LCZero has played over 1 billion self-play games through community distributed computing.
   - ORION trained on thousands of CCRL / Lichess GM games.

---

## 5. Strategic Recommendations for ORION

To bridge the strength gap and climb the Elo ladder as outlined in `docs/HIGHER_ELO_ROADMAP.md`:

1. **Adopt NNUE-Style Evaluation for the Alpha-Beta Search Engine:**
   - Implementing a lightweight NNUE architecture (e.g. 768 $\to$ 256 $\to$ 32 $\to$ 1) in C++ or vectorized Python with incremental updates would immediately boost ORION's NPS from 200 to **1,000,000+**.
2. **Use the 21M ResNet Exclusively as an Opening Book / Move Prior:**
   - Rather than calling the heavy PyTorch model at every node in the search tree, run the 21M ResNet **once at the root position** to generate candidate move priors ($P(a|s)$) to guide the move ordering of a fast NNUE search.
3. **Transition to GPU-Batched MCTS if Retaining Deep Neural Architecture:**
   - If maintaining the AlphaZero paradigm, replace the serial Alpha-Beta search with batched MCTS using PUCT, asynchronous queueing, and tensor batching.
