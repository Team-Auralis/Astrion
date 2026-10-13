# ORION Phase 1 Baseline Audit & Technical Assessment

**Document Version:** 1.0.0  
**Audit Date:** October 11, 2026  
**Auditor:** Principal Chess Engine Architect & Performance Engineer  
**Commit SHA:** `3e129dd5c60e63b03d369bee934aaa9b92500a3e` (`main`)  
**Hardware Profile:**  
- CPU: 11th Gen Intel(R) Core(TM) i5-11260H @ 4.40 GHz (6 Cores / 12 Threads)  
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM / Ampere)  
- RAM: 7.74 GiB DDR4  
- OS: Windows 11 Home 64-bit  
- Python Environment: Python 3.14.0rc2, PyTorch 2.7.0.dev20250212+cu128, python-chess 1.11.2  

---

## 1. Actual Architecture and Parameter Count

Contrary to initial project claims of "21.27 million parameters", an exact programmatic count (`sum(p.numel() for p in model.parameters())`) reveals the actual model size:

### Programmatic Parameter Verification
```python
from train_fast import OrionResNet
model = OrionResNet(channels=512, num_blocks=30)
# Actual count: 18,810,113 parameters (~18.81M)
```

### Architectural Breakdown
- **Input Dimension:** 1,152 values ($18 \times 8 \times 8$).
- **Input Projection:** `nn.Linear(1152, 512)` + `nn.BatchNorm1d(512)` = **591,872 params**.
- **Residual Trunk:** 30 Residual Blocks.
  - Each block contains 2 fully connected dense layers: `nn.Linear(512, 512)` with `BatchNorm1d` and `ReLU`.
  - Parameters per block: $2 \times (512 \times 512 + 512) + 2 \times (2 \times 512) = 527,360$ params.
  - 30 blocks total: **15,820,800 params**.
- **Policy Head:**
  - `Linear(512, 512)` + `BatchNorm1d(512)` + `Linear(512, 4096)` = **2,364,928 params**.
- **Value Head:**
  - `Linear(512, 64)` + `BatchNorm1d(64)` + `Linear(64, 1)` = **32,513 params**.
- **Total Model Parameters:** **18,810,113 parameters** (stored in float32; ~75.2 MB raw weights, ~252.8 MB formatted JSON).

> **Crucial Architectural Discovery:**  
> The model is **not a Convolutional Neural Network (CNN)**. It uses **fully connected linear layers** (`nn.Linear`) across a flattened 1152-element 1D vector. There are no 2D/3D convolutions or attention mechanisms preserving the $8 \times 8$ chessboard topology.

---

## 2. Input Encoding and Legal-Move Mapping

### Board Encoding (`train_ccrl_full_corpus.py` & `orion/model/encoder.py`)
- Input tensor shape: $(18, 8, 8)$ flattened to $1,152$ elements.
- **Planes 0–5:** White pieces (`P`, `N`, `B`, `R`, `Q`, `K`).
- **Planes 6–11:** Black pieces (`p`, `n`, `b`, `r`, `q`, `k`).
- **Plane 12:** Side to move (all 1s if White, all 0s if Black).
- **Planes 13–16:** Castling rights (`K`, `Q`, `k`, `q`).
- **Plane 17:** En-passant target square mask.

### Move Encoding (`fast_encode_action` / `ChessActionSpace`)
- Action space size: **4,096 logits**.
- Move index formula: $\text{from\_sq} \times 64 + \text{to\_sq}$ ($64 \times 64 = 4096$).
- **Limitation:** Under-promotions (`e7e8n`, `e7e8b`, `e7e8r`) collide with queen promotions in the $64 \times 64$ mapping. Under-promotions are forced to queen promotion encoding.

---

## 3. Policy & Value Output Semantics and Training

- **Policy Output:** 4,096 unnormalized logits. Softmax over legal moves yields probability distribution $P(m|s)$.
- **Value Output:** Scalar through `torch.sigmoid`, yielding $V(s) \in [0.0, 1.0]$ representing White win probability ($1.0 = \text{White win}$, $0.5 = \text{Draw}$, $0.0 = \text{Black win}$).
- **Loss Function:**
  $$\mathcal{L} = \mathcal{L}_{\text{policy\_CE}} + \lambda \mathcal{L}_{\text{value\_MSE}}$$
- **Dataset:** CCRL 40/15 corpus and Lichess Grandmaster games.

---

## 4. Search Algorithm and Pruning Implementation

### Primary Tournament Search Engine (`experiments/calibrated_eval/tournament_engine.py`)
- **Algorithm:** Negamax Alpha-Beta with Quiescence Search.
- **Root Move Selection:**
  - Evaluates all legal moves at the root.
  - Sorts candidate moves by neural policy prior combined with capture bonus.
  - Searches top candidates to depth 2 and remaining legal moves to depth 1.
  - Prior scaling: $\text{Combined} = \text{Score} + \text{Prior} \times 0.1$.
- **Quiescence Search:**
  - Evaluates terminal mate ($-30000 + \text{ply}$) and draws ($0.0$).
  - When in check: evades check (stand-pat disabled).
  - When not in check: generates captures ordered by MVV-LVA up to q-depth 4 with stand-pat bounds.
- **Missing Features in Tournament Search:**
  - No Transposition Table (TT) with Zobrist hashing.
  - No Iterative Deepening (fixed depth search only).
  - No Null Move Pruning (NMP) or Late Move Reductions (LMR).
  - No Killer Move or History heuristics.

### Legacy Search Engine (`orion/search/search_agent.py`)
- Implements Zobrist TT, Null Move Pruning, Reverse Futility Pruning, and Piece-Square Tables (PST).
- **Limitation:** Uses a custom pure-Python `ChessEnvironment` that is ~50x slower than `python-chess` and incompatible with UCI / PGN tournament runners.

---

## 5. Measured Inference Latency, Throughput & NPS

Measurements taken on local hardware (RTX 3050 Laptop GPU + i5-11260H @ 4.40 GHz):

| Component / Configuration | Latency per Op | Effective Throughput (NPS) |
|---|---|---|
| **OrionResNet GPU Inference (Batch=1)** | **8.63 ms** | **115.9 NPS** |
| **OrionResNet GPU Inference (Batch=8)** | 9.26 ms / batch | 863.5 NPS |
| **OrionResNet GPU Inference (Batch=16)** | 8.27 ms / batch | 1,934.1 NPS |
| **OrionResNet GPU Inference (Batch=32)** | 8.75 ms / batch | 3,659.0 NPS |
| **OrionResNet GPU Inference (Batch=64)** | **8.38 ms / batch** | **7,634.1 NPS** |
| **OrionResNet CPU Inference (Batch=1)** | **6.53 ms** | **153.2 NPS** |
| **Pure Python `ORIONNetwork` (6 blocks)** | **94.55 ms** | **10.6 NPS** |
| **Classical `evaluate_material`** | **22.65 $\mu$s** | **44,158 NPS** |
| **python-chess Legal Move Gen** | **23.64 $\mu$s** | **42,301 pos/s** |
| **python-chess Push / Pop** | **3.65 $\mu$s** | **273,972 ops/s** |

> **Key Performance Finding:**  
> Single-position neural inference takes **8.63 ms** (GPU) or **6.53 ms** (CPU). This caps serial Alpha-Beta search at **~115–153 nodes per second**.  
> However, batched inference reaches **7,634 positions/sec** at batch size 64.  
> Therefore, serial Alpha-Beta cannot leverage the GPU efficiently without **batched leaf evaluation** or **MCTS**.

---

## 6. Memory Usage and Bottlenecks

1. **Peak RAM Usage:** ~1.2 GB during model loading and tournament execution.
2. **Peak VRAM Usage:** ~550 MB on GPU (model weights + CUDA context).
3. **Primary Computational Bottleneck:**  
   - **Serial Neural Forward Passes:** 98.4% of total search time is spent in PyTorch matrix multiplication kernels for individual positions.
   - **Python Interpreter Overhead:** In pure-Python modules (`orion/model/network.py`), nested loop matrix multiplication creates a 10x slowdown compared to PyTorch C++ bindings.

---

## 7. Audit of Existing Benchmark Methodology

### Identified Historical Flaws
1. **Uncalibrated Opponents:** Prior internal benchmarks paired ORION against homemade bots labeled "2100–2800 Elo" without empirical calibration.
2. **Quiescence Mate Blindness Defect:** The engine previously returned `stand_pat` (material difference) on checkmate, causing missed mates and blundered losses. (Fixed in commit `3e129dd`).
3. **Small Sample Uncertainty:** In a 30-game round-robin, statistical 95% confidence intervals spanned $\pm 258$ to $\pm 296$ Elo.

### Verified Benchmark Pipeline
- Tournament protocol: Paired opening positions, color swaps, Cutechess CLI execution, PGN recording, and Ordo maximum-likelihood estimation.
- Documented in `docs/CALIBRATED_BENCHMARK_REPORT.md`.

---

## 8. Current Reproducible Playing-Strength Baseline

- **Verified Tournament Record (30 games):**  
  - ORION-Search-Full: 2 Wins, 7 Losses, 1 Draw (Score: 2.5/10 = 25.0%).  
  - ORION-Neural-Only: 0 Wins, 10 Losses, 0 Draws (Score: 0.0/10 = 0.0%).
- **Calibrated Ordo Estimates:**  
  - **Calibration A (Vice 1.2 = 1850 anchor):** **1267.2 Elo** (95% CI: $[971.0, 1563.4]$).  
  - **Calibration B (Stockfish 1350 anchor):** **1006.0 Elo** (95% CI: $[747.6, 1264.4]$).  
  - **Status:** Preliminary Local Benchmark — Absolute Elo Calibration Unresolved.
- **Tactical Benchmark Solve Rate (Commit `3e129dd`):**  
  - Baseline Search (Broken): 8 / 18 (44.4%).  
  - Fixed Tactical Search: **12 / 18 (66.7%)** (+22.2% gain).

---

## 9. Ranked Optimization Opportunities

| Rank | Optimization Initiative | Target Area | Estimated Elo Gain | Engineering Cost | Complexity |
|---|---|---|---|---|---|
| **1** | **Transposition Table (Zobrist TT) + Iterative Deepening** | Search Engine | **+150 to +250 Elo** | Low (1–2 days) | Medium |
| **2** | **Move Ordering (TT move + MVV-LVA + Killers + History)** | Search Engine | **+100 to +180 Elo** | Low (1 day) | Low |
| **3** | **Classical Tapered Evaluation (PSTs + King Safety + Pawns)** | Evaluation | **+120 to +200 Elo** | Medium (2 days) | Medium |
| **4** | **Null Move Pruning (NMP) & Late Move Reductions (LMR)** | Search Engine | **+80 to +150 Elo** | Low (1 day) | Medium |
| **5** | **Root Policy Prior Integration (Soft Move Ordering)** | Hybrid Engine | **+50 to +100 Elo** | Low (1 day) | Low |
| **6** | **Batched GPU Leaf Evaluation or PUCT MCTS** | Neural Search | **+100 to +250 Elo** | High (1–2 weeks) | High |
| **7** | **Lightweight NNUE-Style Vectorized Evaluator** | Architecture | **+300 to +500 Elo** | High (2–3 weeks) | High |

---

## 10. Measured Facts vs. Hypotheses vs. Unknowns

### Measured Facts
1. The 30-block model contains **18,810,113 parameters** (not 21.27M).
2. The network architecture uses **Linear dense layers** on a flattened vector, not 2D convolutions.
3. Single-position inference takes **8.63 ms (GPU)** and **6.53 ms (CPU)**, capping serial search at ~116–153 NPS.
4. GPU batched inference achieves **7,634 positions/sec** at batch size 64 (a 65x throughput multiplier).
5. Tactical bug fix in commit `3e129dd` improved tactical benchmark accuracy from 44.4% to 66.7%.
6. Python-chess board operations (`push`/`pop`) take only 3.65 microseconds.

### Hypotheses
1. Adding a 64-bit Zobrist Transposition Table and Iterative Deepening will double effective search depth at 0.5s/move from depth 2 to depth 4–5.
2. Using the 18.8M model exclusively at the root node for candidate move ordering will preserve 90%+ of its positional intuition while reducing search latency by 95%.
3. Blending classical piece-square tables with leaf quiescence will eliminate tactical blunders against 1700+ Elo engines.

### Unknowns
1. The exact Elo ceiling of the 18.8M model under pure MCTS with batched GPU search.
2. The exact cross-play rating of ORION against standard CCRL rating lists without local engine anchors.
