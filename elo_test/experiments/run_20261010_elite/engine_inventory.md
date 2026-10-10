# ORION Tournament Engine Inventory & Technical Specification

## 1. Engine Under Test A: ORION Policy Only (`ORION_Policy_Only`)
- **Category**: Pure Deep Neural Policy Model (Zero Search)
- **Implementation**: [`elo_test/orion_policy_engine.py`](file:///D:/orion-game-ai/elo_test/orion_policy_engine.py)
- **Neural Checkpoint**: [`checkpoints/distilled/orion_distill_5ep.json`](file:///D:/orion-game-ai/checkpoints/distilled/orion_distill_5ep.json)
- **Checkpoint SHA-256**: `7E88F0F76BD0726CE1A6D413DDF26774E9745D7DB73885CE6F6811E7C329D8AC`
- **Network Architecture**: 6 Residual Blocks, 128 Filters, 22-plane board representation, Value head (tanh) + Policy head (4,672 action space).
- **Inference Mode**: Direct batch forward pass; masks illegal moves and selects `argmax` policy logit.
- **Search Assistance**: Completely disabled (Depth = 0, Plies = 0).
- **Opening Book**: Completely disabled.
- **Tablebases**: None.
- **Latency**: ~27 ms per move.
- **RAM Footprint**: ~30.5 MB.

---

## 2. Engine Under Test B: ORION Complete Engine (`ORION_Complete_Engine`)
- **Category**: Hybrid Neural Evaluation + Alpha-Beta Search Engine
- **Implementation**: [`elo_test/orion_complete_engine.py`](file:///D:/orion-game-ai/elo_test/orion_complete_engine.py)
- **Neural Checkpoint**: [`checkpoints/distilled/orion_distill_5ep.json`](file:///D:/orion-game-ai/checkpoints/distilled/orion_distill_5ep.json)
- **Evaluation Function**:
  - Neural Network Prior & Value weight: 0.35
  - Handcrafted Tapered Piece-Square Tables & Mobility weight: 0.65
- **Search Algorithmic Stack**:
  - Negamax Alpha-Beta with Iterative Deepening
  - Zobrist Transposition Table (64-bit Polyglot hash keys) with Exact, Lowerbound, and Upperbound flags
  - Quiescence Search with tactical capture pruning
  - Static Exchange Evaluation (SEE) pruning
  - Reverse Futility Pruning (RFP / Static Null Move Pruning)
  - Null Move Pruning (NMP) with zugzwang verification
  - Late Move Pruning (LMP) and Late Move Reductions (LMR)
  - Killer Move Heuristic (2 killer slots per ply)
  - History Heuristic table with depth-squared weight decay
- **Opening Book**: Bypassed for independent validation
- **Tablebases**: None
- **Latency**: ~300 ms - 1,000 ms per move
- **RAM Footprint**: ~157.4 MB

---

## 3. Reference Engines Pool

### 3.1 Stockfish 19 (`stockfish.exe`)
- **Binary Path**: `D:\orion-game-ai\stockfish.exe`
- **SHA-256**: `45BC8E4969147DB9C2EB533810637994619BFF0EACC81CCFD9854394901BCBD0`
- **Calibration Engine**: Official Stockfish internal strength limiter
  - Option `UCI_LimitStrength = true`
  - Calibrated Rating Brackets:
    - `Stockfish_1320`: `UCI_Elo = 1320` (Stockfish lower calibration bound)
    - `Stockfish_1500`: `UCI_Elo = 1500`
    - `Stockfish_1700`: `UCI_Elo = 1700`
    - `Stockfish_1900`: `UCI_Elo = 1900`
    - `Stockfish_2100`: `UCI_Elo = 2100`

### 3.2 BBC 1.1 (`bbc_1.1_64bit_windows.exe`)
- **Binary Path**: `D:\orion-game-ai\elo_test\bbc_repo\bin\windows\bbc_1.1_64bit_windows.exe`
- **SHA-256**: `7C635214BBC185259EDC1475240BC3B6D516237D3337797EE34C1BAC82249151`
- **Author**: Maksim Korzh (Code Monkey King)
- **CCRL Published Rating**: 1888 Elo on CCRL 40/40 list
- **Architecture**: Pure 64-bit bitboard alpha-beta engine with PeSTO piece-square evaluation.

---

## 4. Rating Calculation Tool
- **Tool**: Ordo by Miguel A. Ballicora
- **Binary**: `D:\orion-game-ai\ordo.exe`
- **SHA-256**: `E8CEA4244B1185C35D1B98B129E46D9B214465F17A4450F03ED291BDADE43A3B`
- **Error Estimation**: 1,000 Monte Carlo bootstrap simulations with 95.0% confidence intervals.
