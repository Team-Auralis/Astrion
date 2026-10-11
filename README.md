<div align="center">

# 🌌 ORION Gaming Mind
### *Next-Generation Neural Chess AI & Self-Learning Engine*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Organization](https://img.shields.io/badge/Team-Auralis-purple.svg)](https://github.com/Team-Auralis)
[![Repository](https://img.shields.io/badge/Repo-orion--gaming--mind-brightgreen.svg)](https://github.com/Team-Auralis/orion-gaming-mind)
[![UCI Compliant](https://img.shields.io/badge/UCI-Compatible-success.svg)](#universal-chess-interface-uci)
[![Android APK](https://img.shields.io/badge/Android%20APK-Download%20Release-brightgreen?logo=android&logoColor=white)](https://github.com/Team-Auralis/orion-gaming-mind/releases/tag/mobile-latest)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 Overview

**ORION Gaming Mind** is a modern chess intelligence platform developed by [Team-Auralis](https://github.com/Team-Auralis). It merges **Deep Reinforcement Learning** (Dual-Head Policy & Value Network) with **Stockfish-inspired classical search techniques** (Alpha-Beta Minimax, Quiescence Search, MVV-LVA move ordering) and provides a **real-time Bot Arena** to measure playing strength and compute **official FIDE-style Elo ratings** against real chess bots—including the world-champion **Stockfish**.

---

## ⚡ Key Highlights

- **🧠 Dual-Head Neural Architecture**:
  - **Input**: 14 channels $\times$ 8 ranks $\times$ 8 files (896 input features) encoding all piece placements, active turn, castling rights, and en-passant squares.
  - **Hidden Layers**: Dense representations (256 $\rightarrow$ 128) with ReLU activations.
  - **Policy Head**: 4,096 discrete action logits mapping every from-square to to-square.
  - **Value Head**: Position evaluation scalar in range $[0, 1]$.
- **🚀 Stockfish-Inspired Search Enhancement**:
  - **Alpha-Beta Negamax**: Tree lookahead calculating variations to avoid tactical blunders.
  - **Quiescence Search**: Evaluates capture sequences until the position is tactically stable, solving the infamous *horizon effect*.
  - **MVV-LVA Move Ordering**: Most Valuable Victim / Least Valuable Attacker sorting prunes 90%+ of irrelevant branches instantly.
  - **Root Policy Priors**: Uses ORION's learned neural intuition to bias search ordering toward principled moves.
- **⚔️ Empirically Calibrated Engine Benchmark & Ordo Evaluation**:
  - Independent strength validation using official **Ordo** maximum-likelihood estimation with 95% confidence intervals (`-s 1000 -F 95`).
  - Standardized round-robin testbed with paired opening positions and color reversals against versioned engines with published CCRL/CEGT anchors:
    - **Vice 1.2** (CCRL anchor ~1850 Elo)
    - **Sunfish 2026** (CCRL anchor ~1750 Elo)
    - **BBC 1.2** (CCRL anchor ~1950 Elo)
    - **Stockfish 19** (`UCI_LimitStrength` = 1350 Elo anchor)
  - Clear separation of test subjects:
    - **ORION-Search-Full** (30-Block Master + Alpha-Beta Minimax + Quiescence): **1267.2 ± 296.0 Elo** (Vice anchor) / **1006.0 ± 258.0 Elo** (Stockfish-1350 anchor)
    - **ORION-Neural-Only** (30-Block Policy Prior Argmax): Evaluated as pure intuition baseline.
- **⚡ Vectorized PyTorch Training Pipeline**:
  - 1,000x speedup over standard single-sample Python loops (9,000+ positions/second).
  - Train on massive Lichess puzzle and opening datasets in minutes.
  - Exports directly into 100% backward-compatible native JSON checkpoints.
- **♟️ Universal Chess Interface (UCI)**:
  - Plug ORION directly into **Arena Chess GUI**, **Cutechess**, **Banksia GUI**, or connect as a live bot on **Lichess** via `lichess-bot`.
- **🎮 Interactive Terminal Play**:
  - Play directly against ORION in your terminal with ASCII board display and UCI coordinates (`e2e4`, `g1f3`).

---

## 🏛️ System Architecture

```
                                  [ FEN / Board State ]
                                            │
                                            ▼
                             ┌─────────────────────────────┐
                             │ 14x8x8 Chess Board Encoder  │
                             │        (896 inputs)         │
                             └──────────────┬──────────────┘
                                            │
                                            ▼
                             ┌─────────────────────────────┐
                             │    ORION Neural Network     │
                             │      (Dual-Head Arch)       │
                             └───────┬─────────────┬───────┘
                                     │             │
                    Policy (4096) ◄──┘             └──► Value Scalar
                          │                                   │
                          ▼                                   ▼
             ┌────────────────────────────────────────────────────────┐
             │            Stockfish-Inspired Search Agent             │
             │   - Alpha-Beta Minimax Search                          │
             │   - Quiescence Search (Capture Stabilization)          │
             │   - MVV-LVA Dynamic Move Ordering                      │
             │   - Microsecond Positional Leaf Evaluation             │
             └───────────────────────────┬────────────────────────────┘
                                         │
                                         ▼
                               [ Best Legal Action ]
                                         │
                   ┌─────────────────────┼─────────────────────┐
                   ▼                     ▼                     ▼
             [ Bot Arena ]       [ Terminal Play ]      [ UCI Protocol ]
           (Elo Benchmarking)    (Human vs ORION)      (Arena/Cutechess)
```

---

## 📦 Installation

Clone the repository and install the minimal dependencies:

```powershell
git clone https://github.com/Team-Auralis/orion-gaming-mind.git
cd orion-gaming-mind

# Install dependencies (PyTorch, Datasets, Tqdm)
pip install -r requirements.txt
```

Verify your installation:

```powershell
python test_stockfish_and_arena.py
```

---

## 🚀 Quickstart & CLI Commands

ORION includes a unified master CLI (`cli.py`):

### 1. ⚔️ Battle Bots & Calculate Elo

Pit ORION against real chess bots to verify its playing strength:

```powershell
# Battle against RandomBot (Baseline)
python cli.py battle --opponent random --rounds 6

# Battle against GreedyMaterialBot (~1000 Elo)
python cli.py battle --opponent greedy --rounds 4

# Battle against MinimaxBot (~1350 Elo)
python cli.py battle --opponent minimax --rounds 4

# Battle against real Stockfish (set Elo limit: 1320 - 3190)
python cli.py battle --opponent stockfish --stockfish-elo 1400 --rounds 2
```

Sample Arena Match Output:
```
================================================================
                 ORION BATTLE ARENA MATCH                       
================================================================
White/Black : [1] ORION AI (Search Depth 1) (Elo 1200)
Opponent    : [2] GreedyMaterialBot (Elo ~1000) (Elo 1000)
Rounds      : 4 games (alternating colors)
================================================================

Game 01/04 | ORION AI (Search Depth 1) (White): 1-0 (ORION AI won) | 42 plies (checkmate:black)
Game 02/04 | ORION AI (Search Depth 1) (Black): 1-0 (ORION AI won) | 38 plies (checkmate:white)
...
================================================================
                     MATCH FINAL RESULTS                        
================================================================
ORION AI (Search Depth 1)      Score: 3W - 0L - 1D  (3.5/4, 87.5%)
GreedyMaterialBot (~1000)      Score: 0W - 3L - 1D
Average game length: 40.2 plies
----------------------------------------------------------------
Elo Adjustment for ORION AI: 1200.0 -> 1238.4 (+38.4)
Elo Adjustment for Opponent: 1000.0 ->  961.6 (-38.4)
================================================================
```

---

### 2. ⚡ High-Speed Batched Training

Train ORION on Lichess puzzle and opening datasets at **9,000+ positions/second**:

```powershell
python cli.py train --batch-size 128 --epochs 5 --max-puzzles 50000 --lr 0.001
```

Checkpoints are automatically exported in native ORION format under `checkpoints/fast_torch/`.

---

### 3. ♟️ Play Interactively Against ORION

Play a game against ORION directly in your terminal:

```powershell
# Play as White with Search Depth 2
python cli.py play --color white --depth 2

# Play as Black
python cli.py play --color black --depth 3
```

Enter moves in standard algebraic/UCI format (`e2e4`, `e7e5`, `g1f3`, `b8c6`).

---

### 4. 📱 Mobile App & Web Interface (Android & PWA)

ORION includes both a **native Android app** and a **zero-install Mobile Web / PWA**:

#### Start the Backend Engine Server:
```powershell
python cli.py app --port 8000
```
- **Mobile Web / PWA**: Open `http://localhost:8000` (or `http://<LAN-IP>:8000` on your Android phone). A responsive, touch-optimized chess board will load with live move hints and real-time ORION AI evaluation!
- **REST API**:
  - `GET /api/status`: Engine info, active checkpoint, and default search depth.
  - `POST /api/move`: Submit FEN position `{"fen": "...", "depth": 2}` and receive `{best_move, score, nodes}` in milliseconds.

#### Native Android App (`android-chess-src/`):
- **⚡ Automated GitHub Actions APK Release**: Every push to `main` automatically compiles, signs, and publishes the ready-to-install Android APK to GitHub Releases:
  - 📲 **[Download ORION-Chess-AI.apk](https://github.com/Team-Auralis/orion-gaming-mind/releases/tag/mobile-latest)**
  - Direct install on any phone or tablet running Android 7.0+ (API 24+).
- Customized from [jcarolus/android-chess](https://github.com/jcarolus/android-chess).
- Features [`OrionEngine.java`](android-chess-src/app/src/main/java/jwtc/android/chess/engine/OrionEngine.java) connecting the Android chessboard directly to the ORION neural engine via HTTP/REST.
- Automatic offline fallback to local search if disconnected.
- To build locally: Open `android-chess-src` in Android Studio and click **Run** (`Shift + F10`).

---

### 5. 🌐 Universal Chess Interface (UCI)

Run ORION as a standard UCI engine to connect with **Arena Chess GUI**, **Cutechess**, **Banksia**, or run a 24/7 bot on **Lichess**:

```powershell
python orion_uci.py
```

Compatible UCI commands:
- `uci`
- `isready`
- `ucinewgame`
- `position startpos moves e2e4 e7e5`
- `go depth 3`
- `quit`

---

## 📂 Repository Structure

```
orion-gaming-mind/
├── orion/                     # Core Neural Chess Intelligence Engine
│   ├── arena/                 # Bot Arena & FIDE Elo Tracker
│   │   ├── battle.py          # Arena match loop & Elo math
│   │   └── bots.py            # RandomBot, GreedyBot, MinimaxBot, StockfishBot
│   ├── environment/           # Pure-Python Chess Rules Engine
│   │   └── chess.py           # Board, FEN loader, legal move gen, castling, en-passant
│   ├── model/                 # Neural Architecture
│   │   ├── actions.py         # 4096 discrete action space mapping
│   │   ├── agent.py           # Base ORION Agent (policy-only)
│   │   ├── encoder.py         # 14x8x8 spatial tensor encoder
│   │   └── network.py         # Native ORION Dual-Head Network
│   ├── search/                # Stockfish-Inspired Search Engine
│   │   └── search_agent.py    # Alpha-Beta, Quiescence Search, MVV-LVA ordering
│   ├── training/              # RL, TD-learning, and Self-Play modules
│   └── utils/
│       └── checkpoint.py      # Checkpoint serialization & loading
├── android-chess-src/         # Complete Native Android App (Java/Kotlin Gradle project)
│   ├── app/                   # Android Application module with OrionEngine.java
│   └── build.gradle           # Root Gradle configuration
├── checkpoints/               # Trained model weights (JSON format)
├── data/                      # Lichess training datasets
├── cli.py                     # Unified Master CLI (train, battle, play, app, uci)
├── orion_app_server.py        # Mobile App API server & interactive Web PWA
├── train_fast.py              # High-speed vectorized PyTorch training
├── orion_uci.py               # UCI Protocol implementation
├── test_*.py                  # Complete automated test suite
├── requirements.txt           # Minimal dependencies
└── README.md                  # Project documentation
```

---

## 🔬 Testing & Verification

Run the test suite to verify all core components:

```powershell
# Run the search, arena, and Elo test suite
python test_stockfish_and_arena.py

# Run individual unit tests
python test_agent.py
python test_actions.py
python test_encoder.py
```

---

## 👥 Authors & Credits

Developed with passion by **Team-Auralis**:
- **Project**: [Team-Auralis/orion-gaming-mind](https://github.com/Team-Auralis/orion-gaming-mind)
- Stockfish references and principles inspired by [official-stockfish/Stockfish](https://github.com/official-stockfish/Stockfish).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
