# ORION Engine Performance & Profiling Report

## 1. System Profiling & Hardware Specification
- **CPU:** 11th Gen Intel(R) Core(TM) i5-11260H (12 logical processors @ 4.40 GHz)
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (3.87 GiB dedicated GDDR6 VRAM)
- **RAM:** 8.00 GB DDR4 System RAM
- **OS:** Windows 11 Home 64-bit

---

## 2. Move Generation & Perft Benchmarks
Move generation accuracy and throughput were benchmarked using `elo_test/run_perft.py`:

| Position | FEN | Depth | Nodes | Time (ms) | NPS | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Startpos** | Standard Start | 1 | 20 | 0.2 ms | 91,283 | **PASS** |
| | | 2 | 400 | 2.4 ms | 167,378 | **PASS** |
| | | 3 | 8,902 | 45.4 ms | 195,934 | **PASS** |
| | | 4 | 197,281 | 1,119.9 ms | 176,155 | **PASS** |
| **KiwiPete** | Complex Tactical | 1 | 48 | 0.3 ms | 141,343 | **PASS** |
| | | 2 | 2,039 | 12.4 ms | 164,224 | **PASS** |
| | | 3 | 97,862 | 545.6 ms | 179,378 | **PASS** |
| **Position 3** | Promotions & Checks | 1 | 14 | 0.1 ms | 114,192 | **PASS** |
| | | 2 | 191 | 1.4 ms | 140,441 | **PASS** |
| | | 3 | 2,812 | 17.7 ms | 159,241 | **PASS** |
| | | 4 | 43,238 | 285.3 ms | 151,562 | **PASS** |
| **Position 4** | Pins & En-Passant | 1 | 6 | 0.1 ms | 71,344 | **PASS** |
| | | 2 | 264 | 1.7 ms | 156,779 | **PASS** |
| | | 3 | 9,467 | 51.3 ms | 184,635 | **PASS** |
| **Total** | **All 4 Positions** | — | **362,544** | **2,084 ms** | **173,940** | **100% PASS** |

---

## 3. Component Latency & Throughput Breakdown

| Component | Hardware Target | Execution Time / Latency | Throughput |
| :--- | :--- | :---: | :---: |
| **Polyglot Opening Book** | Disk/Memory Cache | **< 1.0 ms** | > 10,000 lookups/s |
| **Tapered Bitboard Eval** | Intel Core i5 CPU | **< 2.0 μs** | **> 500,000 evaluations/s** |
| **Neural Policy Prior (18.8M)**| RTX 3050 CUDA | **12.5 ms** | ~80 inferences/s |
| **Move Generator (python-chess)**| Intel Core i5 CPU | **~5.7 μs** | **173,940 nodes/s** |
| **Alpha-Beta Search Tree** | Hybrid CPU/Cache | Variable per depth | **~10,000 – 15,000 nodes/s** |

---

## 4. Memory Footprint & Resource Utilization
- **System Memory (RAM):** ~240 MB peak RSS during search tree execution.
- **GPU VRAM Allocation:** ~450 MB dedicated VRAM for 18.8M PyTorch model weights and inference buffers.
- **Transposition Table Size:** 500,000 entries $\times$ ~64 bytes $\approx$ **32 MB** memory consumption.
