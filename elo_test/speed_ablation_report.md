# Experiment C: Component Speed & Ablation Study Report

Systematic comparison of throughput (NPS), search depth, move latency, and move accuracy across engine variants.

## 1. Fixed Time Benchmark (0.10s budget per move)

| Configuration | Average NPS | Completed Depth | Avg Move Time | Tactical Accuracy | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Config A: Baseline v2.2** | 10,886 nps | **2.28 plies** | 77.9 ms | **76.0%** (19/25) | Standard Alpha-Beta, No PVS, No Repetition, Infinite Window |
| **Config B: v2.2 + Move Ordering & Repetition Guard** | 10,499 nps | **2.2 plies** | 77.4 ms | **76.0%** (19/25) | Repetition guard + Queen promotion ordering + Countermoves |
| **Config C: v2.2 + PVS & Move Ordering** | 10,675 nps | **2.12 plies** | 76.4 ms | **76.0%** (19/25) | Zero-window search (PVS) + Move ordering + Repetition guard |
| **Config D: Full v2.3 Candidate** | 10,649 nps | **2.24 plies** | 81.0 ms | **76.0%** (19/25) | PVS + Move ordering + Repetition guard + Aspiration Windows |


## 2. Fixed Depth 4 Tree Pruning Comparison (Nodes Searched to Depth 4)

| Configuration | Startpos Nodes | KiwiPete Nodes | Tactical Middlegame Nodes | Node Reduction vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Config A: Baseline v2.2** | 650 nodes | 2,717 nodes | 2,696 nodes | **+0.0%** |
| **Config B: v2.2 + Move Ordering & Repetition Guard** | 650 nodes | 2,717 nodes | 2,696 nodes | **+0.0%** |
| **Config C: v2.2 + PVS & Move Ordering** | 653 nodes | 2,722 nodes | 2,891 nodes | **-3.3%** |
| **Config D: Full v2.3 Candidate** | 1,020 nodes | 4,890 nodes | 5,581 nodes | **-89.5%** |
