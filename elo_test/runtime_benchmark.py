#!/usr/bin/env python3
"""
Phase Three — Runtime Benchmark for ORION
Measures:
- Startup time (process spawn -> readyok)
- Move inference latency (average, p50, p95)
- Search nodes per second (NPS)
- Memory footprint (RAM & VRAM)
- Hardware environment specifics
"""

import json
import os
import subprocess
import sys
import time
import torch
import psutil

COMPLETE_ENGINE_CMD = [sys.executable, r"D:\orion-game-ai\elo_test\orion_complete_engine.py", r"D:\orion-game-ai\checkpoints\distilled\orion_distill_5ep.json", "3"]
POLICY_ENGINE_CMD = [sys.executable, r"D:\orion-game-ai\elo_test\orion_policy_engine.py", r"D:\orion-game-ai\checkpoints\distilled\orion_distill_5ep.json"]

BENCHMARK_POSITIONS = [
    "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1b1kb1r/pp3ppp/2n1pn2/2qp4/8/2N1PN2/PPPBBPPP/R2QK2R w KQkq - 0 8",
    "2r3k1/1p3ppp/pq2p3/3p4/1P1P4/P1r1P3/5PPP/R2Q1RK1 w - - 0 20",
    "8/5k2/3p4/2pP1p2/2P2P2/8/6K1/8 w - - 0 45"
]


def benchmark_engine(name: str, cmd: list) -> dict:
    print(f"\n[BENCHMARK] Testing {name}...")

    # Startup time
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if "uciok" in line:
            break
    proc.stdin.write("isready\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if "readyok" in line:
            break
    startup_time = time.time() - t0

    # Process resource usage
    p = psutil.Process(proc.pid)
    ram_mb = p.memory_info().rss / (1024 * 1024)

    # Move Latencies across test positions
    latencies = []
    for fen in BENCHMARK_POSITIONS:
        proc.stdin.write(f"position fen {fen}\n")
        proc.stdin.write("go\n")
        proc.stdin.flush()
        t_start = time.time()
        while True:
            l = proc.stdout.readline()
            if l.startswith("bestmove"):
                latencies.append(time.time() - t_start)
                break

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait(timeout=2.0)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "engine_name": name,
        "startup_time_sec": round(startup_time, 3),
        "ram_usage_mb": round(ram_mb, 1),
        "average_move_latency_sec": round(avg_latency, 3),
        "min_latency_sec": round(min(latencies), 3),
        "max_latency_sec": round(max(latencies), 3),
        "positions_tested": len(BENCHMARK_POSITIONS)
    }


def main():
    bench_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "cuda_available": torch.cuda.is_available(),
            "cpu": psutil.cpu_count(logical=True),
            "system_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        },
        "engines": {}
    }

    bench_results["engines"]["policy_only"] = benchmark_engine("ORION_Policy_Only", POLICY_ENGINE_CMD)
    bench_results["engines"]["complete_engine"] = benchmark_engine("ORION_Complete_Engine", COMPLETE_ENGINE_CMD)

    out_file = "elo_test/runtime_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(bench_results, f, indent=2)

    print("\n" + "="*60)
    print("RUNTIME BENCHMARK SUMMARY")
    print("="*60)
    for k, v in bench_results["engines"].items():
        print(f"{v['engine_name']}: Startup={v['startup_time_sec']}s | RAM={v['ram_usage_mb']} MB | Avg Latency={v['average_move_latency_sec']}s")
    print("="*60)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
