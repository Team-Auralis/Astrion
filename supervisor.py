"""Supervisor for the ORION Pokemon PPO runs.

Launches each training run as a child process and restarts it (resuming from
the newest checkpoint) if it exits unexpectedly. Start this detached and it
keeps training alive across terminal / server restarts.

Usage:
    python supervisor.py            # Gen 1 batch (red/blue/yellow)
    python supervisor.py --gen 2    # Gen 2 batch (gold/silver/crystal)
"""

import argparse
import datetime
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOGDIR = os.path.join(ROOT, "logs")
os.makedirs(LOGDIR, exist_ok=True)

# Shared hyperparameters for a *persistent*, long-horizon run. `max-steps` is
# the episode/rollout bookkeeping window; persistence means the game itself
# keeps running across those windows instead of rebooting every max-steps.
COMMON = [
    "--rollout-length",
    "2048",
    "--batch-size",
    "64",
    "--lr",
    "3e-4",
    "--entropy-coef",
    "0.03",
    "--max-steps",
    "2048",
    "--frame-skip",
    "16",
    "--persistent",
    "--persist-max-steps",
    "1000000",
    "--episodes",
    "100000",
]

GEN1 = [
    dict(
        name="red",
        rom=r"D:\roms\Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb",
        ckpt="checkpoints/pokemon/batch1_red",
    ),
    dict(
        name="blue",
        rom=r"D:\roms\Pokemon - Blue Version (USA, Europe) (SGB Enhanced).gb",
        ckpt="checkpoints/pokemon/batch1_blue",
    ),
    dict(
        name="yellow",
        rom=r"D:\roms\Pokemon - Yellow Version - Special Pikachu Edition (USA, Europe) (GBC,SGB Enhanced).gb",
        ckpt="checkpoints/pokemon/batch1_yellow",
    ),
]

GEN2 = [
    dict(
        name="gold",
        rom=r"D:\roms\Pokemon - Gold Version (USA, Europe) (SGB Enhanced) (GB Compatible).gbc",
        ckpt="checkpoints/pokemon/batch2_gold",
    ),
    dict(
        name="silver",
        rom=r"D:\roms\Pokemon - Silver Version (USA, Europe) (SGB Enhanced) (GB Compatible).gbc",
        ckpt="checkpoints/pokemon/batch2_silver",
    ),
    dict(
        name="crystal",
        rom=r"D:\roms\Pokemon - Crystal Version (USA, Europe) (Rev 1).gbc",
        ckpt="checkpoints/pokemon/batch2_crystal",
    ),
]


def log(msg: str) -> None:
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGDIR, "supervisor.log"), "a") as f:
        f.write(line + "\n")


def resume_for(ckpt: str, first: bool):
    best = os.path.join(ROOT, ckpt, "ppo_best.pt")
    latest = os.path.join(ROOT, ckpt, "ppo_latest.pt")
    # First launch: warm-start from the best checkpoint we already have.
    # Restarts: continue from the most recent (latest) checkpoint.
    if first and os.path.exists(best):
        return best
    if os.path.exists(latest):
        return latest
    if os.path.exists(best):
        return best
    return None


def launch(run: dict, first: bool):
    os.makedirs(os.path.join(ROOT, run["ckpt"]), exist_ok=True)
    args = [
        PY,
        "-u",
        "train_pokemon.py",
        "--rom",
        run["rom"],
        *COMMON,
        "--checkpoint-dir",
        run["ckpt"],
    ]
    r = resume_for(run["ckpt"], first)
    if r:
        args += ["--resume", r]
        log(f"{run['name']}: resuming from {os.path.relpath(r, ROOT)}")
    out = open(os.path.join(LOGDIR, f"{run['name']}_sup.log"), "w")
    err = open(os.path.join(LOGDIR, f"{run['name']}_sup.err"), "w")
    p = subprocess.Popen(args, cwd=ROOT, stdout=out, stderr=err)
    log(f"{run['name']}: launched pid={p.pid}")
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "--red-only",
        action="store_true",
        help="Run only Pokemon Red lineages (6 seeds) and stop the rest.",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="HH:MM wall-clock stop: gracefully stops all children and exits.",
    )
    args = parser.parse_args()

    if args.red_only:
        rom = GEN1[0]["rom"]
        runs = [
            dict(name="red", rom=rom, ckpt="checkpoints/pokemon/batch1_red"),
            dict(name="red_b", rom=rom, ckpt="checkpoints/pokemon/batch1_red_b"),
            dict(name="red_c", rom=rom, ckpt="checkpoints/pokemon/batch1_red_c"),
            dict(name="red_d", rom=rom, ckpt="checkpoints/pokemon/batch1_red_d"),
            dict(name="red_e", rom=rom, ckpt="checkpoints/pokemon/batch1_red_e"),
            dict(name="red_f", rom=rom, ckpt="checkpoints/pokemon/batch1_red_f"),
        ]
    else:
        runs = GEN1 if args.gen == 1 else GEN2
    log(f"supervisor starting (gen {args.gen}: {[r['name'] for r in runs]})")

    procs = {r["name"]: launch(r, True) for r in runs}
    first_done = {r["name"]: True for r in runs}

    end_dt = None
    if args.end_time:
        hh, mm = args.end_time.split(":")
        end_dt = datetime.datetime.now().replace(
            hour=int(hh), minute=int(mm), second=0, microsecond=0
        )
        log(f"wall-clock stop scheduled for {args.end_time}")

    while True:
        time.sleep(30)
        if end_dt is not None and datetime.datetime.now() >= end_dt:
            log(
                "END-TIME reached; stopping all children (latest checkpoints already on disk)"
            )
            for r in runs:
                p = procs[r["name"]]
                if p.poll() is None:
                    p.terminate()
                    log(f"{r['name']}: terminated pid={p.pid}")
            time.sleep(10)
            for r in runs:
                p = procs[r["name"]]
                if p.poll() is None:
                    p.kill()
                    log(f"{r['name']}: killed pid={p.pid}")
            log("supervisor exiting after stop time")
            return

        for r in runs:
            p = procs[r["name"]]
            rc = p.poll()
            if rc is None:
                continue
            log(f"{r['name']}: exited rc={rc}; restarting")
            first_done[r["name"]] = False
            procs[r["name"]] = launch(r, first_done[r["name"]])


if __name__ == "__main__":
    main()
