"""Supervisor for the ORION Pokemon PPO runs.

Launches each training run as a child process and restarts it (resuming from
the newest checkpoint) if it exits unexpectedly. Start this detached and it
keeps training alive across terminal / server restarts.

Usage:
    python supervisor.py            # Gen 1 batch (red/blue/yellow)
    python supervisor.py --gen 2    # Gen 2 batch (gold/silver/crystal)
    python supervisor.py --red-only --m1 --session-hours 4   # daily M1 session

The daily M1 session runs the 6 Red lineages with the M1 bundle (PokeRL env +
Pallet curriculum + recurrent PPO) into the fresh `batch1_m1_red*` checkpoint
dirs, then stops cleanly after `--session-hours`. Re-running the same command
resumes each lineage from its newest M1 checkpoint. Requires
`states/red_pallet.state` (see `record_intro_state.py`).
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
END_TIME_FILE = os.path.join(ROOT, "end_time.txt")
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

# Extra training flags for the curated "m1" run: PokéRL objective, a Pallet
# Town curriculum, and a recurrent policy (which already defaults gamma to
# 0.998 upstream, so we intentionally do not hard-code --gamma here).
M1_TRAIN_ARGS = [
    "--pokerl",
    "--curriculum",
    "pallet",
    "--recurrent",
    "--bptt-horizon",
    "16",
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


def m1_runs() -> list:
    """Fresh 6-lineage Red fleet for the M1 (PokeRL + Pallet curriculum) runs."""
    rom = GEN1[0]["rom"]
    return [
        dict(name="red", rom=rom, ckpt="checkpoints/pokemon/batch1_m1_red"),
        dict(name="red_b", rom=rom, ckpt="checkpoints/pokemon/batch1_m1_red_b"),
        dict(name="red_c", rom=rom, ckpt="checkpoints/pokemon/batch1_m1_red_c"),
        dict(name="red_d", rom=rom, ckpt="checkpoints/pokemon/batch1_m1_red_d"),
        dict(name="red_e", rom=rom, ckpt="checkpoints/pokemon/batch1_m1_red_e"),
        dict(name="red_f", rom=rom, ckpt="checkpoints/pokemon/batch1_m1_red_f"),
    ]


def log(msg: str) -> None:
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGDIR, "supervisor.log"), "a") as f:
        f.write(line + "\n")


def resume_for(ckpt: str, first: bool):
    best = os.path.join(ROOT, ckpt, "ppo_best.pt")
    latest = os.path.join(ROOT, ckpt, "ppo_latest.pt")
    # Always continue from the most recent checkpoint. `ppo_best.pt` can be
    # much older than `ppo_latest.pt` (best-reward snapshot from an early
    # spike), so preferring it on first launch throws away training progress.
    if os.path.exists(latest):
        return latest
    if os.path.exists(best):
        return best
    return None


def parse_end_time(val: str):
    """HH:MM -> today's datetime at that time, rolling to tomorrow if past."""
    try:
        hh, mm = val.strip().split(":")
        now = datetime.datetime.now()
        dt = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
        if dt <= now:
            dt += datetime.timedelta(days=1)
        return dt
    except (ValueError, AttributeError):
        return None


def compute_session_deadline(hours, now=None):
    """Wall-clock stop time `hours` from now, or None when unset/zero."""
    if not hours:
        return None
    if now is None:
        now = datetime.datetime.now()
    return now + datetime.timedelta(hours=hours)


_END_TIME_CACHE = {"content": None, "dt": None}


def read_end_time_file():
    """Return the end datetime from end_time.txt (HH:MM), or None.

    The parsed value is cached by file content: re-reading an unchanged file
    must not re-roll an already-accepted today-time to tomorrow. Otherwise a
    poll that runs *after* the deadline (e.g. the 30s poll reading 23:00 at
    23:00:30) would parse `23:00` as already-past, roll it to tomorrow, and
    the fleet would never stop.
    """
    try:
        with open(END_TIME_FILE) as f:
            content = f.read().strip()
    except OSError:
        return None
    if content == _END_TIME_CACHE["content"]:
        return _END_TIME_CACHE["dt"]
    _END_TIME_CACHE["content"] = content
    _END_TIME_CACHE["dt"] = parse_end_time(content)
    return _END_TIME_CACHE["dt"]


def write_fleet_report(runs):
    lines = [f"Fleet stopped at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"]
    for r in runs:
        tl = os.path.join(ROOT, r["ckpt"], "training_log.json")
        info = {}
        try:
            with open(tl) as f:
                info = json.load(f)
        except (OSError, ValueError):
            pass
        has_best = os.path.exists(os.path.join(ROOT, r["ckpt"], "ppo_best.pt"))
        has_latest = os.path.exists(os.path.join(ROOT, r["ckpt"], "ppo_latest.pt"))
        lines.append(
            f"{r['name']}: rollout={info.get('rollout', '-')} "
            f"steps={info.get('global_step', 0)} "
            f"reward={info.get('avg_reward', 0):.1f} "
            f"badges={info.get('max_badges', 0)} "
            f"party={info.get('max_party', 0)} "
            f"lvl={info.get('max_level', 0)} "
            f"map={info.get('max_map', 0)} "
            f"best={has_best} latest={has_latest}"
        )
    try:
        with open(os.path.join(LOGDIR, "fleet_report.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        log(f"fleet report written to logs/fleet_report.txt")
    except OSError as e:
        log(f"could not write fleet report: {e}")


def build_child_args(run: dict, m1: bool = False) -> list:
    """Pure argv assembly for a training child process."""
    return [
        PY,
        "-u",
        "train_pokemon.py",
        "--rom",
        run["rom"],
        *COMMON,
        *(M1_TRAIN_ARGS if m1 else []),
        "--checkpoint-dir",
        run["ckpt"],
    ]


def m1_preflight_ok(root: str = ROOT) -> bool:
    """True when the intro state needed by the m1 Pallet curriculum exists."""
    return os.path.isfile(os.path.join(root, "states", "red_pallet.state"))


def launch(run: dict, first: bool, m1: bool = False):
    os.makedirs(os.path.join(ROOT, run["ckpt"]), exist_ok=True)
    args = build_child_args(run, m1)
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
    parser.add_argument(
        "--m1",
        action="store_true",
        help=(
            "Run the 6 M1 Red lineages (PokeRL + Pallet curriculum + recurrent "
            "PPO) into fresh batch1_m1_red* dirs; the older batch1_red* fleet is "
            "preserved. Needs states/red_pallet.state."
        ),
    )
    parser.add_argument(
        "--session-hours",
        type=float,
        default=None,
        help=(
            "Stop the fleet cleanly this many hours from now (must be > 0; "
            "overrides --end-time/end_time.txt)."
        ),
    )
    args = parser.parse_args()

    if args.session_hours is not None and args.session_hours <= 0:
        parser.error("--session-hours must be > 0")

    if args.m1:
        runs = m1_runs()
    elif args.red_only:
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

    if args.m1 and not m1_preflight_ok():
        log(
            "ERROR: M1 run needs states/red_pallet.state; record it first with "
            "`python record_intro_state.py <rom.gb>`"
        )
        sys.exit(2)

    session_deadline = compute_session_deadline(args.session_hours)
    if session_deadline is not None:
        log(
            f"session stop scheduled for {session_deadline:%Y-%m-%d %H:%M} "
            f"({args.session_hours:g}h)"
        )

    procs = {r["name"]: launch(r, True, args.m1) for r in runs}
    first_done = {r["name"]: True for r in runs}

    end_dt = session_deadline
    if end_dt is None:
        end_dt = read_end_time_file()
        if end_dt is None and args.end_time:
            end_dt = parse_end_time(args.end_time)
    if end_dt is not None and session_deadline is None:
        log(f"wall-clock stop scheduled for {end_dt:%H:%M}")

    while True:
        time.sleep(30)
        end_dt = (
            session_deadline
            if session_deadline is not None
            else (
                read_end_time_file()
                or (parse_end_time(args.end_time) if args.end_time else None)
            )
        )
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
            write_fleet_report(runs)
            log("supervisor exiting after stop time")
            return

        for r in runs:
            p = procs[r["name"]]
            rc = p.poll()
            if rc is None:
                continue
            log(f"{r['name']}: exited rc={rc}; restarting")
            first_done[r["name"]] = False
            procs[r["name"]] = launch(r, first_done[r["name"]], args.m1)


if __name__ == "__main__":
    main()
