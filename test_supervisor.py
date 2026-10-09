"""Unit tests for supervisor.py argv/stop-time helpers (no ROM, no child proc)."""

import datetime

import supervisor


# --- 1. M1 argv -------------------------------------------------------------


def test_m1_argv_contains_expected_tokens():
    runs = supervisor.m1_runs()
    argv = supervisor.build_child_args(runs[0], m1=True)

    for token in (
        "--pokerl",
        "--curriculum",
        "pallet",
        "--recurrent",
        "--bptt-horizon",
        "16",
    ):
        assert token in argv, f"missing M1 token: {token!r}"

    ckpt = argv[argv.index("--checkpoint-dir") + 1]
    assert ckpt.endswith("batch1_m1_red"), ckpt
    assert "--gamma" not in argv
    print(f"M1 argv OK: {argv}")


def test_m1_runs_do_not_shadow_red_dirs():
    runs = supervisor.m1_runs()
    assert len(runs) == 6

    old_red = {
        "batch1_red",
        "batch1_red_b",
        "batch1_red_c",
        "batch1_red_d",
        "batch1_red_e",
        "batch1_red_f",
    }
    for r in runs:
        dirname = r["ckpt"].replace("\\", "/").rstrip("/").split("/")[-1]
        assert "batch1_m1_red" in r["ckpt"], r["ckpt"]
        assert dirname not in old_red, f"{dirname} shadows an old red dir"
    print(f"M1 run ckpts OK: {[r['ckpt'] for r in runs]}")


# --- 2. Non-M1 regression ---------------------------------------------------


def test_non_m1_argv_matches_expected():
    run = supervisor.GEN1[0]
    expected = [
        supervisor.PY,
        "-u",
        "train_pokemon.py",
        "--rom",
        run["rom"],
        *supervisor.COMMON,
        "--checkpoint-dir",
        run["ckpt"],
    ]
    argv = supervisor.build_child_args(run, m1=False)

    assert argv == expected
    for token in ("--pokerl", "--curriculum", "--recurrent", "--bptt-horizon"):
        assert token not in argv, f"non-M1 argv leaked M1 token: {token!r}"
    print(f"non-M1 argv OK: {argv}")


# --- 3. Session deadline ----------------------------------------------------


def test_compute_session_deadline_fixed_now():
    now = datetime.datetime(2026, 1, 1, 12, 0)
    assert supervisor.compute_session_deadline(4.0, now) == datetime.datetime(
        2026, 1, 1, 16, 0
    )


def test_compute_session_deadline_none():
    assert supervisor.compute_session_deadline(None) is None
    assert supervisor.compute_session_deadline(0) is None


def test_compute_session_deadline_uses_wall_clock():
    before = datetime.datetime.now()
    deadline = supervisor.compute_session_deadline(2.0)
    after = datetime.datetime.now()

    expected_lo = before + datetime.timedelta(hours=2)
    expected_hi = after + datetime.timedelta(hours=2)
    assert expected_lo <= deadline <= expected_hi
    print(f"deadline for 2h: {deadline}")


# --- 4. M1 preflight --------------------------------------------------------


def test_m1_preflight_ok(tmp_path):
    assert supervisor.m1_preflight_ok(str(tmp_path)) is False

    states = tmp_path / "states"
    states.mkdir()
    (states / "red_pallet.state").write_bytes(b"stub")

    assert supervisor.m1_preflight_ok(str(tmp_path)) is True


# --- 5. Session-hours precedence -------------------------------------------


def test_session_hours_precedence_documented_but_inline():
    """`--session-hours` wins over `--end-time`/`end_time.txt`.

    The precedence is inline in `main()` (session_deadline short-circuits the
    end-time resolution in the polling loop); there is no pure helper to test
    directly, so this is intentionally not exercised here to avoid a
    test-only refactor of `main()`.
    """
    assert "session_deadline" in supervisor.main.__code__.co_varnames
