"""Smoke tests for examples/showcase: every script runs with --quick and writes its result.json (about 45 s)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("hylo.envs.mujoco", reason="the showcase runs on hylo-env-mujoco")
pytest.importorskip("hylo.learn", reason="the showcase trains reference policies (hylo-learn)")
pytest.importorskip("matplotlib", reason="the showcase renders figures")

SHOWCASE = Path(__file__).resolve().parents[3] / "examples" / "showcase"


def _run(script: str, out: Path, *extra: str) -> dict:
    proc = subprocess.run([sys.executable, str(SHOWCASE / script), "--quick", "--out", str(out), *extra],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads((out / "result.json").read_text())


def test_leaderboard_ranks_and_refuses_another_digest(tmp_path):
    r = _run("01_leaderboard.py", tmp_path)
    board = [row["policy"] for row in r["leaderboard"]]
    assert board[0] == "expert-ik" and board[-1] == "random"
    assert r["edited_protocol"]["boards"] == 2 and r["edited_protocol"]["compare_refused"]
    assert r["edited_protocol"]["override_refused"] and not r["workers_change_digest"]
    assert (tmp_path / "leaderboard.png").exists()


def test_pairwise_counts_discordant_pairs(tmp_path):
    r = _run("02_pairwise.py", tmp_path)
    te, aa = r["pairs"]
    assert te["candidate_only"] > te["baseline_only"] and te["mcnemar_p_one_sided"] < 0.05
    assert aa["candidate_only"] == aa["baseline_only"] == 0                # A/A control: deterministic


def test_parallel_workers_give_identical_results(tmp_path):
    r = _run("03_parallel_workers.py", tmp_path)
    assert r["all_identical"] and {x["workers"] for x in r["runs"]} == {1, 4, 16}


def test_libero_recorded_summary_runs_or_skips(tmp_path):
    r = _run("04_libero_recorded.py", tmp_path)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["source"] == "recorded" and all(s["episodes"] >= 10 for s in r["suites"])
    for s in r["suites"]:
        lo, hi = s["ci95"]
        assert lo <= s["success"] <= hi
