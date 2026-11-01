import json
from pathlib import Path

import pytest

from hylo.bench import leaderboard
from hylo.bench.protocol import builtin_protocols, load_protocol
from hylo.bench.run import compare, policy_spec, run_protocol
from hylo.core import cli
from hylo.core.scorecard import Scorecard, wilson_interval

TESTING = str(Path(__file__).with_name("testing-reach.yaml"))     # over the test double; not a shipped protocol
WEAK = {"type": "testing.p_controller", "params": {"kp": 0.5}}
STRONG = {"type": "testing.p_controller", "params": {"kp": 2.0}}


def test_builtin_protocols_load_and_only_measured_fields_change_the_digest():
    assert set(builtin_protocols()) == {"mujoco-reach", "libero-10", "libero-10-heldout", "libero-10-heldout-realtime"}
    p = load_protocol(TESTING)
    assert p.model_copy(update={"workers": 8, "description": "x"}).digest() == p.digest()
    assert p.model_copy(update={"safety": {"max_delta": 1.0}}).digest() != p.digest()
    assert load_protocol("libero-10").digest() != load_protocol("libero-10-heldout").digest()
    assert load_protocol("libero-10-heldout-realtime").digest() != load_protocol("libero-10-heldout").digest()   # timing is measured
    assert p.model_copy(update={"env": p.env.model_copy(update={"remote": {"runtime": "x"}})}).digest() == p.digest()


def test_scorecards_rank_on_a_leaderboard_and_compare_pairwise(tmp_path):
    p = load_protocol(TESTING)
    weak, _ = run_protocol(p, WEAK, "kp-0.5", out=tmp_path)
    strong, path = run_protocol(p, STRONG, "kp-2.0", out=tmp_path)
    assert path == tmp_path / "scorecards" / "testing-reach" / "kp-2.0.json" and Scorecard.load(path) == strong
    assert strong.metrics["success_mean"] > weak.metrics["success_mean"]
    lo, hi = strong.success_ci95
    assert lo <= strong.metrics["success_mean"] <= hi and sum(t.n for t in strong.per_task.values()) == len(strong.episodes)
    (board,) = leaderboard.boards(leaderboard.collect(tmp_path)).values()
    assert [c.policy for c in board] == ["kp-2.0", "kp-0.5"]
    res = compare(strong, weak)
    assert res.n_pairs == len(strong.episodes) and res.p_value < 0.01


def test_overrides_cannot_change_what_is_measured(tmp_path):
    with pytest.raises(ValueError, match="change what the protocol measures"):
        run_protocol(load_protocol(TESTING), WEAK, "x", out=tmp_path, overrides={"safety.max_delta": 9.0})


def test_scorecards_of_different_protocols_are_not_compared(tmp_path):
    a, _ = run_protocol(load_protocol(TESTING), WEAK, "a", out=tmp_path)
    with pytest.raises(ValueError, match="different protocols"):
        compare(a, a.model_copy(update={"protocol_digest": "other"}))


def test_policy_spec_accepts_json_a_config_or_a_plugin_name(tmp_path):
    assert policy_spec(json.dumps(STRONG)) == STRONG
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(json.dumps({"schema_version": "hylo.experiment/v0.1", "name": "e", "env": {"type": "testing.point_reach"},
                               "benchmark": {"type": "testing/reach-v0"}, "agent": STRONG}))
    assert policy_spec(str(cfg)) == STRONG
    assert policy_spec("testing.p_controller") == {"type": "testing.p_controller", "params": {}}


def test_wilson_interval_is_inside_zero_one():
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0 and wilson_interval(0, 0) is None


def test_wc_bench_end_to_end(tmp_path, capsys):
    out = str(tmp_path)
    assert cli.main(["bench", "run", TESTING, "--policy", json.dumps(STRONG), "--name", "strong", "--out", out]) == 0
    assert cli.main(["bench", "run", TESTING, "--policy", json.dumps(WEAK), "--name", "weak", "--out", out]) == 0
    capsys.readouterr()
    assert cli.main(["bench", "leaderboard", out]) == 0
    table = capsys.readouterr().out
    assert table.index("strong") < table.index("weak") and "testing-reach v1" in table
    sc = tmp_path / "scorecards" / "testing-reach"
    assert cli.main(["bench", "compare", str(sc / "strong.json"), str(sc / "weak.json")]) == 0
    assert json.loads(capsys.readouterr().out)["p_value"] < 0.01
