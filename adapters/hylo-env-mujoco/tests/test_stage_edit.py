from hylo.envs.mujoco.world_tools import Workspace


def test_stage_show_position_can_be_used_in_an_edit(tmp_path):
    w = Workspace(tmp_path, "demo")
    added = w.surface.call("stage_edit", {"expected_revision": 0, "ops": [{"op": "add", "entity": {
        "id": "cube", "kind": "shape", "pos": [0, 0, 1], "shape": {"type": "box", "size": [.1, .1, .1]}}}]})
    assert added.ok and w.stage.get("cube").pose.pos == (0, 0, 1)
    moved = w.surface.call("stage_edit", {"expected_revision": 1, "ops": [{
        "op": "update", "id": "cube", "set": {"pos": [1, 2, 3]}}]})
    assert moved.ok and w.stage.get("cube").pose.pos == (1, 2, 3)
    conflict = w.surface.call("stage_edit", {"expected_revision": 2, "ops": [{
        "op": "update", "id": "cube", "set": {"pos": [1, 2, 3], "pose": {"pos": [0, 0, 0]}}}]})
    assert not conflict.ok and w.stage.revision == 2
