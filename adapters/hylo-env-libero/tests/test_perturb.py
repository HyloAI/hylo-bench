"""Perturbed LIBERO tasks: the parts that need no simulator (BDDL rewriting, labels, task variants of the benchmark,
image noise, seeded choices). The simulator side is checked by examples/perturb_check in the LIBERO venv."""

import numpy as np
import pytest

from hylo.envs.libero.perturb import (KINDS, kinds, noisy, paraphrase, region_label, rng, task_variant, variants)
from hylo.envs.libero.suites import libero_benchmark, libero_tasks

BDDL = """(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language put the bowl on the plate)
    (:regions
      (plate_init_region (:target kitchen_table))
    )
  (:fixtures
    kitchen_table - kitchen_table
    wooden_cabinet_1 - wooden_cabinet
  )
  (:objects
    akita_black_bowl_1 - akita_black_bowl
    cream_cheese_1 - cream_cheese
    plate_1 - plate
  )
  (:obj_of_interest
    akita_black_bowl_1
  )
  (:init
    (On plate_1 kitchen_table_plate_init_region)
  )
  (:goal
    (And (On akita_black_bowl_1 plate_1))
  )
)
"""
LABELS = {"akita_black_bowl_1": "black bowl", "cream_cheese_1": "cream cheese box", "plate_1": "plate",
          "wooden_cabinet_1": "wooden cabinet", "kitchen_table": "table"}


def test_task_variant_rewrites_goal_and_language():
    v = task_variant(BDDL, "put the bowl on the plate", LABELS)
    assert v["object"] == "cream_cheese_1" and v["replaces"] == "akita_black_bowl_1"   # the plate is a receptacle
    assert v["language"] == "put the cream cheese box on the plate"
    assert "(On cream_cheese_1 plate_1)" in v["bddl"] and "(:language put the cream cheese box on the plate)" in v["bddl"]
    assert "akita_black_bowl_1 - akita_black_bowl" in v["bddl"]                         # the scene is unchanged
    two = BDDL.replace("(And (On akita_black_bowl_1 plate_1))", "(And (Open wooden_cabinet_1_top_region) "
                                                                "(In akita_black_bowl_1 wooden_cabinet_1_top_region))")
    assert task_variant(two, "x", LABELS) is None                                       # Open would be lost
    table = BDDL.replace("plate_1))", "kitchen_table_stove_front_region))")
    assert task_variant(table, "x", LABELS) is None                                     # an unnamed table region


def test_region_labels():
    assert region_label("flat_stove_1_cook_region", {"flat_stove_1": "stove"}) == "stove"
    assert region_label("wooden_cabinet_1_top_side", LABELS) == "top of the wooden cabinet"
    assert region_label("wooden_cabinet_1_bottom_region", LABELS) == "bottom drawer of the wooden cabinet"
    assert region_label("wine_rack_1_top_region", {"wine_rack_1": "wine rack"}) == "wine rack"
    assert region_label("desk_caddy_1_back_contain_region", {"desk_caddy_1": "caddy"}) == "back compartment of the caddy"
    assert region_label("main_table_stove_front_region", {}) is None


def test_benchmark_variants_have_their_own_ids_and_params():
    base = libero_tasks("libero_goal", [8])
    assert len(base) == 1 and "perturb" not in base[0].params
    pos, lang = libero_tasks("libero_goal", [8], perturb=["position", "language"])
    assert pos.id.endswith("+position") and pos.params["perturb"] == ["position"] and "perturbed" in pos.tags
    assert lang.id.endswith("+language")
    both = libero_tasks("libero_goal", [8], perturb=[["position", "object"]])[0]
    assert both.id.endswith("+position+object")
    if variants():
        task = libero_tasks("libero_goal", [8], perturb="task")[0]
        assert task.instruction != base[0].instruction and "plate_1" in task.success.expr
        assert len(libero_tasks("libero_goal", None, perturb="task")) < 10                # some goals have no variant
    b = libero_benchmark("libero_goal", task_indices=[8], perturb=["camera", "noise"])
    assert len(b.tasks) == 2
    with pytest.raises(ValueError):
        kinds(["wobble"])
    assert set(kinds(list(KINDS))) == set(KINDS)


def test_noise_is_seeded_and_bounded():
    img = np.full((32, 48, 3), 100, np.uint8)
    a, b = noisy(img, 3, 0), noisy(img, 3, 0)
    assert np.array_equal(a, b) and not np.array_equal(a, noisy(img, 3, 1)) and a.dtype == np.uint8
    assert a.max() > 150                                   # the glare spot
    assert rng(1, "x").integers(1000) == rng(1, "x").integers(1000)


def test_paraphrase_is_chosen_by_seed():
    table = {"do it": ["perform it", "carry it out", "get it done"]}
    picks = {paraphrase("do it", s, table) for s in range(20)}
    assert picks <= set(table["do it"]) and len(picks) > 1
    with pytest.raises(KeyError):
        paraphrase("unknown", 0, table)
