"""Unit tests for the label-anchored lift (M2 A2, docs Sec. 4.3). No LLM needed.
Run:  python tests/test_lift.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctxguard.prompts import lift_condition_prompt  # noqa: E402
from ctxguard.vocab import _lift_check, _slots_from_relations  # noqa: E402


def test_lift_check_v4_counterfactual_gate():
    """The original v4 rule, kept as the ablation (method.m2_lift_gate: counterfactual)."""
    g = "counterfactual"
    assert _lift_check("user_configuration", "x", "y", "q?", False, g) == "no_fact"
    assert _lift_check("device_state", "x", "y", "q?", False, g) == "no_fact"
    assert _lift_check("world_state", "A", "A", "q?", None, g) == "world_fact"
    assert _lift_check("sensor_reading", "reading R", "fact F", "q?", False, g) == "lifted"
    assert _lift_check("sensor_reading", "Reading R", "reading  r", "q?", False, g) == "lift_failed"
    assert _lift_check("home_mode", "mode M", "fact F", "q?", True, g) == "lift_failed"
    assert _lift_check("event", "event E", "fact F", "q?", None, g) == "lift_failed"
    assert _lift_check("sensor_reading", "reading R", "fact F", "", False, g) == "no_fact"


def test_lift_check_v41_evidence_gate():
    """v4.1 (default): the counterfactual is recorded only; the fact must not be evidence."""
    assert _lift_check("user_configuration", "x", "y", "q?", False) == "no_fact"
    assert _lift_check("sensor_reading", "reading R", "fact F", "", False) == "no_fact"
    # a correct lift is accepted whatever the (precautionary) counterfactual answer is
    assert _lift_check("sensor_reading", "smoke detector alarm", "smoke is present in the home", "q?", True) == "lifted"
    assert _lift_check("home_mode", "home mode set to away", "nobody is at home", "q?", True) == "lifted"
    assert _lift_check("event", "event E", "fact F", "q?", None) == "lifted"
    # the fact still names evidence or system state -> failed, also when the model said world_state
    assert _lift_check("sensor_reading", "motion sensor", "motion detected", "q?", False) == "lift_failed"
    assert _lift_check("sensor_reading", "motion sensor", "home mode is away", "q?", False) == "lift_failed"
    assert _lift_check("world_state", "away mode", "home is in away mode", "q?", None) == "lift_failed"
    assert _lift_check("sensor_reading", "t reading", "temperature is below the predefined value", "q?", False) == "lift_failed"
    assert _lift_check("sensor_reading", "fan", "the fan is on", "q?", False) == "lift_failed"
    # a question is not a fact
    assert _lift_check("world_state", "user location", "Is anyone at home?", "q?", None) == "lift_failed"
    # repeating the observation is not a lift
    assert _lift_check("sensor_reading", "Reading R", "reading  r", "q?", False) == "lift_failed"
    # world facts pass
    assert _lift_check("world_state", "user sleeping", "a person in the room is asleep", "q?", None) == "world_fact"
    assert _lift_check("world_state", "x", "the house is on fire", "q?", None) == "world_fact"


def test_prompt_has_no_domain_examples():
    """'No hardcode': the template itself names no device, hazard or room; examples may only
    come from the human codebook passed in as data."""
    for ver in ("v4", "v4.1"):
        _check_no_domain_words(*lift_condition_prompt("COND", "PROP", ["ASSET"], None, ver))


def _check_no_domain_words(sys_, user, schema):
    banned = ["smoke", "fire", "gas", "leak", "door", "window", "heater", "sleep", "asleep", "kitchen",
              "bedroom", "cooking", "co2", "camera", "away", "vacation", "lux", "temperature"]
    text = (sys_ + user).lower()
    hits = [w for w in banned if re.search(rf"\b{w}\b", text)]
    assert not hits, f"domain words in the lift template: {hits}"
    assert set(schema["required"]) >= {"reason", "latent_fact", "counterfactual_still_right", "question"}


def test_examples_render_only_given_fields():
    ex = [{"condition": "c1", "kind": "world_state", "observation": "o1", "question": "q1?", "answer": "a1", "scope": "home"},
          {"condition": "c2", "kind": "sensor_reading", "observation": "o2", "reason": "r2", "latent_fact": "f2",
           "question": "q2?", "answer": "a2", "scope": "room"}]
    sys_, _, _ = lift_condition_prompt("COND", "PROP", [], ex)
    line1 = next(l for l in sys_.splitlines() if "'c1'" in l)
    line2 = next(l for l in sys_.splitlines() if "'c2'" in l)
    assert "reason" not in line1 and "latent_fact" not in line1
    assert "reason 'r2'" in line2 and "latent_fact 'f2'" in line2


def test_slots_from_relations():
    F = ["a1", "a2", "a3", "b1", "c1"]
    rel = {("a1", "a2"): "exclusive", ("a1", "a3"): "same", ("a2", "a3"): "exclusive",
           ("a1", "b1"): "independent", ("a2", "b1"): "independent", ("a3", "b1"): "independent",
           ("a1", "c1"): "independent", ("a2", "c1"): "independent", ("a3", "c1"): "independent",
           ("b1", "c1"): "independent"}
    r = _slots_from_relations(F, rel)
    slots = sorted(sorted(s) for s in r["slots"])
    assert slots == [["a1", "a2", "a3"], ["b1"], ["c1"]], slots      # same merges, exclusive groups, independent separates
    assert not r["violations"]
    # implication = finer/coarser value of the same slot
    r = _slots_from_relations(["x", "y", "z"], {("x", "y"): "a_implies_b", ("x", "z"): "independent", ("y", "z"): "independent"})
    assert sorted(sorted(s) for s in r["slots"]) == [["x", "y"], ["z"]]
    assert r["implications"] == [{"finer": "x", "coarser": "y"}]
    # mixed aspects: p-q dependent, q-s dependent, p-s independent -> reported and split into cliques
    r = _slots_from_relations(["p", "q", "s"], {("p", "q"): "exclusive", ("q", "s"): "exclusive", ("p", "s"): "independent"})
    assert len(r["violations"]) == 1 and r["splits"], r
    for sl in r["slots"]:
        assert not ({"p", "s"} <= set(sl)), "independent facts ended in one slot"
    assert sorted(m for sl in r["slots"] for m in sl) == ["p", "q", "s"]   # nothing lost


def test_conforms():
    from ctxguard.llm import _conforms
    from ctxguard.prompts import fact_relation_prompt
    _, _, sch = fact_relation_prompt("home", "A", "B")
    assert _conforms({"relation": "same", "reason": "r"}, sch) is None
    assert _conforms({"relation": "maybe", "reason": "r"}, sch) is not None
    assert _conforms({"reason": "r"}, sch) is not None


if __name__ == "__main__":
    test_lift_check_v4_counterfactual_gate()
    test_lift_check_v41_evidence_gate()
    test_prompt_has_no_domain_examples()
    test_examples_render_only_given_fields()
    test_slots_from_relations()
    test_conforms()
    print("LIFT TESTS PASSED")
