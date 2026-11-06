"""Unit tests for M1 aggregation and scoring (no LLM). Run:  python tests/test_m1.py"""
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctxguard.effects import _aggregate  # noqa: E402
from ctxguard.metrics_m1 import _cells, _gold, _score  # noqa: E402
from ctxguard.prompts import effects_prompt  # noqa: E402


def S(**kw):
    return {k: set(v.split(",")) for k, v in kw.items()}


def test_either_keeps_both_directions():
    samples = [S(indoor_temperature="either", air_exchange="increase"),
               S(indoor_temperature="decrease", air_exchange="increase"),
               S(indoor_temperature="decrease")]
    kept, audit = _aggregate(samples, 3, "unanimity_else_both")
    assert "indoor_temperature:increase" in kept and "indoor_temperature:decrease" in kept
    assert kept.count("air_exchange:increase") == 1 and "air_exchange:decrease" not in kept
    assert audit["indoor_temperature"]["direction_votes"] == {"either": 1, "decrease": 2}
    # the v2 rule (majority per direction) drops the minority direction: the bug this fixes
    kept_v2, _ = _aggregate(samples, 3, "majority")
    assert "indoor_temperature:increase" not in kept_v2 and "indoor_temperature:decrease" in kept_v2


def test_disagreeing_samples_keep_both():
    samples = [S(indoor_humidity="increase"), S(indoor_humidity="decrease"), S(sound="increase")]
    kept, _ = _aggregate(samples, 3, "unanimity_else_both")
    assert {"indoor_humidity:increase", "indoor_humidity:decrease"} <= set(kept)
    assert "sound:increase" not in kept                      # 1 of 3 samples: below the channel vote


def test_both_listed_in_one_sample_counts_as_either():
    kept, audit = _aggregate([S(indoor_temperature="increase,decrease")], 1, "unanimity_else_both")
    assert set(kept) == {"indoor_temperature:increase", "indoor_temperature:decrease"}
    assert audit["indoor_temperature"]["direction_votes"] == {"either": 1}


def test_unanimous_single_direction():
    kept, _ = _aggregate([S(passage_opening="increase")] * 3, 3, "unanimity_else_both")
    assert kept == ["passage_opening:increase"]


def test_prompt_allows_either():
    sys_, user, schema = effects_prompt({"id": "a", "match": {}, "scope": "zone"}, "- c: d")
    enum = schema["properties"]["effects"]["items"]["properties"]["direction"]["enum"]
    assert "either" in enum and "either" in sys_ and "either" in user


def test_scoring_levels():
    chans = ["indoor_temperature", "passage_opening", "sound"]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as f:
        w = csv.writer(f)
        w.writerow(["action", "match"] + chans)
        w.writerow(["open_window", "{}", "ud", "u", ""])
        path = f.name
    assert _cells(path)[("open_window", "indoor_temperature")] == "ud"
    gold = _gold(path)
    # system found the channel but only one direction, plus one extra channel
    sc = _score(gold, {"open_window": ["indoor_temperature:decrease", "passage_opening:increase", "sound:increase"]})
    assert sc["channel_level"]["recall"] == 1.0 and sc["channel_level"]["precision"] == round(2 / 3, 3)
    assert sc["effect_level"]["recall"] == round(2 / 3, 3)
    assert sc["bidirectional_recall"] == 0.0 and sc["gold_bidirectional_cells"] == 1
    sc = _score(gold, {"open_window": ["indoor_temperature:decrease", "indoor_temperature:increase", "passage_opening:increase"]})
    assert sc["bidirectional_recall"] == 1.0 and sc["effect_level"]["f1"] == 1.0
    os.unlink(path)


if __name__ == "__main__":
    test_either_keeps_both_directions()
    test_disagreeing_samples_keep_both()
    test_both_listed_in_one_sample_counts_as_either()
    test_unanimous_single_direction()
    test_prompt_allows_either()
    test_scoring_levels()
    print("M1 TESTS PASSED")
