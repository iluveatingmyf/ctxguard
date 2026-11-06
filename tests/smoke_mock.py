"""End-to-end smoke test with the MOCK backend.
Fills every annotation sheet automatically from the mock oracle ONLY to exercise
the scoring code. Run:  python tests/smoke_mock.py
"""
import csv
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out_mock")
sys.path.insert(0, ROOT)


os.environ["CTXGUARD_PROPERTIES"] = os.path.join(ROOT, "tests", "seed_properties_TEST.csv")


def run(*args, config="config_mock.yaml", env=None):
    cmd = [sys.executable, "-m", "ctxguard.cli", *args, "--backend", "mock", "--config", os.path.join(ROOT, "tests", config)]
    print("$", " ".join(args))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})})
    print(r.stdout[-1500:])
    if r.returncode:
        print(r.stderr[-4000:])
        raise SystemExit(f"step failed: {args}")


def fill_csv(path, fn):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        fn(r)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    cfg["paths"]["out"] = "out_mock"
    cfg["llm"]["backend"] = "mock"
    cfg["eval"]["audit_items"] = 90
    yaml.safe_dump(cfg, open(os.path.join(ROOT, "tests", "config_mock.yaml"), "w"))
    from ctxguard.mock import EFFECTS, oracle

    run("m1")
    run("m1", "--k", "1")
    run("m1-gold-export")
    def g_m1(r):
        for e in EFFECTS.get(r["action"], []):
            ch, d = e.split(":")
            r[ch] = (r[ch] or "") + ("+" if d == "increase" else "-")
    fill_csv(os.path.join(OUT, "annotate", "m1_effects_gold.csv"), g_m1)
    run("m1-score", "--gold", "out_mock/annotate/m1_effects_gold.csv")
    # M2 stages A-C (first build from the seed), freeze test
    run("m2-extract"); run("m2-dims", "--runs", "2"); run("m2-values"); run("m2-seed-check")
    run("m2-elicit"); run("m2-consolidate")
    run("m2-freeze", "--src", "out_mock/m2/C3_factors_candidate.yaml", "--name", "v1test",
        "--dest", "out_mock/vocab/factors_v1test.yaml", "--parent", "out_mock/m2/A3_factors_corpus.yaml")
    # (no human seed in the smoke test: A3's output stands in for data/vocab/factors_seed.yaml)
    run("m2-expand")
    run("m2-slots"); run("m2-cluster"); run("m2-name")   # ablations
    run("m3")
    run("m2-audit")
    run("m3-refute")
    run("m2-refine")
    run("baseline-direct")
    run("m2-map")
    run("vocab-metrics")
    run("m4-export")
    import yaml as Y
    key = {k["item"]: k for k in Y.safe_load(open(os.path.join(OUT, "annotate", "audit_K_key.yaml")))["items"]}
    from ctxguard.model import Vocab
    from ctxguard.common import p
    V = Vocab.load(p("data/vocab/factors_v0.yaml"))
    fill_csv(os.path.join(OUT, "annotate", "audit_K.csv"),
             lambda r: r.__setitem__("GOLD_label(-/+/0)", oracle(key[int(r["item"])]["effect"], key[int(r["item"])]["asset"],
                                                               V.cell_dict(tuple(key[int(r["item"])]["cell"])))))
    run("m4-score", "--gold", "out_mock/annotate/audit_K.csv")
    for lid in ("L2", "L3"):
        run("m5", "--layout", lid)
        run("m6", "--layout", lid)
        run("sim", "--layout", lid)
        run("eval", "--layout", lid)
        # mock gold: execute OK unless the oracle says some effect harms in the true world
        scen = {s["id"]: s for s in Y.safe_load(open(os.path.join(OUT, "sim", f"scenarios_{lid}.yaml")))}
        def g(r):
            s = scen[r["scenario"]]
            r["OK_execute(1/0)"] = "0" if s["forged"] else "1"
            r["OK_block(1/0)"] = "1" if s["forged"] else "0"
            r["OK_ask(1/0)"] = "1" if s["missing"] else "0"
            r["OK_defer(1/0)"] = "0"
            r["OK_substitute(write action or 0)"] = "0"
        fill_csv(os.path.join(OUT, "annotate", f"scenario_gold_{lid}.csv"), g)
    run("eval-score", "--layout", "L2", "L3", "--gold", "out_mock/annotate/scenario_gold_L2.csv", "out_mock/annotate/scenario_gold_L3.csv")
    run("robust", "--layout", "L2", "L3", "--gold", "out_mock/annotate/scenario_gold_L2.csv", "out_mock/annotate/scenario_gold_L3.csv")
    run("rq3", "--layout", "L2", "L3")
    run("rq3-verify", "--layout", "L2", "L3")
    run("report")
    # ---- next vocabulary version from the confirmed counterexample (CEGAR round)
    run("m2-consolidate", "--base", "current", "--dims", "zone_occupant")
    cand = Y.safe_load(open(os.path.join(OUT, "m2", "C3_factors_candidate.yaml")))
    assert "watching_a_movie" in next(f for f in cand["factors"] if f["name"] == "zone_occupant")["values"], cand["refines"]
    run("m2-freeze", "--src", "out_mock/m2/C3_factors_candidate.yaml", "--name", "v0b", "--dest", "out_mock/vocab/factors_v0b.yaml",
        "--no-provenance-check")   # v0 is the hand-made baseline without provenance
    cfg["paths"]["factors"] = "out_mock/vocab/factors_v0b.yaml"
    yaml.safe_dump(cfg, open(os.path.join(ROOT, "tests", "config_mock2.yaml"), "w"))
    env = {"CTXGUARD_MOCK_VOCAB": "out_mock/vocab/factors_v0b.yaml"}
    run("vocab-migrate", "--old", "data/vocab/factors_v0.yaml", config="config_mock2.yaml", env=env)
    run("m3", "--only-stale", config="config_mock2.yaml", env=env)
    run("m3", "--spec-out", "out_mock/m3/spec_scratch.yaml", config="config_mock2.yaml", env=env)
    run("k-compare", "--a", "out_mock/m3/spec.yaml", "--b", "out_mock/m3/spec_scratch.yaml", config="config_mock2.yaml", env=env)
    run("m2-audit", config="config_mock2.yaml", env=env)
    kc = Y.safe_load(open(os.path.join(OUT, "eval", "k_compare.yaml")))
    assert kc["agreement"] == 1.0, kc
    run("m3", "--effects", "sound:increase", "--merge", config="config_mock2.yaml", env=env)
    run("k-compare", "--a", "out_mock/m3/spec.yaml", "--b", "out_mock/m3/spec_scratch.yaml", config="config_mock2.yaml", env=env)
    assert Y.safe_load(open(os.path.join(OUT, "eval", "k_compare.yaml")))["agreement"] == 1.0
    # check sheets: fill a few cells and score them
    def g_rules(r):
        for k in r:
            if k[:2] in ("A_", "B_"):
                r[k] = "1"
    fill_csv(os.path.join(OUT, "annotate", "m2_rules_CHECK_F-v0b.csv"), g_rules)
    fill_csv(os.path.join(OUT, "annotate", "m2_C2_signature_CHECK.csv"), lambda r: r.__setitem__("HUMAN_label(+/-/0)", r["label"]))
    run("m2-check-score", "--gold", "out_mock/annotate/m2_rules_CHECK_F-v0b.csv", "out_mock/annotate/m2_C2_signature_CHECK.csv",
        config="config_mock2.yaml", env=env)
    # ---- expansion trigger 'new property' (docs Sec. 4.6): only inexpressible properties add evidence
    run("m2-ingest", "--src", "tests/ingest_properties_TEST.csv", config="config_mock2.yaml", env=env)
    ing = Y.safe_load(open(os.path.join(OUT, "m2", "I1_ingest.yaml")))
    assert ing["expressible_already"] >= 1 and ing["inexpressible"] >= 1, ing
    ev = Y.safe_load(open(os.path.join(OUT, "m2", "B1_elicitation.yaml")))
    assert all(i.get("source") == "property" for i in ev["items"] if i.get("property_id")), "ingest items mislabelled"
    run("m2-consolidate", "--base", "current", config="config_mock2.yaml", env=env)
    try:
        run("m2-ingest", "--src", "tests/ingest_heldout_TEST.csv", config="config_mock2.yaml", env=env)
        raise AssertionError("held-out property was ingested")
    except SystemExit:
        pass
    # ---- ablation: asset-agnostic question coding (A2 v3) in a separate output folder
    cfg3 = dict(cfg)
    cfg3["paths"] = {**cfg["paths"], "out": "out_mock_q"}
    cfg3["method"] = {**cfg["method"], "m2_coding": "question", "m2_grouping": "llm"}   # = the v3 procedure
    yaml.safe_dump(cfg3, open(os.path.join(ROOT, "tests", "config_mock3.yaml"), "w"))
    shutil.rmtree(os.path.join(ROOT, "out_mock_q"), ignore_errors=True)
    run("m2-extract", config="config_mock3.yaml"); run("m2-dims", config="config_mock3.yaml")
    a2q = Y.safe_load(open(os.path.join(ROOT, "out_mock_q", "m2", "A2_dims.yaml")))
    a2l = Y.safe_load(open(os.path.join(OUT, "m2", "A2_dims.yaml")))
    assert a2q["coding"] == "question" and a2l["coding"] == "lift", (a2q["coding"], a2l["coding"])
    assert a2l["grouping"] == "relation" and a2q["grouping"] == "llm"
    assert os.path.exists(os.path.join(OUT, "annotate", "m2_A2_relations_CHECK.csv"))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
