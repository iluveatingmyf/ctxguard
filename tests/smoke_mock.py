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


def run(*args):
    cmd = [sys.executable, "-m", "ctxguard.cli", *args, "--backend", "mock", "--config", os.path.join(ROOT, "tests", "config_mock.yaml")]
    print("$", " ".join(args))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
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
    run("m2-extract"); run("m2-dims"); run("m2-values"); run("m2-expand")
    run("m2-slots"); run("m2-cluster"); run("m2-name")   # ablations
    run("m3")
    run("m3-refute")
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
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
