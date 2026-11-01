"""M1 evaluation: human gold Eff (action x channel matrix) vs. system Eff."""
from __future__ import annotations

import csv
import os

from .common import Ctx, dump_yaml, load_yaml


def export_m1_gold(ctx: Ctx) -> str:
    chans = [c["id"] for c in ctx.channels["channels"]]
    path = ctx.outp("annotate", "m1_effects_gold.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action", "match"] + chans)
        for a in ctx.actions:
            w.writerow([a["id"], str(a["match"])] + [""] * len(chans))
    return path


def _gold(path):
    g = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        s = set()
        for ch, v in r.items():
            if ch in ("action", "match") or not v:
                continue
            v = v.strip().lower().lstrip("'")
            # accepted: u / d / ud  (or + / - / +-; letters avoid spreadsheet formula errors)
            if "+" in v or "u" in v:
                s.add(f"{ch}:increase")
            if "-" in v or "d" in v:
                s.add(f"{ch}:decrease")
        g[r["action"]] = s
    return g


def _score(gold, eff):
    tp = fp = fn = 0
    per = {}
    for a, gs in gold.items():
        ss = set(eff.get(a, []))
        tp, fp, fn = tp + len(gs & ss), fp + len(ss - gs), fn + len(gs - ss)
        per[a] = {"missed": sorted(gs - ss), "extra": sorted(ss - gs)}
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3),
            "gold_effects_per_action": round(sum(len(v) for v in gold.values()) / max(1, len(gold)), 2), "per_action": per}


def score_m1(ctx: Ctx, gold_path: str) -> dict:
    gold = _gold(gold_path)
    res = {}
    for tag in ("", "_k1"):
        f = ctx.outp("m1", f"effects{tag}.yaml")
        if os.path.exists(f):
            res["k=" + str(load_yaml(f)["samples"])] = _score(gold, load_yaml(f)["eff"])
    dump_yaml(res, ctx.outp("eval", "m1_metrics.yaml"))
    return {k: {kk: vv for kk, vv in v.items() if kk != "per_action"} for k, v in res.items()}
