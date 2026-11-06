"""M1 evaluation: human gold Eff (action x channel matrix) vs. system Eff.

Three levels, reported separately so that channel errors and direction errors are not mixed:
  channel   does the action affect channel c at all (direction ignored)
  effect    (channel, direction) pairs; a gold 'ud' cell is two gold effects
  bidirectional recall   of the gold 'ud' cells, the share where the system keeps BOTH directions
Every effects*.yaml in out/m1 is scored (main run, k=1 ablation, alternative direction
rule), and every score is appended to out/eval/experiment_log.csv.
m1-agree: Cohen's kappa between two annotators over all action x channel cells
(labels: empty, u, d, ud) -- the inter-annotator agreement reviewers ask for."""
from __future__ import annotations

import csv
import glob
import os

from .common import Ctx, dump_yaml, load_yaml
from .explog import log_result


def export_m1_gold(ctx: Ctx) -> str:
    chans = [c["id"] for c in ctx.channels["channels"]]
    path = ctx.outp("annotate", "m1_effects_gold.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action", "match"] + chans)
        for a in ctx.actions:
            w.writerow([a["id"], str(a["match"])] + [""] * len(chans))
    return path


def _cell(v: str) -> str:
    """Normalise one gold cell to '', 'u', 'd' or 'ud'."""
    v = (v or "").strip().lower().lstrip("'")
    up = "+" in v or "u" in v
    dn = "-" in v or "d" in v
    return ("u" if up else "") + ("d" if dn else "")


def _cells(path) -> dict:
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        for ch, v in r.items():
            if ch in ("action", "match") or ch is None:
                continue
            out[(r["action"], ch)] = _cell(v)
    return out


def _gold(path):
    g = {}
    for (a, ch), v in _cells(path).items():
        s = g.setdefault(a, set())
        if "u" in v:
            s.add(f"{ch}:increase")
        if "d" in v:
            s.add(f"{ch}:decrease")
    return g


def _prf(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3),
            "tp": tp, "fp": fp, "fn": fn}


def _score(gold, eff):
    e = [0, 0, 0]
    c = [0, 0, 0]
    ud_total = ud_hit = 0
    per = {}
    for a, gs in gold.items():
        ss = set(eff.get(a, []))
        e[0] += len(gs & ss); e[1] += len(ss - gs); e[2] += len(gs - ss)
        gc, sc = {x.split(":")[0] for x in gs}, {x.split(":")[0] for x in ss}
        c[0] += len(gc & sc); c[1] += len(sc - gc); c[2] += len(gc - sc)
        for ch in gc:
            if {f"{ch}:increase", f"{ch}:decrease"} <= gs:
                ud_total += 1
                ud_hit += {f"{ch}:increase", f"{ch}:decrease"} <= ss
        per[a] = {"missed": sorted(gs - ss), "extra": sorted(ss - gs)}
    return {"effect_level": _prf(*e), "channel_level": _prf(*c),
            "bidirectional_recall": round(ud_hit / ud_total, 3) if ud_total else None,
            "gold_bidirectional_cells": ud_total,
            "gold_effects_per_action": round(sum(len(v) for v in gold.values()) / max(1, len(gold)), 2),
            "per_action": per}


def score_m1(ctx: Ctx, gold_path: str) -> dict:
    gold = _gold(gold_path)
    res = {}
    for f in sorted(glob.glob(ctx.outp("m1", "effects*.yaml"))):
        lib = load_yaml(f)
        name = os.path.basename(f)[:-5]
        sc = _score(gold, lib["eff"])
        sc.update({"file": os.path.basename(f), "samples": lib.get("samples"),
                   "direction_rule": lib.get("direction_rule", "majority (v2)"),
                   "prompt_version": lib.get("prompt_version", "M1-v1")})
        res[name] = sc
        log_result(ctx, "m1-score", f, {k: sc[k] for k in ("effect_level", "channel_level", "bidirectional_recall")},
                   method_version=f"{sc['prompt_version']}; k={sc['samples']}; rule={sc['direction_rule']}",
                   n=len(gold), note=f"gold={os.path.basename(gold_path)}")
    dump_yaml(res, ctx.outp("eval", "m1_metrics.yaml"))
    return {k: {kk: vv for kk, vv in v.items() if kk != "per_action"} for k, v in res.items()}


def agree_m1(ctx: Ctx, path_a: str, path_b: str) -> dict:
    """Cohen's kappa over the cells both annotators filled in the same rows (actions)."""
    from .vocab import _kappa
    a, b = _cells(path_a), _cells(path_b)
    acts_b = {act for (act, _), v in b.items() if v}       # rows the second annotator actually did
    keys = sorted(k for k in a if k in b and k[0] in acts_b)
    la, lb = [a[k] or "none" for k in keys], [b[k] or "none" for k in keys]
    res = {"cells": len(keys), "actions": len(acts_b),
           "raw_agreement": round(sum(x == y for x, y in zip(la, lb)) / max(1, len(keys)), 3),
           "kappa": _kappa(la, lb),
           "disagreements": [{"action": k[0], "channel": k[1], "a": a[k], "b": b[k]}
                             for k in keys if a[k] != b[k]]}
    out = ctx.outp("eval", "m1_agreement.yaml")
    dump_yaml(res, out)
    log_result(ctx, "m1-agree", out, {"kappa": res["kappa"], "raw_agreement": res["raw_agreement"]},
               n=res["cells"], note=f"{os.path.basename(path_a)} vs {os.path.basename(path_b)}")
    return {k: v for k, v in res.items() if k != "disagreements"}
