"""Token/cost accounting and the paper-ready report (out/report.md)."""
from __future__ import annotations

import csv
import os
from collections import defaultdict

from .common import Ctx, dump_yaml, load_yaml

COST_CLASS = {"M1": "one-time knowledge", "M2": "one-time knowledge", "M3": "one-time knowledge",
              "M5": "per-home deployment", "M6": "per-home deployment",
              "B2": "per-trigger (baseline LLM guard)", "BASE": "evaluation baselines"}


def tokens(ctx: Ctx) -> dict:
    path = ctx.llm.ledger_path
    agg = defaultdict(lambda: {"calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0})
    for r in csv.DictReader(open(path)):
        for key in ((r["module"], r["task"]), (r["module"], "*")):
            a = agg[key]
            if r["cached"] == "1":
                a["cached_calls"] += 1
                continue                      # cached replays cost nothing new
            a["calls"] += 1
            a["prompt_tokens"] += int(r["prompt_tokens"] or 0)
            a["completion_tokens"] += int(r["completion_tokens"] or 0)
            a["seconds"] += int(r["duration_ms"] or 0) / 1000
    res = {f"{m}/{t}": {**v, "seconds": round(v["seconds"], 1), "cost_class": COST_CLASS.get(m, "")}
           for (m, t), v in sorted(agg.items())}
    res["_note"] = "Runtime adjudication (ours, M7-M11) makes 0 LLM calls; see runtime latency in runtime_metrics."
    dump_yaml(res, ctx.outp("eval", "tokens.yaml"))
    return res


def _tbl(rows, header):
    out = "| " + " | ".join(header) + " |\n| " + " | ".join("---" for _ in header) + " |\n"
    for r in rows:
        out += "| " + " | ".join(str(x) for x in r) + " |\n"
    return out


def report(ctx: Ctx) -> str:
    e = lambda *p: os.path.join(ctx.out, "eval", *p)
    md = [f"# ctxguard results  (model: {ctx.cfg['llm']['model']}, backend: {ctx.cfg['llm']['backend']}, vocab: {ctx.vocab.version})\n"]
    if ctx.cfg["llm"]["backend"] == "mock":
        md.append("> **MOCK BACKEND - smoke test only, not results.**\n")
    if os.path.exists(e("vocab_metrics.yaml")):
        v = load_yaml(e("vocab_metrics.yaml"))
        md.append("## Vocabulary (RQ1)\n")
        md.append(f"|Sigma| = {v['sigma_size']} of product {v['product_size']}; projected table max {v['projected_table_size']['max']}, "
                  f"mean {v['projected_table_size']['mean']} over {v['projected_table_size']['pairs_with_bindings']} pairs.\n")
        md.append(_tbl([[k, x["conflict_groups"], x["pairs_affected"], "yes" if x["necessary"] else "NO"]
                        for k, x in v["factor_ablation"].items()], ["factor removed", "conflict groups", "pairs affected", "necessary"]))
        if "expressibility_heldout" in v:
            md.append(f"\nHeld-out expressibility: {v['expressibility_heldout']}; K recall: {v['K_recall_of_heldout_properties']}; "
                      f"novel '-' constraints: {v['novel_constraints']}\n")
    for lid_file in sorted(f for f in os.listdir(os.path.join(ctx.out, "eval")) if f.startswith("groundability_")):
        g = load_yaml(e(lid_file))
        md.append(f"\nGroundability {g['layout']}: {g['groundable']}/{g['entries']} = {g['ratio']}; ungroundable: {', '.join(g['ungroundable']) or '-'}\n")
    if os.path.exists(e("audit_metrics.yaml")):
        a = load_yaml(e("audit_metrics.yaml"))
        md.append("\n## Specification quality vs human labels (RQ1)\n")
        rows = []
        for m in ("K", "B-K1_direct", "B-K2_per_cell"):
            x = a.get(m)
            if isinstance(x, dict):
                rows.append([m, x["accuracy"], x["kappa"], x["-"]["precision"], x["-"]["recall"], x["-"]["f1"], x["+"]["f1"]])
        md.append(_tbl(rows, ["method", "acc", "kappa", "P(-)", "R(-)", "F1(-)", "F1(+)"]))
    for f in sorted(os.listdir(os.path.join(ctx.out, "eval"))):
        if f.startswith("intent_metrics_"):
            md.append(f"\nIntent recovery {f[15:-5]}: {load_yaml(e(f))}\n")
    if os.path.exists(e("runtime_metrics.yaml")):
        r = load_yaml(e("runtime_metrics.yaml"))
        md.append(f"\n## Runtime adjudication (RQ2), n = {r['n_scenarios']} scenarios {r['by_type']}\n")
        md.append(_tbl([[m, x["accuracy"], x["unsafe_exec_rate"], x["over_intervention"], x["purpose_retention"],
                         x["mean_cost"].get("n_factors"), x["mean_cost"].get("n_entities_read"), x["mean_cost"].get("latency_ms")]
                        for m, x in r["methods"].items()],
                       ["method", "accuracy", "unsafe exec", "over-intervention", "purpose retention", "factors", "entities", "latency ms"]))
    if os.path.exists(e("robustness.yaml")):
        md.append("\n## Robustness to missing sensors\n")
        rb = load_yaml(e("robustness.yaml"))
        for m, c in rb.items():
            md.append(_tbl([[r, x["acc"], x["unsafe"], x["intervene"]] for r, x in c.items()], [f"{m}: missing ratio", "acc", "unsafe", "intervene"]))
    if os.path.exists(e("rq3_metrics.yaml")):
        q = load_yaml(e("rq3_metrics.yaml"))
        md.append(f"\n## Policy compilation (RQ3)\nrules {q['rules']}, all equivalent: {q['all_equivalent']}, mean branches {q['mean_branches']}, "
                  f"max enumerated domain {q['max_domain']}, distinct abstract policies {q['distinct_abstract_policies']}\n")
    if os.path.exists(e("tokens.yaml")):
        t = load_yaml(e("tokens.yaml"))
        md.append("\n## LLM cost\n")
        md.append(_tbl([[k, v["cost_class"], v["calls"], v["prompt_tokens"], v["completion_tokens"], v["seconds"]]
                        for k, v in t.items() if k.endswith("/*")], ["module", "class", "calls", "prompt tok", "completion tok", "s"]))
    path = os.path.join(ctx.out, "report.md")
    open(path, "w").write("\n".join(md))
    return path
