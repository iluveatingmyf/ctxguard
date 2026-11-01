"""Runtime evaluation (Sec. VI-B, RQ2): methods x scenarios -> metrics.

Methods
  ours            full adjudicator (M7-M11)
  abl_no_intent   without M9 intent check (gap B handling removed)
  abl_ground_all  grounds every factor instead of F_a (cost of non-minimal verification)
  abl_fill_default unknown factors forced to a default value (no uncertainty handling)
  abl_no_corroboration intent grounded WITH the trigger entity (trusts the signal)
  B0_execute      native TAP: always execute
  B1_property     seed properties as static guards: block iff a property constraint
                  is violated; unknown treated as satisfied; no alternatives
  B2_llm          LLM sees the rule + ALL entity states and decides (tokens counted)
Metrics (gold = human-labelled acceptable dispositions per scenario)
  accuracy           system disposition is acceptable
  unsafe_exec_rate   gold forbids execute, system executes
  over_intervention  gold accepts execute, system does not execute
  purpose_retention  among scenarios where execute is forbidden but a substitute is
                     acceptable: system substitutes acceptably
  cost               factors grounded, entities read, latency, LLM tokens
"""
from __future__ import annotations

import csv
import time
from collections import Counter, defaultdict

import numpy as np

from .adjudicate import Adjudicator, load_spec_bindings, property_spec
from .common import Ctx, dump_yaml, load_yaml
from .layout import Layout
from .prompts import llm_adjudicate_prompt
from .simulate import degrade


def _intents(ctx, layout_id, source):
    if source == "system":
        return load_yaml(ctx.outp("m6", f"intents_{layout_id}.yaml"))["intents"]
    from .intent import load_gold_intents
    return load_gold_intents(ctx, layout_id, source)


def methods(ctx, lay, intents):
    spec = load_spec_bindings(ctx)
    ms = {
        "ours": Adjudicator(ctx, lay, spec, intents, name="ours"),
        "abl_no_intent": Adjudicator(ctx, lay, spec, intents, skip_intent=True, name="abl_no_intent"),
        "abl_ground_all": Adjudicator(ctx, lay, spec, intents, ground_all=True, name="abl_ground_all"),
        "abl_fill_default": Adjudicator(ctx, lay, spec, intents, fill_default=True, name="abl_fill_default"),
        "abl_no_corroboration": Adjudicator(ctx, lay, spec, intents, no_corroboration=True, name="abl_no_corroboration"),
    }
    try:
        ms["B1_property"] = Adjudicator(ctx, lay, property_spec(ctx), intents, static_guard=True, name="B1_property")
    except FileNotFoundError:
        print("[eval] B1 skipped: run `m2-map` first")
    return ms


def b2_llm(ctx, lay, rule, obs):
    states = "\n".join(f"{e}: {v['state']} (for {v['minutes']} min)" for e, v in obs["entities"].items())
    sys, user, schema = llm_adjudicate_prompt(lay.rule_yaml(rule), f"time {obs['time']}\n" + states, ctx.priority)
    t0 = time.perf_counter()
    out = ctx.llm.json("B2", "llm_adjudicate", sys, user, schema, context={"rule": rule, "obs": obs})
    return {"method": "B2_llm", "disposition": out.get("disposition", "execute"),
            "alternative": {"type": out.get("alternative", "")} if out.get("alternative") else None,
            "n_entities_read": len(obs["entities"]), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


def run_runtime(ctx: Ctx, layout_id: str, intents_source: str = "system", with_llm: bool = True) -> dict:
    lay = Layout(layout_id, ctx.actions)
    intents = _intents(ctx, layout_id, intents_source)
    scen = load_yaml(ctx.outp("sim", f"scenarios_{layout_id}.yaml"))
    ms = methods(ctx, lay, intents)
    rules = {r["id"]: r for r in lay.rules}
    out = []
    for s in scen:
        if s["rule"] not in intents:
            continue
        row = {"scenario": s["id"], "type": s["type"], "rule": s["rule"], "decisions": {}}
        for name, m in ms.items():
            d = m.run(s["rule"], s["obs"])
            row["decisions"][name] = {k: d.get(k) for k in ("disposition", "alternative", "intent", "harms_true",
                                                             "harms_unknown", "n_factors", "n_entities_read", "latency_ms", "s_hat")}
        row["decisions"]["B0_execute"] = {"disposition": "execute", "n_entities_read": 0, "n_factors": 0}
        if with_llm:
            row["decisions"]["B2_llm"] = b2_llm(ctx, lay, rules[s["rule"]], s["obs"])
        out.append(row)
    dump_yaml(out, ctx.outp("eval", f"decisions_{layout_id}.yaml"))
    return {"layout": layout_id, "scenarios": len(out), "methods": list(out[0]["decisions"]) if out else []}


def _gold(path):
    g = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        vals = [r["OK_execute(1/0)"], r["OK_defer(1/0)"], r["OK_block(1/0)"], r["OK_ask(1/0)"]]
        if not any(v.strip() for v in vals) and not r["OK_substitute(write action or 0)"].strip():
            continue
        sub = r["OK_substitute(write action or 0)"].strip()
        g[r["scenario"]] = {"execute": r["OK_execute(1/0)"].strip() == "1", "defer": r["OK_defer(1/0)"].strip() == "1",
                            "block": r["OK_block(1/0)"].strip() == "1", "ask": r["OK_ask(1/0)"].strip() == "1",
                            "substitute": sub not in ("", "0"), "substitute_text": sub}
    return g


def _acceptable(d, g):
    disp = d["disposition"]
    if disp == "substitute":
        if not g["substitute"]:
            return False
        if g["substitute_text"] == "1":
            return True
        alt = d.get("alternative") or {}
        return any(str(alt.get(k, "")) and str(alt.get(k, "")) in g["substitute_text"] for k in ("type", "entity"))
    return g.get(disp, False)


def score_runtime(ctx: Ctx, layout_ids: list[str], gold_paths: list[str]) -> dict:
    rows, gold = [], {}
    for lid, gp in zip(layout_ids, gold_paths):
        rows += load_yaml(ctx.outp("eval", f"decisions_{lid}.yaml"))
        gold.update(_gold(gp))
    rows = [r for r in rows if r["scenario"] in gold]
    res = {"n_scenarios": len(rows), "by_type": dict(Counter(r["type"] for r in rows)), "methods": {}}
    names = sorted({m for r in rows for m in r["decisions"]})
    for m in names:
        acc, unsafe, over, ret = [], [], [], []
        per_type = defaultdict(list)
        cost = defaultdict(list)
        for r in rows:
            d, g = r["decisions"].get(m), gold[r["scenario"]]
            if d is None:
                continue
            ok = _acceptable(d, g)
            acc.append(ok)
            per_type[r["type"]].append(ok)
            if not g["execute"]:
                unsafe.append(d["disposition"] == "execute")
                if g["substitute"]:
                    ret.append(d["disposition"] == "substitute" and ok)
            else:
                over.append(d["disposition"] != "execute")
            for k in ("n_factors", "n_entities_read", "latency_ms"):
                if d.get(k) is not None:
                    cost[k].append(d[k])
        f = lambda xs: round(float(np.mean(xs)), 3) if xs else None
        res["methods"][m] = {"accuracy": f(acc), "unsafe_exec_rate": f(unsafe), "over_intervention": f(over),
                             "purpose_retention": f(ret), "n_forbidden": len(unsafe), "n_allowed": len(over),
                             "per_type_accuracy": {t: f(v) for t, v in per_type.items()},
                             "mean_cost": {k: f(v) for k, v in cost.items()}}
    dump_yaml(res, ctx.outp("eval", "runtime_metrics.yaml"))
    return res


def run_robustness(ctx: Ctx, layout_ids: list[str], gold_paths: list[str], intents_source="system") -> dict:
    gold = {}
    for gp in gold_paths:
        gold.update(_gold(gp))
    ratios = ctx.cfg["eval"]["missing_ratios"]
    curves = defaultdict(lambda: defaultdict(lambda: {"unsafe": [], "intervene": [], "acc": []}))
    for lid in layout_ids:
        lay = Layout(lid, ctx.actions)
        intents = _intents(ctx, lid, intents_source)
        ms = methods(ctx, lay, intents)
        scen = [s for s in load_yaml(ctx.outp("sim", f"scenarios_{lid}.yaml"))
                if s["type"] in ("random_rho", "targeted_A") and s["id"] in gold and s["rule"] in intents]
        for ratio in ratios:
            rng = ctx.rng(f"robust:{lid}:{ratio}")
            for s in scen:
                obs = degrade(s["obs"], ratio, set(lay.trigger_entities(next(r for r in lay.rules if r["id"] == s["rule"]))), rng)
                g = gold[s["id"]]
                for name in ("ours", "abl_fill_default", "B1_property"):
                    if name not in ms:
                        continue
                    d = ms[name].run(s["rule"], obs)
                    c = curves[name][ratio]
                    c["acc"].append(_acceptable(d, g))
                    c["intervene"].append(d["disposition"] != "execute")
                    if not g["execute"]:
                        c["unsafe"].append(d["disposition"] == "execute")
    res = {m: {str(r): {k: (round(float(np.mean(v)), 3) if v else None) for k, v in c.items()} for r, c in rc.items()}
           for m, rc in curves.items()}
    dump_yaml(res, ctx.outp("eval", "robustness.yaml"))
    return res
