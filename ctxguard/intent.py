"""M6  Intent recovery  I_r = <rho, a, e*, v*>   (Sec. V-B)

The platform stores only trigger, guard and action. Intent is recovered once
per rule at deployment: rho as a pattern over F (the situation the designer
meant, not the sensor event), e* ∈ Eff(a), v* ∈ V, plus alternative readings
of the trigger (inputs to gap-B2 checks). Output is validated against F, Eff,
V; invalid fields are reported, not repaired. Gold intents are annotated by a
human (export_intent_gold) to measure exact-match accuracy.
"""
from __future__ import annotations

import csv

from .common import Ctx, dump_yaml, load_yaml
from .layout import Layout
from .model import normalize, pattern_to_json
from .prompts import intent_prompt


def run_m6(ctx: Ctx, layout_id: str) -> dict:
    lay = Layout(layout_id, ctx.actions)
    lib = ctx.effects()
    eidx = {e["id"]: e for e in lib["effects"]}
    vocab = ctx.vocab
    out_rules, problems = {}, []
    for r in lay.rules:
        a_type, ent, area = lay.rule_actions(r)[0]
        if a_type is None:
            problems.append({"rule": r["id"], "problem": "action not typed"})
            continue
        effs = lib["eff"].get(a_type, [])
        edesc = "\n".join(f"- {e}: {eidx[e]['desc']}" for e in effs)
        sys, user, schema = intent_prompt(lay.rule_yaml(r), a_type, edesc, vocab.describe_vocab())
        out = ctx.llm.json("M6", "intent", sys, user, schema,
                           context={"rule": r, "action_type": a_type, "effects": effs, "layout": lay.id})
        rec = {"action_type": a_type, "entity": ent, "area": area, "valid": True, "errors": []}
        try:
            rec["rho"] = pattern_to_json(normalize(out.get("rho") or {}, vocab), vocab)
        except ValueError as e:
            rec["valid"], rec["rho"] = False, {}
            rec["errors"].append(f"rho: {e}")
        if out.get("intended_effect") in effs:
            rec["intended_effect"] = out["intended_effect"]
        else:
            rec["valid"] = False
            rec["errors"].append(f"intended_effect {out.get('intended_effect')} not in Eff({a_type})")
            rec["intended_effect"] = None
        if out.get("served_asset") in {a["id"] for a in ctx.asset_list}:
            rec["served_asset"] = out["served_asset"]
        else:
            rec["valid"] = False
            rec["errors"].append(f"served_asset {out.get('served_asset')} invalid")
            rec["served_asset"] = None
        alt = {}
        for n, vs in (out.get("alternative_readings") or {}).items():
            if n in vocab.by_name:
                vs = [str(v) for v in (vs if isinstance(vs, list) else [vs]) if str(v) in vocab.domain(n)]
                if vs:
                    alt[n] = vs
        rec["alternative_readings"] = alt
        out_rules[r["id"]] = rec
    res = {"layout": lay.id, "intents": out_rules, "problems": problems}
    dump_yaml(res, ctx.outp("m6", f"intents_{lay.id}.yaml"))
    return res


def export_intent_gold(ctx: Ctx, layout_id: str) -> str:
    lay = Layout(layout_id, ctx.actions)
    lib = ctx.effects()
    path = ctx.outp("annotate", f"intent_gold_{lay.id}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rule_id", "alias", "rule", "action_type", "effects_to_choose_from",
                    "GOLD_rho(e.g. indoor_air=poor; time_period=day|night)", "GOLD_intended_effect", "GOLD_served_asset"])
        for r in lay.rules:
            a_type = lay.rule_actions(r)[0][0]
            w.writerow([r["id"], r["alias"], lay.rule_yaml(r), a_type, " ".join(lib["eff"].get(a_type, [])), "", "", ""])
    return path


def parse_rho(text: str) -> dict:
    out = {}
    for part in [x.strip() for x in text.split(";") if x.strip()]:
        n, vs = part.split("=", 1)
        out[n.strip()] = [v.strip() for v in vs.split("|")]
    return out


def load_gold_intents(ctx: Ctx, layout_id: str, path: str) -> dict:
    vocab = ctx.vocab
    lay = Layout(layout_id, ctx.actions)
    sysint = load_yaml(ctx.outp("m6", f"intents_{layout_id}.yaml"))["intents"]
    gold = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        if not row["GOLD_intended_effect"].strip():
            continue
        a_type, ent, area = lay.rule_actions(next(r for r in lay.rules if r["id"] == row["rule_id"]))[0]
        gold[row["rule_id"]] = {
            "action_type": a_type, "entity": ent, "area": area, "valid": True,
            "rho": pattern_to_json(normalize(parse_rho(row["GOLD_rho(e.g. indoor_air=poor; time_period=day|night)"]), vocab), vocab),
            "intended_effect": row["GOLD_intended_effect"].strip(), "served_asset": row["GOLD_served_asset"].strip(),
            "alternative_readings": sysint.get(row["rule_id"], {}).get("alternative_readings", {})}
    return gold


def score_intents(ctx: Ctx, layout_id: str, path: str) -> dict:
    sysint = load_yaml(ctx.outp("m6", f"intents_{layout_id}.yaml"))["intents"]
    gold = load_gold_intents(ctx, layout_id, path)
    n = len(gold)
    rho = sum(sysint.get(k, {}).get("rho") == g["rho"] for k, g in gold.items())
    eff = sum(sysint.get(k, {}).get("intended_effect") == g["intended_effect"] for k, g in gold.items())
    ass = sum(sysint.get(k, {}).get("served_asset") == g["served_asset"] for k, g in gold.items())
    allm = sum(sysint.get(k, {}).get("rho") == g["rho"] and sysint.get(k, {}).get("intended_effect") == g["intended_effect"]
               and sysint.get(k, {}).get("served_asset") == g["served_asset"] for k, g in gold.items())
    # factor-level F1 of rho (partial credit)
    tp = fp = fn = 0
    for k, g in gold.items():
        s = sysint.get(k, {}).get("rho") or {}
        gs = {(n_, v) for n_, vs in g["rho"].items() for v in vs}
        ss = {(n_, v) for n_, vs in s.items() for v in vs}
        tp, fp, fn = tp + len(gs & ss), fp + len(ss - gs), fn + len(gs - ss)
    res = {"layout": layout_id, "n": n, "rho_exact": round(rho / max(1, n), 3), "effect_exact": round(eff / max(1, n), 3),
           "asset_exact": round(ass / max(1, n), 3), "all_exact": round(allm / max(1, n), 3),
           "rho_literal_f1": round(2 * tp / max(1, 2 * tp + fp + fn), 3)}
    dump_yaml(res, ctx.outp("eval", f"intent_metrics_{layout_id}.yaml"))
    return res
