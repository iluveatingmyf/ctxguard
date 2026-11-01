"""RQ3  Policy compilation and reuse  (Sec. V-D)

Observation: for a rule r, the adjudicator's decision is a pure function of the
three-valued assignment ŝ restricted to the verification set F_a (Sec. V-C):
      Π_r : ∏_{f ∈ F_a} (D_f ∪ {?}) -> Dispositions
The domain is finite, so Π_r can be
  1. enumerated exhaustively (every input, including every unknown pattern),
  2. minimized into guarded branches with the same ESPRESSO procedure as O4
     ('?' is an ordinary value here),
  3. VERIFIED: every input point is re-evaluated through the branches and must
     give the same disposition (equivalence by exhaustion, not by testing),
  4. emitted as a native Home Assistant automation (choose: branches) plus
     template sensors that implement Γ, so the hardened rule runs on stock HA
     with no LLM and no runtime service.
Reuse: the abstract policy (branches over factor values, alternatives by action
TYPE) depends only on (K, Eff, intent, zone type, available alternative types);
it is reused verbatim in another home whose rule has the same key, and only Γ
(sensor bindings) is regenerated. We report the number of distinct abstract
policies vs. the number of rules across layouts.
"""
from __future__ import annotations

import itertools
import json
import time

import yaml

from .adjudicate import Adjudicator, load_spec_bindings
from .common import Ctx, dump_yaml, load_yaml
from .grounding import AGG_ORDER, BAD_STATES
from .layout import Layout
from .minimize import cube_to_pattern, minimize

Q = "?"


def _label(d):
    if d["disposition"] == "substitute":
        return f"substitute:{d['alternative']['type']}"
    return d["disposition"]


def compile_rule(adj: Adjudicator, rule_id: str) -> dict:
    vocab = adj.vocab
    it = adj.intents[rule_id]
    Fa = adj.verification_set(rule_id)
    if adj.uses_corroboration(rule_id):    # intent factors grounded without the trigger
        Fa = Fa + [f"rho:{n}" for n in vocab.names if n in it["rho"]]
    doms = []
    for n in Fa:
        f = vocab.by_name[n.split(":", 1)[-1]]
        if f.scope == "action":
            doms.append((adj.lay.zone_type(it["area"]),))
        else:
            doms.append(tuple(f.values) + (Q,))
    t0 = time.perf_counter()
    table = {}
    for point in itertools.product(*doms):
        s = {n: (None if v == Q else v) for n, v in zip(Fa, point) if not n.startswith("rho:")}
        s_rho = {n[4:]: (None if v == Q else v) for n, v in zip(Fa, point) if n.startswith("rho:")}
        table[point] = _label(adj.decide(rule_id, s, s_rho if adj.uses_corroboration(rule_id) else None))
    t_enum = time.perf_counter() - t0
    branches = []
    labels = sorted(set(table.values()))
    for lab in labels:
        on = [p for p, l in table.items() if l == lab]
        off = [p for p, l in table.items() if l != lab]
        for cube in minimize(on, off, doms):
            branches.append({"disposition": lab,
                             "when": {n: sorted(s) for n, s in cube_to_pattern(cube, Fa, doms).items()}})
    # equivalence by exhaustion
    mism = 0
    for p, lab in table.items():
        hit = [b["disposition"] for b in branches if all(p[Fa.index(n)] in vs for n, vs in b["when"].items())]
        if len(set(hit)) != 1 or hit[0] != lab:
            mism += 1
    key = json.dumps({"a": it["action_type"], "rho": it.get("rho"), "e": it.get("intended_effect"),
                      "v": it.get("served_asset"), "zone": adj.lay.zone_type(it["area"]),
                      "alts": sorted({c["type"] for c in adj.alternatives(it["action_type"], it["area"], it)})},
                     sort_keys=True)
    return {"rule": rule_id, "Fa": Fa, "domain_size": len(table), "dispositions": dict(
        (l, sum(1 for v in table.values() if v == l)) for l in labels),
        "branches": branches, "n_branches": len(branches), "mismatches": mism,
        "enumeration_s": round(t_enum, 3), "reuse_key": key}


# ------------------------------------------------------------------ Γ -> Jinja
def _j_atom(c, mask=frozenset()):
    if ("state" in c and c["state"] in mask) or ("num" in c and c["num"] in mask):
        return "1"                       # masked entity = unknown
    if "state" in c:
        e = c["state"]
        bad = f"states('{e}') in {sorted(x for x in BAD_STATES if x)}"
        test = f"states('{e}') == '{c['eq']}'" if "eq" in c else f"states('{e}') in {[str(x) for x in c['in']]}"
        return f"(1 if {bad} else (2 if {test} else 0))"
    if "num" in c:
        e = c["num"]
        op = f"> {float(c['gt'])}" if "gt" in c else f"< {float(c['lt'])}"
        return f"(1 if not is_number(states('{e}')) else (2 if states('{e}')|float {op} else 0))"
    if "time" in c:
        a, b = c["time"]
        t = "now().strftime('%H:%M')"
        cond = f"('{a}' <= {t} < '{b}')" if a <= b else f"({t} >= '{a}' or {t} < '{b}')"
        return f"(2 if {cond} else 0)"
    if "held_min" in c:
        inner = _j_atom(c["cond"], mask)
        from .grounding import _entities_in
        es = sorted(_entities_in(c["cond"], set()))
        held = " and ".join(f"(now() - states.{e}.last_changed).total_seconds() >= {60 * float(c['held_min'])}" for e in es)
        return f"({inner} if {inner} != 2 else (2 if ({held}) else 0))"
    if "all" in c:
        return "([" + ", ".join(_j_atom(x, mask) for x in c["all"]) + "]|min)"
    if "any" in c:
        return "([" + ", ".join(_j_atom(x, mask) for x in c["any"]) + "]|max)"
    if "not" in c:
        return f"(2 - {_j_atom(c['not'], mask)})"
    raise ValueError(c)


def _j_chain(rules, default, mask=frozenset()):
    if not rules:
        return default or "unknown"
    c = _j_atom(rules[0]["when"], mask)
    return (f"{{% if {c} == 2 %}}{rules[0]['value']}{{% elif {c} == 1 %}}unknown{{% else %}}"
            + _j_chain(rules[1:], default, mask) + "{% endif %}")


def ctx_sensor(factor, area, suffix=""):
    return f"sensor.ctx_{factor}_{area or 'home'}{suffix}"


def gamma_to_templates(gamma: dict, lay: Layout, vocab, mask=frozenset(), suffix="", only=None) -> list[dict]:
    """Γ as HA template sensors. With `mask`/`suffix`: the independent (trigger-masked)
    variant used for intent corroboration of one rule."""
    out = []
    for e in gamma["entries"]:
        if only is not None and e["factor"] not in only:
            continue
        st = _j_chain(e["rules"], e["default"], mask) if e["groundable"] else "unknown"
        out.append({"name": ctx_sensor(e["factor"], e["area"], suffix).split(".", 1)[1], "state": st})
    for f in vocab.factors:   # home aggregation for zone factors
        if f.scope != "zone" or (only is not None and f.name not in only):
            continue
        order = AGG_ORDER.get(f.name, list(f.values))
        vals = "[" + ", ".join(f"states('{ctx_sensor(f.name, a['id'], suffix)}')" for a in lay.indoor_areas()) + "]"
        body = f"{{% set v = {vals} %}}{{% if '{order[0]}' in v %}}{order[0]}{{% elif 'unknown' in v %}}unknown"
        for o in order[1:]:
            body += f"{{% elif '{o}' in v %}}{o}"
        body += "{% else %}unknown{% endif %}"
        out.append({"name": f"ctx_{f.name}_home{suffix}", "state": body})
    return out


def to_ha_automation(rule: dict, comp: dict, adj: Adjudicator) -> dict:
    it = adj.intents[rule["id"]]
    home = adj.scope.get(it["action_type"]) == "home"
    alts = {c["type"]: c for c in adj.alternatives(it["action_type"], it["area"], it)}
    choose = []
    for b in comp["branches"]:
        conds = []
        for n, vs in b["when"].items():
            base = n.split(":", 1)[-1]
            f = adj.vocab.by_name[base]
            if f.scope == "action":
                continue
            suffix = f"_indep_{rule['id'].lower()}" if n.startswith("rho:") else ""
            sensor = ctx_sensor(base, None if (f.scope == "global" or home) else it["area"], suffix)
            states = ["unknown" if v == Q else v for v in vs]
            conds.append(f"states('{sensor}') in {states}")
        tpl = "{{ " + (" and ".join(conds) if conds else "true") + " }}"
        d = b["disposition"]
        if d == "execute":
            seq = rule["action"]
        elif d.startswith("substitute:"):
            seq = [alts[d.split(":", 1)[1]]["call"]]
        elif d == "ask":
            seq = [{"service": "notify.notify", "data": {"message": f"[{rule['alias']}] needs confirmation"}}]
        elif d == "defer":
            seq = [{"service": "notify.notify", "data": {"message": f"[{rule['alias']}] deferred"}}]
        else:
            seq = [{"service": "notify.notify", "data": {"message": f"[{rule['alias']}] blocked"}}]
        choose.append({"conditions": [{"condition": "template", "value_template": tpl}], "sequence": seq,
                       "alias": d})
    out = {k: v for k, v in rule.items() if k != "action"}
    out["id"] = rule["id"] + "_hardened"
    out["action"] = [{"choose": choose}]
    return out


def run_rq3(ctx: Ctx, layout_ids: list[str], intents_source: str = "system") -> dict:
    from .evaluate import _intents
    spec = load_spec_bindings(ctx)
    summary, keys = [], {}
    for lid in layout_ids:
        lay = Layout(lid, ctx.actions)
        intents = {k: v for k, v in _intents(ctx, lid, intents_source).items() if v.get("valid", True) and v.get("area")}
        adj = Adjudicator(ctx, lay, spec, intents)
        autos, comps = [], []
        for r in lay.rules:
            if r["id"] not in intents:
                continue
            c = compile_rule(adj, r["id"])
            comps.append(c)
            autos.append(to_ha_automation(r, c, adj))
            keys.setdefault(c["reuse_key"], []).append(f"{lid}:{r['id']}")
        gamma = load_yaml(ctx.outp("m5", f"gamma_{lid}.yaml"))
        templates = gamma_to_templates(gamma, lay, ctx.vocab)
        for r in lay.rules:
            if r["id"] in intents and adj.uses_corroboration(r["id"]):
                templates += gamma_to_templates(gamma, lay, ctx.vocab, mask=frozenset(lay.trigger_entities(r)),
                                                suffix=f"_indep_{r['id'].lower()}", only=set(intents[r["id"]]["rho"]))
        with open(ctx.outp("rq3", f"ha_package_{lid}.yaml"), "w") as f:
            yaml.safe_dump({"template": [{"sensor": templates}], "automation": autos}, f, sort_keys=False,
                           allow_unicode=True, width=10000)
        dump_yaml(comps, ctx.outp("rq3", f"policies_{lid}.yaml"))
        for c in comps:
            summary.append({"layout": lid, "rule": c["rule"], "|Fa|": len(c["Fa"]), "domain": c["domain_size"],
                            "branches": c["n_branches"], "mismatches": c["mismatches"],
                            "enumeration_s": c["enumeration_s"]})
    n_rules = len(summary)
    res = {"rules": n_rules, "all_equivalent": all(s["mismatches"] == 0 for s in summary),
           "mean_branches": round(sum(s["branches"] for s in summary) / max(1, n_rules), 2),
           "max_domain": max((s["domain"] for s in summary), default=0),
           "distinct_abstract_policies": len(keys),
           "reused_policies": [{"action": json.loads(k)["a"], "zone": json.loads(k)["zone"], "rules": v}
                               for k, v in keys.items() if len(v) > 1],
           "per_rule": summary}
    dump_yaml(res, ctx.outp("eval", "rq3_metrics.yaml"))
    return res


# ------------------------------------------------------------------ HA package check
def verify_ha_package(ctx: Ctx, layout_ids: list[str], intents_source: str = "system") -> dict:
    """Render the emitted HA package with Jinja2 (HA's template engine, with
    stubs for states()/is_number()/now()) on every simulated scenario, and check
    that the branch HA would take equals the runtime adjudicator's decision."""
    import datetime as dt

    import jinja2

    from .evaluate import _intents
    spec = load_spec_bindings(ctx)
    env = jinja2.Environment()
    res = {"layouts": {}}
    for lid in layout_ids:
        lay = Layout(lid, ctx.actions)
        intents = {k: v for k, v in _intents(ctx, lid, intents_source).items() if v.get("valid", True) and v.get("area")}
        adj = Adjudicator(ctx, lay, spec, intents)
        pkg = yaml.safe_load(open(ctx.outp("rq3", f"ha_package_{lid}.yaml")))
        sensors = pkg["template"][0]["sensor"]
        compiled = [(s["name"], env.from_string(s["state"])) for s in sensors]
        autos = {a["id"][: -len("_hardened")]: a for a in pkg["automation"]}
        scen = load_yaml(ctx.outp("sim", f"scenarios_{lid}.yaml"))
        agree = total = 0
        bad = []
        for sc in scen:
            if sc["rule"] not in autos:
                continue
            obs = sc["obs"]
            h, m = map(int, obs["time"].split(":"))
            now = dt.datetime(2026, 1, 15, h, m)
            st = {k: str(v["state"]) for k, v in obs["entities"].items()}

            class _Ent:
                def __init__(self, eid):
                    self.last_changed = now - dt.timedelta(minutes=float(obs["entities"].get(eid, {}).get("minutes", 0)))

            class _Dom:
                def __init__(self, d):
                    self.d = d

                def __getattr__(self, n):
                    return _Ent(f"{self.d}.{n}")

            class _States:
                def __call__(self, eid):
                    return st.get(eid, "unknown")

                def __getattr__(self, d):
                    return _Dom(d)

            def is_number(x):
                try:
                    float(x)
                    return True
                except (TypeError, ValueError):
                    return False

            g = {"states": _States(), "is_number": is_number, "now": lambda: now}
            for name, tpl in compiled:
                st[f"sensor.{name}"] = tpl.render(**g).strip()
            taken = None
            for br in autos[sc["rule"]]["action"][0]["choose"]:
                if env.from_string(br["conditions"][0]["value_template"]).render(**g).strip() == "True":
                    taken = br["alias"]
                    break
            want = _label(adj.run(sc["rule"], obs))
            total += 1
            agree += taken == want
            if taken != want and len(bad) < 5:
                bad.append({"scenario": sc["id"], "ha": taken, "adjudicator": want})
        res["layouts"][lid] = {"scenarios": total, "agree": agree, "examples_disagree": bad}
    dump_yaml(res, ctx.outp("eval", "rq3_ha_check.yaml"))
    return res
