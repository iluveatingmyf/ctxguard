"""M5  Grounding profile Γ (per deployment)  +  M8 runtime grounding  (Sec. V-A)

Knowledge layer (K, F) never mentions devices. Γ binds each factor (and each
area for zone factors) to the home's entities via ordered rules in a small
condition DSL with Kleene three-valued semantics:
  - a missing / 'unavailable' / 'unknown' entity makes its atom UNKNOWN;
  - rules are ordered; value = first TRUE rule, but an earlier UNKNOWN rule
    makes the result UNKNOWN (we cannot exclude the more specific value);
  - all rules FALSE -> `default` (may be null = UNKNOWN).
Γ is generated once per home by the LLM, validated (entity ids exist, values in
the domain, DSL well-formed) and frozen; the runtime executor is deterministic.
effect_zone is derived from the layout (no LLM).
"""
from __future__ import annotations

from .common import Ctx, dump_yaml, load_yaml
from .layout import Layout
from .model import UNKNOWN
from .prompts import grounding_prompt

BAD_STATES = {"unavailable", "unknown", "none", "", None}

# worst-case aggregation order for home-scope actions (first = most severe)
AGG_ORDER = {"zone_occupant": ["asleep", "awake", "empty"], "vulnerable_present": ["yes", "no"],
             "indoor_air": ["poor", "good"], "indoor_thermal": ["cold", "hot", "comfortable"]}


# ------------------------------------------------------------------ DSL
def _entities_in(cond, acc):
    if not isinstance(cond, dict):
        return acc
    for k in ("state", "num"):
        if k in cond:
            acc.add(cond[k])
    for k in ("all", "any"):
        for c in cond.get(k, []) or []:
            _entities_in(c, acc)
    for k in ("not", "cond"):
        if k in cond:
            _entities_in(cond[k], acc)
    return acc


def validate_cond(cond, layout: Layout) -> bool:
    if not isinstance(cond, dict) or not cond:
        return False
    if "state" in cond:
        return cond["state"] in layout.entities and ("eq" in cond or "in" in cond)
    if "num" in cond:
        return cond["num"] in layout.entities and ("gt" in cond or "lt" in cond)
    if "time" in cond:
        t = cond["time"]
        return isinstance(t, list) and len(t) == 2
    if "held_min" in cond:
        return "cond" in cond and validate_cond(cond["cond"], layout)
    for k in ("all", "any"):
        if k in cond:
            return isinstance(cond[k], list) and len(cond[k]) > 0 and all(validate_cond(c, layout) for c in cond[k])
    if "not" in cond:
        return validate_cond(cond["not"], layout)
    return False


def _min(a, b):
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return True


def _max(a, b):
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return False


def eval_cond(cond, obs, read: set):
    ents = obs["entities"]
    if "state" in cond or "num" in cond:
        eid = cond.get("state", cond.get("num"))
        read.add(eid)
        st = ents.get(eid, {}).get("state")
        if st in BAD_STATES:
            return None
        if "state" in cond:
            if "eq" in cond:
                return str(st) == str(cond["eq"])
            return str(st) in [str(x) for x in cond["in"]]
        try:
            x = float(st)
        except (TypeError, ValueError):
            return None
        if "gt" in cond:
            return x > float(cond["gt"])
        return x < float(cond["lt"])
    if "held_min" in cond:
        inner = eval_cond(cond["cond"], obs, read)
        if inner is not True:
            return inner
        mins = [ents.get(e, {}).get("minutes") for e in _entities_in(cond["cond"], set())]
        if any(m is None for m in mins):
            return None
        return min(mins) >= float(cond["held_min"])
    if "time" in cond:
        a, b = cond["time"]
        t = obs["time"]
        return (a <= t < b) if a <= b else (t >= a or t < b)
    if "all" in cond:
        r = True
        for c in cond["all"]:
            r = _min(r, eval_cond(c, obs, read))
        return r
    if "any" in cond:
        r = False
        for c in cond["any"]:
            r = _max(r, eval_cond(c, obs, read))
        return r
    if "not" in cond:
        r = eval_cond(cond["not"], obs, read)
        return None if r is None else (not r)
    return None


# ------------------------------------------------------------------ Γ build
def run_m5(ctx: Ctx, layout_id: str) -> dict:
    lay = Layout(layout_id, ctx.actions)
    vocab = ctx.vocab
    entries = []
    edesc = lay.describe_entities()
    for f in vocab.factors:
        if f.scope == "action":
            continue
        targets = [None] if f.scope == "global" else lay.indoor_areas()
        for area in targets:
            sys, user, schema = grounding_prompt(f, area, edesc)
            out = ctx.llm.json("M5", "grounding", sys, user, schema,
                               context={"factor": f.name, "area": area, "layout": lay.id, "entities": list(lay.entities)})
            rules, invalid = [], 0
            for r in out.get("rules", []):
                if str(r.get("value")) in f.values and validate_cond(r.get("when"), lay):
                    rules.append({"value": str(r["value"]), "when": r["when"]})
                else:
                    invalid += 1
            default = out.get("default")
            default = str(default) if default is not None and str(default) in f.values else None
            groundable = bool(out.get("groundable")) and bool(rules or default)
            entries.append({"factor": f.name, "area": area["id"] if area else None, "groundable": groundable,
                            "rules": rules, "default": default, "invalid_rules": invalid,
                            "entities": sorted(set().union(*[_entities_in(r["when"], set()) for r in rules])) if rules else [],
                            "note": out.get("note", "")})
    gam = {"layout": lay.id, "vocab_version": vocab.version, "entries": entries}
    dump_yaml(gam, ctx.outp("m5", f"gamma_{lay.id}.yaml"))
    n = len(entries)
    g = sum(e["groundable"] for e in entries)
    rep = {"layout": lay.id, "entries": n, "groundable": g, "ratio": round(g / max(1, n), 3),
           "invalid_rules_dropped": sum(e["invalid_rules"] for e in entries),
           "ungroundable": [f"{e['factor']}@{e['area']}" for e in entries if not e["groundable"]]}
    dump_yaml(rep, ctx.outp("eval", f"groundability_{lay.id}.yaml"))
    return rep


class Grounder:
    def __init__(self, ctx: Ctx, layout: Layout, gamma: dict | None = None):
        self.vocab = ctx.vocab
        self.lay = layout
        self.g = gamma or load_yaml(ctx.outp("m5", f"gamma_{layout.id}.yaml"))
        self.by = {(e["factor"], e["area"]): e for e in self.g["entries"]}

    def _one(self, factor, area, obs, read):
        e = self.by.get((factor, area))
        if not e or not e["groundable"]:
            return UNKNOWN
        for r in e["rules"]:
            v = eval_cond(r["when"], obs, read)
            if v is True:
                return r["value"]
            if v is None:
                return UNKNOWN
        return e["default"]

    def ground(self, factors, area, home_scope: bool, obs, mask: set | None = None):
        """Partial assignment ŝ over `factors` for an action acting in `area`.
        `mask`: entities treated as unavailable (used to ground rho WITHOUT the
        trigger entity -> independent corroboration of the intent, gap B1)."""
        if mask:
            obs = {"time": obs["time"], "entities": {k: ({"state": "unavailable", "minutes": 0} if k in mask else v)
                                                     for k, v in obs["entities"].items()}}
        read, s = set(), {}
        for n in factors:
            f = self.vocab.by_name[n]
            if f.scope == "action":
                s[n] = self.lay.zone_type(area)      # target device's area type
                if s[n] not in f.values:
                    s[n] = UNKNOWN
            elif f.scope == "global":
                s[n] = self._one(n, None, obs, read)
            elif home_scope:
                vals = [self._one(n, a["id"], obs, read) for a in self.lay.indoor_areas()]
                order = AGG_ORDER.get(n, list(f.values))
                known = [v for v in vals if v is not UNKNOWN]
                worst = min(known, key=lambda v: order.index(v) if v in order else 99) if known else UNKNOWN
                s[n] = worst if (worst is not UNKNOWN and (worst == order[0] or UNKNOWN not in vals)) else UNKNOWN
            else:
                s[n] = self._one(n, area, obs, read)
        return s, read
