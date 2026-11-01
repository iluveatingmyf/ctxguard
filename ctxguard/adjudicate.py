"""M7-M11  Runtime adjudication  (Sec. V-C)

M7  verification set  F_a = dom(rho) ∪ ⋃{dom(phi) : b ∈ K, b.effect ∈ Eff(a)}
                      (∪ the same for same-area alternatives when needed)
M8  grounding         ŝ = Γ(obs) restricted to F_a (three-valued, see grounding.py)
M9  intent check      gap B. Two groundings of dom(rho):
                        full   = with every entity (incl. the trigger)
                        indep  = with the TRIGGER entities masked
                      status = false        if indep (or full) contradicts rho
                               corroborated if indep satisfies rho
                               signal_only  if only full satisfies rho
                               unknown      otherwise
M10 effect check      for every e ∈ Eff(a), every binding of e:
                      match3(phi, ŝ) -> harms True / harms Unknown / benefits (gap A)
M11 disposition       deterministic decision procedure (R1-R4 in Sec. III).
  Let H = assets possibly harmed by a (True or Unknown), worst(X) = most
  important asset in X under the fixed priority order.
  1. intent false                          -> block
  2. H empty                               -> execute
  3. intent unknown                        -> ask
  4. an alternative a' (same area, e* ∈ Eff(a')) with worst(H') strictly less
     important than worst(H) -- and, if intent is signal_only, H' free of
     protective assets (life_health, security, property)  -> substitute (best a')
  5. intent signal_only and H contains a protective asset -> ask
     (an uncorroborated signal never causes protective harm)
  6. some harm True: compare worst(harms True) with worst({v*} ∪ benefits True):
       harm outranks    -> block if v* protective else defer
       benefit outranks -> execute (+notify)
       same asset       -> ask
  7. only Unknown harms                     -> ask
No LLM call happens at runtime.
"""
from __future__ import annotations

import time

from .common import Ctx, load_yaml
from .grounding import Grounder
from .layout import Layout
from .model import UNKNOWN, Binding, match3, normalize

PROTECTIVE = {"life_health", "security", "property"}


class Adjudicator:
    def __init__(self, ctx: Ctx, layout: Layout, spec: list[Binding], intents: dict, *,
                 gamma=None, skip_intent=False, ground_all=False, fill_default=False,
                 no_alternatives=False, static_guard=False, no_corroboration=False, name="ours"):
        self.ctx, self.lay, self.vocab = ctx, layout, ctx.vocab
        self.spec = spec
        self.lib = ctx.effects()
        self.intents = intents
        self.grounder = Grounder(ctx, layout, gamma)
        self.priority = ctx.priority
        self.opts = dict(skip_intent=skip_intent, ground_all=ground_all, fill_default=fill_default,
                         no_alternatives=no_alternatives, static_guard=static_guard,
                         no_corroboration=no_corroboration)
        self.name = name
        self.by_effect = {}
        for b in spec:
            self.by_effect.setdefault(b.effect, []).append(b)
        self.scope = {a["id"]: a["scope"] for a in ctx.actions}
        self.cands = layout.candidate_actions()

    # ---------------------------------------------------------------- M7
    def factors_for_type(self, a_type):
        fs = set()
        for e in self.lib["eff"].get(a_type, []):
            for b in self.by_effect.get(e, []):
                fs |= set(b.phi)
        return fs

    def alternatives(self, a_type, area, intent):
        if self.opts["no_alternatives"] or self.opts["static_guard"] or not intent.get("intended_effect"):
            return []
        es = intent["intended_effect"]
        out = [c for c in self.cands if c["type"] != a_type and c["area"] == area
               and self.scope.get(c["type"]) == "local" and es in self.lib["eff"].get(c["type"], [])]
        return sorted(out, key=lambda c: (c["type"], c["entity"]))

    def verification_set(self, rule_id):
        it = self.intents[rule_id]
        a_type, area = it["action_type"], it["area"]
        if self.opts["ground_all"]:
            return list(self.vocab.names)
        fs = set(it.get("rho", {})) if not (self.opts["skip_intent"] or self.opts["static_guard"]) else set()
        fs |= self.factors_for_type(a_type)
        for c in self.alternatives(a_type, area, it):
            fs |= self.factors_for_type(c["type"])
        return [n for n in self.vocab.names if n in fs]

    # ---------------------------------------------------------------- M9-M11 (pure)
    def assess(self, a_type, s):
        hT, hU, bT, ev = set(), set(), set(), []
        for e in self.lib["eff"].get(a_type, []):
            for b in self.by_effect.get(e, []):
                m = match3(b.phi, s)
                if self.opts["static_guard"] and m is None:
                    m = False
                if m is False:
                    continue
                ev.append({"effect": e, "asset": b.asset, "label": b.label, "status": "true" if m else "unknown"})
                if b.label == "-":
                    (hT if m else hU).add(b.asset)
                elif m:
                    bT.add(b.asset)
        return hT, hU - hT, bT, ev

    def top(self, assets):
        return min((self.priority.index(a) for a in assets), default=len(self.priority))

    def decide(self, rule_id, s, s_rho=None):
        it = self.intents[rule_id]
        a_type, area = it["action_type"], it["area"]
        hT, hU, bT, ev = self.assess(a_type, s)
        res = {"harms_true": sorted(hT), "harms_unknown": sorted(hU), "benefits_true": sorted(bT), "evidence": ev}
        if self.opts["static_guard"]:
            res.update(intent="n/a", disposition="block" if hT else "execute", alternative=None)
            return res
        rho = normalize(it.get("rho") or {}, self.vocab)
        if self.opts["skip_intent"]:
            status = "corroborated"
        else:
            full = match3(rho, s)
            ind = full if s_rho is None else match3(rho, {**s, **s_rho})
            if ind is False or full is False:
                status = "false"
            elif ind is True:
                status = "corroborated"
            elif full is True:
                status = "signal_only"
            else:
                status = "unknown"
        res["intent"] = status
        res["alternative"] = None
        H = hT | hU
        if status == "false":
            res["disposition"] = "block"
            return res
        if not H:
            res["disposition"] = "execute"
            return res
        if status == "unknown":
            res["disposition"] = "ask"
            return res
        worst = self.top(H)
        prot = lambda i: i < len(self.priority) and self.priority[i] in PROTECTIVE
        best = None
        for c in self.alternatives(a_type, area, it):
            cT, cU, _, _ = self.assess(c["type"], s)
            w = self.top(cT | cU)                  # alternative's worst possible harm
            if w > worst and (status == "corroborated" or not prot(w)) and (best is None or w > best[0]):
                best = (w, c, sorted(cT | cU))
        if best:
            res["disposition"], res["alternative"] = "substitute", best[1]
            res["alternative_residual_harms"] = best[2]
            return res
        if status == "signal_only" and prot(worst):
            res["disposition"] = "ask"
            return res
        if hT:
            lost = ({it["served_asset"]} | bT) if it.get("served_asset") else bT
            if self.top(hT) < self.top(lost):
                res["disposition"] = "block" if it.get("served_asset") in PROTECTIVE else "defer"
            elif self.top(hT) > self.top(lost):
                res["disposition"] = "execute"
            else:
                res["disposition"] = "ask"
            return res
        res["disposition"] = "ask"
        return res

    def uses_corroboration(self, rule_id):
        return not (self.opts["skip_intent"] or self.opts["static_guard"] or self.opts["no_corroboration"]) \
            and bool(self.intents[rule_id].get("rho"))

    def lay_rule(self, rule_id):
        return next(r for r in self.lay.rules if r["id"] == rule_id)

    # ---------------------------------------------------------------- full run
    def run(self, rule_id, obs):
        t0 = time.perf_counter()
        it = self.intents[rule_id]
        Fa = self.verification_set(rule_id)
        s, read = self.grounder.ground(Fa, it["area"], self.scope.get(it["action_type"]) == "home", obs)
        if self.opts["fill_default"]:
            for n in Fa:
                if s[n] is UNKNOWN:
                    e = self.grounder.by.get((n, None)) or self.grounder.by.get((n, it["area"])) or {}
                    s[n] = e.get("default") or self.vocab.domain(n)[0]
        s_rho, read2 = None, set()
        if self.uses_corroboration(rule_id):
            rho_f = [n for n in self.vocab.names if n in (it.get("rho") or {})]
            s_rho, read2 = self.grounder.ground(rho_f, it["area"], self.scope.get(it["action_type"]) == "home", obs,
                                                mask=set(self.lay.trigger_entities(self.lay_rule(rule_id))))
            if self.opts["fill_default"]:
                for n in rho_f:
                    if s_rho[n] is UNKNOWN:
                        s_rho[n] = s[n]
        res = self.decide(rule_id, s, s_rho)
        read |= read2
        res.update({"s_rho_independent": s_rho, "method": self.name, "rule": rule_id, "action_type": it["action_type"], "area": it["area"],
                    "Fa": Fa, "s_hat": {k: v for k, v in s.items()}, "n_factors": len(Fa),
                    "n_entities_read": len(read), "latency_ms": round((time.perf_counter() - t0) * 1000, 3)})
        return res


def load_spec_bindings(ctx: Ctx, path=None):
    return [Binding.from_json(b, ctx.vocab) for b in load_yaml(path or ctx.spec_path())["bindings"]]


def property_spec(ctx: Ctx):
    """K_P for baseline B1: '-' constraints obtained by mapping the seed properties."""
    rows = load_yaml(ctx.outp("m2", "property_mapping.yaml"))
    out = []
    for r in rows:
        if r.get("expressible") and r.get("label") == "-":
            for e in r["effects"]:
                out.append(Binding(normalize(r["pattern"], ctx.vocab), e, r["asset"], "-", r["text"]))
    return out
