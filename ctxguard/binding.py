"""M3  Binding discovery  ->  specification K   (Sec. IV-C)

For every (effect e, asset v):
  O2  relevance gate      one abstract query "can e ever be +/- for v?";
                          a 'no' is verified by k random point labels.
  O3  top-down splitting  start from the whole Sigma. Ask "is the label constant
                          on this region?" If yes -> spot-check k random cells of
                          the region with point labels (abstract claim must agree
                          with concrete cells, CEGAR-style); if they disagree the
                          region is split on a factor separating the disagreeing
                          cells. If no -> split on the factor the model names.
                          Leaves partition Sigma; factors never split on are
                          irrelevant to (e,v) (implicit projection).
  O4  minimization        ESPRESSO-style prime cover per label (minimize.py);
                          lossless w.r.t. the leaf labelling.
  O5  refutation          an adversarial query per binding searches for real
                          situations inside the pattern whose expert label
                          differs because of an attribute outside F.
Output: K = bindings <phi, e, v, label, harm> with label in {+,-}; '0' is the
closed-world default.
"""
from __future__ import annotations

import time
from collections import Counter

from .common import Ctx, dump_yaml, load_yaml
from .effects import effect_index
from .minimize import cube_to_pattern, minimize, verify
from .model import Binding, contains, normalize, pattern_to_json
from .prompts import (direct_constraints_prompt, point_label_prompt, refute_prompt,
                      region_prompt, relevance_prompt)


class PairRun:
    def __init__(self, ctx: Ctx, effect: dict, asset: dict):
        self.ctx, self.e, self.v = ctx, effect, asset
        self.vocab = ctx.vocab
        self.vdesc = self.vocab.describe_vocab()
        self.k = ctx.cfg["method"]["spot_check_k"]
        self.budget = ctx.cfg["method"]["max_nodes_per_pair"]
        self.rng = ctx.rng(f"{effect['id']}|{asset['id']}")
        self.stats = Counter()
        self.point_cache = {}

    def ctxinfo(self, **kw):
        return {"effect": self.e["id"], "asset": self.v["id"], **kw}

    # ------------------------------------------------------------ queries
    def point(self, cell):
        if cell in self.point_cache:
            return self.point_cache[cell]
        sys, user, schema = point_label_prompt(self.vdesc, self.e, self.v, self.vocab.describe_cell(cell))
        out = self.ctx.llm.json("M3", "point_label", sys, user, schema,
                                context=self.ctxinfo(cell=self.vocab.cell_dict(cell)))
        lab = out.get("label", "0")
        lab = lab if lab in ("+", "-", "0") else "0"
        harm = out.get("harm", "")
        if lab == "-" and not harm.strip():
            lab = "0"                       # no mechanism -> no harm claim
            self.stats["minus_without_harm"] += 1
        self.stats["point_calls"] += 1
        self.point_cache[cell] = (lab, harm)
        return lab, harm

    def relevance(self):
        sys, user, schema = relevance_prompt(self.vdesc, self.e, self.v)
        out = self.ctx.llm.json("M3", "relevance", sys, user, schema, context=self.ctxinfo())
        self.stats["relevance_calls"] += 1
        return bool(out.get("can_affect", True)), out.get("reason", "")

    def region(self, phi, free):
        sys, user, schema = region_prompt(self.vdesc, self.e, self.v,
                                          self.vocab.describe_pattern(phi), free)
        out = self.ctx.llm.json("M3", "region", sys, user, schema,
                                context=self.ctxinfo(phi=pattern_to_json(phi, self.vocab), free=free))
        self.stats["region_calls"] += 1
        return out

    # ------------------------------------------------------------ helpers
    def allowed(self, phi, n):
        return phi.get(n, frozenset(self.vocab.domain(n)))

    def free(self, phi):
        return [n for n in self.vocab.names if len(self.allowed(phi, n)) > 1]

    def children(self, phi, n, groups):
        out = []
        for g in groups:
            c = dict(phi)
            c[n] = frozenset(g)
            out.append(normalize({k: list(v) for k, v in c.items()}, self.vocab))
        return out

    def valid_groups(self, phi, n, groups):
        allowed = self.allowed(phi, n)
        flat = [str(v) for g in groups for v in g]
        return (len(groups) >= 2 and all(groups) and len(flat) == len(set(flat))
                and set(flat) == set(allowed))

    # ------------------------------------------------------------ O2 + O3
    def run(self):
        t0 = time.time()
        leaves = []
        relevant, reason = self.relevance()
        if not relevant:
            sample = self.vocab.sample_cells({}, self.k, self.rng)
            labs = [self.point(c) for c in sample]
            if all(l == "0" for l, _ in labs):
                self.stats["irrelevant"] = 1
                return self.finish([({}, "0", reason, "relevance")], t0)
            self.stats["relevance_overridden"] = 1
        stack = [{}]
        nodes = 0
        while stack:
            phi = stack.pop()
            cells = self.vocab.cells_in(phi)
            if not cells:
                continue
            free = self.free(phi)
            if len(cells) == 1 or not free:
                lab, harm = self.point(cells[0])
                leaves.append((phi, lab, harm, "point"))
                continue
            nodes += 1
            if nodes > self.budget:                       # budget: majority of k points
                labs = [self.point(c) for c in self.vocab.sample_cells(phi, self.k, self.rng)]
                lab, _ = Counter(l for l, _ in labs).most_common(1)[0]
                harm = next((h for l, h in labs if l == lab), "")
                leaves.append((phi, lab, harm, "budget"))
                self.stats["budget_leaves"] += 1
                continue
            out = self.region(phi, free)
            if out.get("consistent") and out.get("label") in ("+", "-", "0"):
                lab, harm = out["label"], out.get("harm", "")
                sample = self.vocab.sample_cells(phi, self.k, self.rng)
                labs = [(c, *self.point(c)) for c in sample]
                if all(l == lab for _, l, _ in labs):
                    if lab == "-" and not harm.strip():
                        harm = next((h for _, l, h in labs if h), "")
                    leaves.append((phi, lab, harm, "region"))
                    continue
                self.stats["spot_check_overrides"] += 1
                n = self.separating_factor(phi, labs, lab)
                stack.extend(self.children(phi, n, [[v] for v in self.allowed(phi, n)]))
                continue
            n = out.get("split_factor")
            groups = [[str(v) for v in g] for g in (out.get("value_groups") or [])]
            if n not in free or not self.valid_groups(phi, n, groups):
                self.stats["invalid_split_repaired"] += 1
                n = n if n in free else free[0]
                groups = [[v] for v in self.allowed(phi, n)]
            stack.extend(self.children(phi, n, groups))
        self.stats["nodes"] = nodes
        return self.finish(leaves, t0)

    def separating_factor(self, phi, labs, claimed):
        cells = [c for c, _, _ in labs]
        diff = [c for c, l, _ in labs if l != claimed]
        ref = next((c for c, l, _ in labs if l == claimed), None)
        for n in self.free(phi):
            i = self.vocab.index[n]
            vals = {c[i] for c in cells}
            if ref is not None and any(d[i] != ref[i] for d in diff):
                return n
            if ref is None and len(vals) > 1:
                return n
        return self.free(phi)[0]

    # ------------------------------------------------------------ O4
    def finish(self, leaves, t0):
        vocab = self.vocab
        label = {}
        for phi, lab, harm, src in leaves:
            for c in vocab.cells_in(phi):
                label[c] = lab
        doms = [f.values for f in vocab.factors]
        bindings = []
        for lab in ("-", "+"):
            on = [c for c, l in label.items() if l == lab]
            if not on:
                continue
            off = [c for c, l in label.items() if l != lab]
            cubes = minimize(on, off, doms)
            assert verify(cubes, on, vocab.sigma()), "minimization not lossless"
            for cube in cubes:
                phi = cube_to_pattern(cube, vocab.names, doms)
                harms = [h for p, l, h, _ in leaves if l == lab and h and
                         any(contains(phi, c, vocab) for c in vocab.cells_in(p)[:200])]
                harm = " | ".join(dict.fromkeys(harms))[:600]
                bindings.append(Binding(phi, self.e["id"], self.v["id"], lab, harm,
                                        {"cells": sum(1 for c in on if contains(phi, c, vocab))}))
        self.stats["leaves"] = len(leaves)
        self.stats["seconds"] = round(time.time() - t0, 2)
        return {"bindings": bindings, "leaves": [
            {"pattern": pattern_to_json(p, vocab), "label": l, "harm": h, "source": s} for p, l, h, s in leaves],
            "stats": dict(self.stats)}


def run_m3(ctx: Ctx, limit_pairs: int | None = None, only_effects: list | None = None) -> dict:
    lib = ctx.effects()
    effects = [e for e in lib["effects"] if not only_effects or e["id"] in only_effects]
    pairs = [(e, v) for e in effects for v in ctx.asset_list]
    if limit_pairs:
        pairs = pairs[:limit_pairs]
    all_b, per_pair = [], {}
    for i, (e, v) in enumerate(pairs):
        r = PairRun(ctx, e, v).run()
        all_b.extend(r["bindings"])
        per_pair[f"{e['id']}|{v['id']}"] = {"stats": r["stats"], "leaves": r["leaves"]}
        print(f"[M3] {i + 1}/{len(pairs)} {e['id']} x {v['id']}: "
              f"{len(r['bindings'])} bindings, {r['stats'].get('region_calls', 0)} region + "
              f"{r['stats'].get('point_calls', 0)} point calls")
    spec = {"vocab_version": ctx.vocab.version, "effects_version": lib["version"],
            "bindings": [b.to_json(ctx.vocab) for b in all_b]}
    dump_yaml(spec, ctx.spec_path())
    dump_yaml(per_pair, ctx.outp("m3", "pairs_trace.yaml"))
    return spec


# ------------------------------------------------------------------- O5
def run_refute(ctx: Ctx) -> dict:
    vocab = ctx.vocab
    spec = load_yaml(ctx.spec_path())
    eidx = effect_index(ctx.effects())
    aidx = {a["id"]: a for a in ctx.asset_list}
    k = ctx.cfg["method"]["refute_max_counterexamples"]
    rows, gaps = [], Counter()
    for i, bd in enumerate(spec["bindings"]):
        b = Binding.from_json(bd, vocab)
        sys, user, schema = refute_prompt(vocab.describe_vocab(), eidx[b.effect], aidx[b.asset],
                                          vocab.describe_pattern(b.phi), b.label, b.harm, k)
        out = ctx.llm.json("M3", "refute", sys, user, schema,
                           context={"effect": b.effect, "asset": b.asset, "phi": bd["pattern"], "label": b.label})
        valid = []
        for ce in out.get("counterexamples", [])[:k]:
            a = {n: str(v) for n, v in (ce.get("assignment") or {}).items()}
            ok = (set(a) == set(vocab.names) and all(a[n] in vocab.domain(n) for n in vocab.names))
            if ok:
                cell = tuple(a[n] for n in vocab.names)
                ok = vocab.feasible(cell) and contains(b.phi, cell, vocab) and ce.get("expected_label") != b.label
            if ok:
                valid.append(ce)
                gaps[ce.get("missing_attribute", "").strip().lower()] += 1
        rows.append({"binding": i, "effect": b.effect, "asset": b.asset, "label": b.label,
                     "pattern": bd["pattern"], "refuted": bool(valid), "counterexamples": valid})
    res = {"n_bindings": len(rows), "n_refuted": sum(r["refuted"] for r in rows),
           "gap_report": gaps.most_common(), "items": rows}
    dump_yaml(res, ctx.outp("m3", "refutation.yaml"))
    return res


# ------------------------------------------------------------------- B-K1
def run_direct_baseline(ctx: Ctx, pairs: list[tuple[str, str]]) -> dict:
    """Prior-work style: one free-form generation of constraints per pair."""
    vocab = ctx.vocab
    eidx = effect_index(ctx.effects())
    aidx = {a["id"]: a for a in ctx.asset_list}
    out_b, invalid = [], 0
    for e, v in pairs:
        sys, user, schema = direct_constraints_prompt(vocab.describe_vocab(), eidx[e], aidx[v])
        out = ctx.llm.json("BASE", "direct_constraints", sys, user, schema, context={"effect": e, "asset": v})
        for it in out.get("items", []):
            try:
                phi = normalize(it.get("pattern") or {}, vocab)
            except ValueError:
                invalid += 1
                continue
            if it.get("label") in ("+", "-"):
                out_b.append(Binding(phi, e, v, it["label"], it.get("harm", "")).to_json(vocab))
    res = {"bindings": out_b, "invalid_patterns": invalid}
    dump_yaml(res, ctx.outp("baselines", "spec_direct.yaml"))
    return res
