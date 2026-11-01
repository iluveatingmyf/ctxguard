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

import numpy as np

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


def run_m3(ctx: Ctx, limit_pairs: int | None = None, only_effects: list | None = None,
           only_stale: bool = False, spec_out: str | None = None, merge: bool = False) -> dict:
    """Build K. only_stale: rebuild only the pairs listed by vocab-migrate in
    m3/stale_pairs.yaml and keep the migrated bindings of every other pair.
    spec_out: write to another file (e.g. a from-scratch K for k-compare)."""
    from .manifest import record, save_yaml
    lib = ctx.effects()
    effects = [e for e in lib["effects"] if not only_effects or e["id"] in only_effects]
    pairs = [(e, v) for e in effects for v in ctx.asset_list]
    stale = set()
    if only_stale:
        stale = {s["pair"] for s in load_yaml(ctx.outp("m3", "stale_pairs.yaml"))["stale_pairs"]}
        pairs = [(e, v) for e, v in pairs if f"{e['id']}|{v['id']}" in stale]
    if limit_pairs:
        pairs = pairs[:limit_pairs]
    all_b, per_pair = [], {}
    calls = Counter()
    for i, (e, v) in enumerate(pairs):
        r = PairRun(ctx, e, v).run()
        all_b.extend(r["bindings"])
        per_pair[f"{e['id']}|{v['id']}"] = {"stats": r["stats"], "leaves": r["leaves"]}
        for k in ("relevance_calls", "region_calls", "point_calls"):
            calls[k] += r["stats"].get(k, 0)
        print(f"[M3] {i + 1}/{len(pairs)} {e['id']} x {v['id']}: "
              f"{len(r['bindings'])} bindings, {r['stats'].get('region_calls', 0)} region + "
              f"{r['stats'].get('point_calls', 0)} point calls")
    out_b = [b.to_json(ctx.vocab) for b in all_b]
    spec = {"vocab_version": ctx.vocab.version, "effects_version": lib["version"]}
    trace_path = ctx.outp("m3", "pairs_trace.yaml")
    if only_stale or merge:
        old = load_yaml(ctx.spec_path())
        done = {f"{e['id']}|{v['id']}" for e, v in pairs}
        if old.get("vocab_version") != ctx.vocab.version:
            raise SystemExit(f"existing K uses {old.get('vocab_version')}, config uses {ctx.vocab.version}: run vocab-migrate first")
        kept = [b for b in old["bindings"] if f"{b['effect']}|{b['asset']}" not in done]
        spec.update({"migrated_from": old.get("migrated_from"), "rebuilt_pairs": len(pairs)})
        out_b = kept + out_b
        try:
            tr = load_yaml(trace_path) or {}
        except FileNotFoundError:
            tr = {}
        tr.update(per_pair)
        per_pair = tr
        save_yaml(ctx, "m3 incremental", {"rebuilt_pairs": len(pairs), "kept_bindings": len(kept),
                                           "new_bindings": len(all_b), "llm_calls": dict(calls)},
                  "m3", "only_stale_stats.yaml")
    spec["bindings"] = out_b
    spec["llm_calls"] = dict(calls)
    path = spec_out or ctx.spec_path()
    dump_yaml(spec, path)
    record(ctx, "m3", path)
    if not spec_out:
        dump_yaml(per_pair, trace_path)
        record(ctx, "m3", trace_path)
    return spec


# ------------------------------------------------------------------- O5 (CEGAR)
def _claims(ctx: Ctx, spec: dict, zero_max: int) -> list[dict]:
    """Claims to refute: every '+'/'-' binding AND the closed-world '0' region of every
    (effect, asset) pair, written as a prime cover of its 0-cells (at most zero_max largest
    cubes per pair; the whole Σ when a pair has no binding at all)."""
    from .minimize import minimize
    from .roughset import Tables
    vocab = ctx.vocab
    bs = [Binding.from_json(b, vocab) for b in spec["bindings"]]
    claims = [{"kind": "binding", "effect": b.effect, "asset": b.asset, "label": b.label, "phi": b.phi, "harm": b.harm}
              for b in bs]
    pairs = [(e["id"], a["id"]) for e in ctx.effects()["effects"] for a in ctx.asset_list]
    T = Tables(vocab, bs, pairs)
    doms = [f.values for f in vocab.factors]
    for pi, (e, a) in enumerate(T.pairs):
        zero = T.L[pi] == 0
        if not zero.any():
            continue
        if zero.all():
            cubes = [{}]
        else:
            on = [T.sigma[i] for i in np.flatnonzero(zero)]
            off = [T.sigma[i] for i in np.flatnonzero(~zero)]
            cubes = [cube_to_pattern(c, vocab.names, doms) for c in minimize(on, off, doms)]
            cubes.sort(key=lambda phi: -int(T.mask(phi).sum()))
            cubes = cubes[:zero_max]
        for phi in cubes:
            claims.append({"kind": "zero", "effect": e, "asset": a, "label": "0", "phi": phi,
                           "harm": "no relevant influence (closed-world default)"})
    return claims


def run_refute(ctx: Ctx, zero_max: int | None = None) -> dict:
    """CEGAR-style refutation (thesis Sec. 2.5.4). An abstract cell s of Σ merges many real
    situations. For each claim 'label is l everywhere in [phi]' a refuter proposes real
    situations inside phi that deserve another label because of an attribute outside F.
    A proposal is VALID if its factor assignment really lies in phi (and in Σ); it is
    CONFIRMED (not spurious) if an independent judge, who sees only the narrative, gives the
    expected label (different from l). Confirmed counterexamples = the abstraction is too
    coarse; m2-refine turns them into evidence for the next vocabulary version."""
    from .manifest import save_yaml
    from .prompts import judge_situation_prompt
    vocab = ctx.vocab
    spec = load_yaml(ctx.spec_path())
    eidx = effect_index(ctx.effects())
    aidx = {a["id"]: a for a in ctx.asset_list}
    k = ctx.cfg["method"]["refute_max_counterexamples"]
    zm = zero_max if zero_max is not None else ctx.cfg["method"].get("refute_zero_max_per_pair", 4)
    rows, gaps = [], Counter()
    for ci, c in enumerate(_claims(ctx, spec, zm)):
        pj = pattern_to_json(c["phi"], vocab)
        sys, user, schema = refute_prompt(vocab.describe_vocab(), eidx[c["effect"]], aidx[c["asset"]],
                                          vocab.describe_pattern(c["phi"]), c["label"], c["harm"], k)
        out = ctx.llm.json("M3", "refute", sys, user, schema,
                           context={"effect": c["effect"], "asset": c["asset"], "phi": pj, "label": c["label"]})
        ces = []
        for ce in out.get("counterexamples", [])[:k]:
            a = {n: str(v) for n, v in (ce.get("assignment") or {}).items()}
            ok = (set(a) == set(vocab.names) and all(a[n] in vocab.domain(n) for n in vocab.names))
            if ok:
                cell = tuple(a[n] for n in vocab.names)
                ok = vocab.feasible(cell) and contains(c["phi"], cell, vocab) and ce.get("expected_label") != c["label"]
            row = {**ce, "valid": bool(ok)}
            if ok:
                js, ju, jsch = judge_situation_prompt(eidx[c["effect"]], aidx[c["asset"]], ce.get("narrative", ""))
                j = ctx.llm.json("M3", "judge", js, ju, jsch,
                                 context={"effect": c["effect"], "asset": c["asset"], "narrative": ce.get("narrative", "")})
                row["judge_label"] = j.get("label")
                row["confirmed"] = j.get("label") == ce.get("expected_label")
                if row["confirmed"]:
                    gaps[str(ce.get("missing_attribute", "")).strip().lower()] += 1
            ces.append(row)
        rows.append({"claim": ci, "kind": c["kind"], "effect": c["effect"], "asset": c["asset"], "label": c["label"],
                     "pattern": pj, "refuted": any(x["valid"] for x in ces),
                     "confirmed": any(x.get("confirmed") for x in ces), "counterexamples": ces})
    by = lambda kind, key: sum(1 for r in rows if r["kind"] == kind and (key is None or r[key]))
    res = {"n_claims": len(rows), "n_bindings": by("binding", None), "n_zero_claims": by("zero", None),
           "n_refuted": sum(r["refuted"] for r in rows), "n_confirmed": sum(r["confirmed"] for r in rows),
           "confirmed_binding_claims": by("binding", "confirmed"), "confirmed_zero_claims": by("zero", "confirmed"),
           "valid_counterexamples": sum(x["valid"] for r in rows for x in r["counterexamples"]),
           "spurious_counterexamples": sum(x["valid"] and not x.get("confirmed") for r in rows for x in r["counterexamples"]),
           "gap_report": gaps.most_common(), "items": rows}
    save_yaml(ctx, "m3-refute", res, "m3", "refutation.yaml")
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
