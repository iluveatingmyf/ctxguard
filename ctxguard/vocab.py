"""M2  Context-factor vocabulary: construction, audit, versioned extension  (thesis Ch. 4, Sec. 4.2)

The factors are NOT fixed a priori. They are the output of the stages below and become
fixed only when a reviewed candidate is frozen as a version (F-v1, F-v2, ...). Every stage
writes its full intermediate result (out/m2/<stage>_*.yaml|csv, listed in out/MANIFEST.csv),
every LLM answer is cached (out/llm_cache.sqlite) and every call is in out/ledger.csv.

Stage A  dimension discovery from the property corpus (asset-agnostic)
  A1 m2-extract      property -> atomic situation conditions; per-property asset metadata
  A2 m2-dims         iterative open coding: conditions -> dimensions (+ excluded categories);
                     --runs R repeats it on permuted condition orders (stability)
  A3 m2-values       per dimension: values + assignment of every coded condition to a value
  A4 (human)         review with rules R1-R4 -> data/vocab/factors_seed.yaml (provenance per
                     factor); m2-seed-check validates it and reports the edits vs A3
Stage B  effect-anchored elicitation (asset-aware), bounded by E x V
  m2-elicit          for each (effect, asset): attributes that flip the label, filed under an
                     existing dimension or NEW:<name>
Stage C  label-signature consolidation (rough-set value granulation, marginal)
  m2-consolidate     canonical states -> signature over the dimension's evidence pairs ->
                     states with identical signatures merged; existing values never merged
                     here; outputs a candidate with value provenance and a `refines` map
Stage D  freeze, bind, audit, refine (CEGAR loop)
  m2-freeze          validate + freeze a reviewed candidate as F-<name> (versioning.py)
  m3                 bindings over the frozen Σ (prime-implicant covers, minimize.py)
  m2-audit           rough-set audit of K: core / reduct / value partition / prime check;
                     one dimension card per factor (evidence + verdict)
  m3-refute          counterexample search on '+', '-' AND '0' regions, independent judge
  m2-refine          confirmed counterexamples -> new evidence -> m2-consolidate --base current
  vocab-migrate, m3 --only-stale   carry K to the next version, rebuild only stale pairs
Check sheets (annotate/m2_*_CHECK.csv) + m2-check-score give the human validation numbers.
Ablations kept: m2-slots (top-down slots), m2-cluster + m2-name (embedding clustering),
m2-expand (factor-anchored expansion).
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import os
import random
from collections import Counter, defaultdict

import numpy as np

from .common import Ctx, dump_yaml, load_yaml, p
from .manifest import record, save_csv, save_yaml
from .model import Binding, Vocab, contains, index_spec, label_of, normalize, pattern_to_json
from .prompts import (attribute_dimension_prompt, canonicalize_prompt, classify_condition_prompt,
                      contrast_label_prompt, elicit_prompt, expand_values_prompt, extract_predicates_prompt,
                      induce_dims_prompt, map_property_prompt, name_factor_prompt, property_assets_prompt,
                      slot_values_prompt)
from .roughset import INV, Tables


# ------------------------------------------------------------------ helpers
def _norm(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(s).lower()).strip("_")


def _as_dict(vals) -> dict:
    return {str(k): (v or "") for k, v in vals.items()} if isinstance(vals, dict) else {str(v): str(v) for v in vals}


def _base(ctx: Ctx, base: str = "seed") -> tuple[dict, str]:
    """Vocabulary a round of stages B/C starts from: 'seed' (A4 output, first build),
    'current' (the frozen version in config.yaml, later rounds) or a file path."""
    if base == "seed":
        path = p("data/vocab/factors_seed.yaml")
        if not os.path.exists(path):
            path = ctx.outp("m2", "A3_factors_corpus.yaml")
    elif base == "current":
        path = p(ctx.cfg["paths"]["factors"])
    else:
        path = base
    d = load_yaml(path)
    for f in d["factors"]:
        f["values"] = _as_dict(f["values"])
    return d, path


def _sample(rows: list, k: int, key, seed: str) -> list:
    """Deterministic stratified sample (round-robin over strata)."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    for v in by.values():
        rng.shuffle(v)
    out, strata = [], sorted(by)
    while len(out) < k and any(by[s] for s in strata):
        for s in strata:
            if by[s] and len(out) < k:
                out.append(by[s].pop())
    return out


def _kappa(a: list, b: list) -> float | None:
    if not a:
        return None
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def _ari(x: list, y: list) -> float:
    """Adjusted Rand index of two labelings of the same items."""
    from math import comb
    n = len(x)
    cont = Counter(zip(x, y))
    a, b = Counter(x), Counter(y)
    s_ij = sum(comb(v, 2) for v in cont.values())
    s_a = sum(comb(v, 2) for v in a.values())
    s_b = sum(comb(v, 2) for v in b.values())
    exp = s_a * s_b / comb(n, 2) if n > 1 else 0
    mx = (s_a + s_b) / 2
    return round((s_ij - exp) / (mx - exp), 4) if mx != exp else 1.0


# ------------------------------------------------------------------ corpus
def load_properties(ctx: Ctx) -> list[dict]:
    path = os.environ.get("CTXGUARD_PROPERTIES") or p("data/properties/seed_properties.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    rows = [r for r in rows if r.get("scope", "in") == "in"]      # out-of-scope rows are reported, not used
    ratio = ctx.cfg["method"]["property_test_ratio"]
    for r in rows:   # deterministic split by hash of id
        h = int(hashlib.md5(f"{ctx.seed}:{r['id']}".encode()).hexdigest(), 16) % 1000
        r["split"] = "test" if h < ratio * 1000 else "train"
    return rows


# ------------------------------------------------------------------ A1
def run_extract(ctx: Ctx) -> dict:
    """A1: every TRAIN property -> atomic, device-agnostic situation conditions.
    The asset(s) a property protects are recorded as metadata only (coverage report);
    the discovery of dimensions does not filter by asset."""
    props = load_properties(ctx)
    rows, pmeta, empty = [], [], []
    for prop in props:
        if prop["split"] != "train":
            continue
        sys, user, schema = extract_predicates_prompt(prop["text"])
        out = ctx.llm.json("M2", "extract_predicates", sys, user, schema, context={"property": prop})
        sys, user, schema = property_assets_prompt(prop["text"])
        assets = ctx.llm.json("M2", "property_assets", sys, user, schema, context={"property": prop}).get("assets", [])
        conds = [c.strip() for c in out.get("conditions", []) if c and c.strip()]
        pmeta.append({"property_id": prop["id"], "source": prop.get("source", ""), "assets": assets, "n_conditions": len(conds)})
        if not conds:
            empty.append(prop["id"])
        for c in conds:
            rows.append({"property_id": prop["id"], "condition": c, "assets": assets, "property_text": prop["text"]})
    save_yaml(ctx, "m2-extract", rows, "m2", "A1_conditions.yaml")
    per_asset = {a["id"]: {"properties": sum(a["id"] in m["assets"] for m in pmeta),
                           "conditions": sum(a["id"] in r["assets"] for r in rows)} for a in ctx.asset_list}
    rep = {"properties_in_scope": len(props), "train": len(pmeta), "test_heldout": len(props) - len(pmeta),
           "conditions": len(rows), "distinct_conditions": len({r["condition"] for r in rows}),
           "properties_without_condition": empty, "coverage_per_asset": per_asset, "properties": pmeta}
    save_yaml(ctx, "m2-extract", rep, "m2", "A1_report.yaml")
    sample = _sample(rows, 50, lambda r: r["property_id"][:1], "A1")
    save_csv(ctx, "m2-extract", ["property_id", "property_text", "condition",
                                 "HUMAN_faithful(1/0)", "HUMAN_world_state(1/0)", "note"],
             [[r["property_id"], r["property_text"], r["condition"], "", "", ""] for r in sample],
             "annotate", "m2_A1_extraction_CHECK.csv")
    return rep


def _cond_props(ctx: Ctx) -> dict:
    m = defaultdict(set)
    for r in load_yaml(ctx.outp("m2", "A1_conditions.yaml")):
        m[r["condition"]].add(r["property_id"])
    return m


def _cond_texts(ctx: Ctx) -> list[str]:
    return sorted({c["condition"] for c in load_yaml(ctx.outp("m2", "A1_conditions.yaml"))})


# ------------------------------------------------------------------ ablation: embedding clustering
def _silhouette(D: np.ndarray, labels: np.ndarray) -> float:
    ids = np.unique(labels)
    if len(ids) < 2 or len(ids) >= len(labels):
        return -1.0
    s = np.zeros(len(labels))
    for i in range(len(labels)):
        same = labels == labels[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, labels == j].mean() for j in ids if j != labels[i])
        s[i] = 0.0 if not same.any() else (b - a) / max(a, b, 1e-9)
    return float(s.mean())


def run_cluster(ctx: Ctx) -> dict:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist, squareform
    conds = load_yaml(ctx.outp("m2", "A1_conditions.yaml"))
    texts = sorted({c["condition"] for c in conds})
    X = np.array(ctx.llm.embed("M2", texts), dtype=float)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    d = pdist(X, metric="cosine")
    Z = linkage(d, method="average")
    D = squareform(d)
    grid = [round(float(x), 2) for x in np.arange(0.20, 0.81, 0.05)]
    scores = []
    for t in grid:
        lab = fcluster(Z, t=t, criterion="distance")
        scores.append((float(_silhouette(D, lab)), t, int(len(set(lab)))))
    best = max(scores)
    lab = fcluster(Z, t=best[1], criterion="distance")
    clusters = defaultdict(list)
    for t_, l in zip(texts, lab):
        clusters[int(l)].append(t_)
    res = {"cut": float(best[1]), "silhouette": round(float(best[0]), 4), "n_clusters": len(clusters),
           "grid": [{"cut": t, "silhouette": round(s, 4), "k": k} for s, t, k in scores],
           "clusters": [{"id": k, "conditions": v} for k, v in sorted(clusters.items(), key=lambda kv: -len(kv[1]))]}
    save_yaml(ctx, "m2-cluster", res, "m2", "ablation_clusters.yaml")
    return res


def run_name(ctx: Ctx, min_size: int = 2) -> dict:
    cl = load_yaml(ctx.outp("m2", "ablation_clusters.yaml"))
    factors, small = [], []
    for c in cl["clusters"]:
        if len(c["conditions"]) < min_size:
            small.extend(c["conditions"])
            continue
        sys, user, schema = name_factor_prompt(c["conditions"][:40])
        out = ctx.llm.json("M2", "name_factor", sys, user, schema, context={"conditions": c["conditions"]})
        factors.append({"name": out["name"], "scope": out.get("scope", "global"),
                        "semantics": out.get("semantics", ""),
                        "values": {v["value"]: v["semantics"] for v in out.get("values", [])},
                        "provenance": {"cluster": c["id"], "n_conditions": len(c["conditions"])}})
    res = {"version": "F-ablation-clustering", "status": "ABLATION ONLY (bottom-up clustering); not the vocabulary",
           "factors": factors, "feasibility": [], "unclustered_conditions": small}
    save_yaml(ctx, "m2-name", res, "m2", "ablation_clustering_factors.yaml")
    return res




# ------------------------------------------------------------------ A2
def _induce(ctx: Ctx, texts: list[str], excluded: list[dict], max_rounds: int, stop_uncovered: float):
    """One open-coding run. Round r: the LLM proposes dimensions from the pool; every
    condition is then coded into one dimension or an excluded category or 'other';
    'other' conditions form the next pool. Stops when the uncovered share <= stop_uncovered,
    when no new dimension appears, or after max_rounds."""
    ex_ids = {e["id"] for e in excluded}
    dims, pool, rounds, coding, proposals = {}, texts, [], [], []
    for r in range(max_rounds):
        sys, user, sch = induce_dims_prompt(pool, list(dims.values()))
        out = ctx.llm.json("M2", "induce_dims", sys, user, sch, context={"conditions": pool, "existing": list(dims)})
        new = 0
        for d in out.get("dimensions", []):
            did = _norm(d.get("id", ""))
            prop = {"round": r + 1, "proposed_id": d.get("id"), "desc": d.get("desc", ""),
                    "covers": [pool[i] for i in d.get("covers", []) if isinstance(i, int) and 0 <= i < len(pool)]}
            if not did or did in dims or did in ex_ids:
                prop["decision"] = "rejected: empty, duplicate or excluded category"
            else:
                dims[did] = {"id": did, "kind": "situation", "scope": d.get("scope", "global"), "desc": d.get("desc", ""),
                             "max_factors": max(1, min(4, int(d.get("n_variables") or 1))), "round": r + 1}
                prop["decision"] = "accepted"
                new += 1
            proposals.append(prop)
        sdesc = "\n".join(f"- {s['id']} ({s['kind']}): {s['desc']}" for s in list(dims.values()) + excluded)
        coding = []
        for t in texts:
            sys, user, sch = classify_condition_prompt(t, sdesc)
            o = ctx.llm.json("M2", "classify_condition", sys, user, sch, context={"condition": t, "dims": list(dims)})
            slot = o.get("slot") if o.get("slot") in dims or o.get("slot") in ex_ids else "other"
            coding.append({"condition": t, "slot": slot, "statement": o.get("situation_statement") or t,
                           "reason": o.get("reason", ""), "round": r + 1})
        uncovered = [c["condition"] for c in coding if c["slot"] == "other"]
        rounds.append({"round": r + 1, "new_dimensions": new, "n_dimensions": len(dims),
                       "uncovered": len(uncovered), "uncovered_share": round(len(uncovered) / max(1, len(texts)), 3)})
        if not new or len(uncovered) / max(1, len(texts)) <= stop_uncovered:
            break
        pool = uncovered
    per = Counter(c["slot"] for c in coding)
    for d in dims.values():
        d["n_conditions"] = per.get(d["id"], 0)
    return dims, rounds, coding, proposals, per


def run_dims(ctx: Ctx, runs: int = 1, max_rounds: int = 3, stop_uncovered: float = 0.05) -> dict:
    """A2: dimensions induced from the corpus. Run 0 uses the sorted condition order and is
    THE result; runs 1..R-1 use seeded permutations of the order (the prompt changes, so the
    LLM answer is re-drawn) and measure stability: ARI of the codings vs run 0 and, per
    dimension of run 0, its recurrence = share of runs with a dimension whose coded
    conditions overlap it with Jaccard >= 0.5."""
    ref = load_yaml(p("data/vocab/slots.yaml"))
    excluded = [s for s in ref["slots"] if s["kind"] == "excluded"]
    texts = _cond_texts(ctx)
    cp = _cond_props(ctx)
    dims, rounds, coding, proposals, per = _induce(ctx, texts, excluded, max_rounds, stop_uncovered)
    for d in dims.values():
        d["property_ids"] = sorted({pid for c in coding if c["slot"] == d["id"] for pid in cp.get(c["condition"], ())})
    kept = [d for d in dims.values() if d["n_conditions"] > 0]
    res = {"version": "S-induced", "rounds": rounds, "slots": kept + excluded, "derived": ref.get("derived", []),
           "coverage": dict(per), "dropped_empty_dimensions": [d["id"] for d in dims.values() if d["n_conditions"] == 0]}
    save_yaml(ctx, "m2-dims", res, "m2", "A2_dims.yaml")
    save_yaml(ctx, "m2-dims", coding, "m2", "A2_coding.yaml")
    save_yaml(ctx, "m2-dims", proposals, "m2", "A2_proposals.yaml")
    dlookup = {d["id"]: d["desc"] for d in kept + excluded}
    sample = _sample(coding, 50, lambda r: r["slot"], "A2")
    save_csv(ctx, "m2-dims", ["condition", "assigned_dimension", "dimension_definition", "HUMAN_coding_ok(1/0)",
                              "HUMAN_better_dimension", "note"],
             [[c["condition"], c["slot"], dlookup.get(c["slot"], "(no situation dimension fits)"), "", "", ""] for c in sample],
             "annotate", "m2_A2_coding_CHECK.csv")
    if runs > 1:
        base = {c["condition"]: c["slot"] for c in coding}
        sets0 = {d["id"]: {c["condition"] for c in coding if c["slot"] == d["id"]} for d in kept}
        stab = {"runs": runs, "per_run": [], "dimension_recurrence": {}}
        hits = defaultdict(list)
        for i in range(1, runs):
            perm = texts[:]
            random.Random(f"{ctx.seed}:A2:{i}").shuffle(perm)
            d_i, r_i, c_i, p_i, _ = _induce(ctx, perm, excluded, max_rounds, stop_uncovered)
            save_yaml(ctx, "m2-dims", {"dims": list(d_i.values()), "rounds": r_i, "proposals": p_i},
                      "m2", "A2_runs", f"run{i}_dims.yaml")
            save_yaml(ctx, "m2-dims", c_i, "m2", "A2_runs", f"run{i}_coding.yaml")
            lab_i = {c["condition"]: c["slot"] for c in c_i}
            sets_i = defaultdict(set)
            for c in c_i:
                if c["slot"] in d_i:
                    sets_i[c["slot"]].add(c["condition"])
            stab["per_run"].append({"run": i, "n_dimensions": len([d for d in d_i.values() if d["n_conditions"]]),
                                    "ari_vs_run0": _ari([base[t] for t in texts], [lab_i[t] for t in texts])})
            for did, s0 in sets0.items():
                best = max(((len(s0 & s) / max(1, len(s0 | s)), k) for k, s in sets_i.items()), default=(0.0, None))
                hits[did].append({"run": i, "best_match": best[1], "jaccard": round(best[0], 3)})
        for did, h in hits.items():
            stab["dimension_recurrence"][did] = {"recurrence": round(sum(x["jaccard"] >= 0.5 for x in h) / len(h), 3),
                                                 "matches": h}
        stab["mean_ari"] = round(float(np.mean([x["ari_vs_run0"] for x in stab["per_run"]])), 4)
        save_yaml(ctx, "m2-dims", stab, "m2", "A2_stability.yaml")
        res["stability"] = {"mean_ari": stab["mean_ari"],
                            "recurrence": {k: v["recurrence"] for k, v in stab["dimension_recurrence"].items()}}
    return res


# ------------------------------------------------------------------ A3 (and the top-down ablation)
def run_slots(ctx: Ctx, schema_path: str | None = None, tag: str = "ablation_topdown") -> dict:
    """A3 (tag='A3', schema = A2_dims.yaml): per dimension, induce the factor(s) and their
    values from the restated conditions and assign every condition to factor=value.
    Ablation (tag='ablation_topdown', schema = data/vocab/slots.yaml): same procedure with
    dimensions fixed top-down."""
    schema = load_yaml(schema_path or p("data/vocab/slots.yaml"))
    slots = {s["id"]: s for s in schema["slots"]}
    sdesc = "\n".join(f"- {s['id']} ({s['kind']}): {s['desc']}" for s in schema["slots"])
    texts = _cond_texts(ctx)
    cp = _cond_props(ctx)
    if tag == "A3" and os.path.exists(ctx.outp("m2", "A2_coding.yaml")):
        placed = load_yaml(ctx.outp("m2", "A2_coding.yaml"))          # reuse the A2 coding (no re-classification)
    else:
        placed = []
        for t in texts:
            sys, user, sch = classify_condition_prompt(t, sdesc)
            out = ctx.llm.json("M2", "classify_condition", sys, user, sch, context={"condition": t})
            slot = out.get("slot") if out.get("slot") in slots else "other"
            placed.append({"condition": t, "slot": slot, "statement": out.get("situation_statement") or t})
    by = defaultdict(list)
    for r in placed:
        by[r["slot"]].append(r)
    factors, assign = [], []
    report = {"n_conditions": len(texts), "per_slot": {}}
    for sid, s in slots.items():
        rows = by.get(sid, [])
        entry = {"kind": s["kind"], "n_conditions": len(rows), "examples": [r["condition"] for r in rows[:6]]}
        if s["kind"] == "situation" and rows:
            stmts = sorted({r["statement"] for r in rows})
            src = defaultdict(set)
            for r in rows:
                src[r["statement"]].add(r["condition"])
            sys, user, sch = slot_values_prompt(s, stmts, s.get("max_factors", 1))
            out = ctx.llm.json("M2", "slot_values", sys, user, sch, context={"slot": s, "statements": stmts})
            fs = out.get("factors", [])[: s.get("max_factors", 1)]
            names = {f["name"] for f in fs}
            vcount = Counter()
            for a in out.get("assignment", []):
                i = a.get("statement")
                if not isinstance(i, int) or not 0 <= i < len(stmts):
                    continue
                ok = a.get("factor") in names and a.get("value")
                conds = sorted(src[stmts[i]])
                assign.append({"dimension": sid, "statement": stmts[i], "factor": a.get("factor") if ok else None,
                               "value": a.get("value") if ok else None, "conditions": conds,
                               "property_ids": sorted({x for c in conds for x in cp.get(c, ())})})
                if ok:
                    vcount[(a["factor"], str(a["value"]))] += 1
            covered = sum(vcount.values())
            entry.update({"n_statements": len(stmts), "covered_statements": covered,
                          "coverage": round(covered / max(1, len(stmts)), 3), "factors": [f["name"] for f in fs]})
            for f in fs:
                vals = {str(x["value"]): x["semantics"] for x in f.get("values", [])}
                factors.append({"name": f["name"], "scope": s.get("scope", "global"), "semantics": f.get("semantics", ""),
                                "values": vals,
                                "provenance": {"slot": sid, "n_conditions": len(rows), "dey": s.get("dey"),
                                               "statements_per_value": {v: vcount[(f["name"], v)] for v in vals}}})
        elif s["kind"] == "situation":
            entry["note"] = "no corpus evidence; dimension kept for review"
        report["per_slot"][sid] = entry
    for d in schema.get("derived", []):
        factors.append({"name": d["name"], "scope": d["scope"], "semantics": d["semantics"], "values": d["values"],
                        "provenance": {"derived": "action target area"}})
    n_ex = sum(len(by.get(sid, [])) for sid, s in slots.items() if s["kind"] == "excluded")
    report["excluded_share"] = round(n_ex / max(1, len(texts)), 3)
    report["excluded_per_category"] = {sid: len(by.get(sid, [])) for sid, s in slots.items() if s["kind"] == "excluded"}
    res = {"version": f"F-{tag}", "schema": schema.get("version"), "factors": factors, "feasibility": []}
    if tag == "A3":
        save_yaml(ctx, "m2-values", res, "m2", "A3_factors_corpus.yaml")
        save_yaml(ctx, "m2-values", report, "m2", "A3_values_report.yaml")
        save_yaml(ctx, "m2-values", assign, "m2", "A3_value_assignment.yaml")
    else:
        save_yaml(ctx, "m2-slots", res, "m2", f"{tag}_factors.yaml")
        save_yaml(ctx, "m2-slots", report, "m2", f"{tag}_values_report.yaml")
        save_yaml(ctx, "m2-slots", placed, "m2", f"{tag}_condition_coding.yaml")
    return report


# ------------------------------------------------------------------ A4
def run_seed_check(ctx: Ctx) -> dict:
    """A4: machine checks of the human-reviewed seed (data/vocab/factors_seed.yaml) and the
    record of what the review changed relative to the corpus output A3."""
    seed, path = _base(ctx, "seed")
    a3 = load_yaml(ctx.outp("m2", "A3_factors_corpus.yaml"))
    a2 = {d["id"] for d in load_yaml(ctx.outp("m2", "A2_dims.yaml"))["slots"]}
    issues, rows = [], []
    for f in seed["factors"]:
        prov = f.get("provenance") or {}
        if f.get("scope") != "action" and not prov:
            issues.append(f"{f['name']}: no provenance")
        src = prov.get("dimension") or prov.get("slot") or []
        src = [src] if isinstance(src, str) else list(src)
        known = [d for d in src if d in a2]
        if f.get("scope") != "action" and src and not known:
            issues.append(f"{f['name']}: provenance dimension(s) {src} not among the A2 dimensions")
        corp = [c for c in a3["factors"] if (c.get("provenance") or {}).get("slot") in known]
        cvals = {v for c in corp for v in c["values"]}
        rows.append({"factor": f["name"], "from_dimensions": known, "corpus_factors": [c["name"] for c in corp],
                     "values": list(f["values"]), "values_also_in_corpus": sorted(set(f["values"]) & cvals),
                     "values_new_or_renamed": sorted(set(f["values"]) - cvals),
                     "corpus_values_not_kept": sorted(cvals - set(f["values"])),
                     "removed_with_reason": prov.get("removed", [])})
    used = {d for r in rows for d in r["from_dimensions"]}
    try:
        V = Vocab.load(path)
        sigma = len(V.sigma())
    except Exception as e:  # noqa: BLE001
        issues.append(f"does not load: {e}")
        sigma = 0
    res = {"seed": os.path.relpath(path, p()), "n_factors_corpus": len(a3["factors"]), "n_factors_seed": len(seed["factors"]),
           "a2_dimensions_not_used": sorted(a2 - used - {"device_state", "user_configuration", "other"}),
           "sigma": sigma, "issues": issues, "per_factor": rows}
    save_yaml(ctx, "m2-seed-check", res, "m2", "A4_seed_check.yaml")
    return res


# ------------------------------------------------------------------ ablation: factor-anchored expansion
def run_expand(ctx: Ctx, max_new: int = 4) -> dict:
    """M2 step 4: agent expansion of value sets beyond the corpus.
    For every factor the agent proposes up to max_new missing values, each with a
    witness (effect e, asset v, closest existing value). A value is ACCEPTED only if
    two independent contrast queries give different labels for (e, v) under the new
    value and under the closest existing value, i.e. the value changes some safety
    judgment (the same necessity criterion used for the vocabulary metrics)."""
    base = _base(ctx, "seed")[0]
    lib = ctx.effects()
    eidx = {e["id"]: e for e in lib["effects"]}
    aidx = {a["id"]: a for a in ctx.asset_list}
    edesc = "\n".join(f"- {e['id']}: {e['desc']}" for e in lib["effects"])
    report, factors = [], []
    for f in base["factors"]:
        f = dict(f)
        f["values"] = dict(f["values"])
        f.setdefault("value_source", {k: "corpus" for k in f["values"]})
        if f.get("scope") == "action":
            factors.append(f)
            continue
        sys, user, sch = expand_values_prompt(f, edesc, list(aidx), max_new)
        out = ctx.llm.json("M2", "expand_values", sys, user, sch, context={"factor": f})
        for c in out.get("candidates", [])[:max_new]:
            val = "".join(ch if ch.isalnum() else "_" for ch in str(c.get("value", "")).lower()).strip("_")
            row = {"factor": f["name"], "value": val, "witness": {k: c.get(k) for k in ("effect", "asset", "closest_existing", "why")}}
            if not val or val in f["values"]:
                row["decision"] = "rejected: empty or duplicate"
            elif c.get("closest_existing") not in f["values"] or c.get("effect") not in eidx or c.get("asset") not in aidx:
                row["decision"] = "rejected: invalid witness"
            else:
                e, a, old = eidx[c["effect"]], aidx[c["asset"]], c["closest_existing"]
                l_new = ctx.llm.json("M2", "contrast_label", *contrast_label_prompt(e, a, f, val, c.get("semantics", "")),
                                     context={"effect": e["id"], "asset": a["id"], "factor": f["name"], "value": val}).get("label")
                l_old = ctx.llm.json("M2", "contrast_label", *contrast_label_prompt(e, a, f, old, f["values"][old]),
                                     context={"effect": e["id"], "asset": a["id"], "factor": f["name"], "value": old}).get("label")
                row["labels"] = {"new": l_new, "closest_existing": l_old}
                if l_new != l_old:
                    row["decision"] = "accepted"
                    f["values"][val] = c.get("semantics", "")
                    f["value_source"][val] = f"agent (witness {e['id']} x {a['id']}: {l_old} -> {l_new}, refines {old})"
                else:
                    row["decision"] = "rejected: no label change"
            report.append(row)
        factors.append(f)
    res = {"version": "F-ablation-expand", "status": "ABLATION ONLY (factor-anchored expansion)",
           "factors": factors, "feasibility": []}
    save_yaml(ctx, "m2-expand", res, "m2", "ablation_expand_factors.yaml")
    summ = {"proposed": len(report), "accepted": sum(r["decision"] == "accepted" for r in report), "items": report}
    save_yaml(ctx, "m2-expand", summ, "m2", "ablation_expansion_report.yaml")
    return summ




# ------------------------------------------------------------------ B
def _dims_desc(dims: list[dict]) -> str:
    return "\n".join(f"- {f['name']}: {f.get('semantics', '')}; values: {', '.join(f['values'])}" for f in dims)


def _evidence(ctx: Ctx) -> dict:
    path = ctx.outp("m2", "B1_elicitation.yaml")
    return load_yaml(path) if os.path.exists(path) else {"items": [], "rejected": [], "rounds": []}


def run_elicit(ctx: Ctx, max_items: int = 6, base: str = "seed", effects: list | None = None) -> dict:
    """Stage B: for every effect e and asset v (finite E x V; or only the given effects in an
    incremental round) the agent recalls the situation attributes that make e unacceptable
    ('-') or clearly beneficial ('+') for v and files each under an existing dimension or a
    NEW one. Items of re-queried pairs replace the old ones; other items are kept."""
    B, bpath = _base(ctx, base)
    lib = ctx.effects()
    dims = [f for f in B["factors"] if f.get("scope") != "action"]
    names = {f["name"] for f in dims}
    ddesc = _dims_desc(dims)
    E = [e for e in lib["effects"] if not effects or e["id"] in effects]
    queried = {(e["id"], a["id"]) for e in E for a in ctx.asset_list}
    ev = _evidence(ctx)
    keep = [i for i in ev["items"] if i.get("source") == "refutation" or (i["effect"], i["asset"]) not in queried]
    rej = [i for i in ev.get("rejected", []) if (i["effect"], i["asset"]) not in queried]
    rnd = len(ev.get("rounds", [])) + 1
    items = []
    for e in E:
        for a in ctx.asset_list:
            sys, user, sch = elicit_prompt(e, a, ddesc, max_items)
            out = ctx.llm.json("M2", "elicit", sys, user, sch, context={"effect": e["id"], "asset": a["id"], "dims": sorted(names)})
            for it in out.get("items", [])[:max_items]:
                row = {"effect": e["id"], "asset": a["id"], "attribute": str(it.get("attribute", "")).strip(),
                       "label": it.get("label"), "dimension": str(it.get("dimension", "")), "why": it.get("why", ""),
                       "source": "elicit", "round": rnd}
                d = row["dimension"]
                if d.upper().startswith("NEW:"):
                    row["dimension"] = "NEW:" + _norm(d[4:])
                elif d not in names:
                    row["reject_reason"] = "dimension is neither existing nor NEW:<name>"
                    rej.append(row)
                    continue
                if not row["attribute"]:
                    row["reject_reason"] = "empty attribute"
                    rej.append(row)
                    continue
                items.append(row)
    allitems = keep + items
    per = Counter(i["dimension"] for i in allitems)
    pairs = defaultdict(set)
    for i in allitems:
        pairs[i["dimension"]].add((i["effect"], i["asset"]))
    rep = {"round": rnd, "base": os.path.relpath(bpath, p()), "pairs_queried": len(queried), "new_items": len(items),
           "rejected_new": sum(1 for r in rej if r.get("round") == rnd), "items_total": len(allitems),
           "pairs_with_items": len({(i["effect"], i["asset"]) for i in allitems}),
           "per_dimension": {d: {"attributes": n, "eliciting_pairs": len(pairs[d])} for d, n in per.most_common()},
           "per_asset": dict(Counter(i["asset"] for i in allitems))}
    ev = {"items": allitems, "rejected": rej, "rounds": ev.get("rounds", []) + [rep]}
    save_yaml(ctx, "m2-elicit", ev, "m2", "B1_elicitation.yaml")
    sample = _sample([i for i in items], 60, lambda r: r["dimension"], f"B{rnd}")
    save_csv(ctx, "m2-elicit", ["effect", "asset", "attribute", "label", "dimension", "why",
                                "HUMAN_world_state(1/0)", "HUMAN_label_ok(1/0)", "HUMAN_dimension_ok(1/0)", "note"],
             [[r["effect"], r["asset"], r["attribute"], r["label"], r["dimension"], r["why"], "", "", "", ""] for r in sample],
             "annotate", f"m2_B1_elicit_CHECK_round{rnd}.csv")
    return rep


# ------------------------------------------------------------------ C
def run_consolidate(ctx: Ctx, base: str = "seed", dims: list | None = None,
                    min_support: int = 2, max_pairs: int = 30) -> dict:
    """Stage C: label-signature consolidation = rough-set value granulation on the marginal
    decision table of one dimension. For a dimension D (existing, or NEW with >= min_support
    distinct evidence pairs):
      C1 canonicalize existing values + evidence phrases into mutually exclusive states,
         each with its parent (the existing value whose meaning contains it);
      C2 R_D = the evidence pairs of D (elicitation + refutation), at most max_pairs;
         sig(c) = (L(e, v | D = c, everything else ordinary))_{(e,v) in R_D};
      C3 states with identical signatures are indiscernible for every judgment in R_D and
         are merged; an existing value is never merged or removed here (its removal needs
         the exact rough-set evidence of m2-audit and a human decision).
    New values produce the `refines` map of the candidate (split / extend / added).
    Interactions between dimensions are NOT seen here (marginal table); they are handled by
    M3 over the full Σ and checked by m2-audit and refutation."""
    B, bpath = _base(ctx, base)
    ev = _evidence(ctx)
    lib = ctx.effects()
    eidx = {e["id"]: e for e in lib["effects"]}
    aidx = {a["id"]: a for a in ctx.asset_list}
    by_dim = defaultdict(list)
    for i in ev["items"]:
        by_dim[i["dimension"]].append(i)
    existing = [f for f in B["factors"] if f.get("scope") != "action"]
    names = {f["name"] for f in existing}
    want = None if not dims else {d[4:] if d.startswith("NEW:") else d for d in dims}
    work = [dict(f) for f in existing if want is None or f["name"] in want]
    for d, its in by_dim.items():
        if d.startswith("NEW:") and d[4:] not in names and (want is None or d[4:] in want):
            sup = len({(i["effect"], i["asset"]) for i in its})
            if sup >= min_support:
                work.append({"name": d[4:], "scope": "zone", "values": {}, "new": True,
                             "semantics": f"New dimension elicited from {len(its)} attributes in {sup} effect x asset pairs."})
    rebuilt, c1, sig_rows, report, refines = {}, {}, [], {}, {}
    for f in work:
        key = ("NEW:" + f["name"]) if f.get("new") else f["name"]
        its = by_dim.get(key, [])
        ex_keys = list(f["values"])
        phrases = [f"{k}: {s}" for k, s in f["values"].items()] + sorted({i["attribute"] for i in its})
        if len(phrases) <= len(ex_keys):               # no new evidence: keep as is
            rebuilt[f["name"]] = dict(f)
            report[f["name"]] = {"evidence_items": len(its), "changed": False}
            continue
        sys, user, sch = canonicalize_prompt(f, phrases, len(ex_keys))
        can = ctx.llm.json("M2", "canonicalize", sys, user, sch,
                           context={"dim": f["name"], "phrases": phrases, "n_existing": len(ex_keys)})
        states = [s for s in can.get("states", []) if s.get("name")]
        c1[f["name"]] = {"phrases": phrases, "n_existing": len(ex_keys), "states": states, "dropped": can.get("dropped", [])}
        pc = Counter((i["effect"], i["asset"]) for i in its)
        R = [pa for pa, _ in pc.most_common(max_pairs) if pa[0] in eidx and pa[1] in aidx]
        sig, is_ex, parent, members = {}, {}, {}, {}

        def signature(sname, sem):
            vec = []
            for e, a in R:
                o = ctx.llm.json("M2", "contrast_label", *contrast_label_prompt(eidx[e], aidx[a], f, sname, sem),
                                 context={"effect": e, "asset": a, "factor": f["name"], "value": sname})
                lab = o.get("label") if o.get("label") in ("+", "-", "0") else "0"
                vec.append(lab)
                sig_rows.append([f["name"], sname, int(sname in f["values"]), e, a, lab, o.get("harm", "")])
            return tuple(vec)

        for s in states:
            mem = [i for i in s.get("members", []) if isinstance(i, int) and 0 <= i < len(phrases)]
            ex_mem = [i for i in mem if i < len(ex_keys)]
            sname = ex_keys[ex_mem[0]] if ex_mem else _norm(s["name"])
            if sname in sig:
                members[sname].extend(phrases[i] for i in mem)
                continue
            sem = f["values"].get(sname) or s.get("semantics", "")
            sig[sname] = signature(sname, sem)
            is_ex[sname] = sname in f["values"]
            members[sname] = [phrases[i] for i in mem]
            par = str(s.get("parent") or "")
            parent[sname] = sname if is_ex[sname] else (par if par in f["values"] else None)
            if not is_ex[sname]:
                f.setdefault("_sem", {})[sname] = sem
        for k in ex_keys:                              # existing values are always kept and labelled
            if k not in sig:
                sig[k] = signature(k, f["values"][k])
                is_ex[k], parent[k], members[k] = True, k, [f"{k}: {f['values'][k]}"]
        groups = defaultdict(list)
        for sname, vec in sig.items():
            groups[vec].append(sname)
        # C3 merge rule. A new state joins an existing value only if nothing changes meaning:
        #  - its parent IS that value and the signatures agree (a sub-case that never changes a label), or
        #  - it had no parent (the situation was not representable) and the signatures agree (extends the value).
        # A state whose parent is P but whose signature equals another existing value Q stays a NEW value
        # (split of P) flagged indiscernible_from Q: merging it into Q would silently move situations
        # from P to Q; the exact decision is left to the rough-set audit of K (m2-audit) and the reviewer.
        values, prov, pending = {}, {}, defaultdict(list)
        split, extend, ext_meaning = defaultdict(list), [], defaultdict(list)
        for vec, mem in groups.items():
            exs = [m for m in mem if is_ex[m]]
            for m in exs:
                values[m] = f["values"][m]
                prov[m] = {"signature": "".join(vec), "source": "existing", "absorbed_states": [],
                           "extended_by": [], "phrases": list(members[m])}
            for x in (m for m in mem if not is_ex[m]):
                if parent[x] in exs:
                    prov[parent[x]]["absorbed_states"].append(x)
                    prov[parent[x]]["phrases"] += members[x]
                elif parent[x] is None and exs:
                    prov[exs[0]]["extended_by"].append(x)
                    prov[exs[0]]["phrases"] += members[x]
                    ext_meaning[exs[0]].append(x)
                else:
                    pending[(vec, parent[x])].append(x)
        for (vec, par), xs in pending.items():
            name = xs[0]
            values[name] = "; ".join(dict.fromkeys(f["_sem"][x] for x in xs))[:300]
            prov[name] = {"signature": "".join(vec), "source": "evidence", "merged_states": xs[1:], "parent": par,
                          "indiscernible_from": [m for m in groups[vec] if is_ex[m]],
                          "phrases": [ph for x in xs for ph in members[x]]}
            if f.get("new"):
                continue
            if par:
                split[par].append(name)
            else:
                extend.append(name)
        if f.get("new") and not any(set(pv["signature"]) <= {"0"} for pv in prov.values()):
            values["none"] = "None of the elicited states holds."          # exhaustiveness (rule R3)
            prov["none"] = {"signature": "0" * len(R), "source": "baseline added (R3)", "phrases": []}
        changed = f.get("new") or bool(split) or bool(extend) or bool(ext_meaning)
        if f.get("new"):
            refines[f["name"]] = {"added": True}
        elif changed:
            refines[f["name"]] = {"split": {u: [u] + ch for u, ch in split.items()}, "extend": extend,
                                  "extend_meaning": dict(ext_meaning)}
        nf = {k: v for k, v in f.items() if k not in ("values", "new", "_sem", "value_provenance", "relevant_pairs")}
        nf.update({"values": values, "value_provenance": prov, "relevant_pairs": [f"{e} x {a}" for e, a in R],
                   "evidence": dict(Counter(i.get("source", "elicit") for i in its))})
        if f.get("new"):
            nf["new_dimension"] = True
            nf["provenance"] = {"elicited": True, "support_pairs": len({(i["effect"], i["asset"]) for i in its})}
        rebuilt[f["name"]] = nf
        report[f["name"]] = {"evidence_items": len(its), "states_after_canonicalize": len(states),
                             "values_before": len(ex_keys), "values_after": len(values), "relevant_pairs": len(R),
                             "dropped_phrases": len(can.get("dropped", [])), "new_dimension": bool(f.get("new")),
                             "changed": bool(changed)}
    out_f = []
    for f in B["factors"]:
        out_f.append(rebuilt.get(f["name"], f))
    out_f += [f for n, f in rebuilt.items() if n not in {x["name"] for x in B["factors"]}]
    res = {"version": "F-candidate", "status": "REVIEW, then m2-freeze --src <this file or your edited copy>",
           "base": os.path.relpath(bpath, p()), "base_version": B.get("version"), "rebuilt_dimensions": sorted(rebuilt),
           "refines": refines, "factors": out_f, "feasibility": B.get("feasibility", [])}
    save_yaml(ctx, "m2-consolidate", c1, "m2", "C1_canonical.yaml")
    save_csv(ctx, "m2-consolidate", ["dimension", "state", "existing_value", "effect", "asset", "label", "harm"], sig_rows,
             "m2", "C2_signatures.csv")
    save_yaml(ctx, "m2-consolidate", res, "m2", "C3_factors_candidate.yaml")
    save_yaml(ctx, "m2-consolidate", report, "m2", "C3_report.yaml")
    sample = _sample(sig_rows, 60, lambda r: r[0], "C2")
    save_csv(ctx, "m2-consolidate", ["dimension", "value", "effect", "asset", "label", "HUMAN_label(+/-/0)", "note"],
             [[r[0], r[1], r[3], r[4], r[5], "", ""] for r in sample], "annotate", "m2_C2_signature_CHECK.csv")
    return report


# ------------------------------------------------------------------ D: audit
def _spec_tables(ctx: Ctx, spec_path: str | None = None) -> Tables:
    vocab = ctx.vocab
    spec_raw = load_yaml(spec_path or ctx.spec_path())
    if spec_raw.get("vocab_version") not in (None, vocab.version):
        raise SystemExit(f"K was built with {spec_raw.get('vocab_version')} but config uses {vocab.version}; "
                         "run vocab-migrate (and m3 --only-stale) first")
    spec = [Binding.from_json(b, vocab) for b in spec_raw["bindings"]]
    lib = ctx.effects()
    pairs = [(e["id"], a["id"]) for e in lib["effects"] for a in ctx.asset_list]
    return Tables(vocab, spec, pairs)


def _rules_sheet(ctx: Ctx, path: str | None) -> dict:
    path = path or ctx.outp("annotate", f"m2_rules_CHECK_{ctx.vocab.version}.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        cols = {k: (v or "").strip() for k, v in r.items() if k[:2] in ("A_", "B_")}
        if any(cols.values()):
            out[r["factor"]] = cols
    return out


def run_audit(ctx: Ctx, rules_csv: str | None = None) -> dict:
    """D2 (after M3): exact rough-set audit of K on the frozen Σ and one dimension card per
    factor. Writes the full decision tables so every number can be recomputed."""
    T = _spec_tables(ctx)
    V = ctx.vocab
    vfile = load_yaml(p(ctx.cfg["paths"]["factors"]))
    fentry = {f["name"]: f for f in vfile["factors"]}
    names = V.names
    nz = [i for i in range(len(T.pairs)) if T.L[i].any()]
    # decision tables (only pairs with a non-0 label; all others are constant 0)
    path = ctx.outp("m2", "D2_decision_tables.csv.gz")
    with gzip.open(path, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cell"] + names + [f"{T.pairs[i][0]}|{T.pairs[i][1]}" for i in nz])
        for c, cell in enumerate(T.sigma):
            w.writerow([c, *cell] + [INV[int(T.L[i, c])] for i in nz])
    record(ctx, "m2-audit", path)
    # core (per pair and global)
    core = T.core()
    save_csv(ctx, "m2-audit", ["pair"] + names,
             [[f"{T.pairs[i][0]} x {T.pairs[i][1]}"] + [int(core[n][i]) for n in names] for i in nz],
             "m2", "D2_core_matrix.csv")
    gcore = [n for n in names if core[n].sum() > 0]
    # reducts
    red = {}
    for i in nz:
        pair = T.pairs[i]
        r = T.reduct([i])
        red[f"{pair[0]} x {pair[1]}"] = {"core": [n for n in names if core[n][i] > 0], "reduct": r,
                                         "pattern_support": T.support(pair),
                                         "bindings": len(T.bindings.get(pair, []))}
    greduct = T.reduct(nz)
    save_yaml(ctx, "m2-audit", {"global_core": gcore, "global_reduct": greduct, "per_pair": red}, "m2", "D2_reducts.yaml")
    # value granulation
    vp = {}
    for n in names:
        groups, M = T.value_partition(n)
        dom = list(V.domain(n))
        vp[n] = {"values": dom, "decision_preserving_groups": groups, "mergeable": len(groups) < len(dom),
                 "merge_conflicts": {f"{dom[u]}+{dom[w]}": int(M[u, w]) for u in range(len(dom)) for w in range(u + 1, len(dom))}}
    save_yaml(ctx, "m2-audit", vp, "m2", "D2_value_partition.yaml")
    prime = T.prime_check()
    save_yaml(ctx, "m2-audit", prime, "m2", "D2_prime_check.yaml")
    # evidence for the cards
    a2 = load_yaml(ctx.outp("m2", "A2_dims.yaml")) if os.path.exists(ctx.outp("m2", "A2_dims.yaml")) else {"slots": []}
    a2d = {d["id"]: d for d in a2["slots"]}
    stab = load_yaml(ctx.outp("m2", "A2_stability.yaml")) if os.path.exists(ctx.outp("m2", "A2_stability.yaml")) else {}
    ev = _evidence(ctx)["items"]
    refi = load_yaml(ctx.outp("m2", "D3_refinement.yaml")) if os.path.exists(ctx.outp("m2", "D3_refinement.yaml")) else {"items": []}
    rules = _rules_sheet(ctx, rules_csv)
    cards = []
    for n in names:
        fe = fentry.get(n, {})
        prov = fe.get("provenance") or {}
        src = prov.get("dimension") or prov.get("slot") or []
        src = [src] if isinstance(src, str) else list(src)
        corpus = {d: {"conditions": a2d[d].get("n_conditions"), "property_ids": a2d[d].get("property_ids", [])}
                  for d in src if d in a2d}
        el = [i for i in ev if i["dimension"] in (n, "NEW:" + n)]
        rf = [i for i in refi.get("items", []) if i.get("dimension") in (n, "NEW:" + n)]
        k_pairs = [f"{T.pairs[i][0]} x {T.pairs[i][1]}" for i in nz if core[n][i] > 0]
        rc = rules.get(n, {})
        failed = sorted({k[2:4] for k, v in rc.items() if v == "0"})
        verdict = []
        if n not in gcore:
            verdict.append("DROP candidate: no label in K depends on this factor (not in the global core)")
        if 1 < len(vp[n]["decision_preserving_groups"]) < len(V.domain(n)):
            verdict.append(f"MERGE candidate: decision-preserving value groups {vp[n]['decision_preserving_groups']}")
        if failed:
            verdict.append(f"REVIEW: rule(s) {failed} failed by an annotator")
        cards.append({
            "factor": n, "scope": V.by_name[n].scope, "semantics": V.by_name[n].semantics,
            "values": {v: V.by_name[n].value_sem.get(v, "") for v in V.domain(n)},
            "origin": ("derived" if V.by_name[n].scope == "action" else "elicited" if fe.get("new_dimension") else "corpus"),
            "evidence": {
                "corpus": corpus or ("derived from the action's target area" if V.by_name[n].scope == "action" else "none"),
                "corpus_properties": len({x for c in corpus.values() for x in c["property_ids"]}),
                "elicitation": {"items": len(el), "pairs": len({(i['effect'], i['asset']) for i in el}),
                                "by_source": dict(Counter(i.get("source", "elicit") for i in el))},
                "refutation_counterexamples": len(rf),
                "value_provenance": fe.get("value_provenance", {}),
                "stability_recurrence": {d: stab.get("dimension_recurrence", {}).get(d, {}).get("recurrence") for d in src} if stab else "not run",
            },
            "necessity": {"in_global_core": n in gcore, "pairs_where_necessary": len(k_pairs),
                          "conflict_classes_if_removed": int(core[n].sum()), "pairs": k_pairs[:30]},
            "value_partition": vp[n]["decision_preserving_groups"],
            "rules_check": rc or "not annotated",
            "verdict": verdict or ["KEEP"]})
    save_yaml(ctx, "m2-audit", cards, "m2", "D2_dimension_cards.yaml")
    md = [f"# Dimension cards — {V.version}\n", f"|Σ| = {len(T.sigma)}; pairs with a non-0 label: {len(nz)}/{len(T.pairs)}; "
          f"global core: {gcore}; global reduct: {greduct}; prime bindings {prime['prime']}/{prime['bindings']}, "
          f"irredundant {prime['irredundant']}/{prime['bindings']}\n"]
    for c in cards:
        e = c["evidence"]
        md.append(f"\n## {c['factor']}  ({c['scope']}, {c['origin']})\n{c['semantics']}\n")
        md.append("- values: " + "; ".join(f"`{k}` {v}" for k, v in c["values"].items()))
        md.append(f"- corpus: {e['corpus_properties']} properties; elicitation: {e['elicitation']['items']} items from "
                  f"{e['elicitation']['pairs']} pairs; refutation counterexamples: {e['refutation_counterexamples']}")
        md.append(f"- stability: {e['stability_recurrence']}")
        md.append(f"- necessity: in global core = {c['necessity']['in_global_core']}, necessary for "
                  f"{c['necessity']['pairs_where_necessary']} pairs ({c['necessity']['conflict_classes_if_removed']} conflict classes if removed)")
        md.append(f"- decision-preserving value groups: {c['value_partition']}")
        md.append(f"- rules: {c['rules_check']}")
        md.append(f"- **verdict: {' | '.join(c['verdict'])}**")
    path = ctx.outp("m2", "D2_dimension_cards.md")
    open(path, "w", encoding="utf-8").write("\n".join(md) + "\n")
    record(ctx, "m2-audit", path)
    summ = {"vocab_version": V.version, "sigma": len(T.sigma), "pairs": len(T.pairs), "pairs_nonzero": len(nz),
            "global_core": gcore, "outside_core": [n for n in names if n not in gcore], "global_reduct": greduct,
            "mean_pair_reduct": round(float(np.mean([len(x["reduct"]) for x in red.values()])), 2) if red else 0,
            "mean_pair_support": round(float(np.mean([len(x["pattern_support"]) for x in red.values()])), 2) if red else 0,
            "mergeable_factors": [n for n in names if vp[n]["mergeable"]],
            "prime": f"{prime['prime']}/{prime['bindings']}", "irredundant": f"{prime['irredundant']}/{prime['bindings']}",
            "verdicts": {c["factor"]: c["verdict"] for c in cards}}
    save_yaml(ctx, "m2-audit", summ, "m2", "D2_audit_summary.yaml")
    return summ


# ------------------------------------------------------------------ D: refine (CEGAR)
def run_refine(ctx: Ctx) -> dict:
    """D3: confirmed counterexamples of m3-refute (two real situations collapse onto one
    abstract cell but deserve different labels = the abstraction is too coarse) become new
    evidence items (source 'refutation') filed under an existing factor (-> split a value)
    or a NEW dimension. Next: m2-consolidate --base current --dims <affected>."""
    ref = load_yaml(ctx.outp("m3", "refutation.yaml"))
    B, _ = _base(ctx, "current")
    dims = [f for f in B["factors"] if f.get("scope") != "action"]
    names = {f["name"] for f in dims}
    ddesc = _dims_desc(dims)
    rows, items = [], []
    for it in ref["items"]:
        for ce in it["counterexamples"]:
            if not ce.get("confirmed"):
                continue
            attr = str(ce.get("missing_attribute", "")).strip()
            sys, user, sch = attribute_dimension_prompt(attr, ddesc)
            o = ctx.llm.json("M2", "attribute_dimension", sys, user, sch, context={"attribute": attr, "dims": sorted(names)})
            d = str(o.get("dimension", ""))
            d = ("NEW:" + _norm(d[4:])) if d.upper().startswith("NEW:") else d
            valid = d in names or d.startswith("NEW:")
            row = {"claim": it["claim"], "effect": it["effect"], "asset": it["asset"], "pattern": it["pattern"],
                   "claimed_label": it["label"], "narrative": ce.get("narrative", ""), "expected_label": ce.get("expected_label"),
                   "judge_label": ce.get("judge_label"), "missing_attribute": attr, "dimension": d,
                   "is_world_state": bool(o.get("is_world_state")), "used": bool(valid and o.get("is_world_state"))}
            rows.append(row)
            if row["used"]:
                items.append({"effect": it["effect"], "asset": it["asset"], "attribute": attr, "label": ce.get("expected_label"),
                              "dimension": d, "why": ce.get("narrative", ""), "source": "refutation", "claim": it["claim"]})
    ev = _evidence(ctx)
    ev["items"] = [i for i in ev["items"] if i.get("source") != "refutation"] + items
    ev.setdefault("rounds", []).append({"refinement_items": len(items)})
    save_yaml(ctx, "m2-refine", ev, "m2", "B1_elicitation.yaml")
    affected = sorted({i["dimension"] for i in items})
    res = {"confirmed_counterexamples": len(rows), "used_as_evidence": len(items), "affected_dimensions": affected,
           "affected_pairs": sorted({f"{i['effect']} x {i['asset']}" for i in items}),
           "next": (f"m2-consolidate --base current --dims {' '.join(affected)}" if affected else "nothing to refine: stop"),
           "items": rows}
    save_yaml(ctx, "m2-refine", res, "m2", "D3_refinement.yaml")
    save_csv(ctx, "m2-refine", ["claim", "effect", "asset", "pattern", "claimed_label", "narrative", "expected_label",
                                "judge_label", "missing_attribute", "dimension", "HUMAN_real_counterexample(1/0)",
                                "HUMAN_dimension_ok(1/0)", "note"],
             [[r["claim"], r["effect"], r["asset"], r["pattern"], r["claimed_label"], r["narrative"], r["expected_label"],
               r["judge_label"], r["missing_attribute"], r["dimension"], "", "", ""] for r in rows],
             "annotate", "m2_D3_counterexample_CHECK.csv")
    return {k: v for k, v in res.items() if k != "items"}


# ------------------------------------------------------------------ check sheets
def run_check_score(ctx: Ctx, paths: list[str]) -> dict:
    """Scores any filled m2_*_CHECK.csv: mean of every HUMAN_*(1/0) column; accuracy and
    kappa of HUMAN_label vs the system label; agreement and Cohen's kappa of A_*/B_* columns."""
    res = {}
    for path in paths:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        r = {"rows": len(rows)}
        cols = rows[0].keys() if rows else []
        for c in cols:
            if c.startswith("HUMAN_") and c.endswith("(1/0)"):
                v = [int(x[c]) for x in rows if (x[c] or "").strip() in ("0", "1")]
                if v:
                    r[c] = {"n": len(v), "mean": round(sum(v) / len(v), 3)}
        if "HUMAN_label(+/-/0)" in cols:
            pr = [(x["label"], x["HUMAN_label(+/-/0)"].strip()) for x in rows if x["HUMAN_label(+/-/0)"].strip() in ("+", "-", "0")]
            if pr:
                r["label_vs_human"] = {"n": len(pr), "accuracy": round(sum(a == b for a, b in pr) / len(pr), 3),
                                       "kappa": _kappa([a for a, _ in pr], [b for _, b in pr])}
        for c in cols:
            if c.startswith("A_") and ("B_" + c[2:]) in cols:
                pr = [(x[c].strip(), x["B_" + c[2:]].strip()) for x in rows if x[c].strip() and x["B_" + c[2:]].strip()]
                a_only = [int(x[c]) for x in rows if x[c].strip() in ("0", "1")]
                r[c[2:]] = {"pass_rate_A": round(sum(a_only) / len(a_only), 3) if a_only else None, "n_both": len(pr),
                            "agreement": round(sum(a == b for a, b in pr) / len(pr), 3) if pr else None,
                            "kappa": _kappa([a for a, _ in pr], [b for _, b in pr]) if pr else None}
        res[os.path.basename(path)] = r
    save_yaml(ctx, "m2-check-score", res, "eval", "m2_checks.yaml")
    return res


# ------------------------------------------------------------------ property mapping
def run_map_properties(ctx: Ctx) -> dict:
    """Map every seed property into the vocabulary (used for expressibility on the
    held-out split, for the property-guard baseline B1, and for novelty of K)."""
    vocab = ctx.vocab
    lib = ctx.effects()
    edesc = "\n".join(f"- {e['id']}: {e['desc']}" for e in lib["effects"])
    eids = {e["id"] for e in lib["effects"]}
    aids = {a["id"] for a in ctx.asset_list}
    rows = []
    for prop in load_properties(ctx):
        sys, user, schema = map_property_prompt(prop["text"], vocab.describe_vocab(), edesc)
        out = ctx.llm.json("M2", "map_property", sys, user, schema, context={"property": prop})
        row = {"id": prop["id"], "split": prop["split"], "text": prop["text"], "source": prop.get("source", ""),
               "expressible": bool(out.get("expressible")), "missing": out.get("missing", "")}
        if row["expressible"]:
            try:
                phi = normalize(out.get("pattern") or {}, vocab)
                effs = [e for e in out.get("effects", []) if e in eids]
                if not effs or out.get("asset") not in aids or out.get("label") not in ("+", "-"):
                    raise ValueError("effect/asset/label invalid")
                row.update({"pattern": pattern_to_json(phi, vocab), "effects": effs,
                            "asset": out["asset"], "label": out["label"]})
            except ValueError as e:
                row.update({"expressible": False, "missing": f"invalid mapping: {e}"})
        rows.append(row)
    save_yaml(ctx, "m2-map", rows, "m2", "property_mapping.yaml")
    with open(ctx.outp("annotate", "property_mapping_CHECK.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "text", "system_expressible", "system_mapping", "missing",
                    "HUMAN_mapping_correct(1/0)", "HUMAN_expressible(1/0)", "note"])
        for r in rows:
            m = "" if not r["expressible"] else f"{r['effects']} {r['label']} {r['asset']} IF {r['pattern']}"
            w.writerow([r["id"], r["split"], r["text"], int(r["expressible"]), m, r["missing"], "", "", ""])
    return {"n": len(rows), "expressible": sum(r["expressible"] for r in rows)}




# ------------------------------------------------------------------ metrics (Sec. VI-A)
def vocab_metrics(ctx: Ctx, human_check_csv: str | None = None) -> dict:
    """Vocabulary metrics. Necessity uses the exact rough-set tables (roughset.Tables), the
    same computation as m2-audit."""
    vocab = ctx.vocab
    spec_raw = load_yaml(ctx.spec_path())
    spec = [Binding.from_json(b, vocab) for b in spec_raw["bindings"]]
    T = Tables(vocab, spec)
    res = {"vocab_version": vocab.version, "n_factors": len(vocab.factors),
           "domain_sizes": {f.name: len(f.values) for f in vocab.factors},
           "product_size": vocab.product_size(), "sigma_size": len(vocab.sigma())}
    proj = []
    for pair in T.pairs:
        size = 1
        for n in T.support(pair):
            size *= len(vocab.domain(n))
        proj.append(size)
    res["projected_table_size"] = {"pairs_with_bindings": len(proj), "max": max(proj, default=0),
                                   "mean": round(float(np.mean(proj)), 1) if proj else 0}
    core = T.core()
    res["factor_ablation"] = {n: {"conflict_groups": int(core[n].sum()), "pairs_affected": int((core[n] > 0).sum()),
                                  "necessary": bool(core[n].sum() > 0)} for n in vocab.names}
    vm = {}
    for n in vocab.names:
        dom = list(vocab.domain(n))
        M = T.value_merge_conflicts(n)
        merges = {f"{dom[u]}+{dom[w]}": int(M[u, w]) for u in range(len(dom)) for w in range(u + 1, len(dom))}
        vm[n] = {"merge_conflicts": merges,
                 "values_with_free_merge": [v for v in dom if any(merges[k] == 0 for k in merges if v in k.split("+"))]}
    res["value_merge_ablation"] = vm
    # expressibility on held-out properties + K coverage
    mp = ctx.outp("m2", "property_mapping.yaml")
    try:
        rows = load_yaml(mp)
    except FileNotFoundError:
        rows = []
    if rows:
        human = {}
        if human_check_csv:
            for r in csv.DictReader(open(human_check_csv, encoding="utf-8")):
                human[r["id"]] = r
        test = [r for r in rows if r["split"] == "test"]
        expr = [r for r in test if (human.get(r["id"], {}).get("HUMAN_expressible(1/0)", "") or str(int(r["expressible"]))) == "1"]
        idx = index_spec(spec)
        cov = []
        for r in test:
            if not r["expressible"]:
                continue
            phi = normalize(r["pattern"], vocab)
            cells = vocab.cells_in(phi)
            for e in r["effects"]:
                if not cells:
                    continue
                hit = sum(label_of(idx, e, r["asset"], c, vocab) == r["label"] for c in cells)
                cov.append(hit / len(cells))
        res["expressibility_heldout"] = {"n_test": len(test), "expressible": len(expr),
                                         "ratio": round(len(expr) / max(1, len(test)), 3),
                                         "human_checked": bool(human)}
        res["K_recall_of_heldout_properties"] = {"mean_cell_coverage": round(float(np.mean(cov)), 3) if cov else None,
                                                 "share_covered_ge_0.9": round(float(np.mean([c >= 0.9 for c in cov])), 3) if cov else None}
        # novelty: '-' bindings of K disjoint from every mapped property region
        prop_regions = defaultdict(list)
        for r in rows:
            if r["expressible"]:
                for e in r["effects"]:
                    prop_regions[(e, r["asset"])].append(normalize(r["pattern"], vocab))
        novel = []
        for b in spec:
            if b.label != "-":
                continue
            cells = vocab.cells_in(b.phi)
            overl = any(contains(q, c, vocab) for q in prop_regions.get((b.effect, b.asset), []) for c in cells[:500])
            if not overl:
                novel.append(b)
        res["novel_constraints"] = {"n_minus_bindings": sum(b.label == "-" for b in spec), "not_in_any_property": len(novel)}
        with open(ctx.outp("annotate", "novel_constraints_CHECK.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["effect", "asset", "pattern", "harm", "HUMAN_valid(1/0)", "note"])
            for b in novel:
                w.writerow([b.effect, b.asset, pattern_to_json(b.phi, vocab), b.harm, "", ""])
    dump_yaml(res, ctx.outp("eval", "vocab_metrics.yaml"))
    return res
