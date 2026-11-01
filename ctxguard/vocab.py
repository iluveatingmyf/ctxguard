"""M2  Context-factor vocabulary construction + vocabulary metrics  (Sec. IV-B, VI-A)

Construction (main method):
  m2-extract  conditions from properties
  m2-dims     dimensions induced from the corpus by iterative open coding (run_dims)
  m2-values   values per dimension induced from the corpus (run_slots on dims.yaml)
  m2-expand   agent expansion of values, accepted only if they change a label (run_expand)
  then human freeze -> factors_v1.yaml; M3 binds effects to the frozen vocabulary.
Ablations: m2-slots (dimensions fixed top-down from data/vocab/slots.yaml, Dey 2001);
           m2-cluster + m2-name (embedding clustering, bottom-up only):
  1. extract   : LLM turns each property into atomic, device-agnostic situation
                 conditions ("someone is asleep in the room").
  2. cluster   : embed conditions (nomic-embed-text), average-linkage
                 agglomerative clustering on cosine distance; the cut height is
                 chosen by maximal silhouette over a fixed grid (no hand tuning).
  3. name      : LLM proposes one factor per cluster (name, scope, 2-6 values
                 with decidable semantics).
  4. freeze    : a human merges/renames -> factors_v1.yaml (edits are counted
                 and reported).
Validation (the four criteria, all computed, none hand-asserted):
  expressibility  share of HELD-OUT properties expressible as patterns over F
  necessity       per-factor / per-value ablation on K's decision tables:
                  conflict cells after removing a factor / merging two values
                  (rough-set core / reduct test)
  computability   |Sigma|, and projected table size per (e,v)
  groundability   share of (factor, area) groundable per layout (from M5)
"""
from __future__ import annotations

import csv
import os
import hashlib
import itertools
from collections import Counter, defaultdict

import numpy as np

from .common import Ctx, dump_yaml, load_yaml, p
from .effects import effect_index
from .model import Binding, contains, index_spec, label_of, normalize, pattern_to_json
from .prompts import (classify_condition_prompt, contrast_label_prompt, expand_values_prompt,
                      extract_predicates_prompt, induce_dims_prompt, map_property_prompt,
                      name_factor_prompt, slot_values_prompt)


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


# ------------------------------------------------------------------ build
def run_extract(ctx: Ctx) -> list[dict]:
    rows = []
    for prop in load_properties(ctx):
        if prop["split"] != "train":
            continue
        sys, user, schema = extract_predicates_prompt(prop["text"])
        out = ctx.llm.json("M2", "extract_predicates", sys, user, schema, context={"property": prop})
        for c in out.get("conditions", []):
            c = c.strip()
            if c:
                rows.append({"property_id": prop["id"], "condition": c})
    dump_yaml(rows, ctx.outp("m2", "conditions.yaml"))
    return rows


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
    conds = load_yaml(ctx.outp("m2", "conditions.yaml"))
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
    dump_yaml(res, ctx.outp("m2", "clusters.yaml"))
    return res


def run_name(ctx: Ctx, min_size: int = 2) -> dict:
    cl = load_yaml(ctx.outp("m2", "clusters.yaml"))
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
    dump_yaml(res, ctx.outp("m2", "ablation_clustering_factors.yaml"))
    return res


# ------------------------------------------------------------------ slot-based induction (main method)
def run_slots(ctx: Ctx, schema_path: str | None = None, tag: str = "ablation_topdown") -> dict:
    """M2 main method: top-down dimensions (data/vocab/slots.yaml), bottom-up values.
    1. classify every extracted condition into one slot (situation slots or excluded
       device_state / user_configuration / other), restated device-agnostically;
    2. per situation slot, induce <= max_factors factors and their value sets from the
       restated statements, and assign each statement to a value (coverage check);
    3. add derived factors (effect_zone) and write factors_candidate.yaml for human freeze.
    The embedding clustering (m2-cluster/m2-name) is kept only as an ablation."""
    schema = load_yaml(schema_path or p("data/vocab/slots.yaml"))
    slots = {s["id"]: s for s in schema["slots"]}
    sdesc = "\n".join(f"- {s['id']} ({s['kind']}): {s['desc']}" for s in schema["slots"])
    conds = load_yaml(ctx.outp("m2", "conditions.yaml"))
    texts = sorted({c["condition"] for c in conds})
    placed = []
    for t in texts:
        sys, user, sch = classify_condition_prompt(t, sdesc)
        out = ctx.llm.json("M2", "classify_condition", sys, user, sch, context={"condition": t})
        slot = out.get("slot") if out.get("slot") in slots else "other"
        placed.append({"condition": t, "slot": slot, "statement": out.get("situation_statement") or t})
    by = defaultdict(list)
    for r in placed:
        by[r["slot"]].append(r)
    factors, report = [], {"n_conditions": len(texts), "per_slot": {}}
    for sid, s in slots.items():
        rows = by.get(sid, [])
        entry = {"kind": s["kind"], "n_conditions": len(rows), "examples": [r["condition"] for r in rows[:6]]}
        if s["kind"] == "situation" and rows:
            stmts = sorted({r["statement"] for r in rows})
            sys, user, sch = slot_values_prompt(s, stmts, s.get("max_factors", 1))
            out = ctx.llm.json("M2", "slot_values", sys, user, sch, context={"slot": s, "statements": stmts})
            fs = out.get("factors", [])[: s.get("max_factors", 1)]
            names = {f["name"] for f in fs}
            covered = sum(1 for a in out.get("assignment", []) if a.get("factor") in names and a.get("value"))
            entry.update({"n_statements": len(stmts), "covered_statements": covered,
                          "coverage": round(covered / max(1, len(stmts)), 3), "factors": [f["name"] for f in fs]})
            for f in fs:
                factors.append({"name": f["name"], "scope": s["scope"], "semantics": f.get("semantics", ""),
                                "values": {str(x["value"]): x["semantics"] for x in f.get("values", [])},
                                "provenance": {"slot": sid, "n_conditions": len(rows), "dey": s.get("dey")}})
        elif s["kind"] == "situation":
            entry["note"] = "no corpus evidence; schema dimension kept for review"
        report["per_slot"][sid] = entry
    for d in schema.get("derived", []):
        factors.append({"name": d["name"], "scope": d["scope"], "semantics": d["semantics"], "values": d["values"],
                        "provenance": {"derived": "action target area"}})
    n_ex = sum(len(by.get(sid, [])) for sid, s in slots.items() if s["kind"] == "excluded")
    report["excluded_share"] = round(n_ex / max(1, len(texts)), 3)
    res = {"version": f"F-{tag}", "schema": schema["version"], "factors": factors, "feasibility": []}
    dump_yaml(res, ctx.outp("m2", f"factors_{tag}.yaml"))
    dump_yaml(report, ctx.outp("m2", f"values_report_{tag}.yaml"))
    dump_yaml(placed, ctx.outp("m2", f"condition_coding_{tag}.yaml"))
    return report


# ------------------------------------------------------------------ main method
def run_dims(ctx: Ctx, max_rounds: int = 3, stop_uncovered: float = 0.05) -> dict:
    """M2 step 2: induce situation DIMENSIONS from the corpus by iterative open coding.
    Round r: the LLM proposes dimensions from the pool of conditions; every condition is
    then coded into one dimension or into the fixed non-situation categories
    (device_state, user_configuration) or 'other'. Conditions coded 'other' form the
    pool of the next round. Stop when the uncovered share <= stop_uncovered, when no
    new dimension appears, or after max_rounds. Output: out/m2/dims.yaml."""
    ref = load_yaml(p("data/vocab/slots.yaml"))
    excluded = [s for s in ref["slots"] if s["kind"] == "excluded"]
    conds = load_yaml(ctx.outp("m2", "conditions.yaml"))
    texts = sorted({c["condition"] for c in conds})
    dims, pool, rounds, coding = {}, texts, [], []
    for r in range(max_rounds):
        sys, user, sch = induce_dims_prompt(pool, list(dims.values()))
        out = ctx.llm.json("M2", "induce_dims", sys, user, sch, context={"conditions": pool, "existing": list(dims)})
        new = 0
        for d in out.get("dimensions", []):
            did = "".join(ch if ch.isalnum() else "_" for ch in str(d.get("id", "")).lower()).strip("_")
            if not did or did in dims or did in {e["id"] for e in excluded}:
                continue
            dims[did] = {"id": did, "kind": "situation", "scope": d.get("scope", "global"), "desc": d.get("desc", ""),
                         "max_factors": max(1, min(4, int(d.get("n_variables") or 1))), "round": r + 1}
            new += 1
        sdesc = "\n".join(f"- {s['id']} ({s['kind']}): {s['desc']}" for s in list(dims.values()) + excluded)
        coding = []
        for t in texts:
            sys, user, sch = classify_condition_prompt(t, sdesc)
            o = ctx.llm.json("M2", "classify_condition", sys, user, sch, context={"condition": t, "dims": list(dims)})
            slot = o.get("slot") if o.get("slot") in dims or o.get("slot") in {e["id"] for e in excluded} else "other"
            coding.append({"condition": t, "slot": slot, "statement": o.get("situation_statement") or t})
        uncovered = [c["condition"] for c in coding if c["slot"] == "other"]
        rounds.append({"round": r + 1, "new_dimensions": new, "n_dimensions": len(dims),
                       "uncovered": len(uncovered), "uncovered_share": round(len(uncovered) / max(1, len(texts)), 3)})
        if not new or len(uncovered) / max(1, len(texts)) <= stop_uncovered:
            break
        pool = uncovered
    per = Counter(c["slot"] for c in coding)
    for d in dims.values():
        d["n_conditions"] = per.get(d["id"], 0)
    kept = [d for d in dims.values() if d["n_conditions"] > 0]
    res = {"version": "S-induced", "rounds": rounds, "slots": kept + excluded, "derived": ref.get("derived", []),
           "coverage": {k: v for k, v in per.items()},
           "dropped_empty_dimensions": [d["id"] for d in dims.values() if d["n_conditions"] == 0]}
    dump_yaml(res, ctx.outp("m2", "dims.yaml"))
    dump_yaml(coding, ctx.outp("m2", "condition_dims.yaml"))
    return res


def run_expand(ctx: Ctx, max_new: int = 4) -> dict:
    """M2 step 4: agent expansion of value sets beyond the corpus.
    For every factor the agent proposes up to max_new missing values, each with a
    witness (effect e, asset v, closest existing value). A value is ACCEPTED only if
    two independent contrast queries give different labels for (e, v) under the new
    value and under the closest existing value, i.e. the value changes some safety
    judgment (the same necessity criterion used for the vocabulary metrics)."""
    base = load_yaml(ctx.outp("m2", "factors_corpus.yaml"))
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
    res = {"version": "F-candidate", "status": "REVIEW: merge/rename, add feasibility, save as data/vocab/factors_v1.yaml",
           "factors": factors, "feasibility": []}
    dump_yaml(res, ctx.outp("m2", "factors_candidate.yaml"))
    summ = {"proposed": len(report), "accepted": sum(r["decision"] == "accepted" for r in report), "items": report}
    dump_yaml(summ, ctx.outp("m2", "expansion_report.yaml"))
    return summ


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
    dump_yaml(rows, ctx.outp("m2", "property_mapping.yaml"))
    with open(ctx.outp("annotate", "property_mapping_CHECK.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "text", "system_expressible", "system_mapping", "missing",
                    "HUMAN_mapping_correct(1/0)", "HUMAN_expressible(1/0)", "note"])
        for r in rows:
            m = "" if not r["expressible"] else f"{r['effects']} {r['label']} {r['asset']} IF {r['pattern']}"
            w.writerow([r["id"], r["split"], r["text"], int(r["expressible"]), m, r["missing"], "", "", ""])
    return {"n": len(rows), "expressible": sum(r["expressible"] for r in rows)}


# ------------------------------------------------------------------ metrics
def _pair_tables(ctx: Ctx, spec):
    vocab = ctx.vocab
    idx = index_spec(spec)
    sigma = vocab.sigma()
    return {pair: {c: label_of(idx, pair[0], pair[1], c, vocab) for c in sigma} for pair in idx}


def vocab_metrics(ctx: Ctx, human_check_csv: str | None = None) -> dict:
    vocab = ctx.vocab
    spec_raw = load_yaml(ctx.spec_path())
    spec = [Binding.from_json(b, vocab) for b in spec_raw["bindings"]]
    tables = _pair_tables(ctx, spec)
    res = {"vocab_version": vocab.version, "n_factors": len(vocab.factors),
           "domain_sizes": {f.name: len(f.values) for f in vocab.factors},
           "product_size": vocab.product_size(), "sigma_size": len(vocab.sigma())}
    # computability: projected table per pair = product of domains of used factors
    proj = []
    for pair in tables:
        used = {n for b in spec if (b.effect, b.asset) == pair for n in b.phi}
        size = 1
        for n in used:
            size *= len(vocab.domain(n))
        proj.append(size)
    res["projected_table_size"] = {"pairs_with_bindings": len(proj), "max": max(proj, default=0),
                                   "mean": round(float(np.mean(proj)), 1) if proj else 0}
    # necessity: factor removal
    nec = {}
    for f in vocab.factors:
        i = vocab.index[f.name]
        conflicts, pairs_hit = 0, 0
        for pair, tab in tables.items():
            groups = defaultdict(set)
            for c, lab in tab.items():
                groups[c[:i] + c[i + 1:]].add(lab)
            k = sum(1 for g in groups.values() if len(g) > 1)
            conflicts += k
            pairs_hit += k > 0
        nec[f.name] = {"conflict_groups": conflicts, "pairs_affected": pairs_hit,
                       "necessary": conflicts > 0}
    res["factor_ablation"] = nec
    # necessity: value merges
    vm = {}
    for f in vocab.factors:
        i = vocab.index[f.name]
        merges = {}
        for u, w in itertools.combinations(f.values, 2):
            conflicts = 0
            for tab in tables.values():
                for c, lab in tab.items():
                    if c[i] == u:
                        c2 = c[:i] + (w,) + c[i + 1:]
                        if c2 in tab and tab[c2] != lab:
                            conflicts += 1
            merges[f"{u}+{w}"] = conflicts
        redundant = [v for v in f.values if any(merges[k] == 0 for k in merges if v in k.split("+"))]
        vm[f.name] = {"merge_conflicts": merges, "values_with_free_merge": redundant}
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
