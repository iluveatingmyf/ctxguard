"""M2  Context-factor vocabulary: construction, audit, versioned extension  (thesis Ch. 4, Sec. 4.2)

The factors are NOT fixed a priori. They are the output of the stages below and become
fixed only when a reviewed candidate is frozen as a version (F-v1, F-v2, ...). Every stage
writes its full intermediate result (out/m2/<stage>_*.yaml|csv, listed in out/MANIFEST.csv),
every LLM answer is cached (out/llm_cache.sqlite) and every call is in out/ledger.csv.

Stage A  dimension discovery from the property corpus (asset-agnostic)
  A1 m2-extract      property -> atomic situation conditions; per-property asset metadata
  A2 m2-dims         variable-level coding: per condition kind / world fact / variable / value
                     (one call each), axial merge of variable phrases, one-variable check per
                     dimension; --runs R repeats it with sampling (stability)
  A3 m2-values       per dimension: value set (one call) + closed-choice assignment of every
                     coded fact (one call each)
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
import re
from collections import Counter, defaultdict

import numpy as np

from .common import Ctx, dump_yaml, load_yaml, p
from .explog import log_result
from .manifest import _sha as _file_sha, record, save_csv, save_yaml
from .model import Binding, Vocab, contains, index_spec, label_of, normalize, pattern_to_json
from .prompts import (CONDITION_KINDS, FACT_RELATIONS, assign_value_prompt, fact_relation_prompt, name_slot_prompt, attribute_dimension_prompt, canonicalize_prompt,
                      check_dimension_prompt, classify_condition_prompt, code_condition_prompt, contrast_label_prompt,
                      dimension_values_prompt, elicit_prompt, expand_values_prompt, extract_predicates_prompt,
                      group_variables_prompt, lift_condition_prompt, map_property_prompt, name_factor_prompt,
                      property_assets_prompt, slot_values_prompt)
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




# ------------------------------------------------------------------ A2 (variable-level open + axial coding)
NO_VARIABLE_KINDS = ("device_state", "user_configuration", "obligation", "unclear")
SCOPE_MAP = {"room": "zone", "home": "global", "person": "zone", "outdoor": "global"}


def _cond_property_text(ctx: Ctx) -> dict:
    m = {}
    for r in load_yaml(ctx.outp("m2", "A1_conditions.yaml")):
        m.setdefault(r["condition"], r["property_text"])
    return m


def _codebook(ctx: Ctx | None = None) -> dict:
    """The human A2 codebook. method.m2_codebook selects the file (versioned, so every
    A2 result can name the codebook it was produced with); default data/vocab/a2_codebook.yaml."""
    rel = (ctx.cfg.get("method", {}).get("m2_codebook") if ctx else None) or "data/vocab/a2_codebook.yaml"
    path = p(rel)
    return load_yaml(path) if os.path.exists(path) else {"examples": [], "dev": []}


def _lift_cfg(ctx: Ctx) -> tuple[str, str]:
    """method.m2_lift_prompt: v4.1 (default) | v4 (ablation, original wording).
    method.m2_lift_gate:   evidence (default) | counterfactual (v4 ablation)."""
    m = ctx.cfg.get("method", {})
    ver, gate = m.get("m2_lift_prompt", "v4.1"), m.get("m2_lift_gate", "evidence")
    if gate not in LIFT_GATES:
        raise SystemExit(f"method.m2_lift_gate must be one of {LIFT_GATES}, got {gate!r}")
    return ver, gate


def _units(ctx: Ctx) -> list[dict]:
    """Evidence units of A2: one per (property, condition). A condition shared by several
    properties is coded once PER property, because the reason it matters (and so the fact it
    is lifted to) depends on the property's action and asset."""
    units, seen = [], set()
    for r in load_yaml(ctx.outp("m2", "A1_conditions.yaml")):
        key = f"{r['property_id']}|{r['condition']}"
        if key in seen:
            continue
        seen.add(key)
        units.append({"unit": key, "property_id": r["property_id"], "condition": r["condition"],
                      "property_text": r["property_text"], "assets": list(r.get("assets") or [])})
    return units


def _coding_mode(ctx: Ctx) -> str:
    """method.m2_coding: 'lift' (default, label-anchored, docs Sec. 4.3) or 'question'
    (v3: asset-agnostic situation-question coding, kept as the ablation)."""
    mode = ctx.cfg.get("method", {}).get("m2_coding", "lift")
    if mode not in ("lift", "question"):
        raise SystemExit(f"method.m2_coding must be 'lift' or 'question', got {mode!r}")
    return mode


# Words that mark a statement as evidence or system state rather than a world fact (rule R1/R4).
# This is a program check on the MODEL OUTPUT, not a prompt: no domain example is shown to the model.
EVIDENCE_WORDS = re.compile(
    r"\b(sensors?|detectors?|alarms?|modes?|thresholds?|predefined|pre-defined|readings?|detected|detects|detection|"
    r"lux|ppm|settings?|configured|user[- ]specified|user[- ]set)\b"
    r"|\b(turned|switched) (on|off)\b|\b(is|are) (on|off)\s*$", re.I)
LIFT_GATES = ("evidence", "counterfactual")


def _lift_check(kind: str, observation: str, latent: str, question: str, still_right, gate: str = "evidence") -> str:
    """Programmatic check of one lift (no LLM). Returns the unit's lift status:
      no_fact      device state / configuration / obligation / unclear, or no fact given
      world_fact   the condition already states a world fact
      lifted       evidence (reading, mode, event) lifted to a world fact
      lift_failed  the 'fact' is still evidence or system state, or repeats the observation,
                   or is written as a question
    gate='evidence' (v4.1, default): the fact must not be a question, must not contain
      evidence/system words (EVIDENCE_WORDS) and must differ from the observation; this
      applies to world_state too (v4 let any unit the model called world_state through,
      e.g. 'home is in away mode'). The counterfactual answer is recorded only.
      Why: v4's counterfactual asks whether acting on the evidence is still right if the
      fact is false. For safety properties a careful coder says yes (precaution under a
      possible false alarm), so 52 of 63 v4 failures were correct lifts rejected by it,
      including 38 of 42 hazard units. That question is gap B1, decided at runtime (M9).
    gate='counterfactual' (v4, ablation): the original rule."""
    if kind in NO_VARIABLE_KINDS or not latent or not question:
        return "no_fact"
    if gate == "counterfactual":
        if kind == "world_state":
            return "world_fact"
        same = _norm(" ".join(latent.split())) == _norm(" ".join(observation.split()))
        return "lift_failed" if (same or still_right is not False) else "lifted"
    if latent.strip().endswith("?") or EVIDENCE_WORDS.search(latent):
        return "lift_failed"
    if kind == "world_state":
        return "world_fact"
    if _norm(" ".join(latent.split())) == _norm(" ".join(observation.split())):
        return "lift_failed"
    return "lifted"


def _code_conditions(ctx: Ctx, units: list[dict], temperature: float = 0.0, seed: int = 0) -> list[dict]:
    """A2a: ONE call per evidence unit.
    lift mode (default): kind, observation (evidence; kept for grounding/M5), reason (why the
      property's action is right under the condition, for its asset), latent fact (what the
      reason depends on), counterfactual, and the situation question the LATENT FACT answers.
      The question is the variable that A2b groups, so dimensions are at the level of the
      facts judgments depend on, not of the sensors that report them. _lift_check decides
      whether a unit enters the grouping; with method.m2_lift_strict (default true) failed
      lifts are kept out and listed.
    question mode (ablation, v3): the question the condition itself answers, asset-agnostic.
    Rule R1: device states, configurations and obligations carry no question. Few-shot
    examples come only from the human codebook (data/vocab/a2_codebook.yaml)."""
    mode = _coding_mode(ctx)
    strict = ctx.cfg.get("method", {}).get("m2_lift_strict", True)
    examples = _codebook(ctx).get("examples", [])
    lift_ver, gate = _lift_cfg(ctx)
    rows = []
    for u in units:
        if mode == "lift":
            sys, user, sch = lift_condition_prompt(u["condition"], u["property_text"], u["assets"], examples, lift_ver)
            task = "lift_condition"
        else:
            sys, user, sch = code_condition_prompt(u["condition"], u["property_text"], examples)
            task = "code_condition"
        o = ctx.llm.json("M2", task, sys, user, sch, temperature=temperature, seed=seed,
                         context={"condition": u["condition"], "property_id": u["property_id"], "assets": u["assets"]})
        kind = o.get("kind") if o.get("kind") in CONDITION_KINDS else "unclear"
        obs = str(o.get("observation", "")).strip()
        q = " ".join(str(o.get("question", "")).split())
        ans = str(o.get("answer", "")).strip()
        latent = " ".join(str(o.get("latent_fact", "")).split()) if mode == "lift" else ""
        still = o.get("counterfactual_still_right") if mode == "lift" else None
        status = _lift_check(kind, obs, latent, q, still, gate) if mode == "lift" else "n/a"
        if kind in NO_VARIABLE_KINDS or status == "no_fact" or (strict and status == "lift_failed"):
            var, value = "", ""
        else:
            var, value = q.lower(), ans
        rows.append({"unit": u["unit"], "property_id": u["property_id"], "property_ids": [u["property_id"]],
                     "condition": u["condition"], "assets": u["assets"], "kind": kind, "observation": obs,
                     "reason": str(o.get("reason", "")).strip(), "latent_fact": latent,
                     "counterfactual_still_right": still, "lift_status": status,
                     "question": q, "answer": ans, "variable": var, "value": value,
                     "world_fact": (latent or f"{q} {ans}").strip() if var else "", "scope": o.get("scope", "home")})
    return rows


def _group_variables(ctx: Ctx, rows: list[dict], temperature: float = 0.0, seed: int = 0,
                     order_seed: str | None = None) -> tuple[dict, dict, dict]:
    """A2b axial coding (merge variable phrases naming the same quantity) + A2c verification
    of every multi-member dimension (one variable? not an asset?), splitting when it fails.
    Returns dims, variable -> dimension id, and the full record."""
    cnt = Counter(r["variable"] for r in rows if r["variable"])
    ex = defaultdict(list)
    for r in rows:
        if r["variable"] and r["value"] and r["value"] not in ex[r["variable"]]:
            ex[r["variable"]].append(r["value"])
    var = sorted(cnt, key=lambda v: (-cnt[v], v))
    if order_seed:
        random.Random(order_seed).shuffle(var)
    lines = [f"{v} (x{cnt[v]}; values e.g. {', '.join(ex[v][:3]) or '-'})" for v in var]
    rec = {"variables": [{"variable": v, "count": cnt[v], "example_values": ex[v][:6]} for v in var]}
    out = ctx.llm.json("M2", "group_variables", *group_variables_prompt(lines), temperature=temperature, seed=seed,
                       context={"variables": var})
    assets = {a["id"] for a in ctx.asset_list}
    v2d, dims, dup = {}, {}, []
    for g in out.get("groups", []):
        did = _norm(g.get("id", "")) or "dimension"
        while did in dims:
            did += "_x"
        mem = []
        for i in g.get("members", []):
            if isinstance(i, int) and 0 <= i < len(var):
                if var[i] in v2d:
                    dup.append(var[i])
                else:
                    v2d[var[i]] = did
                    mem.append(var[i])
        if mem:
            dims[did] = {"id": did, "desc": g.get("definition", ""), "variables": mem, "origin": "grouped"}
    for v in var:                                          # every variable ends up in exactly one dimension
        if v not in v2d:
            did = _norm(v) or "dimension"
            while did in dims:
                did += "_x"
            dims[did] = {"id": did, "desc": v, "variables": [v], "origin": "unassigned by the grouping call"}
            v2d[v] = did
    checks = []
    for did in list(dims):
        d = dims[did]
        issues = [f"named after protected asset '{a}'" for a in assets if a in did.split("_")]
        if len(d["variables"]) < 2:
            d["verification"] = {"single_variable": True, "checked": False, "issues": issues}
            continue
        o = ctx.llm.json("M2", "check_dimension", *check_dimension_prompt(did, d["desc"], d["variables"]),
                         temperature=temperature, seed=seed, context={"dim": did, "members": d["variables"]})
        split = o.get("split") or []
        idx = [i for g in split for i in g.get("members", []) if isinstance(i, int)]
        valid = (not o.get("single_variable") and len(split) >= 2 and sorted(idx) == list(range(len(d["variables"]))))
        checks.append({"dimension": did, "members": d["variables"], "single_variable": bool(o.get("single_variable")),
                       "reason": o.get("reason", ""), "split_applied": valid, "issues": issues})
        if valid:
            del dims[did]
            for g in split:
                nid = _norm(g.get("id", "")) or did
                while nid in dims:
                    nid += "_x"
                mem = [d["variables"][i] for i in g["members"]]
                dims[nid] = {"id": nid, "desc": f"{d['desc']} (split: {'; '.join(mem)})", "variables": mem,
                             "origin": f"split of {did}", "verification": {"single_variable": True, "checked": True,
                                                                          "issues": issues}}
                for v in mem:
                    v2d[v] = nid
        else:
            d["verification"] = {"single_variable": bool(o.get("single_variable")), "checked": True,
                                 "reason": o.get("reason", ""), "issues": issues}
    rec.update({"grouping_answer": out.get("groups", []), "duplicate_assignments": dup, "verification": checks})
    return dims, v2d, rec


DEPENDENT = ("exclusive", "a_implies_b", "b_implies_a")
# Factor name = <subject prefix>_<aspect>. The prefix is the runtime grounding scope:
# home/outdoor/person -> one value for the home (scope global), zone -> one value per room,
# evaluated in the room where the action's effect acts (scope zone).
SUBJECT_PREFIX = {"home": "home", "room": "zone", "outdoor": "outdoor", "person": "person"}


def _grouping_mode(ctx: Ctx) -> str:
    """method.m2_grouping: 'relation' (default; slots from pairwise logical relations of facts,
    docs Sec. 4.3 operator G) or 'llm' (one grouping call + one-question check, the v3 way)."""
    mode = ctx.cfg.get("method", {}).get("m2_grouping", "relation")
    if mode not in ("relation", "llm"):
        raise SystemExit(f"method.m2_grouping must be 'relation' or 'llm', got {mode!r}")
    return mode


def _slots_from_relations(facts: list[str], rel: dict) -> dict:
    """Deterministic slot construction from pairwise relations (no LLM).
    facts: fact keys of ONE subject. rel: {(a, b): relation} for queried pairs (a < b).
      1. 'same' pairs are merged (union-find) into values;
      2. values linked by 'exclusive' / 'implies' form one candidate slot (connected component);
      3. a slot must be a set of mutually DEPENDENT values: if two values in a component were
         judged 'independent', the component mixes two aspects of the situation and is split
         greedily into cliques of the dependency graph; every such case is reported.
    Returns {"slots": [[fact, ...], ...], "violations": [...], "splits": [...], "implications": [...]}."""
    parent = {f: f for f in facts}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (a, b), r in sorted(rel.items()):
        if r == "same":
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    cl = defaultdict(list)
    for f in facts:
        cl[find(f)].append(f)
    roots = sorted(cl)
    dep, ind, impl = defaultdict(set), set(), []
    for (a, b), r in sorted(rel.items()):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if r in DEPENDENT:
            dep[ra].add(rb)
            dep[rb].add(ra)
            if r != "exclusive":
                impl.append({"finer": a if r == "a_implies_b" else b, "coarser": b if r == "a_implies_b" else a})
        elif r == "independent":
            ind.add(frozenset((ra, rb)))
    seen, comps = set(), []
    for r0 in roots:                                   # connected components over dependency edges
        if r0 in seen:
            continue
        comp, stack = [], [r0]
        seen.add(r0)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in sorted(dep[x]):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(sorted(comp))
    slots, violations, splits = [], [], []
    for comp in comps:
        bad = [sorted(pair) for pair in ind if pair <= set(comp)]
        if not bad:
            slots.append(comp)
            continue
        violations.append({"component": [m for x in comp for m in cl[x]],
                           "independent_pairs": [[cl[a][0], cl[b][0]] for a, b in sorted(bad)]})
        left = sorted(comp, key=lambda x: (-len(dep[x] & set(comp)), x))
        parts = []
        while left:                                    # greedy clique cover of the dependency graph
            clique = [left.pop(0)]
            for y in list(left):
                if all(y in dep[c] for c in clique):
                    clique.append(y)
                    left.remove(y)
            parts.append(sorted(clique))
        splits.append({"from": [m for x in comp for m in cl[x]], "into": [[m for x in pt for m in cl[x]] for pt in parts]})
        slots.extend(parts)
    return {"slots": [[m for x in s for m in cl[x]] for s in slots], "violations": violations,
            "splits": splits, "implications": impl}


def _group_relations(ctx: Ctx, rows: list[dict], temperature: float = 0.0, seed: int = 0,
                     order_seed: str | None = None) -> tuple[dict, dict, dict]:
    """A2b by pairwise relations (operator G). Every coded fact gets a subject (its scope:
    home / room / outdoor / person); facts of different subjects are never in one slot.
    For each subject, the relation of every PAIR of facts is asked (one small call each)
    when the number of pairs is within method.m2_pair_budget; above it, candidate pairs come
    from one grouping call (pairs inside a candidate group, plus all cross pairs of two
    groups whose first members turn out dependent). _slots_from_relations then builds the
    slots without any LLM; each slot is named by one call (the name changes nothing)."""
    budget = ctx.cfg.get("method", {}).get("m2_pair_budget", 2000)
    assets = {a["id"] for a in ctx.asset_list}
    text, subj_of = {}, {}
    for r in rows:
        if not r["variable"]:
            continue
        subj = r.get("scope") if r.get("scope") in ("home", "room", "outdoor", "person") else "home"
        key = f"{subj}|{_norm(r['world_fact'])}"
        r["variable"] = key
        text.setdefault(key, r["world_fact"])
        subj_of[key] = subj
    rec = {"pair_budget": budget, "subjects": {}, "relations": [], "violations": [], "splits": [], "implications": []}
    dims, v2d = {}, {}
    for subj in sorted(set(subj_of.values())):
        facts = sorted(k for k, s in subj_of.items() if s == subj)
        if order_seed:
            random.Random(f"{order_seed}:{subj}").shuffle(facts)
        n = len(facts)
        rel = {}

        def ask(a, b):
            a, b = min(a, b), max(a, b)
            if (a, b) in rel:
                return rel[(a, b)]
            o = ctx.llm.json("M2", "fact_relation", *fact_relation_prompt(subj, text[a], text[b]),
                             temperature=temperature, seed=seed, context={"subject": subj, "a": text[a], "b": text[b]})
            r_ = o.get("relation") if o.get("relation") in FACT_RELATIONS else "independent"
            rel[(a, b)] = r_
            rec["relations"].append({"subject": subj, "a": text[a], "b": text[b], "relation": r_,
                                     "raw": o.get("relation"), "reason": o.get("reason", "")})
            return r_
        if n * (n - 1) // 2 <= budget:
            mode = "exhaustive"
            for i in range(n):
                for j in range(i + 1, n):
                    ask(facts[i], facts[j])
        else:
            mode = "candidate"
            out = ctx.llm.json("M2", "group_variables", *group_variables_prompt([text[f] for f in facts]),
                               temperature=temperature, seed=seed, context={"variables": [text[f] for f in facts]})
            groups, used = [], set()
            for g in out.get("groups", []):
                mem = [facts[i] for i in g.get("members", []) if isinstance(i, int) and 0 <= i < n and facts[i] not in used]
                used.update(mem)
                if mem:
                    groups.append(mem)
            groups += [[f] for f in facts if f not in used]
            for g in groups:
                for i in range(len(g)):
                    for j in range(i + 1, len(g)):
                        ask(g[i], g[j])
            for gi in range(len(groups)):
                for gj in range(gi + 1, len(groups)):
                    if ask(groups[gi][0], groups[gj][0]) in DEPENDENT:
                        for a in groups[gi]:
                            for b in groups[gj]:
                                ask(a, b)
        res = _slots_from_relations(facts, rel)
        rec["subjects"][subj] = {"facts": n, "pairs_asked": len(rel), "pair_mode": mode,
                                 "relations": dict(Counter(rel.values())), "slots": len(res["slots"])}
        rec["violations"] += [{"subject": subj, **v} for v in res["violations"]]
        rec["splits"] += [{"subject": subj, **s} for s in res["splits"]]
        rec["implications"] += [{"subject": subj, "finer": text[x["finer"]], "coarser": text[x["coarser"]]}
                                for x in res["implications"]]
        for members in res["slots"]:
            o = ctx.llm.json("M2", "name_slot", *name_slot_prompt(subj, [text[m] for m in members]),
                             temperature=temperature, seed=seed, context={"subject": subj, "facts": [text[m] for m in members]})
            pre = SUBJECT_PREFIX.get(subj, subj)
            words = [w for w in _norm(o.get("aspect", "")).split("_") if w and w not in (pre, subj)][:3]
            did = f"{pre}_{'_'.join(words) or 'aspect'}"
            while did in dims:
                did += "_x"
            issues = [f"named after protected asset '{a}'" for a in assets if a in did.split("_")]
            dims[did] = {"id": did, "desc": o.get("question", ""), "subject": subj, "variables": [text[m] for m in members],
                         "origin": "relation grouping", "verification": {"single_variable": True, "checked": True,
                                                                         "method": "pairwise dependency", "issues": issues}}
            for m in members:
                v2d[m] = did
    return dims, v2d, rec


def _a2_once(ctx: Ctx, units, temperature=0.0, seed=0, order_seed=None):
    rows = _code_conditions(ctx, units, temperature, seed)
    if _grouping_mode(ctx) == "relation":
        dims, v2d, rec = _group_relations(ctx, rows, temperature, seed, order_seed)
    else:
        dims, v2d, rec = _group_variables(ctx, rows, temperature, seed, order_seed)
    for r in rows:
        if r["variable"]:
            r["slot"] = v2d.get(r["variable"])
        elif r["kind"] in NO_VARIABLE_KINDS:
            r["slot"] = r["kind"]
        elif r["lift_status"] == "lift_failed":
            r["slot"] = "lift_failed"
        else:
            r["slot"] = "no_variable_given"
    return rows, dims, rec


def run_dims(ctx: Ctx, runs: int = 1) -> dict:
    """A2: dimensions discovered from the corpus as WORLD VARIABLES.
      A2a open coding   one call per condition: kind, world fact, variable, value, scope;
      A2b axial coding  one call merging the (short) list of variable phrases;
      A2c verification  one call per multi-member dimension: one variable? else split.
    Every condition is coded independently, so there is no 'uncovered pool' and no
    stopping rule; quality is measured by the check sheet and by stability.
    Stability: runs 1..R-1 repeat A2a-c with temperature 0.7 and other seeds (and a
    shuffled variable list); reported: kind agreement (kappa), ARI of the condition ->
    dimension labelling vs run 0, and per-dimension recurrence (Jaccard >= 0.5)."""
    units = _units(ctx)
    keys = [u["unit"] for u in units]
    mode = _coding_mode(ctx)
    rows, dims, rec = _a2_once(ctx, units)
    for did, d in dims.items():
        mine = [r for r in rows if r["slot"] == did]
        d.update({"kind": "situation", "n_conditions": len(mine),
                  "scope": SCOPE_MAP.get(Counter(r["scope"] for r in mine).most_common(1)[0][0], "global") if mine else "global",
                  "property_ids": sorted({x for r in mine for x in r["property_ids"]})})
    kinds = Counter(r["kind"] for r in rows)
    excluded = Counter(r["slot"] for r in rows if not r["variable"])
    slots = sorted(dims.values(), key=lambda d: -d["n_conditions"]) + \
        [{"id": k, "kind": "excluded", "desc": f"conditions coded as {k} (no world variable)", "n_conditions": n}
         for k, n in sorted(excluded.items())]
    lift = Counter(r["lift_status"] for r in rows)
    lift_ver, gate = _lift_cfg(ctx)
    cb = _codebook(ctx)
    res = {"version": (f"S-induced-{lift_ver} ({mode} coding, gate={gate})" if mode == "lift"
                       else "S-induced-v3 (question coding, ablation)"),
           "codebook": cb.get("version"),
           "coding": mode, "n_conditions": len(rows), "n_units": len(rows),
           "n_distinct_conditions": len({r["condition"] for r in rows}), "kinds": dict(kinds),
           "lift_status": dict(lift) if mode == "lift" else None,
           "lift_failed": [{"unit": r["unit"], "kind": r["kind"], "observation": r["observation"],
                            "latent_fact": r["latent_fact"], "counterfactual_still_right": r["counterfactual_still_right"]}
                           for r in rows if r["lift_status"] == "lift_failed"],
           "grouping": _grouping_mode(ctx),
           "relation_grouping": ({"subjects": rec.get("subjects"), "mixed_aspect_components": len(rec.get("violations", [])),
                                  "splits": rec.get("splits"), "implications": len(rec.get("implications", []))}
                                 if _grouping_mode(ctx) == "relation" else None),
           "with_world_variable": sum(1 for r in rows if r["variable"]), "n_dimensions": len(dims),
           "unverified_or_failed": [d["id"] for d in dims.values() if not d.get("verification", {}).get("single_variable", True)],
           "asset_named": [d["id"] for d in dims.values() if d.get("verification", {}).get("issues")],
           "slots": slots, "coverage": dict(Counter(r["slot"] for r in rows))}
    save_yaml(ctx, "m2-dims", res, "m2", "A2_dims.yaml")
    save_yaml(ctx, "m2-dims", rows, "m2", "A2_coding.yaml")
    save_yaml(ctx, "m2-dims", rec, "m2", "A2_grouping.yaml")
    if rec.get("relations"):
        rs = _sample(rec["relations"], 60, lambda r: r["relation"], "A2rel")
        save_csv(ctx, "m2-dims", ["subject", "fact_a", "fact_b", "relation", "reason",
                                  "HUMAN_relation_ok(1/0)", "HUMAN_relation(same/exclusive/a_implies_b/b_implies_a/independent)", "note"],
                 [[r["subject"], r["a"], r["b"], r["relation"], r["reason"], "", "", ""] for r in rs],
                 "annotate", "m2_A2_relations_CHECK.csv")
    sample = _sample(rows, 50, lambda r: r["slot"], "A2")
    save_csv(ctx, "m2-dims", ["property_id", "condition", "assets", "kind", "observation", "reason", "latent_fact",
                              "counterfactual_still_right", "lift_status", "question", "dimension",
                              "HUMAN_kind_ok(1/0)", "HUMAN_lift_ok(1/0)", "HUMAN_dimension_ok(1/0)", "HUMAN_better", "note"],
             [[r["property_id"], r["condition"], ";".join(r["assets"]), r["kind"], r["observation"], r["reason"],
               r["latent_fact"], r["counterfactual_still_right"], r["lift_status"], r["question"], r["slot"],
               "", "", "", "", ""] for r in sample],
             "annotate", "m2_A2_coding_CHECK.csv")
    if runs > 1:
        lab0 = {r["unit"]: r["slot"] for r in rows}
        kind0 = {r["unit"]: r["kind"] for r in rows}
        sets0 = {did: {r["unit"] for r in rows if r["slot"] == did} for did in dims}
        stab = {"runs": runs, "per_run": [], "dimension_recurrence": {}}
        hits = defaultdict(list)
        for i in range(1, runs):
            r_i, d_i, rec_i = _a2_once(ctx, units, temperature=0.7, seed=i, order_seed=f"{ctx.seed}:A2:{i}")
            save_yaml(ctx, "m2-dims", {"dims": list(d_i.values()), "grouping": rec_i}, "m2", "A2_runs", f"run{i}_dims.yaml")
            save_yaml(ctx, "m2-dims", r_i, "m2", "A2_runs", f"run{i}_coding.yaml")
            lab = {r["unit"]: r["slot"] for r in r_i}
            kd = {r["unit"]: r["kind"] for r in r_i}
            sets_i = defaultdict(set)
            for r in r_i:
                if r["variable"]:
                    sets_i[r["slot"]].add(r["unit"])
            stab["per_run"].append({"run": i, "n_dimensions": len(d_i),
                                    "ari_vs_run0": _ari([lab0[t] for t in keys], [lab[t] for t in keys]),
                                    "kind_kappa_vs_run0": _kappa([kind0[t] for t in keys], [kd[t] for t in keys])})
            for did, s0 in sets0.items():
                best = max(((len(s0 & s) / max(1, len(s0 | s)), k) for k, s in sets_i.items()), default=(0.0, None))
                hits[did].append({"run": i, "best_match": best[1], "jaccard": round(best[0], 3)})
        for did, h in hits.items():
            stab["dimension_recurrence"][did] = {"recurrence": round(sum(x["jaccard"] >= 0.5 for x in h) / len(h), 3),
                                                 "matches": h}
        stab["mean_ari"] = round(float(np.mean([x["ari_vs_run0"] for x in stab["per_run"]])), 4)
        stab["mean_kind_kappa"] = round(float(np.mean([x["kind_kappa_vs_run0"] or 0 for x in stab["per_run"]])), 4)
        save_yaml(ctx, "m2-dims", stab, "m2", "A2_stability.yaml")
        res["stability"] = {"mean_ari": stab["mean_ari"], "mean_kind_kappa": stab["mean_kind_kappa"],
                            "recurrence": {k: v["recurrence"] for k, v in stab["dimension_recurrence"].items()}}
    log_result(ctx, "m2-dims", ctx.outp("m2", "A2_dims.yaml"),
               {"n_dimensions": res["n_dimensions"], "lift_status": res["lift_status"], "kinds": res["kinds"],
                "with_world_variable": res["with_world_variable"],
                "mean_ari": (res.get("stability") or {}).get("mean_ari"),
                "mean_kind_kappa": (res.get("stability") or {}).get("mean_kind_kappa")},
               method_version=res["version"], n=res["n_units"], codebook=str(cb.get("version")), note=f"runs={runs}")
    # A3 is derived from A2: once A2 changes, the old A3 files no longer describe it.
    if os.path.exists(ctx.outp("m2", "A3_values_report.yaml")):
        with open(ctx.outp("m2", "A3_STALE.txt"), "w", encoding="utf-8") as f:
            f.write(f"A2 was re-run ({res['version']}); A3_* files predate it. Run m2-values before using them.\n")
        print("NOTE: out/m2/A3_* now describe an older A2 -> run m2-values (marker: out/m2/A3_STALE.txt)")
    return res


def run_dev(ctx: Ctx) -> dict:
    """Score A2 against the held-out part of the human codebook (never shown to the model):
    kind accuracy, 'has a situation question' accuracy, and ARI between the produced
    condition -> dimension grouping and the human grouping (all dev items, seen-only,
    unseen-only). Run after every prompt change BEFORE trusting a full A2 run."""
    cb = _codebook(ctx)
    coding = load_yaml(ctx.outp("m2", "A2_coding.yaml"))
    a2 = load_yaml(ctx.outp("m2", "A2_dims.yaml"))
    by_prop = defaultdict(list)
    for r in coding:
        for pid in r.get("property_ids", []):
            by_prop[pid].append(r)
    rows = []
    for d in cb.get("dev", []):
        cands = by_prop.get(d["property"], [])
        words = set(d["condition"].lower().replace("_", " ").split())
        best = max(cands, key=lambda r: len(words & set(r["condition"].lower().replace("_", " ").split())), default=None)
        if best is None:
            rows.append({**d, "found": False})
            continue
        rows.append({**d, "found": True, "matched_condition": best["condition"], "got_kind": best["kind"],
                     "got_dim": best["slot"] if best["variable"] else "none", "got_question": best.get("question", "")})
    f = [r for r in rows if r["found"]]

    def ari(sub):
        return _ari([r["dim"] for r in sub], [r["got_dim"] for r in sub]) if len(sub) > 1 else None
    res = {"codebook": cb.get("version"), "dev_items": len(rows), "found": len(f),
           "kind_accuracy": round(sum(r["kind"] == r["got_kind"] for r in f) / max(1, len(f)), 3),
           "has_question_accuracy": round(sum((r["dim"] != "none") == (r["got_dim"] != "none") for r in f) / max(1, len(f)), 3),
           "ari_all": ari(f), "ari_seen_dims": ari([r for r in f if r["seen"]]),
           "ari_unseen_dims": ari([r for r in f if not r["seen"]]), "items": rows}
    res["scored_A2"] = {"version": a2.get("version"), "A2_coding_sha": _file_sha(ctx.outp("m2", "A2_coding.yaml"))}
    save_yaml(ctx, "m2-dev", res, "eval", "m2_A2_dev.yaml")
    # keep every dev score (the prompt-development history), named by A2 version and model
    tag = _norm(f"{a2.get('version', '')}_{ctx.cfg['llm']['model']}")[:80]
    hist = save_yaml(ctx, "m2-dev", res, "eval", "dev_history", f"m2_A2_dev_{tag}_{res['scored_A2']['A2_coding_sha']}.yaml")
    log_result(ctx, "m2-dev", hist, {k: res[k] for k in ("kind_accuracy", "has_question_accuracy", "ari_all",
                                                          "ari_seen_dims", "ari_unseen_dims")},
               method_version=str(a2.get("version")), n=res["found"], codebook=str(cb.get("version")))
    return res


# ------------------------------------------------------------------ A3
def run_values(ctx: Ctx) -> dict:
    """A3: values of every A2 dimension.
      A3a one call per dimension: value set (mutually exclusive, with a baseline) from the
          observed values;
      A3b one closed-choice call per distinct coded fact: which value it states (enum, or
          none_fits). Coverage is therefore exact and every unassigned fact is listed.
    One dimension = one variable (A2c), so one dimension = one factor (no max_factors).
    The action-derived factor effect_zone is added separately and marked as not from the corpus."""
    a2 = load_yaml(ctx.outp("m2", "A2_dims.yaml"))
    coding = load_yaml(ctx.outp("m2", "A2_coding.yaml"))
    factors, assign, report = [], [], {"per_dimension": {}}
    for d in [s for s in a2["slots"] if s["kind"] == "situation"]:
        rows = [r for r in coding if r["slot"] == d["id"]]
        if not rows:
            continue
        vc = Counter(r["value"] or r["world_fact"] for r in rows)
        phrases = [f"{v} (x{n})" for v, n in vc.most_common()]
        o = ctx.llm.json("M2", "dimension_values", *dimension_values_prompt(d, phrases), context={"dim": d, "phrases": list(vc)})
        values = {}
        for v in o.get("values", []):
            k = _norm(v.get("value", ""))
            if k and k not in values:
                values[k] = v.get("semantics", "")
        if len(values) < 2:
            values.setdefault("none", "Baseline: the dimension's condition does not hold.")
        facts = sorted({(r["world_fact"], r["value"]) for r in rows})
        fmap = {}
        for fact, val in facts:
            a = ctx.llm.json("M2", "assign_value", *assign_value_prompt(d, values, f"{fact} (stated value: {val or '-'})"),
                             context={"dim": d["id"], "values": list(values), "fact": fact, "value": val})
            fmap[(fact, val)] = a.get("value") if a.get("value") in values else None
        per_val = Counter()
        props_val = defaultdict(set)
        for r in rows:
            v = fmap.get((r["world_fact"], r["value"]))
            per_val[v] += 1
            if v:
                props_val[v].update(r.get("property_ids", []))
            assign.append({"dimension": d["id"], "condition": r["condition"], "world_fact": r["world_fact"],
                           "observation": r.get("observation", ""),      # evidence layer: seeds the grounding of this value (M5)
                           "observed_value": r["value"], "value": v, "property_ids": r.get("property_ids", [])})
        covered = sum(n for v, n in per_val.items() if v)
        factors.append({"name": d["id"], "scope": d.get("scope", "global"), "semantics": d.get("desc", ""),
                        "values": values,
                        "provenance": {"dimension": d["id"], "variables": d.get("variables", []),
                                       "n_conditions": len(rows),
                                       "conditions_per_value": {v: per_val.get(v, 0) for v in values},
                                       "property_ids_per_value": {v: sorted(props_val.get(v, ())) for v in values}}})
        report["per_dimension"][d["id"]] = {"n_conditions": len(rows), "values": list(values),
                                            "covered": covered, "coverage": round(covered / len(rows), 3),
                                            "unassigned": sorted({r["condition"] for r in rows
                                                                  if not fmap.get((r["world_fact"], r["value"]))}),
                                            "values_without_evidence": [v for v in values if not per_val.get(v)]}
    for dv in load_yaml(p("data/vocab/slots.yaml")).get("derived", []):
        factors.append({"name": dv["name"], "scope": dv["scope"], "semantics": dv["semantics"], "values": dv["values"],
                        "provenance": {"derived": "action target area (NOT induced from the corpus)"}})
    report["n_conditions_with_variable"] = sum(x["n_conditions"] for x in report["per_dimension"].values())
    report["overall_coverage"] = round(sum(x["covered"] for x in report["per_dimension"].values()) /
                                       max(1, report["n_conditions_with_variable"]), 3)
    report["source_A2"] = {"version": a2.get("version"), "A2_dims_sha": _file_sha(ctx.outp("m2", "A2_dims.yaml"))}
    res = {"version": "F-A3", "schema": a2.get("version"), "source_A2": report["source_A2"],
           "factors": factors, "feasibility": []}
    save_yaml(ctx, "m2-values", res, "m2", "A3_factors_corpus.yaml")
    save_yaml(ctx, "m2-values", report, "m2", "A3_values_report.yaml")
    save_yaml(ctx, "m2-values", assign, "m2", "A3_value_assignment.yaml")
    if os.path.exists(ctx.outp("m2", "A3_STALE.txt")):
        os.remove(ctx.outp("m2", "A3_STALE.txt"))
    return report


# ------------------------------------------------------------------ ablation: top-down slots (Dey 2001)
def run_slots(ctx: Ctx, schema_path: str | None = None, tag: str = "ablation_topdown") -> dict:
    """ABLATION ONLY: dimensions fixed top-down (data/vocab/slots.yaml); conditions classified
    into slots, values induced per slot in one call (the pre-v2 procedure)."""
    schema = load_yaml(schema_path or p("data/vocab/slots.yaml"))
    slots = {s["id"]: s for s in schema["slots"]}
    sdesc = "\n".join(f"- {s['id']} ({s['kind']}): {s['desc']}" for s in schema["slots"])
    texts = _cond_texts(ctx)
    cp = _cond_props(ctx)
    placed = []
    if True:
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
        corp = [c for c in a3["factors"] if ((c.get("provenance") or {}).get("dimension")
                                              or (c.get("provenance") or {}).get("slot")) in known]
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
    # only this stage's own items are replaced; refutation and property evidence is kept
    keep = [i for i in ev["items"] if i.get("source", "elicit") != "elicit" or (i["effect"], i["asset"]) not in queried]
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


# ------------------------------------------------------------------ expansion: new properties
def run_ingest(ctx: Ctx, src: str, allow_heldout: bool = False) -> dict:
    """Expansion trigger 'new property' (docs Sec. 4.6). New properties (CSV with id, source,
    text[, scope]) are first mapped into the CURRENT vocabulary (config paths.factors):
      expressible   -> the vocabulary already covers it: nothing changes (no growth without need);
      inexpressible -> its conditions are lifted (same L operator and _lift_check as A2), each
                       lifted fact is filed under an existing factor or NEW:<name>, and becomes an
                       evidence item (source 'property') for m2-consolidate --base current, where
                       the witness rule (a label signature differing from the parent value)
                       decides whether it becomes a value.
    Held-out properties of the corpus must not be ingested (they measure expressibility);
    this is refused unless allow_heldout."""
    vocab = ctx.vocab
    lib = ctx.effects()
    eids = {e["id"] for e in lib["effects"]}
    aids = {a["id"] for a in ctx.asset_list}
    edesc = "\n".join(f"- {e['id']}: {e['desc']}" for e in lib["effects"])
    B, bpath = _base(ctx, "current")
    dims = [f for f in B["factors"] if f.get("scope") != "action"]
    names = {f["name"] for f in dims}
    ddesc = _dims_desc(dims)
    examples = _codebook(ctx).get("examples", [])
    lift_ver, gate = _lift_cfg(ctx)
    new = [r for r in csv.DictReader(open(src, encoding="utf-8")) if r.get("scope", "in") == "in"]
    split = {r["id"]: r["split"] for r in load_properties(ctx)}
    leak = [r["id"] for r in new if split.get(r["id"]) == "test"]
    if leak and not allow_heldout:
        raise SystemExit(f"refused: {leak} are held-out properties of the corpus (they measure expressibility)")
    rows, items = [], []
    for prop in new:
        out = ctx.llm.json("M2", "map_property", *map_property_prompt(prop["text"], vocab.describe_vocab(), edesc),
                           context={"property": prop})
        effs = [e for e in out.get("effects", []) or [] if e in eids]
        asset = out.get("asset") if out.get("asset") in aids else None
        label = out.get("label") if out.get("label") in ("+", "-") else None
        row = {"property_id": prop["id"], "text": prop["text"], "source": prop.get("source", ""),
               "already_in_corpus": prop["id"] in split, "expressible": False, "missing": out.get("missing", ""),
               "units": []}
        if out.get("expressible"):
            try:
                phi = normalize(out.get("pattern") or {}, vocab)
                if not effs or not asset or not label:
                    raise ValueError("effect/asset/label invalid")
                row.update({"expressible": True, "pattern": pattern_to_json(phi, vocab), "effects": effs,
                            "asset": asset, "label": label})
            except ValueError as e:
                row["missing"] = f"invalid mapping: {e}"
        if not row["expressible"]:
            conds = [c.strip() for c in ctx.llm.json("M2", "extract_predicates", *extract_predicates_prompt(prop["text"]),
                                                      context={"property": prop}).get("conditions", []) if c and c.strip()]
            assets = ctx.llm.json("M2", "property_assets", *property_assets_prompt(prop["text"]),
                                  context={"property": prop}).get("assets", [])
            for c in conds:
                o = ctx.llm.json("M2", "lift_condition", *lift_condition_prompt(c, prop["text"], assets, examples, lift_ver),
                                 context={"condition": c, "property_id": prop["id"], "assets": assets})
                kind = o.get("kind") if o.get("kind") in CONDITION_KINDS else "unclear"
                latent = " ".join(str(o.get("latent_fact", "")).split())
                status = _lift_check(kind, str(o.get("observation", "")).strip(), latent,
                                     " ".join(str(o.get("question", "")).split()), o.get("counterfactual_still_right"), gate)
                u = {"condition": c, "kind": kind, "observation": o.get("observation", ""), "latent_fact": latent,
                     "reason": o.get("reason", ""), "lift_status": status}
                if status in ("lifted", "world_fact"):
                    a = ctx.llm.json("M2", "attribute_dimension", *attribute_dimension_prompt(latent, ddesc),
                                     context={"attribute": latent, "dims": sorted(names)})
                    d = str(a.get("dimension", ""))
                    d = ("NEW:" + _norm(d[4:])) if d.upper().startswith("NEW:") else d
                    u["dimension"] = d
                    u["used"] = (d in names or d.startswith("NEW:")) and bool(a.get("is_world_state"))
                    if u["used"]:
                        pairs = [(e, asset) for e in effs] if (effs and asset) else []
                        for e, v in pairs or [("", asset or (assets[0] if assets else ""))]:
                            items.append({"effect": e, "asset": v, "attribute": latent, "label": label, "dimension": d,
                                          "why": u["reason"], "source": "property", "property_id": prop["id"],
                                          "has_pair": bool(pairs)})
                row["units"].append(u)
        rows.append(row)
    ids = {r["property_id"] for r in rows}
    ev = _evidence(ctx)
    ev["items"] = [i for i in ev["items"] if not (i.get("source") == "property" and i.get("property_id") in ids)] + items
    ev.setdefault("rounds", []).append({"ingest": os.path.relpath(src, p()), "properties": len(rows), "items": len(items)})
    save_yaml(ctx, "m2-ingest", ev, "m2", "B1_elicitation.yaml")
    affected = sorted({i["dimension"] for i in items})
    res = {"base": os.path.relpath(bpath, p()), "base_version": B.get("version"), "src": os.path.relpath(src, p()),
           "properties": len(rows), "expressible_already": sum(r["expressible"] for r in rows),
           "inexpressible": sum(not r["expressible"] for r in rows),
           "lift_status": dict(Counter(u["lift_status"] for r in rows for u in r["units"])),
           "evidence_items": len(items), "items_without_effect_pair": sum(not i["has_pair"] for i in items),
           "affected_dimensions": affected,
           "next": (f"m2-consolidate --base current --dims {' '.join(affected)}" if affected
                    else "nothing to add: the current vocabulary expresses every new property"),
           "per_property": rows}
    save_yaml(ctx, "m2-ingest", res, "m2", "I1_ingest.yaml")
    save_csv(ctx, "m2-ingest", ["property_id", "text", "expressible", "condition", "observation", "latent_fact",
                                "lift_status", "dimension", "HUMAN_expressible(1/0)", "HUMAN_lift_ok(1/0)",
                                "HUMAN_dimension_ok(1/0)", "note"],
             [[r["property_id"], r["text"], int(r["expressible"]), u["condition"], u["observation"], u["latent_fact"],
               u["lift_status"], u.get("dimension", ""), "", "", "", ""]
              for r in rows for u in (r["units"] or [{"condition": "", "observation": "", "latent_fact": "",
                                                      "lift_status": "", "dimension": ""}])],
             "annotate", "m2_I1_ingest_CHECK.csv")
    return {k: v for k, v in res.items() if k != "per_property"}


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
