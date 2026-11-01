"""Versioned, extensible vocabulary (M2 stage D).

The context factors are not fixed a priori: they are the OUTPUT of M2 and become fixed
only when a reviewed candidate is frozen as a version F-<name>. A later round (new
effects, new properties, confirmed refutation counterexamples) produces the next
candidate from the current version; `freeze` records how the new version refines the
parent, `migrate` carries K over and marks the (effect, asset) pairs that must be
rebuilt, and `m3 --only-stale` rebuilds only those.

refines (written by m2-consolidate, checked by the reviewer, stored in the version):
  <factor>:
    split:  {old_value: [child, child, ...]}   # old meaning = union of the children
    merge:  {absorbed_value: kept_value}       # value removed by a merge
    extend: [new_value, ...]                   # new value with no parent (domain grows)
    extend_meaning: {value: [state, ...]}      # an existing value now also covers these states
    added:  true                               # the whole factor is new
"""
from __future__ import annotations

import csv
import os
import time

import yaml

from .common import ROOT, dump_yaml, load_yaml, p
from .manifest import record, save_yaml
from .model import Binding, Vocab, normalize
from .roughset import Tables


def _vals(f):
    v = f["values"]
    return [str(k) for k in (v if isinstance(v, dict) else v)]


def _factors(d):
    return {f["name"]: f for f in d["factors"]}


def diff(old: dict, new: dict) -> dict:
    fo, fn = _factors(old), _factors(new)
    out = {"added_factors": [n for n in fn if n not in fo], "removed_factors": [n for n in fo if n not in fn],
           "changed": {}}
    for n in fn:
        if n in fo:
            a, b = set(_vals(fo[n])), set(_vals(fn[n]))
            if a != b:
                out["changed"][n] = {"added_values": sorted(b - a), "removed_values": sorted(a - b)}
    return out


def _check_refines(old: dict, new: dict, ref: dict) -> list[str]:
    fo, fn = _factors(old), _factors(new)
    errs = []
    for n, r in (ref or {}).items():
        if n not in fn:
            errs.append(f"refines: unknown factor {n}")
            continue
        newd = set(_vals(fn[n]))
        oldd = set(_vals(fo[n])) if n in fo else set()
        for u, ch in (r.get("split") or {}).items():
            if u not in oldd:
                errs.append(f"refines.{n}.split: {u} not an old value")
            if not set(map(str, ch)) <= newd:
                errs.append(f"refines.{n}.split.{u}: children {set(ch) - newd} not in new domain")
        for w, u in (r.get("merge") or {}).items():
            if w not in oldd or str(u) not in newd:
                errs.append(f"refines.{n}.merge: {w}->{u} invalid")
        for w in r.get("extend") or []:
            if str(w) not in newd:
                errs.append(f"refines.{n}.extend: {w} not in new domain")
    return errs


def freeze(ctx, src: str, name: str, dest: str | None = None, parent: str | None = None,
           check_provenance: bool = True) -> dict:
    """Validate a reviewed candidate and freeze it as version F-<name>."""
    d = load_yaml(src)
    parent_path = parent or p(ctx.cfg["paths"]["factors"])
    old = load_yaml(parent_path)
    errs = []
    for f in d["factors"]:
        vals = f.get("values") or {}
        if len(vals) < 2:
            errs.append(f"{f['name']}: fewer than 2 values")
        if isinstance(vals, dict) and any(not str(s or "").strip() for s in vals.values()):
            errs.append(f"{f['name']}: a value has no semantics")
        if check_provenance and f.get("scope") != "action" and not (f.get("provenance") or f.get("value_provenance")):
            errs.append(f"{f['name']}: no provenance (every factor must cite its evidence)")
    ref = d.get("refines") or {}
    errs += _check_refines(old, d, ref)
    dest = dest or p("data/vocab", f"factors_{name}.yaml")
    out = {k: v for k, v in d.items() if k not in ("status", "version", "parent", "refines", "frozen_at")}
    out = {"version": f"F-{name}", "parent": old.get("version", "?"),
           "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "refines": ref, **out}
    tmp = dest + ".tmp"
    dump_yaml(out, tmp)
    try:
        V = Vocab.load(tmp)
        n_sigma = len(V.sigma())
        if not n_sigma:
            errs.append("Sigma is empty (feasibility rules exclude everything)")
    except Exception as e:  # noqa: BLE001
        errs.append(f"vocabulary does not load: {e}")
        n_sigma = 0
    if errs:
        os.remove(tmp)
        raise SystemExit("freeze refused:\n  " + "\n  ".join(errs))
    os.replace(tmp, dest)
    record(ctx, "m2-freeze", dest)
    dif = diff(old, out)
    undeclared = [n for n in dif["changed"] if n not in ref] + [n for n in dif["added_factors"] if n not in ref]
    rec = {"version": out["version"], "parent": out["parent"], "file": os.path.relpath(dest, ROOT), "source": src,
           "n_factors": len(out["factors"]), "sigma": n_sigma, "product": V.product_size(),
           "diff_vs_parent": dif, "provenance_check": check_provenance, "refines": ref, "changes_without_refines_entry": undeclared}
    save_yaml(ctx, "m2-freeze", rec, "m2", "versions", f"{out['version']}.yaml")
    # rule-check sheet for two annotators (Cohen's kappa in m2-check-score)
    rows = []
    for f in out["factors"]:
        vals = f["values"] if isinstance(f["values"], dict) else {v: "" for v in f["values"]}
        src_kind = ("derived" if f.get("scope") == "action" else "elicited" if f.get("new_dimension")
                    else "corpus")
        rows.append([f["name"], f.get("scope", ""), f.get("semantics", ""), "; ".join(f"{k} = {v}" for k, v in vals.items()),
                     src_kind, "", "", "", "", "", "", "", "", ""])
    path = ctx.outp("annotate", f"m2_rules_CHECK_{out['version']}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["factor", "scope", "semantics", "values", "source",
                    "A_R1_world_state(1/0)", "A_R2_one_variable(1/0)", "A_R3_exclusive_exhaustive(1/0)", "A_R4_cause_not_reading(1/0)",
                    "B_R1_world_state(1/0)", "B_R2_one_variable(1/0)", "B_R3_exclusive_exhaustive(1/0)", "B_R4_cause_not_reading(1/0)",
                    "note"])
        w.writerows(rows)
    record(ctx, "m2-freeze", path)
    return rec


def migrate(ctx, old_vocab_path: str) -> dict:
    """Carry K (built under the old version) over to the current version ctx.vocab and
    mark the pairs that must be rebuilt. Old K is kept as spec_<old version>.yaml."""
    new_v = ctx.vocab
    new_d = load_yaml(p(ctx.cfg["paths"]["factors"]))
    old_d = load_yaml(old_vocab_path)
    ref = new_d.get("refines") or {}
    dif = diff(old_d, new_d)
    changed = set(dif["changed"]) | set(dif["added_factors"]) | set(dif["removed_factors"]) | set(ref)
    spec = load_yaml(ctx.spec_path())
    backup = ctx.outp("m3", f"spec_{old_d.get('version', 'old')}.yaml")
    dump_yaml(spec, backup)
    record(ctx, "vocab-migrate", backup)
    stale: dict[str, set] = {}

    def mark(pair, why):
        stale.setdefault(pair, set()).add(why)

    for f in new_d["factors"]:      # evidence pairs of changed factors
        if f["name"] in changed:
            for rp in f.get("relevant_pairs", []):
                e, a = [x.strip() for x in rp.split(" x ")]
                mark(f"{e}|{a}", f"evidence pair of changed factor {f['name']}")
    out_b = []
    for bd in spec["bindings"]:
        pair = f"{bd['effect']}|{bd['asset']}"
        pat = {}
        for n, vals in bd["pattern"].items():
            if n not in new_v.by_name:
                mark(pair, f"factor {n} removed")
                continue
            r = ref.get(n, {})
            split, merge = r.get("split") or {}, r.get("merge") or {}
            dom = set(new_v.domain(n))
            mapped = set()
            for u in map(str, vals):
                if u in split:
                    mapped |= set(map(str, split[u]))
                elif u in merge:
                    mapped.add(str(merge[u]))
                elif u in dom:
                    mapped.add(u)
                else:
                    mark(pair, f"value {n}={u} removed without a refines entry")
            if n in changed:
                mark(pair, f"binding constrains changed factor {n}")
            if not mapped:
                mark(pair, f"pattern on {n} became empty")
                pat = None
                break
            pat[n] = sorted(mapped, key=list(new_v.domain(n)).index)
        if pat is None:
            continue
        try:
            phi = normalize(pat, new_v)
        except ValueError as e:
            mark(pair, f"invalid after migration: {e}")
            continue
        meta = dict(bd.get("meta") or {})
        meta["migrated_from"] = old_d.get("version")
        out_b.append(Binding(phi, bd["effect"], bd["asset"], bd["label"], bd.get("harm", ""), meta))
    # a migrated pair must still have disjoint '+' and '-' regions
    T = Tables(new_v, out_b)
    for pair, bs in T.bindings.items():
        plus = [T.mask(b.phi) for b in bs if b.label == "+"]
        minus = [T.mask(b.phi) for b in bs if b.label == "-"]
        if any((x & y).any() for x in plus for y in minus):
            mark(f"{pair[0]}|{pair[1]}", "'+' and '-' regions overlap after migration")
    for b in out_b:
        if f"{b.effect}|{b.asset}" in stale:
            b.meta["stale"] = True
    new_spec = {"vocab_version": new_v.version, "effects_version": spec.get("effects_version"),
                "migrated_from": old_d.get("version"), "bindings": [b.to_json(new_v) for b in out_b]}
    dump_yaml(new_spec, ctx.spec_path())
    record(ctx, "vocab-migrate", ctx.spec_path())
    res = {"old_version": old_d.get("version"), "new_version": new_v.version, "changed_factors": sorted(changed),
           "bindings_before": len(spec["bindings"]), "bindings_after": len(out_b),
           "pairs_with_bindings": len({f"{b['effect']}|{b['asset']}" for b in spec["bindings"]}),
           "stale_pairs": [{"pair": k, "reasons": sorted(v)} for k, v in sorted(stale.items())]}
    save_yaml(ctx, "vocab-migrate", res, "m3", "stale_pairs.yaml")
    return res


def snapshot(ctx, tag: str) -> str:
    """Copy every result needed to reproduce / audit a run into runs/<date>_<tag>/."""
    import shutil
    dst = p("runs", f"{time.strftime('%Y-%m-%d')}_{tag}")
    os.makedirs(dst, exist_ok=True)
    for sub in ("m1", "m2", "m3", "eval", "annotate", "baselines"):
        s = os.path.join(ctx.out, sub)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(dst, "out", sub), dirs_exist_ok=True)
    for f in ("ledger.csv", "MANIFEST.csv"):
        if os.path.exists(os.path.join(ctx.out, f)):
            shutil.copy2(os.path.join(ctx.out, f), os.path.join(dst, "out", f))
    shutil.copytree(p("data", "vocab"), os.path.join(dst, "data_vocab"), dirs_exist_ok=True)
    shutil.copy2(p("ctxguard", "prompts.py"), os.path.join(dst, "prompts.py"))
    with open(os.path.join(dst, "config.yaml"), "w") as f:
        yaml.safe_dump(ctx.cfg, f, sort_keys=False)
    return dst
