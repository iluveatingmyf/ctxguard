"""M4  Audit of K against human labels  (Sec. VI-A, RQ1)

Sampling: items = (effect, asset, abstract situation) drawn from Sigma for the
pairs K covers, stratified by K's predicted label (-, +, 0) in equal thirds so
rare '-' cells are measured. Population-weighted accuracy re-weights strata by
their share in Sigma (unbiased under stratification).
The annotator sees the situation and the effect/asset, never the prediction.
Baselines on the same items:
  B-K1 direct generation  (one free-form constraint list per pair)
  B-K2 per-cell labelling (one LLM call per cell; accurate but its cost to
       cover Sigma is extrapolated: calls x |Sigma| x pairs)
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict

import numpy as np

from .common import Ctx, dump_yaml, load_yaml
from .effects import effect_index
from .model import Binding, index_spec, label_of
from .prompts import point_label_prompt

LABELS = ["-", "+", "0"]


def export_audit(ctx: Ctx, n: int | None = None) -> str:
    n = n or ctx.cfg["eval"]["audit_items"]
    vocab = ctx.vocab
    spec = [Binding.from_json(b, vocab) for b in load_yaml(ctx.spec_path())["bindings"]]
    idx = index_spec(spec)
    rng = ctx.rng("audit")
    sigma = vocab.sigma()
    pairs = sorted(idx)
    strata = defaultdict(list)
    for pair in pairs:
        for c in rng.sample(sigma, min(len(sigma), 400)):
            strata[label_of(idx, pair[0], pair[1], c, vocab)].append((pair, c))
    share = Counter()
    for pair in pairs:
        for c in sigma:
            share[label_of(idx, pair[0], pair[1], c, vocab)] += 1
    items = []
    for lab in LABELS:
        pool = strata.get(lab, [])
        items += [(lab, it) for it in rng.sample(pool, min(len(pool), n // 3))]
    rng.shuffle(items)
    eidx = effect_index(ctx.effects())
    key = []
    path = ctx.outp("annotate", "audit_K.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "effect", "effect_meaning", "asset", "situation",
                    "GOLD_label(-/+/0)", "GOLD_harm_or_reason", "ANNOTATOR_B_label(optional)"])
        for i, (lab, ((e, v), c)) in enumerate(items):
            w.writerow([i, e, eidx[e]["desc"], v, vocab.describe_cell(c), "", "", ""])
            key.append({"item": i, "effect": e, "asset": v, "cell": list(c), "pred": lab})
    total = sum(share.values())
    dump_yaml({"items": key, "stratum_share": {k: v / total for k, v in share.items()}},
              ctx.outp("annotate", "audit_K_key.yaml"))
    return path


def _prf(gold, pred):
    out = {}
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[lab] = {"precision": round(pr, 3), "recall": round(rc, 3),
                    "f1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0, "support": tp + fn}
    out["accuracy"] = round(float(np.mean([g == p for g, p in zip(gold, pred)])), 3) if gold else None
    out["kappa"] = round(kappa(gold, pred), 3) if gold else None
    return out


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score_audit(ctx: Ctx, annotated_csv: str, run_point_baseline: bool = True) -> dict:
    vocab = ctx.vocab
    keyd = load_yaml(ctx.outp("annotate", "audit_K_key.yaml"))
    key = {k["item"]: k for k in keyd["items"]}
    rows = [r for r in csv.DictReader(open(annotated_csv, encoding="utf-8")) if r["GOLD_label(-/+/0)"].strip()]
    gold, pred, B = [], [], []
    for r in rows:
        k = key[int(r["item"])]
        gold.append(r["GOLD_label(-/+/0)"].strip())
        pred.append(k["pred"])
        B.append(r.get("ANNOTATOR_B_label(optional)", "").strip())
    res = {"n_annotated": len(gold), "K": _prf(gold, pred)}
    # population-weighted accuracy (strata = predicted label)
    share = keyd["stratum_share"]
    acc_s = {lab: np.mean([g == p for g, p in zip(gold, pred) if p == lab]) for lab in LABELS if lab in pred}
    res["K"]["population_weighted_accuracy"] = round(float(sum(share.get(l, 0) * a for l, a in acc_s.items())), 3)
    pairs_b = [(g, b) for g, b in zip(gold, B) if b]
    if pairs_b:
        res["inter_annotator_kappa"] = round(kappa([g for g, _ in pairs_b], [b for _, b in pairs_b]), 3)
    # B-K1 direct generation
    try:
        direct = [Binding.from_json(b, vocab) for b in load_yaml(ctx.outp("baselines", "spec_direct.yaml"))["bindings"]]
        didx = index_spec(direct)
        dpred = [label_of(didx, key[int(r["item"])]["effect"], key[int(r["item"])]["asset"],
                          tuple(key[int(r["item"])]["cell"]), vocab) for r in rows]
        res["B-K1_direct"] = _prf(gold, dpred)
    except FileNotFoundError:
        res["B-K1_direct"] = "not run (ctxguard baseline-direct)"
    # B-K2 per-cell labelling
    if run_point_baseline:
        eidx = effect_index(ctx.effects())
        aidx = {a["id"]: a for a in ctx.asset_list}
        ppred = []
        for r in rows:
            k = key[int(r["item"])]
            cell = tuple(k["cell"])
            sys, user, schema = point_label_prompt(vocab.describe_vocab(), eidx[k["effect"]], aidx[k["asset"]],
                                                   vocab.describe_cell(cell))
            out = ctx.llm.json("BASE", "point_label_audit", sys, user, schema,
                               context={"effect": k["effect"], "asset": k["asset"], "cell": vocab.cell_dict(cell)})
            ppred.append(out.get("label", "0"))
        res["B-K2_per_cell"] = _prf(gold, ppred)
    dump_yaml(res, ctx.outp("eval", "audit_metrics.yaml"))
    return res
