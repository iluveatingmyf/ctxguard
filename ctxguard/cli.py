"""Command line. Run `python -m ctxguard.cli --help`.

Order (see README):  check -> m1 -> m2-* -> m3 -> m3-refute -> baseline-direct ->
m4-export -> (annotate) -> m4-score -> m5/m6 per layout -> sim -> (annotate)
-> eval -> robust -> rq3 -> tokens -> report
"""
from __future__ import annotations

import argparse
import json

from .common import Ctx


def main():
    ap = argparse.ArgumentParser(prog="ctxguard")
    ap.add_argument("step")
    ap.add_argument("--layout", "-l", nargs="*", default=[])
    ap.add_argument("--gold", "-g", nargs="*", default=[], help="annotated CSV file(s)")
    ap.add_argument("--intents", default="system", help="system | path to annotated intent CSV")
    ap.add_argument("--pairs", type=int, default=None, help="M3: only first N (effect,asset) pairs (pilot)")
    ap.add_argument("--k", type=int, default=None, help="M1: samples (1 = no self-consistency ablation)")
    ap.add_argument("--backend", default=None, help="override llm.backend (ollama|mock)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-llm", action="store_true", help="eval: skip B2 LLM baseline")
    a = ap.parse_args()
    ctx = Ctx(a.config, a.backend)
    s = a.step
    show = lambda x: print(json.dumps(x, indent=2, ensure_ascii=False, default=str)[:4000])

    if s == "check":
        from .llm import check_ollama
        print(check_ollama(ctx.cfg))
        print(f"vocab {ctx.vocab.version}: {len(ctx.vocab.factors)} factors, |Sigma|={len(ctx.vocab.sigma())}")
    elif s == "m1":
        from .effects import run_m1
        r = run_m1(ctx, k=a.k, tag="" if not a.k or a.k == ctx.cfg["method"]["m1_samples"] else f"_k{a.k}")
        print(f"M1: {len(r['effects'])} effect types for {len(r['eff'])} actions; out-of-vocab rejected {r['rejected_out_of_vocab']}")
    elif s == "m1-gold-export":
        from .metrics_m1 import export_m1_gold
        print(export_m1_gold(ctx))
    elif s == "m1-score":
        from .metrics_m1 import score_m1
        show(score_m1(ctx, a.gold[0]))
    elif s == "m2-extract":
        from .vocab import run_extract
        print(f"{len(run_extract(ctx))} conditions")
    elif s == "m2-cluster":
        from .vocab import run_cluster
        r = run_cluster(ctx)
        print(f"{r['n_clusters']} clusters at cut {r['cut']} (silhouette {r['silhouette']})")
    elif s == "m2-name":
        from .vocab import run_name
        r = run_name(ctx)
        print(f"ABLATION: {len(r['factors'])} factors from clustering -> out/m2/ablation_clustering_factors.yaml")
    elif s == "m2-dims":
        from .vocab import run_dims
        r = run_dims(ctx)
        for x in r["rounds"]:
            print(f"round {x['round']}: +{x['new_dimensions']} dims (total {x['n_dimensions']}), uncovered {x['uncovered']} ({x['uncovered_share']})")
        print("dimensions:", ", ".join(f"{d['id']}({d.get('n_conditions', '-')})" for d in r["slots"] if d["kind"] == "situation"))
        print("coding:", r["coverage"])
    elif s in ("m2-values", "m2-slots"):
        from .vocab import run_slots
        if s == "m2-values":
            r = run_slots(ctx, ctx.outp("m2", "dims.yaml"), tag="corpus")
        else:
            r = run_slots(ctx, None, tag="ablation_topdown")
        print(f"{r['n_conditions']} conditions; excluded share {r['excluded_share']}")
        for k, v in r["per_slot"].items():
            print(f"  {k:24s} {v['n_conditions']:3d} conditions  {v.get('factors', '')}  coverage {v.get('coverage', '-')}")
    elif s == "m2-expand":
        from .vocab import run_expand
        r = run_expand(ctx)
        print(f"agent proposed {r['proposed']} values, accepted {r['accepted']} -> out/m2/factors_candidate.yaml")
    elif s == "m2-map":
        from .vocab import run_map_properties
        show(run_map_properties(ctx))
    elif s == "m3":
        from .binding import run_m3
        r = run_m3(ctx, limit_pairs=a.pairs)
        print(f"K: {len(r['bindings'])} bindings")
    elif s == "m3-refute":
        from .binding import run_refute
        r = run_refute(ctx)
        print(f"refuted {r['n_refuted']}/{r['n_bindings']}; top gaps {r['gap_report'][:5]}")
    elif s == "baseline-direct":
        from .binding import run_direct_baseline
        from .common import load_yaml
        pairs = sorted({(b["effect"], b["asset"]) for b in load_yaml(ctx.spec_path())["bindings"]})
        r = run_direct_baseline(ctx, pairs)
        print(f"B-K1: {len(r['bindings'])} bindings, invalid {r['invalid_patterns']}")
    elif s == "vocab-metrics":
        from .vocab import vocab_metrics
        show(vocab_metrics(ctx, a.gold[0] if a.gold else None))
    elif s == "m4-export":
        from .audit import export_audit
        print(export_audit(ctx))
    elif s == "m4-score":
        from .audit import score_audit
        show(score_audit(ctx, a.gold[0]))
    elif s == "m5":
        from .grounding import run_m5
        for lid in a.layout:
            show(run_m5(ctx, lid))
    elif s == "m6":
        from .intent import export_intent_gold, run_m6
        for lid in a.layout:
            r = run_m6(ctx, lid)
            print(f"M6 {lid}: {len(r['intents'])} intents, problems {r['problems']}; gold sheet {export_intent_gold(ctx, lid)}")
    elif s == "m6-score":
        from .intent import score_intents
        for lid, g in zip(a.layout, a.gold):
            show(score_intents(ctx, lid, g))
    elif s == "sim":
        from .simulate import run_sim
        from .intent import load_gold_intents
        for i, lid in enumerate(a.layout):
            ints = None
            if a.intents != "system":      # --intents takes one gold CSV per layout, same order
                paths = a.intents.split(",")
                ints = load_gold_intents(ctx, lid, paths[i])
            show(run_sim(ctx, lid, ints))
    elif s == "eval":
        from .evaluate import run_runtime
        for lid in a.layout:
            show(run_runtime(ctx, lid, a.intents, with_llm=not a.no_llm))
    elif s == "eval-score":
        from .evaluate import score_runtime
        show(score_runtime(ctx, a.layout, a.gold))
    elif s == "robust":
        from .evaluate import run_robustness
        show(run_robustness(ctx, a.layout, a.gold, a.intents))
    elif s == "rq3":
        from .compile_policy import run_rq3
        r = run_rq3(ctx, a.layout, a.intents)
        show({k: v for k, v in r.items() if k != "per_rule"})
    elif s == "rq3-verify":
        from .compile_policy import verify_ha_package
        show(verify_ha_package(ctx, a.layout, a.intents))
    elif s == "tokens":
        from .report import tokens
        show(tokens(ctx))
    elif s == "report":
        from .report import report, tokens
        tokens(ctx)
        print(report(ctx))
    else:
        raise SystemExit(f"unknown step {s}")


if __name__ == "__main__":
    main()
