"""Command line. Run `python -m ctxguard.cli --help`.

Order (see README):  check -> m1 -> m2 (A: m2-extract, m2-dims, m2-values, [human seed], m2-seed-check;
B: m2-elicit; C: m2-consolidate; D: m2-freeze) -> m3 -> m2-audit -> m3-refute -> m2-refine ->
(next round: m2-consolidate --base current, m2-freeze, vocab-migrate, m3 --only-stale) -> baseline-direct ->
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
    # M2 / versioning
    ap.add_argument("--runs", type=int, default=1, help="m2-dims: number of open-coding runs (stability)")
    ap.add_argument("--base", default="seed", help="m2-elicit/m2-consolidate: seed | current | <vocab file>")
    ap.add_argument("--effects", nargs="*", default=None, help="m2-elicit / m3: only these effects (incremental round)")
    ap.add_argument("--dims", nargs="*", default=None, help="m2-consolidate: only these dimensions (name or NEW:name)")
    ap.add_argument("--src", default=None, help="m2-freeze: reviewed candidate file")
    ap.add_argument("--name", default=None, help="m2-freeze: version name, e.g. v1 -> F-v1")
    ap.add_argument("--dest", default=None, help="m2-freeze: output vocab file (default data/vocab/factors_<name>.yaml)")
    ap.add_argument("--parent", default=None, help="m2-freeze: parent vocab file (default: the one in config)")
    ap.add_argument("--no-provenance-check", action="store_true", help="m2-freeze: only for the hand-made v0 baseline / tests")
    ap.add_argument("--old", default=None, help="vocab-migrate: the vocab file K was built with")
    ap.add_argument("--only-stale", action="store_true", help="m3: rebuild only pairs in m3/stale_pairs.yaml")
    ap.add_argument("--merge", action="store_true", help="m3 --effects: keep the bindings of all other pairs")
    ap.add_argument("--spec-out", default=None, help="m3: write K to this file instead of out/m3/spec.yaml")
    ap.add_argument("--a", default=None, help="k-compare: spec A")
    ap.add_argument("--b", default=None, help="k-compare: spec B")
    ap.add_argument("--tag", default=None, help="snapshot: run name")
    ap.add_argument("--zero-max", type=int, default=None, help="m3-refute: max '0'-region claims per pair")
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
        r = run_extract(ctx)
        print(f"A1: {r['conditions']} conditions ({r['distinct_conditions']} distinct) from {r['train']} train properties; "
              f"{r['test_heldout']} held out; per asset {r['coverage_per_asset']}")
    elif s == "m2-cluster":
        from .vocab import run_cluster
        r = run_cluster(ctx)
        print(f"ABLATION: {r['n_clusters']} clusters at cut {r['cut']} (silhouette {r['silhouette']})")
    elif s == "m2-name":
        from .vocab import run_name
        r = run_name(ctx)
        print(f"ABLATION: {len(r['factors'])} factors from clustering -> out/m2/ablation_clustering_factors.yaml")
    elif s == "m2-dims":
        from .vocab import run_dims
        r = run_dims(ctx, runs=a.runs)
        for x in r["rounds"]:
            print(f"round {x['round']}: +{x['new_dimensions']} dims (total {x['n_dimensions']}), uncovered {x['uncovered']} ({x['uncovered_share']})")
        print("dimensions:", ", ".join(f"{d['id']}({d.get('n_conditions', '-')})" for d in r["slots"] if d["kind"] == "situation"))
        print("coding:", r["coverage"])
        if "stability" in r:
            print("stability:", r["stability"])
    elif s in ("m2-values", "m2-slots"):
        from .vocab import run_slots
        if s == "m2-values":
            r = run_slots(ctx, ctx.outp("m2", "A2_dims.yaml"), tag="A3")
        else:
            r = run_slots(ctx, None, tag="ablation_topdown")
        print(f"{r['n_conditions']} conditions; excluded share {r['excluded_share']}")
        for k, v in r["per_slot"].items():
            print(f"  {k:24s} {v['n_conditions']:3d} conditions  {v.get('factors', '')}  coverage {v.get('coverage', '-')}")
    elif s == "m2-seed-check":
        from .vocab import run_seed_check
        show({k: v for k, v in run_seed_check(ctx).items() if k != "per_factor"})
    elif s == "m2-expand":
        from .vocab import run_expand
        r = run_expand(ctx)
        print(f"ABLATION: agent proposed {r['proposed']} values, accepted {r['accepted']} -> out/m2/ablation_expand_factors.yaml")
    elif s == "m2-elicit":
        from .vocab import run_elicit
        r = run_elicit(ctx, base=a.base, effects=a.effects)
        print(f"round {r['round']}: {r['new_items']} new attributes from {r['pairs_queried']} effect x asset pairs "
              f"(rejected {r['rejected_new']}); total {r['items_total']}")
        for d, x in r["per_dimension"].items():
            print(f"  {d:30s} {x['attributes']:4d} attributes from {x['eliciting_pairs']} pairs")
    elif s == "m2-consolidate":
        from .vocab import run_consolidate
        r = run_consolidate(ctx, base=a.base, dims=a.dims)
        for d, x in r.items():
            if x.get("changed") is False and "values_after" not in x:
                print(f"  {d:30s} unchanged (no new evidence)")
            else:
                print(f"  {d:30s} values {x['values_before']} -> {x['values_after']} (states {x['states_after_canonicalize']}, "
                      f"pairs {x['relevant_pairs']}{', NEW dimension' if x['new_dimension'] else ''})")
        print("-> out/m2/C3_factors_candidate.yaml (review it, then m2-freeze)")
    elif s == "m2-freeze":
        from .versioning import freeze
        r = freeze(ctx, a.src or ctx.outp("m2", "C3_factors_candidate.yaml"), a.name, a.dest, a.parent,
                   check_provenance=not a.no_provenance_check)
        show(r)
        print(f"-> set config.yaml paths.factors to {r['file']}")
    elif s == "m2-audit":
        from .vocab import run_audit
        show(run_audit(ctx, a.gold[0] if a.gold else None))
    elif s == "m2-refine":
        from .vocab import run_refine
        show(run_refine(ctx))
    elif s == "m2-check-score":
        from .vocab import run_check_score
        show(run_check_score(ctx, a.gold))
    elif s == "vocab-migrate":
        from .versioning import migrate
        r = migrate(ctx, a.old)
        show({k: v for k, v in r.items() if k != "stale_pairs"})
        print(f"stale pairs: {len(r['stale_pairs'])} -> run: m3 --only-stale")
    elif s == "k-compare":
        from .common import load_yaml as _ly
        from .model import Binding as _B
        from .roughset import Tables, compare
        from .manifest import save_yaml as _sy
        tabs = [Tables(ctx.vocab, [_B.from_json(b, ctx.vocab) for b in _ly(x)["bindings"]]) for x in (a.a, a.b)]
        r = {"A": a.a, "B": a.b, **compare(*tabs)}
        _sy(ctx, "k-compare", r, "eval", "k_compare.yaml")
        show(r)
    elif s == "snapshot":
        from .versioning import snapshot
        print(snapshot(ctx, a.tag or "run"))
    elif s == "m2-map":
        from .vocab import run_map_properties
        show(run_map_properties(ctx))
    elif s == "m3":
        from .binding import run_m3
        r = run_m3(ctx, limit_pairs=a.pairs, only_effects=a.effects, only_stale=a.only_stale, spec_out=a.spec_out,
                   merge=a.merge)
        print(f"K: {len(r['bindings'])} bindings; LLM calls {r.get('llm_calls')}")
    elif s == "m3-refute":
        from .binding import run_refute
        r = run_refute(ctx, a.zero_max)
        print(f"claims {r['n_claims']} ({r['n_bindings']} bindings + {r['n_zero_claims']} '0' regions); "
              f"refuted {r['n_refuted']}, confirmed by judge {r['n_confirmed']} "
              f"(spurious counterexamples {r['spurious_counterexamples']}); top gaps {r['gap_report'][:5]}")
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
