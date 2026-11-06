"""M1  Effect mapping  Eff: A -> 2^E  with occurrence preconditions  (Sec. IV-A)

Method (self-consistency over a closed channel set):
  for each action type a:
     draw k independent samples (temperature t, different seeds) of
     "list every effect of a" constrained to channel ids in C; a direction is
     increase, decrease, or either (= the sign depends on the situation, e.g. a window
     warms OR cools the room depending on the outdoor temperature);
     CHANNEL vote: keep channel c iff it is mentioned (any direction) in >= ceil(k/2) samples;
     DIRECTION rule (method.m1_direction_rule):
       unanimity_else_both (default, v3): if every kept sample gives the same single
         direction, keep that direction; if any sample says 'either', or samples
         disagree, keep BOTH directions. Disagreement between samples is evidence that the
         sign is situation-dependent (or that the model is unsure); keeping both is the
         conservative choice: the runtime checks both, so a harm is never missed, at the
         cost of possible over-intervention. Measured against the gold 'ud' cells.
       majority (v2 behaviour, ablation): vote each (channel, direction) separately.
         This silently drops the minority direction (v2 kept 0 bidirectional effects).
  Effect type id = "<channel>:<direction>"; effects of different actions with the
  same (channel, direction) share one id, so E is the union over actions.
  Both rules are computed from the SAME raw samples (no extra LLM calls); the
  non-default one is written as effects{tag}_alt-<rule>.yaml for the ablation table.
Prompt v3: adds direction 'either' (v2 had only increase|decrease, so a sample had to
  list two entries to express situation dependence, which a 7B model rarely does).
Prompt v2: a dedicated system prompt without assets/labels (v1 shared the
  analysis preamble and the model confused assets with channels), and effects
  that need an extra device or unknown load are flagged and discarded.
Why: single LLM samples both hallucinate and omit side effects; voting over samples
trades a little recall for precision and is reported as an ablation (k=1 vs k=3)
against the human gold Eff (eval: channel-level and effect-level P/R/F1).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from .common import Ctx
from .manifest import save_yaml
from .prompts import effects_prompt

PROMPT_VERSION = "M1-v3 (direction 'either'; channel vote + direction rule)"
RULES = ("unanimity_else_both", "majority")
DIRS = ("increase", "decrease")


def _aggregate(samples: list[dict], k: int, rule: str) -> tuple[list[str], dict]:
    """samples: per sample {channel: set of directions in {increase, decrease, either}}.
    Returns the kept effect ids and, per kept channel, the votes behind the decision."""
    need = math.ceil(k / 2)
    if rule == "majority":
        votes = Counter()
        for s in samples:
            for ch, ds in s.items():
                for d in ({"increase", "decrease"} if "either" in ds else ds):
                    votes[(ch, d)] += 1
        kept = sorted(f"{c}:{d}" for (c, d), n in votes.items() if n >= need)
        return kept, {f"{c}:{d}": n for (c, d), n in sorted(votes.items())}
    presence, labels = Counter(), defaultdict(Counter)
    for s in samples:
        for ch, ds in s.items():
            presence[ch] += 1
            lab = "either" if ("either" in ds or set(DIRS) <= ds) else next(iter(ds))
            labels[ch][lab] += 1
    kept, audit = [], {}
    for ch in sorted(presence):
        if presence[ch] < need:
            continue
        labs = labels[ch]
        dirs = [next(iter(labs))] if len(labs) == 1 and "either" not in labs else list(DIRS)
        kept += [f"{ch}:{d}" for d in dirs]
        audit[ch] = {"mentioned_in": presence[ch], "direction_votes": dict(labs), "kept": dirs}
    return sorted(kept), audit


def run_m1(ctx: Ctx, k: int | None = None, tag: str = "") -> dict:
    k = k or ctx.cfg["method"]["m1_samples"]
    rule = ctx.cfg["method"].get("m1_direction_rule", "unanimity_else_both")
    if rule not in RULES:
        raise SystemExit(f"method.m1_direction_rule must be one of {RULES}")
    t = ctx.cfg["method"]["m1_temperature"] if k > 1 else 0.0
    ch_ids = {c["id"] for c in ctx.channels["channels"]}
    cdesc = {c["id"]: c["desc"] for c in ctx.channels["channels"]}
    raw, per_action = {}, {}
    rejected = extra_device = 0
    for a in ctx.actions:
        raw[a["id"]], samples, texts = [], [], defaultdict(list)
        for s in range(k):
            sys, user, schema = effects_prompt(a, ctx.channels_desc())
            out = ctx.llm.json("M1", "effects", sys, user, schema, temperature=t,
                               seed=ctx.seed + s, context={"action": a})
            raw[a["id"]].append(out)
            sample = defaultdict(set)
            for e in out.get("effects", []):
                d = e.get("direction")
                if e.get("channel") not in ch_ids or d not in ("increase", "decrease", "either"):
                    rejected += 1
                    continue
                if e.get("requires_extra_device") is True:   # speculative effect, discarded
                    extra_device += 1
                    continue
                sample[e["channel"]].add(d)
                pre = str(e.get("precondition") or "always")
                if pre not in texts[e["channel"]]:
                    texts[e["channel"]].append(pre)
            samples.append(sample)
        per_action[a["id"]] = (samples, texts)

    results = {}
    for r in RULES:
        eff, effects, audit = {}, {}, {}
        for aid, (samples, texts) in per_action.items():
            kept, aud = _aggregate(samples, k, r)
            eff[aid], audit[aid] = kept, aud
            for eid in kept:
                c, d = eid.split(":")
                effects.setdefault(eid, {"id": eid, "channel": c, "direction": d,
                                         "desc": f"{d} of {c.replace('_', ' ')} ({cdesc[c]})",
                                         "preconditions": {}})
                effects[eid]["preconditions"][aid] = " | ".join(texts[c]) or "always"
        bidir = sorted(f"{aid}:{c}" for aid, ks in eff.items() for c in {x.split(':')[0] for x in ks}
                       if f"{c}:increase" in ks and f"{c}:decrease" in ks)
        results[r] = {"version": f"E-k{k}", "samples": k, "temperature": t, "direction_rule": r,
                      "prompt_version": PROMPT_VERSION, "rejected_out_of_vocab": rejected,
                      "discarded_extra_device": extra_device, "n_bidirectional": len(bidir),
                      "bidirectional": bidir, "effects": sorted(effects.values(), key=lambda e: e["id"]),
                      "eff": eff, "votes": audit}
    save_yaml(ctx, "m1", results[rule], "m1", f"effects{tag}.yaml")
    for r in RULES:
        if r != rule:
            save_yaml(ctx, "m1", results[r], "m1", f"effects{tag}_alt-{r}.yaml")
    save_yaml(ctx, "m1", raw, "m1", f"raw_samples{tag}.yaml")
    return results[rule]


def effect_index(eff_lib: dict) -> dict:
    return {e["id"]: e for e in eff_lib["effects"]}
