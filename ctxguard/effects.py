"""M1  Effect mapping  Eff: A -> 2^E  with occurrence preconditions  (Sec. IV-A)

Method (self-consistency over a closed channel set):
  for each action type a:
     draw k independent samples (temperature t, different seeds) of
     "list every effect of a" constrained to channel ids in C;
     keep effect (channel, direction) iff it appears in >= ceil(k/2) samples.
  Effect type id = "<channel>:<direction>"; effects of different actions with the
  same (channel, direction) share one id, so E is the union over actions.
Direction-dependent effects (a window warms OR cools the room) are kept as two
  effects; both are assumed possible at runtime (conservative approximation:
  never misses a harm, may over-intervene). M1 uses no context factor.
Prompt v2: a dedicated system prompt without assets/labels (v1 shared the
  analysis preamble and the model confused assets with channels), and effects
  that need an extra device or unknown load are flagged and discarded.
Why: single LLM samples both hallucinate and omit side effects; majority vote
over samples trades a little recall for precision and is reported as an
ablation (k=1 vs k=3) against the human gold Eff (eval: P/R/F1).
"""
from __future__ import annotations

import math
from collections import Counter

from .common import Ctx, dump_yaml
from .prompts import effects_prompt


def run_m1(ctx: Ctx, k: int | None = None, tag: str = "") -> dict:
    k = k or ctx.cfg["method"]["m1_samples"]
    t = ctx.cfg["method"]["m1_temperature"] if k > 1 else 0.0
    ch_ids = {c["id"] for c in ctx.channels["channels"]}
    cdesc = {c["id"]: c["desc"] for c in ctx.channels["channels"]}
    eff, effects, raw = {}, {}, {}
    rejected = extra_device = 0
    for a in ctx.actions:
        votes, texts = Counter(), {}
        raw[a["id"]] = []
        for s in range(k):
            sys, user, schema = effects_prompt(a, ctx.channels_desc())
            out = ctx.llm.json("M1", "effects", sys, user, schema, temperature=t,
                               seed=ctx.seed + s, context={"action": a})
            raw[a["id"]].append(out)
            seen = set()
            for e in out.get("effects", []):
                if e.get("channel") not in ch_ids or e.get("direction") not in ("increase", "decrease"):
                    rejected += 1
                    continue
                if e.get("requires_extra_device") is True:   # speculative effect, discarded
                    extra_device += 1
                    continue
                key = (e["channel"], e["direction"])
                texts.setdefault(key, str(e.get("precondition") or "always"))
                if key in seen:
                    continue
                seen.add(key)
                votes[key] += 1
        need = math.ceil(k / 2)
        kept = sorted(key for key, n in votes.items() if n >= need)
        eff[a["id"]] = [f"{c}:{d}" for c, d in kept]
        for c, d in kept:
            eid = f"{c}:{d}"
            effects.setdefault(eid, {"id": eid, "channel": c, "direction": d,
                                     "desc": f"{d} of {c.replace('_', ' ')} ({cdesc[c]})",
                                     "preconditions": {}})
            effects[eid]["preconditions"][a["id"]] = texts[(c, d)]
    res = {"version": f"E-k{k}", "samples": k, "rejected_out_of_vocab": rejected, "discarded_extra_device": extra_device,
           "prompt_version": "M1-v2 (dedicated system prompt)",
           "effects": sorted(effects.values(), key=lambda e: e["id"]), "eff": eff}
    dump_yaml(res, ctx.outp("m1", f"effects{tag}.yaml"))
    dump_yaml(raw, ctx.outp("m1", f"raw_samples{tag}.yaml"))
    return res


def effect_index(eff_lib: dict) -> dict:
    return {e["id"]: e for e in eff_lib["effects"]}
