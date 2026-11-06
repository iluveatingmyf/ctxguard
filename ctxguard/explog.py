"""Experiment log: the record standard for every result that may be cited in the paper.

One row per run or score, appended to out/eval/experiment_log.csv and never rewritten:
  ts            local time of the run
  step          CLI step (m1, m1-score, m2-dims, m2-dev, ...)
  result_file   the file the numbers come from (its hash is in MANIFEST.csv)
  model,backend the LLM that produced the scored output
  method_version the method/prompt version string of the scored output (e.g. M1-v3, A2 lift-v4.1)
  prompts_sha   hash of ctxguard/prompts.py at scoring time
  config_sha    hash of the method section of the config (all switches that change results)
  codebook      A2 codebook version (M2 only)
  n             number of scored items
  metrics       the headline numbers, as JSON
  note          free text (e.g. "ablation: m1_direction_rule=majority")

Rule for the paper: a number goes into a table only if it has a row here. A prompt or
method change is judged on the dev/gold rows, never on the held-out properties.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time

FIELDS = ["ts", "step", "result_file", "model", "backend", "method_version", "prompts_sha", "config_sha",
          "codebook", "n", "metrics", "note"]


def _sha_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def _sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def log_result(ctx, step: str, result_file: str, metrics: dict, method_version: str = "", n=None,
               codebook: str = "", note: str = "") -> str:
    from .common import ROOT
    path = ctx.outp("eval", "experiment_log.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    method_cfg = json.dumps(ctx.cfg.get("method", {}), sort_keys=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), step, os.path.relpath(result_file, ROOT),
                    ctx.cfg["llm"]["model"], ctx.cfg["llm"]["backend"], method_version,
                    _sha_file(os.path.join(ROOT, "ctxguard", "prompts.py")), _sha_text(method_cfg),
                    codebook, "" if n is None else n, json.dumps(metrics, ensure_ascii=False, sort_keys=True), note])
    return path
