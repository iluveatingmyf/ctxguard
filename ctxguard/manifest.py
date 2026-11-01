"""Audit trail: every file written by an M2 step (and by freeze / migrate) is recorded in
out/MANIFEST.csv with the step, content hash, model, backend, vocabulary version and the
hash of prompts.py. Together with out/llm_cache.sqlite (every raw LLM answer) and
out/ledger.csv (every call's tokens) this makes each intermediate result traceable."""
from __future__ import annotations

import csv
import hashlib
import os
import time

from .common import ROOT, dump_yaml

FIELDS = ["ts", "step", "file", "sha256", "bytes", "model", "backend", "vocab_version", "prompts_sha"]


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def record(ctx, step: str, path: str) -> str:
    man = os.path.join(ctx.out, "MANIFEST.csv")
    new = not os.path.exists(man)
    with open(man, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), step, os.path.relpath(path, ROOT), _sha(path),
                    os.path.getsize(path), ctx.cfg["llm"]["model"], ctx.cfg["llm"]["backend"], ctx.vocab.version,
                    _sha(os.path.join(ROOT, "ctxguard", "prompts.py"))])
    return path


def save_yaml(ctx, step: str, obj, *parts) -> str:
    path = ctx.outp(*parts)
    dump_yaml(obj, path)
    return record(ctx, step, path)


def save_csv(ctx, step: str, header: list, rows: list, *parts) -> str:
    path = ctx.outp(*parts)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return record(ctx, step, path)
