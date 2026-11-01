"""LLM access layer: ollama client with JSON-schema structured output,
disk cache (re-runs are free and identical), and a token ledger.

Every call is recorded in out/ledger.csv:
  ts, module, task, model, prompt_tokens, completion_tokens, duration_ms, cached
`ctxguard tokens` aggregates it per module and per cost class.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Callable

import requests

LEDGER_FIELDS = ["ts", "module", "task", "model", "prompt_tokens",
                 "completion_tokens", "duration_ms", "cached"]


class LLM:
    def __init__(self, cfg: dict, out_dir: str, mock: Callable | None = None):
        self.cfg = cfg["llm"]
        self.backend = self.cfg["backend"]
        self.model = self.cfg["model"]
        self.mock = mock
        os.makedirs(out_dir, exist_ok=True)
        self.ledger_path = os.path.join(out_dir, "ledger.csv")
        self.db = sqlite3.connect(os.path.join(out_dir, "llm_cache.sqlite"))
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, "w", newline="") as f:
                csv.writer(f).writerow(LEDGER_FIELDS)

    # ------------------------------------------------------------------ utils
    def _log(self, module, task, model, pt, ct, ms, cached):
        with open(self.ledger_path, "a", newline="") as f:
            csv.writer(f).writerow([round(time.time(), 3), module, task, model,
                                    pt, ct, ms, int(cached)])

    def _key(self, payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         ensure_ascii=False).encode()).hexdigest()

    def _get(self, k):
        r = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def _put(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v)))
        self.db.commit()

    # ------------------------------------------------------------- json call
    def json(self, module: str, task: str, system: str, user: str,
             schema: dict, temperature: float = 0.0, seed: int = 0,
             context: dict | None = None) -> dict:
        """One structured call. `context` is only passed to the mock backend."""
        payload = {"model": self.model, "backend": self.backend, "system": system,
                   "user": user, "schema": schema, "t": temperature, "seed": seed}
        k = self._key(payload)
        hit = self._get(k)
        if hit is not None:
            self._log(module, task, self.model, hit["pt"], hit["ct"], 0, True)
            return hit["out"]

        if self.backend == "mock":
            out = self.mock(task, context or {}, seed)
            pt, ct, ms = len(system + user) // 4, len(json.dumps(out)) // 4, 0
        else:
            out, pt, ct, ms = self._ollama(system, user, schema, temperature, seed)
        self._put(k, {"out": out, "pt": pt, "ct": ct})
        self._log(module, task, self.model, pt, ct, ms, False)
        return out

    def _ollama(self, system, user, schema, temperature, seed):
        url = self.cfg["host"].rstrip("/") + "/api/chat"
        last = None
        for attempt in range(self.cfg.get("max_retries", 2) + 1):
            body = {
                "model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "format": schema, "stream": False,
                "options": {"temperature": temperature, "seed": seed + attempt,
                            "num_ctx": self.cfg.get("num_ctx", 8192)},
            }
            t0 = time.time()
            r = requests.post(url, json=body, timeout=self.cfg.get("timeout_s", 600))
            r.raise_for_status()
            j = r.json()
            ms = int((time.time() - t0) * 1000)
            try:
                out = json.loads(j["message"]["content"])
                return out, j.get("prompt_eval_count", 0), j.get("eval_count", 0), ms
            except json.JSONDecodeError as e:   # retry with a new seed
                last = e
        raise RuntimeError(f"model returned invalid JSON 3 times: {last}")

    # ------------------------------------------------------------- embedding
    def embed(self, module: str, texts: list[str]) -> list[list[float]]:
        model = self.cfg.get("embed_model", "nomic-embed-text")
        payload = {"embed": model, "texts": texts, "backend": self.backend}
        k = self._key(payload)
        hit = self._get(k)
        if hit is not None:
            self._log(module, "embed", model, hit["pt"], 0, 0, True)
            return hit["out"]
        if self.backend == "mock":
            out = [_hash_embed(t) for t in texts]
            pt, ms = sum(len(t) for t in texts) // 4, 0
        else:
            t0 = time.time()
            r = requests.post(self.cfg["host"].rstrip("/") + "/api/embed",
                              json={"model": model, "input": texts},
                              timeout=self.cfg.get("timeout_s", 600))
            r.raise_for_status()
            j = r.json()
            out, pt = j["embeddings"], j.get("prompt_eval_count", 0)
            ms = int((time.time() - t0) * 1000)
        self._put(k, {"out": out, "pt": pt, "ct": 0})
        self._log(module, "embed", model, pt, 0, ms, False)
        return out


def _hash_embed(t: str, dim: int = 64) -> list[float]:
    """Mock embedding: bag of word hashes (only for offline smoke tests)."""
    v = [0.0] * dim
    for w in t.lower().split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    return v


def check_ollama(cfg: dict) -> str:
    host = cfg["llm"]["host"].rstrip("/")
    tags = requests.get(host + "/api/tags", timeout=10).json()
    names = [m["name"] for m in tags.get("models", [])]
    msg = [f"ollama reachable at {host}", "installed models: " + ", ".join(names)]
    for need in (cfg["llm"]["model"], cfg["llm"].get("embed_model", "nomic-embed-text")):
        ok = any(n == need or n.startswith(need + ":") or n.split(":")[0] == need for n in names)
        msg.append(f"  {'OK ' if ok else 'MISSING'} {need}" + ("" if ok else f"   -> run: ollama pull {need}"))
    return "\n".join(msg)
