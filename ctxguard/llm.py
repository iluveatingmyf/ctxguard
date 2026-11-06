"""LLM access layer: ollama or DeepSeek (OpenAI-compatible) client with structured
JSON output, disk cache (re-runs are free and identical), and a token ledger.

Backends (config llm.backend):
  ollama    local; the JSON schema is enforced by ollama's `format`.
  deepseek  https://api.deepseek.com chat/completions with response_format json_object.
            JSON mode does not enforce a schema, so the schema is put in the system
            prompt and every answer is checked locally (_conforms); a non-conforming
            answer is retried. The key is read from the environment variable named in
            llm.api_key_env (default DEEPSEEK_API_KEY); it is never written to disk.
            DeepSeek has no `seed`: run-to-run identity comes from the cache.
            Thinking (DeepSeek V4): llm.thinking = disabled (default) | enabled. V4 thinks by
            default; in thinking mode the API IGNORES temperature (so the k-sample and
            stability designs lose their control variable), each call spends thousands of
            reasoning tokens, and long reasoning can exhaust max_tokens (finish_reason=length,
            truncated JSON). 'disabled' is sent explicitly. A non-default thinking setting is part
            of the cache key, so answers of the two modes never mix; entries written before this
            switch existed were made with V4's default (thinking on) and stay keyed that way.
            Debug: export CTXGUARD_DEBUG=1 prints finish_reason, token counts and a preview of
            every DeepSeek answer.
  mock      offline smoke tests only.
The cache key contains backend and model, so results of different models never mix.

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
        self.thinking = str(self.cfg.get("thinking", "disabled")).lower()
        if self.backend == "deepseek" and self.thinking not in ("enabled", "disabled"):
            raise SystemExit(f"llm.thinking must be 'enabled' or 'disabled', got {self.thinking!r}")
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
        if self.backend == "deepseek" and self.thinking != "enabled":
            payload["thinking"] = self.thinking      # 'enabled' = V4 default = how older entries were keyed
        k = self._key(payload)
        hit = self._get(k)
        if hit is not None:
            self._log(module, task, self.model, hit["pt"], hit["ct"], 0, True)
            return hit["out"]

        if self.backend == "mock":
            out = self.mock(task, context or {}, seed)
            pt, ct, ms = len(system + user) // 4, len(json.dumps(out)) // 4, 0
        elif self.backend == "deepseek":
            out, pt, ct, ms = self._deepseek(system, user, schema, temperature)
        elif self.backend == "ollama":
            out, pt, ct, ms = self._ollama(system, user, schema, temperature, seed)
        else:
            raise SystemExit(f"unknown llm.backend {self.backend!r} (ollama | deepseek | mock)")
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

    def _deepseek(self, system, user, schema, temperature):
        key = os.environ.get(self.cfg.get("api_key_env", "DEEPSEEK_API_KEY"), "")
        if not key:
            raise SystemExit(f"set the environment variable {self.cfg.get('api_key_env', 'DEEPSEEK_API_KEY')} "
                             "to your DeepSeek API key")
        url = self.cfg.get("base_url", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
        sys_ = (system + "\n\nReturn ONE JSON object that conforms to this JSON schema (no prose, no code fence):\n" +
                json.dumps(schema, ensure_ascii=False))
        debug = os.environ.get("CTXGUARD_DEBUG") == "1"
        last = None
        for attempt in range(self.cfg.get("max_retries", 2) + 1):
            body = {"model": self.model, "temperature": temperature,
                    "max_tokens": self.cfg.get("max_tokens", 4096),
                    "response_format": {"type": "json_object"},
                    "thinking": {"type": self.thinking},
                    "messages": [{"role": "system", "content": sys_}, {"role": "user", "content": user}]}
            t0 = time.time()
            r = requests.post(url, json=body, timeout=self.cfg.get("timeout_s", 600),
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            if r.status_code == 429 or r.status_code >= 500:      # rate limit / server busy: back off and retry
                last = f"HTTP {r.status_code}"
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            j = r.json()
            ms = int((time.time() - t0) * 1000)
            choice = (j.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            finish = choice.get("finish_reason")
            usage = j.get("usage") or {}
            if debug:
                print(f"[deepseek] attempt {attempt + 1} model={j.get('model')} finish={finish} "
                      f"prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} "
                      f"reasoning_chars={len(message.get('reasoning_content') or '')} content_chars={len(content)}\n"
                      f"  preview: {content[:300]!r}")
            if finish == "length":
                last = (f"answer cut off at max_tokens={body['max_tokens']} (finish_reason=length; thinking="
                        f"{self.thinking}). Raise llm.max_tokens, or keep llm.thinking: disabled")
                continue
            if not content.strip():
                last = f"empty content (finish_reason={finish})"
                continue
            try:
                out = json.loads(content)
            except json.JSONDecodeError as e:
                last = f"invalid JSON: {e}; preview={content[:200]!r}"
                continue
            problem = _conforms(out, schema)
            if problem is None:
                return out, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), ms
            last = f"does not conform to the schema: {problem}"
        raise RuntimeError(f"DeepSeek answer unusable after retries: {last}")

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
        else:   # embeddings always come from ollama (DeepSeek has none); only the m2-cluster ablation uses them
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


def _conforms(obj, schema, path="$"):
    """Minimal JSON-schema check for the subset used in prompts.py (object/required/
    properties, array/items, string/boolean/integer/number, enum, nullable type lists).
    Returns None if obj conforms, else a short description of the first problem."""
    t = schema.get("type")
    types = t if isinstance(t, list) else [t] if t else []
    if obj is None:
        return None if "null" in types else f"{path} is null"
    if "enum" in schema and obj not in schema["enum"]:
        return f"{path}={obj!r} not in {schema['enum']}"
    kind = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int, "number": (int, float)}
    want = [kind[x] for x in types if x in kind]
    if want and not isinstance(obj, tuple(want)):
        return f"{path} has type {type(obj).__name__}, expected {types}"
    if isinstance(obj, bool) and "boolean" not in types and types:
        return f"{path} is boolean, expected {types}"
    if isinstance(obj, dict):
        for k in schema.get("required", []):
            if k not in obj:
                return f"{path}.{k} missing"
        for k, sub in (schema.get("properties") or {}).items():
            if k in obj:
                prob = _conforms(obj[k], sub, f"{path}.{k}")
                if prob:
                    return prob
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            for k, v in obj.items():
                if k not in (schema.get("properties") or {}):
                    prob = _conforms(v, extra, f"{path}.{k}")
                    if prob:
                        return prob
    if isinstance(obj, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(obj):
            prob = _conforms(v, schema["items"], f"{path}[{i}]")
            if prob:
                return prob
    return None


def _hash_embed(t: str, dim: int = 64) -> list[float]:
    """Mock embedding: bag of word hashes (only for offline smoke tests)."""
    v = [0.0] * dim
    for w in t.lower().split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    return v


def check_backend(cfg: dict) -> str:
    if cfg["llm"]["backend"] == "deepseek":
        env = cfg["llm"].get("api_key_env", "DEEPSEEK_API_KEY")
        if not os.environ.get(env):
            return f"MISSING environment variable {env} (export {env}=sk-...)"
        url = cfg["llm"].get("base_url", "https://api.deepseek.com").rstrip("/") + "/models"
        r = requests.get(url, headers={"Authorization": f"Bearer {os.environ[env]}"}, timeout=20)
        if r.status_code != 200:
            return f"DeepSeek reachable but returned HTTP {r.status_code}: {r.text[:200]}"
        names = [m.get("id") for m in r.json().get("data", [])]
        ok = cfg["llm"]["model"] in names
        th = str(cfg["llm"].get("thinking", "disabled"))
        return (f"DeepSeek reachable; models: {', '.join(names)}\n  {'OK ' if ok else 'MISSING'} {cfg['llm']['model']}"
                + ("" if ok else "   -> set llm.model to one of the listed models")
                + f"\n  thinking: {th}" + ("  (temperature is IGNORED in thinking mode)" if th == "enabled" else ""))
    return check_ollama(cfg)


def check_ollama(cfg: dict) -> str:
    host = cfg["llm"]["host"].rstrip("/")
    tags = requests.get(host + "/api/tags", timeout=10).json()
    names = [m["name"] for m in tags.get("models", [])]
    msg = [f"ollama reachable at {host}", "installed models: " + ", ".join(names)]
    for need in (cfg["llm"]["model"], cfg["llm"].get("embed_model", "nomic-embed-text")):
        ok = any(n == need or n.startswith(need + ":") or n.split(":")[0] == need for n in names)
        msg.append(f"  {'OK ' if ok else 'MISSING'} {need}" + ("" if ok else f"   -> run: ollama pull {need}"))
    return "\n".join(msg)
