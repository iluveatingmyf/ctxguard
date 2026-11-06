"""DeepSeek backend against a local fake server (no network, no key needed).
Checks: request format (JSON mode, schema in the system prompt, bearer key), retry on a
non-conforming answer, token accounting, cache hit on the second call.
Run:  python tests/test_deepseek_backend.py
"""
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["CTXGUARD_TEST_KEY"] = "sk-test"

from ctxguard.llm import LLM  # noqa: E402
from ctxguard.prompts import fact_relation_prompt  # noqa: E402

SEEN = []


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        # first answer violates the enum, second is fine -> the client must retry once
        content = {"relation": "probably", "reason": "x"} if len(SEEN) == 1 else {"relation": "exclusive", "reason": "ok"}
        out = json.dumps({"choices": [{"message": {"content": json.dumps(content)}}],
                          "usage": {"prompt_tokens": 11, "completion_tokens": 3}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main():
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = {"llm": {"backend": "deepseek", "model": "deepseek-chat", "api_key_env": "CTXGUARD_TEST_KEY",
                   "base_url": f"http://127.0.0.1:{srv.server_port}", "max_retries": 2, "timeout_s": 10}}
    with tempfile.TemporaryDirectory() as d:
        llm = LLM(cfg, d)
        sys_, user, schema = fact_relation_prompt("home", "nobody is at home", "someone is at home")
        out = llm.json("M2", "fact_relation", sys_, user, schema)
        assert out == {"relation": "exclusive", "reason": "ok"}, out
        assert len(SEEN) == 2, "non-conforming answer was not retried"
        b = SEEN[0]["body"]
        assert SEEN[0]["path"] == "/chat/completions" and SEEN[0]["auth"] == "Bearer sk-test"
        assert b["response_format"] == {"type": "json_object"} and b["model"] == "deepseek-chat"
        assert "JSON schema" in b["messages"][0]["content"] and '"relation"' in b["messages"][0]["content"]
        out2 = llm.json("M2", "fact_relation", sys_, user, schema)      # cached: no new request
        assert out2 == out and len(SEEN) == 2
        rows = open(os.path.join(d, "ledger.csv")).read().strip().splitlines()
        assert rows[1].split(",")[4:6] == ["11", "3"] and rows[2].endswith(",1"), rows
    srv.shutdown()
    print("DEEPSEEK BACKEND TEST PASSED")


if __name__ == "__main__":
    main()
