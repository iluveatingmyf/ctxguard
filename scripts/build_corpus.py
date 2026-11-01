"""Build data/properties/seed_properties.csv from the raw corpora (reproducible).

Inputs (data/properties/raw/):
  cleaned_corpus.json  externally collected + merged corpus (TAPFixer, IoTSafe,
                       IoTSeer, TAPInspector, EnvGuard, IoTGuard), normalized wording
  seed_v0_62.csv       first hand-collected set (Soteria, IoTCom, IoTGuard*, IoTSan, VetIoT)
  ../splits.yaml       hand splits of compound properties
Steps:
  1. drop templates (meta-properties without concrete devices/conditions)
  2. split compound properties (splits.yaml)
  3. drop seed_v0 IoTGuard rows (paraphrased; the external corpus has the originals)
  4. exact de-duplication on normalized text (lowercase, alphanumerics only)
  5. scope tagging by fixed rules (below). Only scope == "in" is used by M2;
     the other categories are counted and reported in the paper.
Scope rules (first match wins):
  out:temporal        timing / liveness ("within t seconds", "eventually", "after X, ...")
  out:device_health   battery level
  out:outdoor         garden / soil / rain-sprinkler irrigation
  out:virtual_state   location/home mode bookkeeping ("set mode", "locationMode")
  out:device_conflict two actuators that must not run together (cross-app conflicts,
                      handled by interaction-threat analyses, orthogonal to this work)
  in                  everything else
Run:  python scripts/build_corpus.py
"""
import csv
import json
import os
import re
from collections import Counter

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "properties", "raw")
OUT = os.path.join(ROOT, "data", "properties", "seed_properties.csv")
SRC = {"tapfixer": "TAPFixer", "iotsafe": "IoTSafe", "iotseer": "IoTSeer", "tapinspector": "TAPInspector",
       "envguard": "EnvGuard", "iotguard": "IoTGuard (NDSS 2019)"}

RULES = [
    ("out:temporal", r"within (t|\d+) (seconds?|minutes?)|\beventually\b|for more than|for at least|at most \d|"
                     r"\bt seconds\b|permitted time|for too long|\bleads? to\b|^(after|once) "),
    ("out:device_health", r"\bbattery\b"),
    ("out:outdoor", r"\bsoil\b|\bgarden\b|sprinkler system must not be on when it rains|"
                    r"weather is raining, the sprinkler|it rains.*sprinkler"),
    ("out:virtual_state", r"set mode|locationmode|\bmode should be\b|mode must be set|\bset mode\b"),
    ("out:device_conflict", r"at the same time|simultaneously|both the heater and ac|"
                            r"when the heater is on|when the ac is on|the heater is on"),
]


def scope(text):
    t = text.lower()
    for name, rx in RULES:
        if re.search(rx, t):
            return name
    return "in"


def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def main():
    corpus = json.load(open(os.path.join(RAW, "cleaned_corpus.json")))
    splits = yaml.safe_load(open(os.path.join(ROOT, "data", "properties", "splits.yaml")))
    rows, stats = [], Counter()
    for p in corpus["properties"]:
        if p["flags"].get("template"):
            stats["dropped_template"] += 1
            continue
        src = "+".join(SRC.get(s, s) for s in p["sources"])
        prov = "cleaned_corpus:" + p["cid"] + " ids=" + ",".join(map(str, p["property_ids"]))
        parts = splits.get(p["cid"]) if p["flags"].get("needs_split") else None
        if p["flags"].get("needs_split") and not parts:
            raise SystemExit(f"{p['cid']} needs a split in splits.yaml")
        if parts:
            stats["split_from"] += 1
            for i, t in enumerate(parts):
                rows.append({"id": f"{p['cid']}{'abc'[i]}", "source": src, "verbatim": "0", "text": t, "provenance": prov + " (split)"})
        else:
            rows.append({"id": p["cid"], "source": src, "verbatim": "0", "text": p["text"], "provenance": prov})
    for r in csv.DictReader(open(os.path.join(RAW, "seed_v0_62.csv"), encoding="utf-8")):
        if r["source"].startswith("IoTGuard"):
            stats["dropped_paraphrased_iotguard"] += 1
            continue
        rows.append({"id": r["id"], "source": r["source"], "verbatim": r["verbatim"], "text": r["text"], "provenance": "seed_v0"})
    seen, out = {}, []
    for r in rows:
        k = norm(r["text"])
        if k in seen:
            stats["dropped_exact_duplicate"] += 1
            continue
        seen[k] = r["id"]
        r["scope"] = scope(r["text"])
        out.append(r)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "verbatim", "scope", "text", "provenance"])
        w.writeheader()
        w.writerows(out)
    sc = Counter(r["scope"] for r in out)
    src = Counter(s for r in out if r["scope"] == "in" for s in r["source"].split("+"))
    report = {"rows_total": len(out), "by_scope": dict(sc), "in_scope_by_source": dict(src), "steps": dict(stats)}
    yaml.safe_dump(report, open(os.path.join(ROOT, "data", "properties", "corpus_report.yaml"), "w"), sort_keys=False)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
