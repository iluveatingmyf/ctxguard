"""m2-report: tables and figures for the vocabulary-construction section (Sec. VI-A).

Compares one or more finished A2/A3 runs, e.g. the old asset-agnostic coding (v3) against
the label-anchored lift + relation grouping (v4), or the same method with two models
(qwen2.5-7B via ollama vs DeepSeek). Reads only files the pipeline already wrote; no LLM.

  python -m ctxguard.cli m2-report --compare v3=runs/<old_out> v4-qwen=out v4-deepseek=out_deepseek

Writes out/eval/m2_report/:
  report.md            summary table, per-slot tables, agreement between runs
  fig_slot_sizes.png   slots of each run sorted by how many evidence units support them
  fig_unit_flow.png    where the evidence units of each run go (slot / excluded kinds / failed lift)
  summary.csv, slots_<run>.csv   the numbers behind every table and figure
What a reviewer gets from it: how many slots the corpus supports and how concentrated the
evidence is (fragmentation), what share of the corpus is device language that never becomes
a factor, whether the factors are stable across runs and across models (ARI on the same
evidence units), and, per factor, its supporting properties and example facts.
"""
from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict

from .common import load_yaml, p
from .vocab import _ari

EXCLUDED = ("device_state", "user_configuration", "obligation", "unclear")
# reference categorical order (dataviz skill, references/palette.md), fixed, never cycled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _load_run(label: str, out_dir: str) -> dict:
    base = out_dir if os.path.isabs(out_dir) else p(out_dir)
    m2 = os.path.join(base, "m2")
    dims = load_yaml(os.path.join(m2, "A2_dims.yaml"))
    coding = load_yaml(os.path.join(m2, "A2_coding.yaml"))
    a3 = os.path.join(m2, "A3_factors_corpus.yaml")
    a3r = os.path.join(m2, "A3_values_report.yaml")
    stab = os.path.join(m2, "A2_stability.yaml")
    units = {}
    for r in coding:                       # v4 rows are units; v3 rows are conditions with several properties
        pids = r.get("property_ids") or [r.get("property_id")]
        for pid in pids:
            units[f"{pid}|{r['condition']}"] = r
    slots = [d for d in dims["slots"] if d.get("kind") == "situation"]
    values = {}
    if os.path.exists(a3):
        for f in load_yaml(a3).get("factors", []):
            values[f["name"]] = list(f["values"])
    return {"label": label, "dir": base, "dims": dims, "coding": coding, "units": units, "slots": slots,
            "values": values, "a3": load_yaml(a3r) if os.path.exists(a3r) else None,
            "stability": load_yaml(stab) if os.path.exists(stab) else None,
            "coding_mode": dims.get("coding") or "question (v3)", "grouping": dims.get("grouping") or "llm (v3)"}


def _flow(run: dict) -> Counter:
    c = Counter()
    for r in run["units"].values():
        slot = r.get("slot")
        if r.get("variable"):
            c["in a slot"] += 1
        elif r.get("kind") in EXCLUDED or slot in EXCLUDED:
            c["device / configuration / obligation"] += 1
        elif r.get("lift_status") == "lift_failed" or slot == "lift_failed":
            c["lift failed"] += 1
        else:
            c["no situation fact"] += 1
    return c


def _summary(run: dict) -> dict:
    sizes = sorted((sum(1 for u in run["units"].values() if u.get("slot") == d["id"] and u.get("variable"))
                    for d in run["slots"]), reverse=True)
    n_units = len(run["units"])
    in_slot = sum(sizes)
    top = sum(sizes[:5])
    st = run["stability"] or {}
    return {"run": run["label"], "coding": run["coding_mode"], "grouping": run["grouping"],
            "evidence_units": n_units, "units_in_a_slot": in_slot,
            "slots": len(sizes), "singleton_slots": sum(1 for s in sizes if s == 1),
            "share_in_top5_slots": round(top / in_slot, 3) if in_slot else None,
            "a3_value_coverage": (run["a3"] or {}).get("overall_coverage"),
            "within_run_mean_ari": st.get("mean_ari"),
            "lift_failed": _flow(run).get("lift failed", 0)}


def _agreement(a: dict, b: dict) -> dict:
    common = sorted(set(a["units"]) & set(b["units"]))
    lab = lambda run, k: run["units"][k].get("slot") or "none"
    both = [k for k in common if a["units"][k].get("variable") and b["units"][k].get("variable")]
    return {"pair": f"{a['label']} vs {b['label']}", "common_units": len(common),
            "ari_all_units": _ari([lab(a, k) for k in common], [lab(b, k) for k in common]) if len(common) > 1 else None,
            "units_in_a_slot_in_both": len(both),
            "ari_slotted_units": _ari([lab(a, k) for k in both], [lab(b, k) for k in both]) if len(both) > 1 else None}


def _slot_rows(run: dict) -> list[dict]:
    rows = []
    for d in run["slots"]:
        mem = [u for u in run["units"].values() if u.get("slot") == d["id"] and u.get("variable")]
        pids = sorted({k.split("|")[0] for k, u in run["units"].items() if u.get("slot") == d["id"] and u.get("variable")})
        facts = [f for f, _ in Counter(u.get("latent_fact") or u.get("world_fact") or u.get("condition") for u in mem).most_common(3)]
        rows.append({"slot": d["id"], "subject": d.get("subject") or d.get("scope", ""), "question": d.get("desc", ""),
                     "units": len(mem), "properties": len(pids), "values_A3": ", ".join(run["values"].get(d["id"], [])),
                     "example_facts": " | ".join(facts), "property_ids": " ".join(pids)})
    return sorted(rows, key=lambda r: (-r["units"], r["slot"]))


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _figures(runs: list[dict], outdir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    # Fig 1: slots sorted by support, one panel per run (single series per panel -> no legend)
    n = len(runs)
    rows_per = [_slot_rows(r) for r in runs]
    h = max(2.2, 0.2 * max((len(x) for x in rows_per), default=1) + 0.9)
    fig, axes = plt.subplots(1, n, figsize=(max(5.2, 4.4 * n), h + 0.3), dpi=200, squeeze=False, facecolor=SURFACE)
    xmax = max((x["units"] for rr in rows_per for x in rr), default=1)
    for ax, run, rr in zip(axes[0], runs, rows_per):
        _style(ax)
        names = [x["slot"] for x in rr][::-1]
        vals = [x["units"] for x in rr][::-1]
        ax.barh(range(len(vals)), vals, height=0.62, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=7, color=INK)
        ax.set_xlim(0, xmax * 1.15)
        for i, v in enumerate(vals):
            ax.text(v + xmax * 0.02, i, str(v), va="center", fontsize=6.5, color=INK2)
        s = _summary(run)
        ax.set_title(f"{run['label']}\n{s['slots']} slots, {s['singleton_slots']} supported by a single unit",
                     fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("evidence units supporting the slot", fontsize=7.5, color=INK2)
    fig.tight_layout()
    f1 = os.path.join(outdir, "fig_slot_sizes.png")
    fig.savefig(f1, facecolor=SURFACE)
    plt.close(fig)
    files.append(f1)
    # Fig 2: where the units go (parts of a whole per run -> one stacked bar per run, fixed category order)
    cats = ["in a slot", "device / configuration / obligation", "lift failed", "no situation fact"]
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * n + 1.3), dpi=200, facecolor=SURFACE)
    _style(ax)
    for i, run in enumerate(runs):
        c = _flow(run)
        tot = sum(c.values()) or 1
        left = 0.0
        for j, cat in enumerate(cats):
            w = c.get(cat, 0) / tot
            if w:
                ax.barh(i, w, left=left, height=0.55, color=SERIES[j], edgecolor=SURFACE, linewidth=2)
                if w >= 0.07:
                    ax.text(left + w / 2, i, f"{c[cat]}", ha="center", va="center", fontsize=7,
                            color="white" if j in (0, 1) else INK)
                left += w
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in runs], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of evidence units (counts printed in the bars)", fontsize=7.5, color=INK2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=SERIES[j], label=cat) for j, cat in enumerate(cats)],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=7, labelcolor=INK)
    fig.tight_layout()
    f2 = os.path.join(outdir, "fig_unit_flow.png")
    fig.savefig(f2, facecolor=SURFACE)
    plt.close(fig)
    files.append(f2)
    return files


def _md_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |")
    return "\n".join(out)


def run_m2_report(ctx, compare: list[str] | None) -> dict:
    specs = compare or [f"current={ctx.cfg['paths']['out']}"]
    runs = []
    for s in specs:
        label, _, d = s.partition("=")
        runs.append(_load_run(label, d or label))
    outdir = os.path.join(ctx.out, "eval", "m2_report")
    os.makedirs(outdir, exist_ok=True)
    summ = [_summary(r) for r in runs]
    agree = [_agreement(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
    with open(os.path.join(outdir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    md = ["# M2 vocabulary construction: run comparison", "",
          "Generated by `m2-report` from finished A2/A3 outputs (no LLM call).", "",
          "## Summary", "", _md_table(summ, list(summ[0])), ""]
    if agree:
        md += ["## Agreement between runs on the same evidence units (ARI of unit -> slot)", "",
               _md_table(agree, list(agree[0])), ""]
    for r in runs:
        sr = _slot_rows(r)
        with open(os.path.join(outdir, f"slots_{r['label']}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sr[0]) if sr else ["slot"])
            w.writeheader()
            w.writerows(sr)
        md += [f"## Slots of {r['label']} ({r['coding_mode']} coding, {r['grouping']} grouping)", "",
               _md_table(sr, ["slot", "subject", "units", "properties", "values_A3", "example_facts"]), ""]
    figs = _figures(runs, outdir)
    md += ["## Figures", ""] + [f"![{os.path.basename(x)}]({os.path.basename(x)})" for x in figs]
    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return {"dir": outdir, "summary": summ, "agreement": agree, "figures": figs}
