"""
arch_cost/analyze.py — From raw records to defensible claims.

Statistical choices (the point of the talk):
  * Unit of analysis = TASK, not run: repeats of a task are averaged
    first (no pseudo-replication from repeats).
  * Model-vs-model cost comparisons are PAIRED on task identity, with
    a paired bootstrap CI on the mean per-task cost difference.
  * Headline metric: cost per SOLVED task = total cost / total correct.
    Cheap-but-wrong is not cheap.
"""
import plotly.graph_objects as go

import itertools
import json
from collections import defaultdict
from pathlib import Path


import numpy as np

from registry import by_id, colour_lbl_map

RNG = np.random.default_rng(0)
N_BOOT = 5000

color_map  = colour_lbl_map()
def create_plotly_table(acc: list[int], cost_per_run: list[float], color_label:list[str], suite:str) -> go.Figure:
    shape_map = {"easy": "circle", "reasoning": "triangle-up", "tool": "square", "all": "diamond"}

    fig = go.Figure()
    # for lbl in np.unique(color_label):
    for i in range(len(color_label)):
        # m = color_label == lbl
        lbl = color_label[i]
        fig.add_trace(go.Scatter(
            x=[acc[i]], y=[cost_per_run[i]], mode="markers", name=str(lbl) + f" ({suite})",
            marker=dict(
                size=10,
                symbol=shape_map[suite],
                color=color_map[lbl]
            ),
        ))
    return fig

def load(path: str = "results.jsonl") -> list[dict]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return [r for r in records if r.get("error") is None]


def per_task_table(records: list[dict]) -> dict:
    """{(model, suite, task_id): {"cost": mean, "acc": mean, "tokens": mean}}"""
    groups = defaultdict(list)
    for r in records:
        groups[(r["model"], r["suite"], r["task_id"])].append(r)
    table = {}
    for key, rs in groups.items():
        table[key] = {
            "cost": float(np.mean([r["cost_usd"] for r in rs])),
            "acc": float(np.mean([r["correct"] for r in rs])),
            "out_tokens": float(np.mean([r["completion_tokens"] for r in rs])),
            "latency": float(np.median([r["latency_ms"] for r in rs])),
            "n": len(rs),
        }
    return table

def arch_summary(table: dict, suite: str | None = None) -> list[dict]:
    agg = defaultdict(lambda: {"cost": [], "acc": [], "out_tokens": [], "latency": []})
    for (model, s, _tid), row in table.items():
        if suite and s != suite:
            continue
        spec = by_id(model)
        arch = spec.arch_label if spec else "?"
        for k in agg[arch]:
            agg[arch][k].append(row[k])
    out = []
    for arch, cols in agg.items():
        total_cost = float(np.sum(cols["cost"]))
        mean_acc = float(np.mean(cols["acc"]))
        solved = mean_acc * len(cols["acc"])
        out.append({
            "arch": arch,
            "acc": mean_acc,
            "mean_cost": float(np.mean(cols["cost"])),
            "cost_per_solved": total_cost / solved if solved else float("inf"),
            "mean_out_tokens": float(np.mean(cols["out_tokens"])),
            "median_latency_ms": float(np.median(cols["latency"])),
        })
    data_out = sorted(out, key=lambda r: r["cost_per_solved"])

    return data_out


def model_summary(table: dict, suite: str | None = None) -> list[dict]:
    agg = defaultdict(lambda: {"cost": [], "acc": [], "out_tokens": [], "latency": []})
    for (model, s, _tid), row in table.items():
        if suite and s != suite:
            continue
        for k in agg[model]:
            agg[model][k].append(row[k])
    out = []
    for model, cols in agg.items():
        spec = by_id(model)
        total_cost = float(np.sum(cols["cost"]))
        mean_acc = float(np.mean(cols["acc"]))
        solved = mean_acc * len(cols["acc"])
        out.append({
            "model": model,
            "arch": spec.arch_label if spec else "?",
            "active_b": spec.active_params_b if spec else None,
            "acc": mean_acc,
            "mean_cost": float(np.mean(cols["cost"])),
            "cost_per_solved": total_cost / solved if solved else float("inf"),
            "mean_out_tokens": float(np.mean(cols["out_tokens"])),
            "median_latency_ms": float(np.median(cols["latency"])),
        })
    return sorted(out, key=lambda r: r["cost_per_solved"])


def paired_cost_compare(table: dict, model_a: str, model_b: str,
                        suite: str | None = None) -> dict | None:
    """Paired bootstrap on per-task cost deltas (a - b), shared tasks only."""
    deltas = []
    for (model, s, tid), row in table.items():
        if model != model_a or (suite and s != suite):
            continue
        other = table.get((model_b, s, tid))
        if other:
            deltas.append(row["cost"] - other["cost"])
    if len(deltas) < 3:
        return None
    d = np.array(deltas)
    boots = np.array([RNG.choice(d, size=len(d), replace=True).mean()
                      for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "a": model_a, "b": model_b, "n_tasks": len(d),
        "mean_delta_usd": float(d.mean()),
        "ci95": (float(lo), float(hi)),
        "conclusive": bool(lo > 0 or hi < 0),  # CI excludes zero
    }


def report(path: str = "results.jsonl") -> None:
    records = load(path)
    if not records:
        print("No successful records found.")
        return
    table = per_task_table(records)
    suites = sorted({s for (_m, s, _t) in table})
    figs=[]
    for suite in suites + [None]:
        label = suite or "ALL SUITES"
        print(f"\n== {label} " + "=" * (70 - len(label)))
        print(f"{'model':34} {'arch':22} {'acc':>5} {'$/task':>9} "
              f"{'$/solved':>9} {'out_tok':>8}")
        for row in model_summary(table, suite):
            
            print(f"{row['model'].split('/')[-1]:34} {row['arch']:22} "
                  f"{row['acc']:>5.0%} {row['mean_cost']:>9.5f} "
                  f"{row['cost_per_solved']:>9.5f} {row['mean_out_tokens']:>8.0f}")
        data_out = arch_summary(table, suite)
        c=0
        print("\n== Architecture summary ==")
        suites_in = []
        for row in data_out:
            print(f"{row['arch']:22} "
                  f"{row['acc']:>5.0%} {row['mean_cost']:>9.5f} "
                  f"{row['cost_per_solved']:>9.5f} {row['mean_out_tokens']:>8.0f}")
        figs.append(create_plotly_table([row['acc'] for row in data_out], [row['cost_per_solved'] for row in data_out], [row['arch'] for row in data_out], suite if suite else 'all'))

    merged = go.Figure()
    for f in figs:
        merged.add_traces(f.data)

    merged.update_layout(figs[0].layout)   # reuse title, axis titles, etc. from the first figure
    merged.show()

    models = sorted({m for (m, _s, _t) in table})
    print("\n== Paired cost comparisons (bootstrap 95% CI on per-task delta) ==")
    for a, b in itertools.combinations(models, 2):
        c = paired_cost_compare(table, a, b)
        if c:
            verdict = "conclusive" if c["conclusive"] else "CI crosses 0"
            print(f"{a.split('/')[-1]:28} vs {b.split('/')[-1]:28} "
                  f"Δ=${c['mean_delta_usd']:+.5f} "
                  f"CI[{c['ci95'][0]:+.5f},{c['ci95'][1]:+.5f}] ({verdict})")
