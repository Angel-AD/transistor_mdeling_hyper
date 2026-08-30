"""Two-stage report builder for architecture_search_strategies.py's results.csv:
  1. Scans results.csv, computes the "best runs" groupings (global top-N, best per H
     architecture, best per H-architecture x strategy, summary stats), and writes them to a
     JSON file -- a stable, inspectable intermediate artifact, not just an HTML side effect.
  2. Renders that JSON into a self-contained report.html with one table per grouping.

Safe/cheap to re-run anytime, including mid-sweep (just reads results.csv, never touches
metrics.json or models_dir). Run `python architecture_search_strategies.py populate` first to
(re)build results.csv from whatever combos have finished so far.

Usage:
    python build_search_report.py
    python build_search_report.py --results_csv <path> --out_dir <dir> --top 30
"""
import argparse
import csv
import html as _html
import json
import os
from datetime import datetime, timezone

import numpy as np


def load_rows(results_csv):
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("status") == "ok" and r.get("loo_mean_rel_rmse")]
    for r in rows:
        r["loo_mean_rel_rmse"] = float(r["loo_mean_rel_rmse"])
        r["full_fit_mean_rel_rmse"] = float(r["full_fit_mean_rel_rmse"])
    return rows


def _slim(r):
    """Only the fields worth keeping in the JSON/HTML tables -- results.csv carries a lot more
    (per-run hyperparameters, timestamps, etc.) that isn't useful to skim in a "best runs" view."""
    return {
        "combo_id": r["combo_id"],
        "f_arch_hash": r["f_arch_hash"],
        "h_arch_hash": r["h_arch_hash"],
        "strategy": r["strategy"],
        "h_label": f"h{r['h_hidden']}_l{r['h_layers']}",
        "n_params_f": int(r["n_params_f"]),
        "loo_mean_rel_rmse": r["loo_mean_rel_rmse"],
        "full_fit_mean_rel_rmse": r["full_fit_mean_rel_rmse"],
    }


def _summary_stats(rows):
    loos = [r["loo_mean_rel_rmse"] for r in rows]
    best = min(rows, key=lambda r: r["loo_mean_rel_rmse"])
    return {
        "n": len(rows),
        "mean": float(np.mean(loos)),
        "median": float(np.median(loos)),
        "best": float(min(loos)),
        "worst": float(max(loos)),
        "best_combo_id": best["combo_id"],
    }


def build_summary(results_csv, f_csv, top_n):
    rows = load_rows(results_csv)
    strategies = ("original", "delta", "ensemble")
    h_arch_hashes = sorted({r["h_arch_hash"] for r in rows},
                            key=lambda h: -int(next(r["h_hidden"] for r in rows if r["h_arch_hash"] == h)))

    h_label_by_hash = {r["h_arch_hash"]: f"h{r['h_hidden']}_l{r['h_layers']}" for r in rows}
    n_f_done = len({r["f_arch_hash"] for r in rows})
    n_f_total = None
    if f_csv and os.path.exists(f_csv):
        with open(f_csv, newline="") as f:
            n_f_total = sum(1 for _ in csv.DictReader(f))

    rows_sorted = sorted(rows, key=lambda r: r["loo_mean_rel_rmse"])

    best_per_strategy = {}
    for strat in strategies:
        srows = [r for r in rows if r["strategy"] == strat]
        if srows:
            best_per_strategy[strat] = _summary_stats(srows)

    best_per_h_arch = {}
    for h in h_arch_hashes:
        hrows = [r for r in rows if r["h_arch_hash"] == h]
        if hrows:
            best_per_h_arch[h] = _summary_stats(hrows)

    best_per_h_and_strategy = {}
    for h in h_arch_hashes:
        best_per_h_and_strategy[h] = {}
        for strat in strategies:
            hsrows = [r for r in rows if r["h_arch_hash"] == h and r["strategy"] == strat]
            if hsrows:
                best = min(hsrows, key=lambda r: r["loo_mean_rel_rmse"])
                best_per_h_and_strategy[h][strat] = _slim(best)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results_csv": os.path.abspath(results_csv),
        "n_combos": len(rows),
        "n_f_architectures_done": n_f_done,
        "n_f_architectures_total": n_f_total,
        "h_label_by_hash": h_label_by_hash,
        "global_top": [_slim(r) for r in rows_sorted[:top_n]],
        "strategy_summary": best_per_strategy,
        "h_arch_summary": best_per_h_arch,
        "best_per_h_and_strategy": best_per_h_and_strategy,
    }
    return summary


_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: -apple-system, "Segoe UI", sans-serif; max-width: 1200px; margin: 2rem auto;
        padding: 0 1rem; line-height: 1.5; }}
h1 {{ border-bottom: 1px solid #88888844; padding-bottom: 0.3rem; }}
h2 {{ border-bottom: 1px solid #88888844; padding-bottom: 0.3rem; margin-top: 2.5rem; }}
p.meta {{ opacity: 0.75; font-size: 0.9rem; }}
table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
th, td {{ border: 1px solid #88888844; padding: 0.4rem 0.7rem; text-align: right; font-variant-numeric: tabular-nums; }}
th {{ background: #80808022; }}
td:first-child, th:first-child, td.left, th.left {{ text-align: left; }}
tr:nth-child(1) td {{ font-weight: 600; }}
code {{ font-family: ui-monospace, Consolas, monospace; font-size: 0.85em; }}
.strategy-original {{ color: #b45309; }}
.strategy-delta {{ color: #1d4ed8; }}
.strategy-ensemble {{ color: #15803d; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _strat_span(strat):
    return f"<span class='strategy-{strat}'>{strat}</span>"


def _combo_row(r, show_h=True, leading_cells=""):
    """leading_cells: extra <td>...</td> markup to prepend (e.g. the H label, when the table's
    own first column is something else and h_label is shown separately)."""
    cells = [leading_cells, f"<td class='left'>{_strat_span(r['strategy'])}</td>"]
    if show_h:
        cells.append(f"<td class='left'>{_html.escape(r['h_label'])}</td>")
    cells += [
        f"<td>{r['loo_mean_rel_rmse']*100:.2f}%</td>",
        f"<td>{r['full_fit_mean_rel_rmse']*100:.2f}%</td>",
        f"<td>{r['n_params_f']}</td>",
        f"<td class='left'><code>{_html.escape(r['combo_id'])}</code></td>",
    ]
    return "<tr>" + "".join(cells) + "</tr>"


def build_html(summary, out_path):
    progress = ""
    if summary["n_f_architectures_total"]:
        pct = 100 * summary["n_f_architectures_done"] / summary["n_f_architectures_total"]
        progress = (f" &mdash; {summary['n_f_architectures_done']}/{summary['n_f_architectures_total']} "
                    f"f-architectures done so far ({pct:.0f}%)")

    body = [
        "<h1>Architecture search results</h1>",
        f"<p class='meta'>Generated {summary['generated_at']} from "
        f"<code>{_html.escape(summary['results_csv'])}</code> &mdash; {summary['n_combos']} combos"
        f"{progress}.</p>",
    ]

    # --- Table 1: global top-N ---
    body.append(f"<h2>Top {len(summary['global_top'])} overall (any strategy, any H)</h2>")
    body.append("<table><tr><th class='left'>strategy</th><th class='left'>H</th><th>LOO mean</th>"
                "<th>full-fit</th><th>n_params(f)</th><th class='left'>combo_id</th></tr>")
    for r in summary["global_top"]:
        body.append(_combo_row(r, show_h=True))
    body.append("</table>")

    # --- Table 2: best per H architecture ---
    body.append("<h2>Best result per H architecture</h2>")
    body.append("<table><tr><th class='left'>H</th><th>combos run</th><th>mean LOO</th>"
                "<th>median LOO</th><th>best LOO</th><th class='left'>best combo_id</th></tr>")
    for h, stats in summary["h_arch_summary"].items():
        h_label = summary["h_label_by_hash"].get(h, h)
        body.append(f"<tr><td class='left'>{_html.escape(h_label)}</td><td>{stats['n']}</td>"
                    f"<td>{stats['mean']*100:.2f}%</td><td>{stats['median']*100:.2f}%</td>"
                    f"<td>{stats['best']*100:.2f}%</td>"
                    f"<td class='left'><code>{_html.escape(stats['best_combo_id'])}</code></td></tr>")
    body.append("</table>")

    # --- Table 3: best per strategy ---
    body.append("<h2>Best result per strategy</h2>")
    body.append("<table><tr><th class='left'>strategy</th><th>combos run</th><th>mean LOO</th>"
                "<th>median LOO</th><th>best LOO</th><th class='left'>best combo_id</th></tr>")
    for strat, stats in summary["strategy_summary"].items():
        body.append(f"<tr><td class='left'>{_strat_span(strat)}</td><td>{stats['n']}</td>"
                    f"<td>{stats['mean']*100:.2f}%</td><td>{stats['median']*100:.2f}%</td>"
                    f"<td>{stats['best']*100:.2f}%</td>"
                    f"<td class='left'><code>{_html.escape(stats['best_combo_id'])}</code></td></tr>")
    body.append("</table>")

    # --- Table 4: best per H architecture x strategy (helps compare strategies within one H) ---
    body.append("<h2>Best result per H architecture &times; strategy</h2>")
    body.append("<table><tr><th class='left'>H</th><th class='left'>strategy</th><th>LOO mean</th>"
                "<th>full-fit</th><th>n_params(f)</th><th class='left'>combo_id</th></tr>")
    for h, by_strat in summary["best_per_h_and_strategy"].items():
        for strat, r in by_strat.items():
            leading = f"<td class='left'>{_html.escape(r['h_label'])}</td>"
            body.append(_combo_row(r, show_h=False, leading_cells=leading))
    body.append("</table>")

    html_text = _PAGE_TEMPLATE.format(title="Architecture search results", body="\n".join(body))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv",
                     default=r"C:\Users\acost\repos\transistor_modeling_hyper_outputs\results.csv")
    ap.add_argument("--f_csv", default="configs/arch_search_1000/f_architectures.csv",
                     help="Used only to report 'X/N f-architectures done so far'.")
    ap.add_argument("--out_dir",
                     default=r"C:\Users\acost\repos\transistor_modeling_hyper_outputs")
    ap.add_argument("--top", type=int, default=30, help="Rows in the global top-N table.")
    args = ap.parse_args()

    summary = build_summary(args.results_csv, args.f_csv, args.top)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "search_report.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    html_path = os.path.join(args.out_dir, "search_report.html")
    build_html(summary, html_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
