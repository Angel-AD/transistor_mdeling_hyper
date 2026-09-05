"""Two-stage report builder for architecture_search_strategies.py's results.csv:
  1. Scans results.csv, computes the "best runs" groupings (global top-N, best per H
     architecture, best per H-architecture x strategy, summary stats), and writes them to a
     JSON file -- a stable, inspectable intermediate artifact, not just an HTML side effect.
  2. Renders that JSON into a self-contained report.html with one table per grouping.

If results.csv carries the optional gm columns (written by `populate --with_gm` for the
full-fit model and `populate --with_loo_gm` for the held-out folds), the report also gets
gm1/gm2/gm3 + combined_gm columns on the existing tables plus dedicated "ranked by gm"
tables. Without those columns the report is exactly as before -- everything gm degrades
away cleanly.

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

# Optional per-combo gm rel-RMSE columns (see architecture_search_strategies.py). "combined_gm"
# is the equal-weight mean of gm1/gm2/gm3. The loo_* variants are the held-out-fold counterparts.
GM_FULLFIT_COLS = ["gm1_rel_rmse", "gm2_rel_rmse", "gm3_rel_rmse", "combined_gm_rel_rmse"]
GM_LOO_COLS = ["loo_gm1_rel_rmse", "loo_gm2_rel_rmse", "loo_gm3_rel_rmse", "loo_combined_gm_rel_rmse"]
_OPTIONAL_FLOAT_COLS = GM_FULLFIT_COLS + GM_LOO_COLS
_GM_COMBINED_FULLFIT = "combined_gm_rel_rmse"
_GM_COMBINED_LOO = "loo_combined_gm_rel_rmse"


def load_rows(results_csv):
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("status") == "ok" and r.get("loo_mean_rel_rmse")]
    for r in rows:
        r["loo_mean_rel_rmse"] = float(r["loo_mean_rel_rmse"])
        r["full_fit_mean_rel_rmse"] = float(r["full_fit_mean_rel_rmse"])
        for c in _OPTIONAL_FLOAT_COLS:
            v = r.get(c, "")
            r[c] = float(v) if v not in (None, "") else None  # absent/blank -> None (no gm for this combo)
    return rows


def _availability(rows):
    """Which gm groupings are worth rendering, from whether any row actually carries the data."""
    return {
        "fullfit_gm": any(r.get(_GM_COMBINED_FULLFIT) is not None for r in rows),
        "loo_gm": any(r.get(_GM_COMBINED_LOO) is not None for r in rows),
    }


def _slim(r):
    """Only the fields worth keeping in the JSON/HTML tables -- results.csv carries a lot more
    (per-run hyperparameters, timestamps, etc.) that isn't useful to skim in a "best runs" view.
    gm fields are carried through as-is (None when that combo has no gm scored)."""
    out = {
        "combo_id": r["combo_id"],
        "f_arch_hash": r["f_arch_hash"],
        "h_arch_hash": r["h_arch_hash"],
        "strategy": r["strategy"],
        "h_label": f"h{r['h_hidden']}_l{r['h_layers']}",
        "n_params_f": int(r["n_params_f"]),
        "loo_mean_rel_rmse": r["loo_mean_rel_rmse"],
        "full_fit_mean_rel_rmse": r["full_fit_mean_rel_rmse"],
    }
    out.update({c: r.get(c) for c in _OPTIONAL_FLOAT_COLS})
    return out


def _col_substats(rows, col):
    """mean/median/best (+ owning combo_id) of `col` over the rows that have it, or None."""
    have = [r for r in rows if r.get(col) is not None]
    if not have:
        return None
    vals = [r[col] for r in have]
    return {
        "n": len(have),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "best": float(min(vals)),
        "best_combo_id": min(have, key=lambda r: r[col])["combo_id"],
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
        # None unless the relevant gm columns are present for at least one row in this group
        "gm_combined": _col_substats(rows, _GM_COMBINED_FULLFIT),
        "loo_gm_combined": _col_substats(rows, _GM_COMBINED_LOO),
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

    avail = _availability(rows)
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

    def _top_by(col):
        have = [r for r in rows if r.get(col) is not None]
        return [_slim(r) for r in sorted(have, key=lambda r: r[col])[:top_n]]

    # gm-ranked H x strategy: best combo of each cell by held-out combined_gm (or full-fit
    # combined_gm when only that is available)
    gm_sort_col = _GM_COMBINED_LOO if avail["loo_gm"] else _GM_COMBINED_FULLFIT
    gm_best_per_h_and_strategy = {}
    if avail["loo_gm"] or avail["fullfit_gm"]:
        for h in h_arch_hashes:
            cell = {}
            for strat in strategies:
                cand = [r for r in rows if r["h_arch_hash"] == h and r["strategy"] == strat
                        and r.get(gm_sort_col) is not None]
                if cand:
                    cell[strat] = _slim(min(cand, key=lambda r: r[gm_sort_col]))
            if cell:
                gm_best_per_h_and_strategy[h] = cell

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results_csv": os.path.abspath(results_csv),
        "n_combos": len(rows),
        "n_f_architectures_done": n_f_done,
        "n_f_architectures_total": n_f_total,
        "h_label_by_hash": h_label_by_hash,
        "gm_available": avail,
        "gm_sort_col": gm_sort_col if (avail["loo_gm"] or avail["fullfit_gm"]) else None,
        "global_top": [_slim(r) for r in rows_sorted[:top_n]],
        "global_top_by_loo_gm": _top_by(_GM_COMBINED_LOO) if avail["loo_gm"] else [],
        "global_top_by_fullfit_gm": _top_by(_GM_COMBINED_FULLFIT) if avail["fullfit_gm"] else [],
        "strategy_summary": best_per_strategy,
        "h_arch_summary": best_per_h_arch,
        "best_per_h_and_strategy": best_per_h_and_strategy,
        "gm_best_per_h_and_strategy": gm_best_per_h_and_strategy,
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
p.note {{ opacity: 0.75; font-size: 0.9rem; font-style: italic; }}
.wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
th, td {{ border: 1px solid #88888844; padding: 0.4rem 0.7rem; text-align: right; font-variant-numeric: tabular-nums; }}
th {{ background: #80808022; }}
td:first-child, th:first-child, td.left, th.left {{ text-align: left; }}
tr:nth-child(1) td {{ font-weight: 600; }}
code {{ font-family: ui-monospace, Consolas, monospace; font-size: 0.85em; }}
.gm {{ background: #8888ff11; }}
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


def _pct(v):
    return "&mdash;" if v is None else f"{v * 100:.2f}%"


def _table(rows_html):
    return "<div class='wrap'><table>" + "".join(rows_html) + "</table></div>"


# ----- combo-row tables (top-N, per-H x strategy): LOO Ids + full-fit Ids, optionally + gm -----

def _combo_header(show_h=True, gm=False):
    cols = ["<th class='left'>strategy</th>"]
    if show_h:
        cols.append("<th class='left'>H</th>")
    cols += ["<th>LOO Ids</th>", "<th>full-fit Ids</th>"]
    if gm:
        cols += ["<th class='gm'>LOO comb.gm</th>", "<th class='gm'>full-fit comb.gm</th>"]
    cols += ["<th>n_params(f)</th>", "<th class='left'>combo_id</th>"]
    return "<tr>" + "".join(cols) + "</tr>"


def _combo_row(r, show_h=True, leading_cells="", gm=False):
    """leading_cells: extra <td>...</td> markup to prepend (e.g. the H label, when the table's
    own first column is something else and h_label is shown separately)."""
    cells = [leading_cells, f"<td class='left'>{_strat_span(r['strategy'])}</td>"]
    if show_h:
        cells.append(f"<td class='left'>{_html.escape(r['h_label'])}</td>")
    cells += [f"<td>{_pct(r['loo_mean_rel_rmse'])}</td>",
              f"<td>{_pct(r['full_fit_mean_rel_rmse'])}</td>"]
    if gm:
        cells += [f"<td class='gm'>{_pct(r.get('loo_combined_gm_rel_rmse'))}</td>",
                  f"<td class='gm'>{_pct(r.get('combined_gm_rel_rmse'))}</td>"]
    cells += [f"<td>{r['n_params_f']}</td>",
              f"<td class='left'><code>{_html.escape(r['combo_id'])}</code></td>"]
    return "<tr>" + "".join(cells) + "</tr>"


# ----- gm-ranked tables: the gm1/gm2/gm3 breakdown for the metric being ranked on -----

def _gm_cols_and_label(avail):
    if avail["loo_gm"]:
        return GM_LOO_COLS, "held-out"
    return GM_FULLFIT_COLS, "full-fit"


def _gm_header(gm_cols, label, show_h=True):
    cols = ["<th class='left'>strategy</th>"]
    if show_h:
        cols.append("<th class='left'>H</th>")
    cols += [f"<th class='gm'>{label} gm1</th>", f"<th class='gm'>{label} gm2</th>",
             f"<th class='gm'>{label} gm3</th>", f"<th class='gm'>{label} comb.gm</th>",
             "<th>LOO Ids</th>", "<th class='left'>combo_id</th>"]
    return "<tr>" + "".join(cols) + "</tr>"


def _gm_row(r, gm_cols, show_h=True, leading_cells=""):
    cells = [leading_cells, f"<td class='left'>{_strat_span(r['strategy'])}</td>"]
    if show_h:
        cells.append(f"<td class='left'>{_html.escape(r['h_label'])}</td>")
    cells += [f"<td class='gm'>{_pct(r.get(c))}</td>" for c in gm_cols]
    cells += [f"<td>{_pct(r['loo_mean_rel_rmse'])}</td>",
              f"<td class='left'><code>{_html.escape(r['combo_id'])}</code></td>"]
    return "<tr>" + "".join(cells) + "</tr>"


# ----- summary tables (per H, per strategy): LOO Ids stats, optionally + combined_gm stats -----

def _summary_header(first_label, avail):
    cols = [f"<th class='left'>{first_label}</th>", "<th>combos</th>", "<th>mean LOO Ids</th>",
            "<th>median</th>", "<th>best</th>"]
    if avail["loo_gm"]:
        cols += ["<th class='gm'>mean LOO comb.gm</th>", "<th class='gm'>best</th>"]
    if avail["fullfit_gm"]:
        cols += ["<th class='gm'>mean full-fit comb.gm</th>", "<th class='gm'>best</th>"]
    cols.append("<th class='left'>best combo_id (by LOO Ids)</th>")
    return "<tr>" + "".join(cols) + "</tr>"


def _summary_row(first_cell, stats, avail):
    cells = [f"<td class='left'>{first_cell}</td>", f"<td>{stats['n']}</td>",
             f"<td>{_pct(stats['mean'])}</td>", f"<td>{_pct(stats['median'])}</td>",
             f"<td>{_pct(stats['best'])}</td>"]
    if avail["loo_gm"]:
        s = stats.get("loo_gm_combined")
        cells += [f"<td class='gm'>{_pct(s['mean']) if s else '&mdash;'}</td>",
                  f"<td class='gm'>{_pct(s['best']) if s else '&mdash;'}</td>"]
    if avail["fullfit_gm"]:
        s = stats.get("gm_combined")
        cells += [f"<td class='gm'>{_pct(s['mean']) if s else '&mdash;'}</td>",
                  f"<td class='gm'>{_pct(s['best']) if s else '&mdash;'}</td>"]
    cells.append(f"<td class='left'><code>{_html.escape(stats['best_combo_id'])}</code></td>")
    return "<tr>" + "".join(cells) + "</tr>"


def build_html(summary, out_path):
    avail = summary["gm_available"]
    any_gm = avail["loo_gm"] or avail["fullfit_gm"]
    gm_cols, gm_label = _gm_cols_and_label(avail)

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
    if not any_gm:
        body.append("<p class='note'>results.csv has no gm columns &mdash; run "
                    "<code>populate --with_gm</code> (full-fit) and/or "
                    "<code>populate --with_loo_gm</code> (held-out folds) to add gm1/gm2/gm3 "
                    "and combined_gm tables here.</p>")
    else:
        body.append("<p class='note'>comb.gm = equal-weight mean of gm1/gm2/gm3 rel-RMSE "
                    "(gm1=dIds/dVgs, gm2=d&sup2;, gm3=d&sup3;). "
                    f"gm-ranked tables below rank on the <b>{gm_label}</b> combined_gm.</p>")

    # --- Table 1: global top-N (by LOO Ids) ---
    body.append(f"<h2>Top {len(summary['global_top'])} overall by LOO Ids (any strategy, any H)</h2>")
    body.append(_table([_combo_header(show_h=True, gm=any_gm)]
                       + [_combo_row(r, show_h=True, gm=any_gm) for r in summary["global_top"]]))

    # --- Table 2: best per H architecture ---
    body.append("<h2>Per H architecture</h2>")
    body.append(_table([_summary_header("H", avail)]
                       + [_summary_row(_html.escape(summary["h_label_by_hash"].get(h, h)), st, avail)
                          for h, st in summary["h_arch_summary"].items()]))

    # --- Table 3: best per strategy ---
    body.append("<h2>Per strategy</h2>")
    body.append(_table([_summary_header("strategy", avail)]
                       + [_summary_row(_strat_span(strat), st, avail)
                          for strat, st in summary["strategy_summary"].items()]))

    # --- Table 4: best per H architecture x strategy (by LOO Ids) ---
    body.append("<h2>Best by LOO Ids, per H architecture &times; strategy</h2>")
    t4 = [_combo_header(show_h=False, gm=any_gm).replace("<tr>", "<tr><th class='left'>H</th>", 1)]
    for h, by_strat in summary["best_per_h_and_strategy"].items():
        for strat, r in by_strat.items():
            t4.append(_combo_row(r, show_h=False,
                                 leading_cells=f"<td class='left'>{_html.escape(r['h_label'])}</td>",
                                 gm=any_gm))
    body.append(_table(t4))

    # --- Table 5: global top-N by held-out combined_gm ---
    if summary["global_top_by_loo_gm"]:
        body.append(f"<h2>Top {len(summary['global_top_by_loo_gm'])} overall by held-out combined_gm</h2>")
        body.append(_table([_gm_header(GM_LOO_COLS, "held-out", show_h=True)]
                           + [_gm_row(r, GM_LOO_COLS, show_h=True)
                              for r in summary["global_top_by_loo_gm"]]))

    # --- Table 6: global top-N by full-fit combined_gm ---
    if summary["global_top_by_fullfit_gm"]:
        body.append(f"<h2>Top {len(summary['global_top_by_fullfit_gm'])} overall by full-fit combined_gm</h2>")
        body.append(_table([_gm_header(GM_FULLFIT_COLS, "full-fit", show_h=True)]
                           + [_gm_row(r, GM_FULLFIT_COLS, show_h=True)
                              for r in summary["global_top_by_fullfit_gm"]]))

    # --- Table 7: best by gm, per H architecture x strategy ---
    if summary["gm_best_per_h_and_strategy"]:
        body.append(f"<h2>Best by {gm_label} combined_gm, per H architecture &times; strategy</h2>")
        t7 = [_gm_header(gm_cols, gm_label, show_h=False).replace(
            "<tr>", "<tr><th class='left'>H</th>", 1)]
        for h, by_strat in summary["gm_best_per_h_and_strategy"].items():
            for strat, r in by_strat.items():
                t7.append(_gm_row(r, gm_cols, show_h=False,
                                  leading_cells=f"<td class='left'>{_html.escape(r['h_label'])}</td>"))
        body.append(_table(t7))

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
    ap.add_argument("--top", type=int, default=30, help="Rows in the global top-N tables.")
    args = ap.parse_args()

    summary = build_summary(args.results_csv, args.f_csv, args.top)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "search_report.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    html_path = os.path.join(args.out_dir, "search_report.html")
    build_html(summary, html_path)

    gm = summary["gm_available"]
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")
    print(f"  gm columns: full-fit={'yes' if gm['fullfit_gm'] else 'no'}, "
          f"held-out(LOO)={'yes' if gm['loo_gm'] else 'no'}")


if __name__ == "__main__":
    main()
