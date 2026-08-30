"""Rank architecture_search_strategies.py's results.csv by LOO mean rel.RMSE (lower = better).
Run `python architecture_search_strategies.py populate` first to (re)build results.csv from
whatever combos have finished so far -- this script only reads it, never touches metrics.json
or models_dir, so it's safe/cheap to re-run anytime, including mid-sweep.

Examples:
  python analyze_results.py                          # top 20 overall
  python analyze_results.py --top 50 --strategy delta # top 50, delta strategy only
  python analyze_results.py --by_strategy             # best result per strategy + summary stats
  python analyze_results.py --h_arch_hash 725b5481c8c7  # only H=h32_l1 combos
"""
import argparse
import csv

import numpy as np


def load_rows(results_csv):
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("status") == "ok" and r.get("loo_mean_rel_rmse")]
    for r in rows:
        r["loo_mean_rel_rmse"] = float(r["loo_mean_rel_rmse"])
        r["full_fit_mean_rel_rmse"] = float(r["full_fit_mean_rel_rmse"])
    return rows


def print_table(rows, n):
    print(f"{'rank':4s} {'strategy':9s} {'loo_mean':>9s} {'full_fit':>9s} {'h_hidden':>8s} "
          f"{'n_params_f':>10s}  f_arch_hash   h_arch_hash   combo_id")
    for i, r in enumerate(rows[:n]):
        print(f"{i+1:4d} {r['strategy']:9s} {r['loo_mean_rel_rmse']*100:8.2f}% "
              f"{r['full_fit_mean_rel_rmse']*100:8.2f}% {r['h_hidden']:>8s} {r['n_params_f']:>10s}  "
              f"{r['f_arch_hash']}  {r['h_arch_hash']}  {r['combo_id']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv",
                     default=r"C:\Users\acost\repos\transistor_modeling_hyper_outputs\results.csv")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--strategy", choices=["original", "delta", "ensemble"], default=None,
                     help="Only show this strategy. Default: all three (ranked together).")
    ap.add_argument("--h_arch_hash", default=None, help="Only show combos using this H architecture.")
    ap.add_argument("--by_strategy", action="store_true",
                     help="Instead of one ranked table, print count/mean/best LOO per strategy, "
                          "then each strategy's own top result.")
    args = ap.parse_args()

    rows = load_rows(args.results_csv)
    print(f"Loaded {len(rows)} completed (status=ok) combos from {args.results_csv}\n")

    if args.strategy:
        rows = [r for r in rows if r["strategy"] == args.strategy]
    if args.h_arch_hash:
        rows = [r for r in rows if r["h_arch_hash"] == args.h_arch_hash]
    rows.sort(key=lambda r: r["loo_mean_rel_rmse"])

    if args.by_strategy:
        for strat in ("original", "delta", "ensemble"):
            srows = [r for r in rows if r["strategy"] == strat]
            if not srows:
                continue
            loos = [r["loo_mean_rel_rmse"] for r in srows]
            best = srows[0]
            print(f"--- {strat} ({len(srows)} combos): mean LOO={np.mean(loos)*100:.2f}%  "
                  f"median={np.median(loos)*100:.2f}%  best={min(loos)*100:.2f}% ---")
            print(f"    best combo: {best['combo_id']}  "
                  f"(h_hidden={best['h_hidden']}, n_params_f={best['n_params_f']})")
        print()

    print_table(rows, args.top)


if __name__ == "__main__":
    main()
