"""
Grid search over (gm1_weight, ids_region_weight) combinations, scored by LOO mean
rel.RMSE, for a FIXED main-net architecture (default: rand032, the best confirmed
candidate from architecture_search.py). Both knobs were tried individually and each
made LOO worse alone (gm1=0.1: 2.22%->2.90% on arch37; ids_region_weight=5.0: 2.33%->
3.03% on rand032) -- this sweeps a grid spanning and interpolating both extremes,
including the untouched (0,0) baseline, to see whether some COMBINATION (or a gentler
value of either) helps where neither did alone.

Fast proxy: reduced epochs (default 2000) and lbfgs_epochs=0, no plots/model-saving --
same rationale as architecture_search.py. Re-run the winning combo through train_loo.py
at the full epoch count (+ L-BFGS) to confirm.

Usage:
    python loss_hparam_search.py --csv_dir "C:\\Users\\acost\\repos\\csvs" --workers 15
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from data_loader import load_all
from model import ARCHITECTURES, main_net_n_params
from plotting import qtag
from train_loo import build_tensors, train_hypernet, evaluate

# Full-factorial grid: 4 gm1_weight values x 5 ids_region_weight values = 20 combinations.
# Includes both previously-tried extremes (gm1=0.1 alone; ids_region_weight=5.0 alone) and
# the untouched (0,0) baseline, plus gentler intermediate values of each.
GM1_WEIGHTS = [0.0, 0.02, 0.05, 0.1]
IDS_REGION_WEIGHTS = [0.0, 0.5, 1.5, 3.0, 5.0]


def generate_combinations():
    combos = {}
    i = 0
    for gm1_w in GM1_WEIGHTS:
        for region_w in IDS_REGION_WEIGHTS:
            combos[f"combo{i:02d}"] = dict(gm1_weight=gm1_w, ids_region_weight=region_w)
            i += 1
    return combos


def run_loo_fold_job(combo_id, gm1_weight, ids_region_weight, held_out, data, architecture,
                      n_params, epochs, lr, ids_region_frac):
    """One (hparam combo, held-out point) LOO fold, in its own process. No plotting/saving --
    this is a fast scoring pass, not the final validated run."""
    torch.set_num_threads(1)
    device = torch.device("cpu")
    keys = sorted(data.keys())
    train_keys = [k for k in keys if k != held_out]
    tensors = build_tensors(data, device, ids_region_frac=ids_region_frac)
    hyper, _ = train_hypernet(train_keys, tensors, n_params, architecture, epochs, lr, device,
                               gm1_weight=gm1_weight, ids_region_weight=ids_region_weight,
                               lbfgs_epochs=0, log_every=0, seed=27)
    err, _ = evaluate(hyper, held_out, tensors, architecture, device)
    return combo_id, held_out, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=r"C:\Users\acost\repos\csvs")
    ap.add_argument("--architecture", default="rand032",
                     help="Preset name from model.ARCHITECTURES, or a raw JSON architecture "
                          "string. Default is the winning architecture found by "
                          "architecture_search.py (87 params).")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=2000,
                     help="Reduced-epoch proxy for ranking combos fast; re-run the winner at "
                          "train_loo.py's full epoch count (+ L-BFGS) to confirm.")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--ids_region_frac", type=float, default=0.05)
    ap.add_argument("--output_json",
                     default=r"C:\Users\acost\repos\transistor_modeling_hyper_outputs\loss_hparam_search_results.json",
                     help="Defaults OUTSIDE this repo (sibling folder) so runs never create "
                          "untracked files inside the git repo.")
    args = ap.parse_args()

    data = load_all(args.csv_dir)
    keys = sorted(data.keys())
    print("Quiescent points found:", keys)

    if args.architecture in ARCHITECTURES:
        architecture = ARCHITECTURES[args.architecture]
    else:
        architecture = json.loads(args.architecture)
    n_params = main_net_n_params(architecture)
    print(f"Fixed architecture: {architecture} ({n_params} params)")

    combos = generate_combinations()
    print(f"Generated {len(combos)} (gm1_weight, ids_region_weight) combinations")

    jobs = []
    for combo_id, hp in combos.items():
        for held_out in keys:
            jobs.append((combo_id, hp["gm1_weight"], hp["ids_region_weight"], held_out))

    print(f"Submitting {len(jobs)} LOO-fold jobs ({args.epochs} epochs each) "
          f"across {args.workers} worker processes ...")

    results = {combo_id: {} for combo_id in combos}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(run_loo_fold_job, combo_id, gm1_w, region_w, held_out, data,
                      architecture, n_params, args.epochs, args.lr, args.ids_region_frac): combo_id
            for combo_id, gm1_w, region_w, held_out in jobs
        }
        for fut in as_completed(futures):
            combo_id, held_out, err = fut.result()
            results[combo_id][held_out] = err
            done += 1
            if done % 20 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                print(f"  {done}/{len(jobs)} folds done ({elapsed:.0f}s elapsed)", flush=True)

    summary = []
    for combo_id, hp in combos.items():
        errs = [results[combo_id][k] for k in keys]
        summary.append(dict(
            combo_id=combo_id,
            gm1_weight=hp["gm1_weight"],
            ids_region_weight=hp["ids_region_weight"],
            loo_mean_rel_rmse=float(np.mean(errs)),
            loo_worst_rel_rmse=float(np.max(errs)),
            loo_per_point={qtag(*k): e for k, e in zip(keys, errs)},
        ))
    summary.sort(key=lambda d: d["loo_mean_rel_rmse"])

    with open(args.output_json, "w") as f:
        json.dump(dict(architecture=architecture, n_params=n_params, epochs=args.epochs,
                        lr=args.lr, ids_region_frac=args.ids_region_frac, results=summary), f, indent=2)

    print(f"\n=== All 20 combinations, ranked by LOO mean rel.RMSE ({args.epochs} epochs, proxy) ===")
    for d in summary:
        print(f"  {d['combo_id']:8s} gm1_weight={d['gm1_weight']:5.2f}  "
              f"ids_region_weight={d['ids_region_weight']:4.1f}  "
              f"LOO mean={d['loo_mean_rel_rmse']*100:6.2f}%  worst={d['loo_worst_rel_rmse']*100:6.2f}%")

    best = summary[0]
    print(f"\nBest: {best['combo_id']} (gm1_weight={best['gm1_weight']}, "
          f"ids_region_weight={best['ids_region_weight']}), "
          f"LOO mean={best['loo_mean_rel_rmse']*100:.2f}% at {args.epochs} epochs, proxy.")
    print("To confirm with the full pipeline (plots, saved models, HTML report, 4000 epochs + L-BFGS):")
    print(f"  python train_loo.py --csv_dir \"{args.csv_dir}\" --epochs 4000 --output_dir results_best "
          f"--architecture '{json.dumps(architecture)}' --gm1_weight {best['gm1_weight']} "
          f"--ids_region_weight {best['ids_region_weight']}")
    print(f"\nFull results saved to {args.output_json}")


if __name__ == "__main__":
    main()
