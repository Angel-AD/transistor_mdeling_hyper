"""
Random search over main-net per-neuron mixed-activation architectures (tanh/swish),
scored by LOO mean rel.RMSE -- the metric that matters (generalization to an unseen
quiescent point), not the in-sample full-fit number.

Fast proxy: fewer epochs than the "confirmed" 4000-epoch runs (default 1500), and no
plots/model-saving per candidate (100 architectures x 6 LOO folds = 600 jobs would
otherwise write 100x the files train_loo.py does for one architecture). Re-run the
winning architecture(s) through train_loo.py at the full epoch count for plots, saved
models, and an HTML report -- train_loo.py's --architecture flag accepts either a
model.ARCHITECTURES preset name or a raw JSON architecture string, so a winner here can
be passed straight through, e.g.:

    python train_loo.py --architecture '[["tanh","swish","tanh"],["swish","swish"]]' ...

Usage:
    python architecture_search.py --csv_dir "C:\\Users\\acost\\repos\\csvs" \\
        --n_architectures 100 --workers 15 --epochs 1500
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from data_loader import load_all
from model import main_net_n_params
from plotting import qtag
from train_loo import build_tensors, train_hypernet, evaluate

ACTIVATIONS = ("tanh", "swish")


def random_architecture(rng, min_layers=1, max_layers=3, min_neurons=2, max_neurons=8):
    n_layers = int(rng.integers(min_layers, max_layers + 1))
    arch = []
    for _ in range(n_layers):
        n_neurons = int(rng.integers(min_neurons, max_neurons + 1))
        layer = [ACTIVATIONS[i] for i in rng.integers(0, len(ACTIVATIONS), size=n_neurons)]
        arch.append(layer)
    return arch


def generate_architectures(n, seed):
    rng = np.random.default_rng(seed)
    archs = {}
    seen = set()
    while len(archs) < n:
        arch = random_architecture(rng)
        sig = json.dumps(arch)
        if sig in seen:
            continue
        seen.add(sig)
        archs[f"rand{len(archs):03d}"] = arch
    return archs


def run_loo_fold_job(arch_id, architecture, held_out, data, n_params, epochs, lr):
    """One (architecture, held-out point) LOO fold, in its own process. No plotting/saving --
    this is a fast scoring pass, not the final validated run."""
    torch.set_num_threads(1)
    device = torch.device("cpu")
    keys = sorted(data.keys())
    train_keys = [k for k in keys if k != held_out]
    tensors = build_tensors(data, device)
    # lbfgs_epochs=0: this is a fast ranking proxy across ~100 candidates -- L-BFGS polishing
    # (added to train_loo.py for the final confirmed run) is skipped here to keep the search
    # fast; re-run the winner through train_loo.py, which polishes by default, to confirm.
    hyper, _ = train_hypernet(train_keys, tensors, n_params, architecture, epochs, lr, device,
                               lbfgs_epochs=0, log_every=0, seed=27)
    err, _ = evaluate(hyper, held_out, tensors, architecture, device)
    return arch_id, held_out, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=r"C:\Users\acost\repos\csvs")
    ap.add_argument("--n_architectures", type=int, default=100)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=1500,
                     help="Reduced-epoch proxy for ranking candidates fast; re-run the winner at "
                          "train_loo.py's full epoch count (default 4000) to confirm.")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json",
                     default=r"C:\Users\acost\repos\transistor_modeling_hyper_outputs\architecture_search_results.json",
                     help="Defaults OUTSIDE this repo (sibling folder) so runs never create "
                          "untracked files inside the git repo.")
    args = ap.parse_args()

    data = load_all(args.csv_dir)
    keys = sorted(data.keys())
    print("Quiescent points found:", keys)

    architectures = generate_architectures(args.n_architectures, args.seed)
    print(f"Generated {len(architectures)} candidate architectures (seed={args.seed})")

    jobs = []
    for arch_id, arch in architectures.items():
        n_params = main_net_n_params(arch)
        for held_out in keys:
            jobs.append((arch_id, arch, held_out, n_params))

    print(f"Submitting {len(jobs)} LOO-fold jobs ({args.epochs} epochs each) "
          f"across {args.workers} worker processes ...")

    results = {arch_id: {} for arch_id in architectures}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_loo_fold_job, arch_id, arch, held_out, data, n_params,
                              args.epochs, args.lr): arch_id
                   for arch_id, arch, held_out, n_params in jobs}
        for fut in as_completed(futures):
            arch_id, held_out, err = fut.result()
            results[arch_id][held_out] = err
            done += 1
            if done % 50 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                print(f"  {done}/{len(jobs)} folds done ({elapsed:.0f}s elapsed)")

    summary = []
    for arch_id, arch in architectures.items():
        errs = [results[arch_id][k] for k in keys]
        summary.append(dict(
            arch_id=arch_id,
            architecture=arch,
            n_params=main_net_n_params(arch),
            loo_mean_rel_rmse=float(np.mean(errs)),
            loo_worst_rel_rmse=float(np.max(errs)),
            loo_per_point={qtag(*k): e for k, e in zip(keys, errs)},
        ))
    summary.sort(key=lambda d: d["loo_mean_rel_rmse"])

    with open(args.output_json, "w") as f:
        json.dump(dict(epochs=args.epochs, lr=args.lr, seed=args.seed, results=summary), f, indent=2)

    print(f"\n=== Top 10 architectures by LOO mean rel.RMSE ({args.epochs} epochs, proxy score) ===")
    for d in summary[:10]:
        print(f"  {d['arch_id']:8s} n_params={d['n_params']:4d}  "
              f"LOO mean={d['loo_mean_rel_rmse']*100:6.2f}%  worst={d['loo_worst_rel_rmse']*100:6.2f}%  "
              f"{d['architecture']}")

    print(f"\n=== Worst 5 (for contrast) ===")
    for d in summary[-5:]:
        print(f"  {d['arch_id']:8s} n_params={d['n_params']:4d}  LOO mean={d['loo_mean_rel_rmse']*100:6.2f}%")

    best = summary[0]
    print(f"\nBest: {best['arch_id']} ({best['n_params']} params, "
          f"LOO mean={best['loo_mean_rel_rmse']*100:.2f}% at {args.epochs} epochs, proxy).")
    print("To confirm with the full pipeline (plots, saved models, HTML report, 4000 epochs):")
    print(f"  python train_loo.py --csv_dir \"{args.csv_dir}\" --epochs 4000 "
          f"--output_dir results_best --architecture '{json.dumps(best['architecture'])}'")
    print(f"\nFull results saved to {args.output_json}")


if __name__ == "__main__":
    main()
