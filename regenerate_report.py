"""Load hypernetwork models already saved by train_loo.py (models/hyper_full.pt,
models/hyper_loo_*.pt) and regenerate plots + report.html WITHOUT retraining. Useful
after a plotting.py change (e.g. adding the gm1/gm2/gm3 panels) when you want updated
figures for an existing results_*/ directory, or just to re-inspect a past run.

Usage:
    python regenerate_report.py --output_dir results_h32_l1
"""
import argparse
import json
import os

import numpy as np
import torch

from data_loader import load_all
from model import HyperNetwork
from plotting import qtag, build_iv_curves, plot_iv_grid, build_html_report
from train_loo import build_tensors, build_qpoint_tensors, evaluate, make_predict_fn, N_PHYSICS_FEATURES


def load_hyper(models_dir, filename, n_params, h_hidden, h_layers, n_in):
    """Recreate a HyperNetwork with the given (saved-run) sizing and load its trained weights."""
    hyper = HyperNetwork(n_params, hidden=h_hidden, n_hidden_layers=h_layers, n_in=n_in)
    hyper.load_state_dict(torch.load(os.path.join(models_dir, filename), map_location="cpu"))
    hyper.eval()
    return hyper


def regenerate_report(output_dir, csv_dir=None):
    """Reads output_dir/run_info.json for the architecture/H-size/etc. this run used, loads
    output_dir/models/*.pt, re-evaluates every fold (full-fit + each LOO hold-out) against
    the real data, and rewrites output_dir/plots/*.png, run_info.json's error fields, and
    report.html. Does not touch the saved model weights or retrain anything."""
    with open(os.path.join(output_dir, "run_info.json")) as f:
        info = json.load(f)

    architecture = info["architecture"]
    n_params = info["n_params"]
    h_hidden = info.get("h_hidden", 32)
    h_layers = info.get("h_layers", 2)
    h_physics = info.get("h_physics", False)
    h_physics_oracle = info.get("h_physics_oracle", False)
    n_in = 2 + (N_PHYSICS_FEATURES if h_physics else 0)

    data = load_all(csv_dir or info["csv_dir"])
    keys = sorted(data.keys())
    device = torch.device("cpu")

    models_dir = os.path.join(output_dir, "models")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    tensors = build_tensors(data, device)

    # --- full-fit: one model, evaluated in-sample on every quiescent point ---
    qpoints = build_qpoint_tensors(data, keys, None, device, h_physics)
    for k in keys:
        tensors[k]["qpoint"] = qpoints[k]
    hyper_full = load_hyper(models_dir, "hyper_full.pt", n_params, h_hidden, h_layers, n_in)

    full_errs_by_key = {}
    for k in keys:
        err, _ = evaluate(hyper_full, k, tensors, architecture, device)
        full_errs_by_key[k] = err
        predict_fn = make_predict_fn(hyper_full, tensors[k]["qpoint"], architecture, device)
        iv_data = build_iv_curves(data[k], predict_fn)
        plot_iv_grid(iv_data, k[0], k[1], err,
                     os.path.join(plots_dir, f"infit_{qtag(*k)}.png"),
                     title_suffix="(in-sample, full-fit)")

    # --- LOO: one model per held-out point, evaluated only on that point ---
    loo_errs_by_key = {}
    for held_out in keys:
        train_keys = [k for k in keys if k != held_out]
        qpoints = build_qpoint_tensors(data, train_keys, held_out, device, h_physics,
                                        oracle_held_out=h_physics_oracle)
        for k, qp in qpoints.items():
            tensors[k]["qpoint"] = qp
        tag = qtag(*held_out)
        hyper_loo = load_hyper(models_dir, f"hyper_loo_{tag}.pt", n_params, h_hidden, h_layers, n_in)
        err, _ = evaluate(hyper_loo, held_out, tensors, architecture, device)
        loo_errs_by_key[held_out] = err
        predict_fn = make_predict_fn(hyper_loo, tensors[held_out]["qpoint"], architecture, device)
        iv_data = build_iv_curves(data[held_out], predict_fn)
        plot_iv_grid(iv_data, held_out[0], held_out[1], err,
                     os.path.join(plots_dir, f"loo_{tag}.png"),
                     title_suffix="(held-out, LOO)")

    full_errs = [full_errs_by_key[k] for k in keys]
    loo_errs = [loo_errs_by_key[k] for k in keys]

    info["full_fit"] = {qtag(*k): e for k, e in zip(keys, full_errs)}
    info["full_fit_mean_rel_rmse"] = float(np.mean(full_errs))
    info["loo"] = {qtag(*k): e for k, e in zip(keys, loo_errs)}
    info["loo_mean_rel_rmse"] = float(np.mean(loo_errs))
    with open(os.path.join(output_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    report_path = build_html_report(output_dir, keys, full_errs, loo_errs,
                                     info["epochs"], info["lr"], n_params,
                                     info.get("architecture_name", ""))

    print(f"Re-evaluated {len(keys)} full-fit + {len(keys)} LOO folds from saved models "
          f"(no retraining).")
    print(f"  full-fit mean rel.RMSE: {info['full_fit_mean_rel_rmse']*100:.2f}%")
    print(f"  LOO mean rel.RMSE:      {info['loo_mean_rel_rmse']*100:.2f}%")
    print(f"Updated plots in {plots_dir}")
    print(f"Updated report at {report_path}")
    return report_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True,
                     help="A results_*/ directory previously produced by train_loo.py "
                          "(must contain run_info.json and models/hyper_full.pt + "
                          "models/hyper_loo_*.pt).")
    ap.add_argument("--csv_dir", default=None,
                     help="Override the csv_dir stored in run_info.json (e.g. if the data "
                          "moved). Defaults to whatever train_loo.py used originally.")
    args = ap.parse_args()
    regenerate_report(args.output_dir, args.csv_dir)


if __name__ == "__main__":
    main()
