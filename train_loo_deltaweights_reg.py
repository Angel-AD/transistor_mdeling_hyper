"""
STANDALONE experiment #4b -- separate from train_loo.py, train_loo_basediff.py,
train_loo_basediff_joint.py, AND train_loo_deltaweights_noreg.py. Only READS (imports,
unmodified) data_loader.py, model.py's HyperNetwork/main_net_forward/main_net_n_params/
ARCHITECTURES, and plotting.py's helpers.

Weight-space delta variant WITH an explicit L2 penalty on ||delta_theta|| (see
train_loo_deltaweights_noreg.py for the un-penalized sibling and the full design rationale).
Idea: fit theta_base ONCE (a flat weight vector, same architecture family as the normal
main-net) directly on BASE_KEY's own data via ordinary gradient optimization -- like training
one individual per-CSV model. Freeze it. Then train a hypernetwork H_comp(Vgsq,Vdsq) ->
delta_theta that generates a small CORRECTION to those SAME weights for every other quiescent
point, added directly in weight space:

    theta_total(qpoint) = theta_base (frozen) + delta_theta(qpoint)   <- H_comp generates this
    Ids_pred = main_net_forward(theta_total, Vgs, Vds, architecture)  <- ONE normal main-net
    loss = MSE(Ids_pred, Ids_real)/norm[k] + delta_reg_weight * mean(delta_theta**2)

The regularization term explicitly pushes H_comp toward "as little correction as possible,"
on the assumption (from the user's original suggestion) that keeping delta_theta small helps
it generalize better across quiescent points -- directly testable against the un-penalized
sibling by comparing LOO numbers.

BASE_KEY is always included in H_comp's training set (its own required delta_theta is ~0 by
construction, a free anchor), so LOO here runs over the other 5 quiescent points, exactly
like train_loo_basediff.py -- 5 training points per fold, directly comparable.

Usage:
    python train_loo_deltaweights_reg.py --csv_dir "C:\\Users\\acost\\repos\\csvs" --architecture arch37 --delta_reg_weight 0.01
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn

from data_loader import load_all
from model import HyperNetwork, main_net_forward, main_net_n_params, ARCHITECTURES
from plotting import qtag, build_iv_curves, plot_iv_grid, build_html_report

VGS_SCALE = 4.0
VDS_SCALE = 45.0
BASE_KEY = (-2.9, 0.0)  # same fixed reference as train_loo_basediff.py


def rel_rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.abs(true).max() + 1e-30))


def fit_theta_base(df, n_params, architecture, epochs=4000, lr=3e-3, lbfgs_epochs=5,
                    lbfgs_max_iter=200, seed=27):
    """Trains a single flat weight vector theta_base directly on BASE_KEY's own data --
    same idea as training one individual per-CSV model, just via raw parameter optimization
    instead of a full nn.Module (main_net_forward doesn't need one)."""
    torch.manual_seed(seed)
    vgs = torch.tensor(df["Vgs"].values / VGS_SCALE, dtype=torch.float32)
    vds = torch.tensor(df["Vds"].values / VDS_SCALE, dtype=torch.float32)
    ids = torch.tensor(df["Ids"].values, dtype=torch.float32)
    norm = (ids.abs().max().item() + 1e-6) ** 2

    theta = nn.Parameter(torch.randn(n_params) * 0.3)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = main_net_forward(theta, vgs, vds, architecture)
        loss = torch.mean((pred - ids) ** 2) / norm
        loss.backward()
        opt.step()

    if lbfgs_epochs > 0:
        lbfgs = torch.optim.LBFGS([theta], max_iter=lbfgs_max_iter, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            pred = main_net_forward(theta, vgs, vds, architecture)
            loss = torch.mean((pred - ids) ** 2) / norm
            loss.backward()
            return loss

        for _ in range(lbfgs_epochs):
            lbfgs.step(closure)

    with torch.no_grad():
        pred = main_net_forward(theta, vgs, vds, architecture).numpy()
    return theta.detach(), rel_rmse(pred, ids.numpy())


def build_tensors(data, keys, device):
    out = {}
    for k in keys:
        df = data[k]
        vgs_t = torch.tensor(df["Vgs"].values / VGS_SCALE, dtype=torch.float32, device=device)
        vds_t = torch.tensor(df["Vds"].values / VDS_SCALE, dtype=torch.float32, device=device)
        ids_np = df["Ids"].values.astype(float)
        ids_t = torch.tensor(ids_np, dtype=torch.float32, device=device)
        qpoint = torch.tensor([k[0] / VGS_SCALE, k[1] / VDS_SCALE], dtype=torch.float32, device=device)
        out[k] = dict(qpoint=qpoint, vgs=vgs_t, vds=vds_t, ids=ids_t, ids_np=ids_np)
    return out


def _total_loss(hcomp, theta_base, train_keys, tensors, architecture, norm, delta_reg_weight):
    total = 0.0
    for k in train_keys:
        t = tensors[k]
        delta = hcomp(t["qpoint"])
        theta_total = theta_base + delta
        pred = main_net_forward(theta_total, t["vgs"], t["vds"], architecture)
        loss = torch.mean((pred - t["ids"]) ** 2) / norm[k]
        if delta_reg_weight > 0:
            # Explicit push toward "as little correction to theta_base as possible" --
            # the user's hypothesis that a smaller delta_theta generalizes better.
            loss = loss + delta_reg_weight * torch.mean(delta ** 2)
        total = total + loss
    return total


def train_hcomp(train_keys, tensors, theta_base, n_params, architecture, epochs, lr, device,
                 lbfgs_epochs=5, lbfgs_max_iter=200, h_hidden=32, h_layers=1, delta_reg_weight=0.01,
                 log_every=0, seed=27):
    torch.manual_seed(seed)
    hcomp = HyperNetwork(n_params, hidden=h_hidden, n_hidden_layers=h_layers, n_in=2).to(device)
    opt = torch.optim.Adam(hcomp.parameters(), lr=lr)
    norm = {k: (np.abs(tensors[k]["ids_np"]).max() + 1e-6) ** 2 for k in train_keys}

    for epoch in range(epochs):
        opt.zero_grad()
        loss = _total_loss(hcomp, theta_base, train_keys, tensors, architecture, norm, delta_reg_weight)
        loss.backward()
        opt.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:5d}  train_loss = {loss.item():.5f}")

    if lbfgs_epochs > 0:
        lbfgs_opt = torch.optim.LBFGS(hcomp.parameters(), max_iter=lbfgs_max_iter, line_search_fn="strong_wolfe")

        def closure():
            lbfgs_opt.zero_grad()
            loss = _total_loss(hcomp, theta_base, train_keys, tensors, architecture, norm, delta_reg_weight)
            loss.backward()
            return loss

        for step in range(lbfgs_epochs):
            lbfgs_opt.step(closure)

    return hcomp


def evaluate(hcomp, theta_base, key, tensors, architecture, device):
    t = tensors[key]
    with torch.no_grad():
        delta = hcomp(t["qpoint"])
        theta_total = theta_base + delta
        pred = main_net_forward(theta_total, t["vgs"], t["vds"], architecture).cpu().numpy()
    return rel_rmse(pred, t["ids_np"]), pred


def make_predict_fn(hcomp, theta_base, qpoint_tensor, architecture, device):
    def predict_fn(vgs_arr, vds_arr):
        with torch.no_grad():
            vgs_t = torch.tensor(vgs_arr / VGS_SCALE, dtype=torch.float32, device=device)
            vds_t = torch.tensor(vds_arr / VDS_SCALE, dtype=torch.float32, device=device)
            delta = hcomp(qpoint_tensor)
            theta_total = theta_base + delta
            return main_net_forward(theta_total, vgs_t, vds_t, architecture).cpu().numpy()
    return predict_fn


def run_full_fit_job(data, theta_base, n_params, architecture, epochs, lr, lbfgs_epochs,
                      lbfgs_max_iter, h_hidden, h_layers, delta_reg_weight, seed, models_dir, plots_dir):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    keys = sorted(data.keys())
    tensors = build_tensors(data, keys, device)
    hcomp = train_hcomp(keys, tensors, theta_base, n_params, architecture, epochs, lr, device,
                         lbfgs_epochs=lbfgs_epochs, lbfgs_max_iter=lbfgs_max_iter,
                         h_hidden=h_hidden, h_layers=h_layers, delta_reg_weight=delta_reg_weight,
                         log_every=0, seed=seed)
    errs = {}
    for k in keys:
        err, _ = evaluate(hcomp, theta_base, k, tensors, architecture, device)
        errs[k] = err
        predict_fn = make_predict_fn(hcomp, theta_base, tensors[k]["qpoint"], architecture, device)
        iv_data = build_iv_curves(data[k], predict_fn)
        plot_iv_grid(iv_data, k[0], k[1], err,
                     os.path.join(plots_dir, f"infit_{qtag(*k)}.png"),
                     title_suffix="(in-sample, full-fit, delta-weights reg)")
    torch.save(hcomp.state_dict(), os.path.join(models_dir, "hcomp_full.pt"))
    return errs


def run_loo_job(held_out, data, theta_base, n_params, architecture, epochs, lr, lbfgs_epochs,
                 lbfgs_max_iter, h_hidden, h_layers, delta_reg_weight, seed, models_dir, plots_dir):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    all_keys = sorted(data.keys())
    other_keys = [k for k in all_keys if k != BASE_KEY]
    train_keys = [BASE_KEY] + [k for k in other_keys if k != held_out]
    tensors = build_tensors(data, all_keys, device)
    hcomp = train_hcomp(train_keys, tensors, theta_base, n_params, architecture, epochs, lr, device,
                         lbfgs_epochs=lbfgs_epochs, lbfgs_max_iter=lbfgs_max_iter,
                         h_hidden=h_hidden, h_layers=h_layers, delta_reg_weight=delta_reg_weight,
                         log_every=0, seed=seed)
    err, _ = evaluate(hcomp, theta_base, held_out, tensors, architecture, device)
    tag = qtag(*held_out)
    torch.save(hcomp.state_dict(), os.path.join(models_dir, f"hcomp_loo_{tag}.pt"))
    predict_fn = make_predict_fn(hcomp, theta_base, tensors[held_out]["qpoint"], architecture, device)
    iv_data = build_iv_curves(data[held_out], predict_fn)
    plot_iv_grid(iv_data, held_out[0], held_out[1], err,
                 os.path.join(plots_dir, f"loo_{tag}.png"),
                 title_suffix="(held-out, LOO, delta-weights reg)")
    return held_out, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=r"C:\Users\acost\repos\csvs")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--architecture", default="arch37",
                     help=f"preset name from model.ARCHITECTURES ({sorted(ARCHITECTURES.keys())}) or raw JSON.")
    ap.add_argument("--lbfgs_epochs", type=int, default=5)
    ap.add_argument("--lbfgs_max_iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--h_hidden", type=int, default=32)
    ap.add_argument("--h_layers", type=int, default=1)
    ap.add_argument("--base_epochs", type=int, default=4000)
    ap.add_argument("--base_lr", type=float, default=3e-3)
    ap.add_argument("--delta_reg_weight", type=float, default=0.01,
                     help="L2 penalty weight on delta_theta (mean(delta_theta**2)), pushing "
                          "H_comp toward the smallest correction that still fits. 0.0 = off "
                          "(equivalent to train_loo_deltaweights_noreg.py).")
    ap.add_argument("--output_dir",
                     default=r"C:\Users\acost\repos\transistor_modeling_hyper_outputs\results_deltaweights_reg",
                     help="Defaults OUTSIDE this repo (sibling folder) so runs never create "
                          "untracked files inside the git repo.")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    models_dir = os.path.join(args.output_dir, "models")
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    data = load_all(args.csv_dir)
    keys = sorted(data.keys())
    other_keys = [k for k in keys if k != BASE_KEY]
    print("Quiescent points found:", keys)
    print("BASE_KEY (fixed reference, never held out):", BASE_KEY)

    if args.architecture in ARCHITECTURES:
        architecture = ARCHITECTURES[args.architecture]
    else:
        architecture = json.loads(args.architecture)
    n_params = main_net_n_params(architecture)
    print(f"architecture={args.architecture} ({architecture}), n_params={n_params}")

    print(f"\nFitting theta_base on {BASE_KEY}'s own data ({args.base_epochs} epochs + "
          f"{args.lbfgs_epochs} L-BFGS steps)...")
    theta_base, base_err = fit_theta_base(data[BASE_KEY], n_params, architecture, epochs=args.base_epochs,
                                           lr=args.base_lr, lbfgs_epochs=args.lbfgs_epochs,
                                           lbfgs_max_iter=args.lbfgs_max_iter, seed=args.seed)
    print(f"  theta_base rel.RMSE on its own base point: {base_err*100:.2f}%")
    torch.save(theta_base, os.path.join(models_dir, "theta_base.pt"))

    n_jobs = 1 + len(other_keys)
    workers = args.workers or min(n_jobs, os.cpu_count() or 1)
    print(f"\nRunning {n_jobs} independent trainings ({args.epochs} epochs each) "
          f"across {workers} worker processes ...")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        full_future = ex.submit(run_full_fit_job, data, theta_base, n_params, architecture, args.epochs,
                                 args.lr, args.lbfgs_epochs, args.lbfgs_max_iter, args.h_hidden,
                                 args.h_layers, args.delta_reg_weight, args.seed, models_dir, plots_dir)
        loo_futures = {ex.submit(run_loo_job, k, data, theta_base, n_params, architecture, args.epochs,
                                  args.lr, args.lbfgs_epochs, args.lbfgs_max_iter, args.h_hidden,
                                  args.h_layers, args.delta_reg_weight, args.seed, models_dir, plots_dir): k
                        for k in other_keys}

        full_errs_by_key = full_future.result()
        print("\n=== Full-fit sanity check (train on BASE_KEY + all 5 others, in-sample) ===")
        for k in keys:
            print(f"  in-sample Vgsq={k[0]:.1f} Vdsq={k[1]:.1f}: rel.RMSE = {full_errs_by_key[k]*100:.2f}%")
        print(f"  --> mean in-sample rel.RMSE: {np.mean(list(full_errs_by_key.values()))*100:.2f}%")

        loo_errs_by_key = {}
        print("\n=== Leave-one-out over the 5 NON-base points (BASE_KEY always in training) ===")
        for fut in as_completed(loo_futures):
            held_out, err = fut.result()
            loo_errs_by_key[held_out] = err
            print(f"  held out Vgsq={held_out[0]:.1f} Vdsq={held_out[1]:.1f}: rel.RMSE(Ids) = {err*100:.2f}%")

    full_errs = [full_errs_by_key[k] for k in keys]
    loo_errs = [loo_errs_by_key[k] for k in other_keys]
    print(f"\n  --> LOO mean rel.RMSE across the 5 non-base held-out points: {np.mean(loo_errs)*100:.2f}%")

    run_info = {
        "pipeline": "train_loo_deltaweights_reg (standalone)",
        "csv_dir": args.csv_dir,
        "base_key": list(BASE_KEY),
        "base_fit_rel_rmse": base_err,
        "delta_reg_weight": args.delta_reg_weight,
        "epochs": args.epochs,
        "lr": args.lr,
        "architecture_name": args.architecture,
        "architecture": architecture,
        "n_params": n_params,
        "lbfgs_epochs": args.lbfgs_epochs,
        "lbfgs_max_iter": args.lbfgs_max_iter,
        "h_hidden": args.h_hidden,
        "h_layers": args.h_layers,
        "seed": args.seed,
        "quiescent_points": [list(k) for k in keys],
        "full_fit": {qtag(*k): e for k, e in zip(keys, full_errs)},
        "full_fit_mean_rel_rmse": float(np.mean(full_errs)),
        "loo": {qtag(*k): e for k, e in zip(other_keys, loo_errs)},
        "loo_mean_rel_rmse": float(np.mean(loo_errs)),
    }
    with open(os.path.join(args.output_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    loo_errs_padded = [loo_errs_by_key.get(k, float("nan")) for k in keys]
    report_path = build_html_report(args.output_dir, keys, full_errs, loo_errs_padded,
                                     args.epochs, args.lr, n_params,
                                     f"{args.architecture} (delta-weights, reg={args.delta_reg_weight}, base={BASE_KEY})")
    print(f"\nSaved models to {models_dir}")
    print(f"Saved plots to {plots_dir}")
    print(f"Saved metrics to {os.path.join(args.output_dir, 'run_info.json')}")
    print(f"Saved HTML report to {report_path}")


if __name__ == "__main__":
    main()
