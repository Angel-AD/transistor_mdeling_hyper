"""
STANDALONE experiment #2 -- separate from BOTH train_loo.py and train_loo_basediff.py.
Only READS (imports, unmodified) data_loader.py, model.py's HyperNetwork/main_net_forward/
main_net_n_params/ARCHITECTURES, and plotting.py's plot helpers.

"Option 2a" (joint training) from the base+diff design discussion -- the alternative to
train_loo_basediff.py's "Option 2b" (fixed real quiescent point as base, frozen f_base,
precomputed diff targets), which was tested and found to fail badly on LOO (up to 52%) for
the quiescent points with the largest/most kink-like diff.

Here f_base is NOT frozen and NOT fit on one specific quiescent point's data alone: it has
its own real trainable weights, and is optimized JOINTLY with H, against Ids DIRECTLY (no
separate diff-target precomputation):

    Ids_pred(Vgs, Vds; qpoint) = f_base(Vgs, Vds) + f_diff(Vgs, Vds; theta_diff(qpoint))
    loss = sum over training quiescent points of MSE(Ids_pred, Ids_real) / norm[k]

f_base sees gradient contributions from EVERY training quiescent point every step (not just
one), so it is free to settle into whatever "shared" curve shape minimizes the total loss --
it is not anchored to any single real curve. Because there is no special "base" quiescent
point that must always stay in the training set, LOO here is the standard 6-fold protocol
(train on 5, hold out 1), directly comparable to train_loo.py's numbers.

Usage:
    python train_loo_basediff_joint.py --csv_dir "C:\\Users\\acost\\repos\\csvs" --architecture arch37
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


class FBase(nn.Module):
    """Small ordinary NN with its OWN trainable weights, shared across every quiescent
    point -- co-optimized with H, not pre-fit on any single point's data."""
    def __init__(self, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, vgs, vds):
        x = torch.stack([vgs, vds], dim=-1)
        return self.net(x).squeeze(-1)


def rel_rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.abs(true).max() + 1e-30))


def build_tensors(data, device):
    out = {}
    for (vgsq, vdsq), df in data.items():
        vgs_t = torch.tensor(df["Vgs"].values / VGS_SCALE, dtype=torch.float32, device=device)
        vds_t = torch.tensor(df["Vds"].values / VDS_SCALE, dtype=torch.float32, device=device)
        ids_np = df["Ids"].values.astype(float)
        ids_t = torch.tensor(ids_np, dtype=torch.float32, device=device)
        qpoint = torch.tensor([vgsq / VGS_SCALE, vdsq / VDS_SCALE], dtype=torch.float32, device=device)
        out[(vgsq, vdsq)] = dict(qpoint=qpoint, vgs=vgs_t, vds=vds_t, ids=ids_t, ids_np=ids_np)
    return out


def _total_loss(hyper, f_base, train_keys, tensors, architecture, norm):
    total = 0.0
    for k in train_keys:
        t = tensors[k]
        base_pred = f_base(t["vgs"], t["vds"])
        theta = hyper(t["qpoint"])
        diff_pred = main_net_forward(theta, t["vgs"], t["vds"], architecture)
        ids_pred = base_pred + diff_pred
        loss = torch.mean((ids_pred - t["ids"]) ** 2) / norm[k]
        total = total + loss
    return total


def train_jointly(train_keys, tensors, n_params, architecture, epochs, lr, device,
                   lbfgs_epochs=5, lbfgs_max_iter=200, h_hidden=32, h_layers=1,
                   base_hidden=8, log_every=0, seed=27):
    torch.manual_seed(seed)
    hyper = HyperNetwork(n_params, hidden=h_hidden, n_hidden_layers=h_layers, n_in=2).to(device)
    f_base = FBase(hidden=base_hidden).to(device)
    params = list(hyper.parameters()) + list(f_base.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    norm = {k: (np.abs(tensors[k]["ids_np"]).max() + 1e-6) ** 2 for k in train_keys}

    for epoch in range(epochs):
        opt.zero_grad()
        loss = _total_loss(hyper, f_base, train_keys, tensors, architecture, norm)
        loss.backward()
        opt.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:5d}  train_loss = {loss.item():.5f}")

    if lbfgs_epochs > 0:
        lbfgs_opt = torch.optim.LBFGS(params, max_iter=lbfgs_max_iter, line_search_fn="strong_wolfe")

        def closure():
            lbfgs_opt.zero_grad()
            loss = _total_loss(hyper, f_base, train_keys, tensors, architecture, norm)
            loss.backward()
            return loss

        for step in range(lbfgs_epochs):
            lbfgs_opt.step(closure)

    return hyper, f_base


def evaluate(hyper, f_base, key, tensors, architecture, device):
    t = tensors[key]
    with torch.no_grad():
        base_pred = f_base(t["vgs"], t["vds"])
        theta = hyper(t["qpoint"])
        diff_pred = main_net_forward(theta, t["vgs"], t["vds"], architecture)
        pred = (base_pred + diff_pred).cpu().numpy()
    return rel_rmse(pred, t["ids_np"]), pred


def make_predict_fn(hyper, f_base, qpoint_tensor, architecture, device):
    def predict_fn(vgs_arr, vds_arr):
        with torch.no_grad():
            vgs_t = torch.tensor(vgs_arr / VGS_SCALE, dtype=torch.float32, device=device)
            vds_t = torch.tensor(vds_arr / VDS_SCALE, dtype=torch.float32, device=device)
            base_pred = f_base(vgs_t, vds_t)
            theta = hyper(qpoint_tensor)
            diff_pred = main_net_forward(theta, vgs_t, vds_t, architecture)
            return (base_pred + diff_pred).cpu().numpy()
    return predict_fn


def run_full_fit_job(data, n_params, architecture, epochs, lr, lbfgs_epochs, lbfgs_max_iter,
                      h_hidden, h_layers, base_hidden, seed, models_dir, plots_dir):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    keys = sorted(data.keys())
    tensors = build_tensors(data, device)
    hyper, f_base = train_jointly(keys, tensors, n_params, architecture, epochs, lr, device,
                                   lbfgs_epochs=lbfgs_epochs, lbfgs_max_iter=lbfgs_max_iter,
                                   h_hidden=h_hidden, h_layers=h_layers, base_hidden=base_hidden,
                                   log_every=0, seed=seed)
    errs = {}
    for k in keys:
        err, _ = evaluate(hyper, f_base, k, tensors, architecture, device)
        errs[k] = err
        predict_fn = make_predict_fn(hyper, f_base, tensors[k]["qpoint"], architecture, device)
        iv_data = build_iv_curves(data[k], predict_fn)
        plot_iv_grid(iv_data, k[0], k[1], err,
                     os.path.join(plots_dir, f"infit_{qtag(*k)}.png"),
                     title_suffix="(in-sample, full-fit, base+diff joint)")
    torch.save(hyper.state_dict(), os.path.join(models_dir, "hyper_full.pt"))
    torch.save(f_base.state_dict(), os.path.join(models_dir, "f_base_full.pt"))
    return errs


def run_loo_job(held_out, data, n_params, architecture, epochs, lr, lbfgs_epochs, lbfgs_max_iter,
                 h_hidden, h_layers, base_hidden, seed, models_dir, plots_dir):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    keys = sorted(data.keys())
    train_keys = [k for k in keys if k != held_out]
    tensors = build_tensors(data, device)
    hyper, f_base = train_jointly(train_keys, tensors, n_params, architecture, epochs, lr, device,
                                   lbfgs_epochs=lbfgs_epochs, lbfgs_max_iter=lbfgs_max_iter,
                                   h_hidden=h_hidden, h_layers=h_layers, base_hidden=base_hidden,
                                   log_every=0, seed=seed)
    err, _ = evaluate(hyper, f_base, held_out, tensors, architecture, device)
    tag = qtag(*held_out)
    torch.save(hyper.state_dict(), os.path.join(models_dir, f"hyper_loo_{tag}.pt"))
    torch.save(f_base.state_dict(), os.path.join(models_dir, f"f_base_loo_{tag}.pt"))
    predict_fn = make_predict_fn(hyper, f_base, tensors[held_out]["qpoint"], architecture, device)
    iv_data = build_iv_curves(data[held_out], predict_fn)
    plot_iv_grid(iv_data, held_out[0], held_out[1], err,
                 os.path.join(plots_dir, f"loo_{tag}.png"),
                 title_suffix="(held-out, LOO, base+diff joint)")
    return held_out, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=r"C:\Users\acost\repos\csvs")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--architecture", default="arch37",
                     help=f"f_diff architecture: preset name from model.ARCHITECTURES "
                          f"({sorted(ARCHITECTURES.keys())}) or raw JSON.")
    ap.add_argument("--lbfgs_epochs", type=int, default=5)
    ap.add_argument("--lbfgs_max_iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--h_hidden", type=int, default=32)
    ap.add_argument("--h_layers", type=int, default=1)
    ap.add_argument("--base_hidden", type=int, default=8,
                     help="Hidden width of f_base (2 hidden layers, tanh), co-trained with H.")
    ap.add_argument("--output_dir",
                     default=r"C:\Users\acost\repos\transistor_modeling_hyper_outputs\results_basediff_joint",
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
    print("Quiescent points found:", keys)

    if args.architecture in ARCHITECTURES:
        architecture = ARCHITECTURES[args.architecture]
    else:
        architecture = json.loads(args.architecture)
    n_params = main_net_n_params(architecture)
    print(f"f_diff architecture={args.architecture} ({architecture}), n_params={n_params}, "
          f"f_base_hidden={args.base_hidden} (jointly trained, standard 6-fold LOO)")

    n_jobs = 1 + len(keys)
    workers = args.workers or min(n_jobs, os.cpu_count() or 1)
    print(f"\nRunning {n_jobs} independent trainings ({args.epochs} epochs each) "
          f"across {workers} worker processes ...")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        full_future = ex.submit(run_full_fit_job, data, n_params, architecture, args.epochs, args.lr,
                                 args.lbfgs_epochs, args.lbfgs_max_iter, args.h_hidden, args.h_layers,
                                 args.base_hidden, args.seed, models_dir, plots_dir)
        loo_futures = {ex.submit(run_loo_job, k, data, n_params, architecture, args.epochs, args.lr,
                                  args.lbfgs_epochs, args.lbfgs_max_iter, args.h_hidden, args.h_layers,
                                  args.base_hidden, args.seed, models_dir, plots_dir): k for k in keys}

        full_errs_by_key = full_future.result()
        print("\n=== Full-fit sanity check (train on all 6, evaluate in-sample) ===")
        for k in keys:
            print(f"  in-sample Vgsq={k[0]:.1f} Vdsq={k[1]:.1f}: rel.RMSE = {full_errs_by_key[k]*100:.2f}%")
        print(f"  --> mean in-sample rel.RMSE: {np.mean(list(full_errs_by_key.values()))*100:.2f}%")

        loo_errs_by_key = {}
        print("\n=== Leave-one-out (standard 6-fold, train on 5, predict the 6th) ===")
        for fut in as_completed(loo_futures):
            held_out, err = fut.result()
            loo_errs_by_key[held_out] = err
            print(f"  held out Vgsq={held_out[0]:.1f} Vdsq={held_out[1]:.1f}: rel.RMSE(Ids) = {err*100:.2f}%")

    full_errs = [full_errs_by_key[k] for k in keys]
    loo_errs = [loo_errs_by_key[k] for k in keys]
    print(f"\n  --> LOO mean rel.RMSE across 6 held-out points: {np.mean(loo_errs)*100:.2f}%")

    run_info = {
        "pipeline": "train_loo_basediff_joint (standalone, separate from train_loo.py and train_loo_basediff.py)",
        "csv_dir": args.csv_dir,
        "base_hidden": args.base_hidden,
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
        "loo": {qtag(*k): e for k, e in zip(keys, loo_errs)},
        "loo_mean_rel_rmse": float(np.mean(loo_errs)),
    }
    with open(os.path.join(args.output_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    report_path = build_html_report(args.output_dir, keys, full_errs, loo_errs,
                                     args.epochs, args.lr, n_params,
                                     f"{args.architecture} (base+diff joint)")
    print(f"\nSaved models to {models_dir}")
    print(f"Saved plots to {plots_dir}")
    print(f"Saved metrics to {os.path.join(args.output_dir, 'run_info.json')}")
    print(f"Saved HTML report to {report_path}")


if __name__ == "__main__":
    main()
