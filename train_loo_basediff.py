"""
STANDALONE experiment -- fully separate from train_loo.py, does NOT modify or share any
state with it. Only READS (imports, unmodified) data_loader.py, model.py's HyperNetwork /
main_net_forward / main_net_n_params / ARCHITECTURES, and plotting.py's plot helpers.

Base+diff hypernetwork variant, inspired by Jarndal (2019), "On Neural Networks Based
Electrothermal Modeling of GaN Devices," IEEE Access -- see the diff-magnitude analysis in
scratch_analyze_diff.py that motivated this. Instead of H(Vgsq,Vdsq) generating a full theta
that describes the ENTIRE Ids(Vgs,Vds) curve for a quiescent point (as train_loo.py does),
this fixes ONE quiescent point as a reference "base" curve, fits a small ordinary NN
f_base(Vgs,Vds) (real trainable weights, NOT hypernetwork-generated -- the one exception to
this project's usual "weightless main-net" convention) on just that point's own data, and
only asks H to generate theta_diff for a small residual network f_diff, trained against
diff = Ids_real - f_base(Vgs,Vds) for the OTHER quiescent points:

    Ids_pred(Vgs, Vds; qpoint) = f_base(Vgs, Vds) + f_diff(Vgs, Vds; theta_diff(qpoint))

f_base is fit ONCE (sequentially, then frozen -- "Option 2b" from the design discussion) on
BASE_KEY's own raw data; it never sees the other points and is IDENTICAL across every fold,
so there is no leakage. Because f_base can never be meaningfully tested this way (there is no
"diff" to predict for its own point), LOO here runs over the OTHER 5 quiescent points only.
BASE_KEY itself IS still included in every fold's training set (its own diff is ~0 by
construction, a free anchor point for H at that qpoint) -- so each LOO fold still trains on
5 real points total (BASE_KEY + 4 of the other 5), matching train_loo.py's LOO training-set
size for a fair comparison.

Usage:
    python train_loo_basediff.py --csv_dir "C:\\Users\\acost\\repos\\csvs" --architecture arch37
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
BASE_KEY = (-2.9, 0.0)  # fixed reference quiescent point -- see scratch_analyze_diff.py:
                        # most negative Vgsq among the Vdsq=0 (zero quiescent power) points.


class FBase(nn.Module):
    """Small ordinary NN with its OWN trainable weights (unlike model.py's weightless
    main-net convention) -- it is the SAME function regardless of quiescent point, fit once
    on BASE_KEY's data and then frozen for every fold of every architecture run."""
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


def fit_f_base(df, hidden=8, epochs=3000, lr=3e-3, lbfgs_epochs=5, lbfgs_max_iter=200, seed=27):
    """Fits f_base ONCE on BASE_KEY's own raw (Vgs,Vds,Ids) data. Returns (model, rel_rmse)."""
    torch.manual_seed(seed)
    vgs = torch.tensor(df["Vgs"].values / VGS_SCALE, dtype=torch.float32)
    vds = torch.tensor(df["Vds"].values / VDS_SCALE, dtype=torch.float32)
    ids = torch.tensor(df["Ids"].values, dtype=torch.float32)
    norm = (ids.abs().max().item() + 1e-6) ** 2

    model = FBase(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((model(vgs, vds) - ids) ** 2) / norm
        loss.backward()
        opt.step()

    if lbfgs_epochs > 0:
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_max_iter, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            loss = torch.mean((model(vgs, vds) - ids) ** 2) / norm
            loss.backward()
            return loss

        for _ in range(lbfgs_epochs):
            lbfgs.step(closure)

    model.eval()
    with torch.no_grad():
        pred = model(vgs, vds).numpy()
    return model, rel_rmse(pred, ids.numpy())


def build_tensors(data, keys, f_base, device):
    """Precomputes, per quiescent point: qpoint, vgs/vds/ids (raw scale for eval), and
    diff_true = Ids_real - f_base(Vgs,Vds) (f_base is frozen, so this is computed once)."""
    out = {}
    with torch.no_grad():
        for k in keys:
            df = data[k]
            vgs_t = torch.tensor(df["Vgs"].values / VGS_SCALE, dtype=torch.float32, device=device)
            vds_t = torch.tensor(df["Vds"].values / VDS_SCALE, dtype=torch.float32, device=device)
            ids_np = df["Ids"].values.astype(float)
            ids_t = torch.tensor(ids_np, dtype=torch.float32, device=device)
            qpoint = torch.tensor([k[0] / VGS_SCALE, k[1] / VDS_SCALE], dtype=torch.float32, device=device)
            base_pred = f_base(vgs_t, vds_t)
            diff_t = ids_t - base_pred
            out[k] = dict(qpoint=qpoint, vgs=vgs_t, vds=vds_t, ids=ids_t, ids_np=ids_np,
                           diff_true=diff_t)
    return out


def _total_loss(hyper, train_keys, tensors, architecture, norm):
    """diff-only MSE (f_base is frozen and already subtracted into diff_true), per-point
    normalized exactly like train_loo.py's Ids loss."""
    total = 0.0
    for k in train_keys:
        t = tensors[k]
        theta = hyper(t["qpoint"])
        diff_pred = main_net_forward(theta, t["vgs"], t["vds"], architecture)
        loss = torch.mean((diff_pred - t["diff_true"]) ** 2) / norm[k]
        total = total + loss
    return total


def train_hypernet(train_keys, tensors, n_params, architecture, epochs, lr, device,
                    lbfgs_epochs=5, lbfgs_max_iter=200, h_hidden=32, h_layers=1,
                    log_every=0, seed=27):
    torch.manual_seed(seed)
    hyper = HyperNetwork(n_params, hidden=h_hidden, n_hidden_layers=h_layers, n_in=2).to(device)
    opt = torch.optim.Adam(hyper.parameters(), lr=lr)

    norm = {k: (tensors[k]["diff_true"].abs().max().item() + 1e-6) ** 2 for k in train_keys}

    for epoch in range(epochs):
        opt.zero_grad()
        loss = _total_loss(hyper, train_keys, tensors, architecture, norm)
        loss.backward()
        opt.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:5d}  train_loss = {loss.item():.5f}")

    if lbfgs_epochs > 0:
        lbfgs_opt = torch.optim.LBFGS(hyper.parameters(), max_iter=lbfgs_max_iter,
                                       line_search_fn="strong_wolfe")

        def closure():
            lbfgs_opt.zero_grad()
            loss = _total_loss(hyper, train_keys, tensors, architecture, norm)
            loss.backward()
            return loss

        for step in range(lbfgs_epochs):
            lbfgs_opt.step(closure)

    return hyper


def evaluate(hyper, key, tensors, architecture, device):
    """Returns (rel_rmse ON Ids -- not diff, pred_np) -- the metric that matters is the
    final Ids accuracy, for a fair comparison against train_loo.py's numbers."""
    t = tensors[key]
    with torch.no_grad():
        theta = hyper(t["qpoint"])
        diff_pred = main_net_forward(theta, t["vgs"], t["vds"], architecture)
        ids_pred = (t["ids"] - t["diff_true"]) + diff_pred  # ids_true - diff_true == f_base(vgs,vds)
        ids_pred = ids_pred.cpu().numpy()
    return rel_rmse(ids_pred, t["ids_np"]), ids_pred


def make_predict_fn(hyper, f_base, qpoint_tensor, architecture, device):
    """Ids(Vgs,Vds) = f_base(Vgs,Vds) + f_diff(Vgs,Vds; theta_diff) at this hyper's fixed
    theta_diff=hyper(qpoint) -- for the smooth model curves in the IV plots."""
    def predict_fn(vgs_arr, vds_arr):
        with torch.no_grad():
            vgs_t = torch.tensor(vgs_arr / VGS_SCALE, dtype=torch.float32, device=device)
            vds_t = torch.tensor(vds_arr / VDS_SCALE, dtype=torch.float32, device=device)
            base_pred = f_base(vgs_t, vds_t)
            theta = hyper(qpoint_tensor)
            diff_pred = main_net_forward(theta, vgs_t, vds_t, architecture)
            return (base_pred + diff_pred).cpu().numpy()
    return predict_fn


def run_full_fit_job(data, f_base_state, base_hidden, n_params, architecture, epochs, lr,
                      lbfgs_epochs, lbfgs_max_iter, h_hidden, h_layers, seed, models_dir, plots_dir):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    f_base = FBase(hidden=base_hidden)
    f_base.load_state_dict(f_base_state)
    f_base.eval()

    keys = sorted(data.keys())  # includes BASE_KEY
    tensors = build_tensors(data, keys, f_base, device)
    hyper = train_hypernet(keys, tensors, n_params, architecture, epochs, lr, device,
                            lbfgs_epochs=lbfgs_epochs, lbfgs_max_iter=lbfgs_max_iter,
                            h_hidden=h_hidden, h_layers=h_layers, log_every=0, seed=seed)

    errs = {}
    for k in keys:
        err, _ = evaluate(hyper, k, tensors, architecture, device)
        errs[k] = err
        predict_fn = make_predict_fn(hyper, f_base, tensors[k]["qpoint"], architecture, device)
        iv_data = build_iv_curves(data[k], predict_fn)
        plot_iv_grid(iv_data, k[0], k[1], err,
                     os.path.join(plots_dir, f"infit_{qtag(*k)}.png"),
                     title_suffix="(in-sample, full-fit, base+diff)")
    torch.save(hyper.state_dict(), os.path.join(models_dir, "hyper_full.pt"))
    return errs


def run_loo_job(held_out, data, f_base_state, base_hidden, n_params, architecture, epochs, lr,
                 lbfgs_epochs, lbfgs_max_iter, h_hidden, h_layers, seed, models_dir, plots_dir):
    """held_out is always one of the 5 non-BASE_KEY points -- BASE_KEY is always included in
    train_keys (its diff is ~0 by construction, a free anchor; never itself held out)."""
    torch.set_num_threads(1)
    device = torch.device("cpu")
    f_base = FBase(hidden=base_hidden)
    f_base.load_state_dict(f_base_state)
    f_base.eval()

    all_keys = sorted(data.keys())
    other_keys = [k for k in all_keys if k != BASE_KEY]
    train_keys = [BASE_KEY] + [k for k in other_keys if k != held_out]
    tensors = build_tensors(data, all_keys, f_base, device)

    hyper = train_hypernet(train_keys, tensors, n_params, architecture, epochs, lr, device,
                            lbfgs_epochs=lbfgs_epochs, lbfgs_max_iter=lbfgs_max_iter,
                            h_hidden=h_hidden, h_layers=h_layers, log_every=0, seed=seed)

    err, _ = evaluate(hyper, held_out, tensors, architecture, device)
    tag = qtag(*held_out)
    torch.save(hyper.state_dict(), os.path.join(models_dir, f"hyper_loo_{tag}.pt"))
    predict_fn = make_predict_fn(hyper, f_base, tensors[held_out]["qpoint"], architecture, device)
    iv_data = build_iv_curves(data[held_out], predict_fn)
    plot_iv_grid(iv_data, held_out[0], held_out[1], err,
                 os.path.join(plots_dir, f"loo_{tag}.png"),
                 title_suffix="(held-out, LOO, base+diff)")
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
    ap.add_argument("--h_layers", type=int, default=1,
                     help="Default 1 (h32_l1) to match this project's best-known H sizing.")
    ap.add_argument("--base_hidden", type=int, default=8,
                     help="Hidden width of f_base (2 hidden layers, tanh). See scratch_analyze_diff.py.")
    ap.add_argument("--base_epochs", type=int, default=3000)
    ap.add_argument("--base_lr", type=float, default=3e-3)
    ap.add_argument("--output_dir",
                     default=r"C:\Users\acost\repos\transistor_modeling_hyper_outputs\results_basediff",
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
    assert BASE_KEY in keys, f"BASE_KEY {BASE_KEY} not found in data keys {keys}"
    other_keys = [k for k in keys if k != BASE_KEY]
    print("Quiescent points found:", keys)
    print("BASE_KEY (fixed reference, never held out):", BASE_KEY)

    print(f"\nFitting f_base on {BASE_KEY}'s own data ({args.base_epochs} epochs + "
          f"{args.lbfgs_epochs} L-BFGS steps)...")
    f_base, base_err = fit_f_base(data[BASE_KEY], hidden=args.base_hidden, epochs=args.base_epochs,
                                   lr=args.base_lr, lbfgs_epochs=args.lbfgs_epochs,
                                   lbfgs_max_iter=args.lbfgs_max_iter, seed=args.seed)
    print(f"  f_base rel.RMSE on its own base point: {base_err*100:.2f}%")
    torch.save(f_base.state_dict(), os.path.join(models_dir, "f_base.pt"))
    f_base_state = f_base.state_dict()

    if args.architecture in ARCHITECTURES:
        architecture = ARCHITECTURES[args.architecture]
    else:
        architecture = json.loads(args.architecture)
    n_params = main_net_n_params(architecture)
    print(f"f_diff architecture={args.architecture} ({architecture}), n_params={n_params}")

    n_jobs = 1 + len(other_keys)  # full-fit + one LOO fold per NON-base quiescent point
    workers = args.workers or min(n_jobs, os.cpu_count() or 1)
    print(f"\nRunning {n_jobs} independent trainings ({args.epochs} epochs each) "
          f"across {workers} worker processes ...")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        full_future = ex.submit(run_full_fit_job, data, f_base_state, args.base_hidden, n_params,
                                 architecture, args.epochs, args.lr, args.lbfgs_epochs,
                                 args.lbfgs_max_iter, args.h_hidden, args.h_layers, args.seed,
                                 models_dir, plots_dir)
        loo_futures = {ex.submit(run_loo_job, k, data, f_base_state, args.base_hidden, n_params,
                                  architecture, args.epochs, args.lr, args.lbfgs_epochs,
                                  args.lbfgs_max_iter, args.h_hidden, args.h_layers, args.seed,
                                  models_dir, plots_dir): k for k in other_keys}

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
        "pipeline": "train_loo_basediff (standalone, separate from train_loo.py)",
        "csv_dir": args.csv_dir,
        "base_key": list(BASE_KEY),
        "base_hidden": args.base_hidden,
        "base_fit_rel_rmse": base_err,
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

    # report over ALL 6 points (full_fit has 6; loo only has the 5 non-base -- build_html_report
    # zips keys/errs positionally, so pass the matching key list for each column explicitly by
    # reusing `keys` for full_fit and inserting a placeholder for BASE_KEY's (non-existent) LOO.
    loo_errs_padded = [loo_errs_by_key.get(k, float("nan")) for k in keys]
    report_path = build_html_report(args.output_dir, keys, full_errs, loo_errs_padded,
                                     args.epochs, args.lr, n_params,
                                     f"{args.architecture} (base+diff, base={BASE_KEY})")
    print(f"\nSaved models to {models_dir}")
    print(f"Saved plots to {plots_dir}")
    print(f"Saved metrics to {os.path.join(args.output_dir, 'run_info.json')}")
    print(f"Saved HTML report to {report_path}")


if __name__ == "__main__":
    main()
