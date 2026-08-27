"""One-off analysis: ensemble the "original" strategy's LOO predictions (results_h32_l1,
arch37) with the "delta" strategy's LOO predictions (results_deltaweights_reg_arch37, arch37,
delta_reg_weight=0.01) by simple averaging, for the 5 quiescent points both strategies have a
held-out model for (delta never holds out its own BASE_KEY). Uses the already-saved per-fold
weights directly -- no retraining."""
import json
import os

import numpy as np
import torch

from data_loader import load_all
from model import HyperNetwork, main_net_forward, main_net_n_params, ARCHITECTURES
from plotting import qtag

VGS_SCALE = 4.0
VDS_SCALE = 45.0
BASE_KEY = (-2.9, 0.0)

ORIG_DIR = "results_h32_l1"
DELTA_DIR = "results_deltaweights_noreg_arch37"


def rel_rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.abs(true).max() + 1e-30))


def main():
    orig_info = json.load(open(os.path.join(ORIG_DIR, "run_info.json")))
    delta_info = json.load(open(os.path.join(DELTA_DIR, "run_info.json")))
    architecture = orig_info["architecture"]
    assert architecture == delta_info["architecture"], "architectures must match for a fair ensemble"
    n_params = main_net_n_params(architecture)
    h_hidden, h_layers = orig_info["h_hidden"], orig_info["h_layers"]

    data = load_all(r"C:\Users\acost\repos\csvs")
    keys = sorted(data.keys())
    other_keys = [k for k in keys if k != BASE_KEY]

    theta_base = torch.load(os.path.join(DELTA_DIR, "models", "theta_base.pt"))

    print(f"{'point':22s} {'original':>10s} {'delta':>10s} {'ensemble':>10s}  best_individual")
    orig_errs, delta_errs, ens_errs = [], [], []
    for held_out in other_keys:
        tag = qtag(*held_out)
        df = data[held_out]
        vgs_t = torch.tensor(df["Vgs"].values / VGS_SCALE, dtype=torch.float32)
        vds_t = torch.tensor(df["Vds"].values / VDS_SCALE, dtype=torch.float32)
        ids_np = df["Ids"].values.astype(float)
        qpoint = torch.tensor([held_out[0] / VGS_SCALE, held_out[1] / VDS_SCALE], dtype=torch.float32)

        hyper = HyperNetwork(n_params, hidden=h_hidden, n_hidden_layers=h_layers, n_in=2)
        hyper.load_state_dict(torch.load(os.path.join(ORIG_DIR, "models", f"hyper_loo_{tag}.pt")))
        hyper.eval()
        with torch.no_grad():
            theta = hyper(qpoint)
            pred_orig = main_net_forward(theta, vgs_t, vds_t, architecture).numpy()

        hcomp = HyperNetwork(n_params, hidden=h_hidden, n_hidden_layers=h_layers, n_in=2)
        hcomp.load_state_dict(torch.load(os.path.join(DELTA_DIR, "models", f"hcomp_loo_{tag}.pt")))
        hcomp.eval()
        with torch.no_grad():
            delta = hcomp(qpoint)
            theta_total = theta_base + delta
            pred_delta = main_net_forward(theta_total, vgs_t, vds_t, architecture).numpy()

        pred_ens = (pred_orig + pred_delta) / 2.0

        e_orig = rel_rmse(pred_orig, ids_np)
        e_delta = rel_rmse(pred_delta, ids_np)
        e_ens = rel_rmse(pred_ens, ids_np)
        orig_errs.append(e_orig)
        delta_errs.append(e_delta)
        ens_errs.append(e_ens)
        best = "ensemble" if e_ens < min(e_orig, e_delta) else ("original" if e_orig < e_delta else "delta")
        print(f"{tag:22s} {e_orig*100:9.2f}% {e_delta*100:9.2f}% {e_ens*100:9.2f}%  {best}")

    print()
    print(f"{'MEAN':22s} {np.mean(orig_errs)*100:9.2f}% {np.mean(delta_errs)*100:9.2f}% {np.mean(ens_errs)*100:9.2f}%")


if __name__ == "__main__":
    main()
