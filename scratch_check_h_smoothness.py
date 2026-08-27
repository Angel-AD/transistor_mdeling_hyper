"""Diagnostic: how smooth is H's qpoint->theta mapping ALREADY, in the baseline (no penalty)
trained model? Compares ||H(q_i) - H(q_j)|| between real training qpoint PAIRS (near vs far)
against ||H(q)|| itself, to see whether there's an actual smoothness deficiency to fix.

Result (arch37+h32_l1, results_h32_l1): theta-distance grows roughly monotonically with
qpoint-distance overall, and the relative change stays small (1-8% of ||theta||) even across
the whole training envelope -- H already looks broadly smooth. But the LOCAL "slope"
(theta-distance / qpoint-distance) is actually STEEPER for the closest pairs (3.8-8.3) than
for the farthest ones (3.7-4.4) -- steepest of all between (-2.4,5.0) and (-2.2,5.0), the two
closest points in the whole dataset (qpoint-distance 0.050). This is best read as H correctly
tracking real point-to-point differences in the measured curves rather than a pathology to
regularize away: forcing smoothness there would stop H from fitting those two curves as
distinctly as their real data actually requires -- consistent with why --smoothness_weight
made FULL-FIT worse in every test, not just LOO (see train_loo.py's smoothness_penalty)."""
import json
import itertools

import numpy as np
import torch

from model import HyperNetwork, main_net_n_params

VGS_SCALE = 4.0
VDS_SCALE = 45.0
DIR = "results_h32_l1"


def main():
    info = json.load(open(f"{DIR}/run_info.json"))
    architecture = info["architecture"]
    n_params = main_net_n_params(architecture)
    hyper = HyperNetwork(n_params, hidden=info["h_hidden"], n_hidden_layers=info["h_layers"], n_in=2)
    hyper.load_state_dict(torch.load(f"{DIR}/models/hyper_full.pt"))
    hyper.eval()

    keys = [tuple(k) for k in info["quiescent_points"]]
    thetas = {}
    with torch.no_grad():
        for k in keys:
            q = torch.tensor([k[0] / VGS_SCALE, k[1] / VDS_SCALE], dtype=torch.float32)
            thetas[k] = hyper(q)

    theta_scale = float(np.mean([t.norm().item() for t in thetas.values()]))
    print(f"mean ||theta|| across the 6 training qpoints: {theta_scale:.3f}\n")

    print(f"{'pair':45s} {'qpoint dist (norm)':>20s} {'||H(a)-H(b)||':>16s} {'ratio to ||theta||':>20s}")
    rows = []
    for a, b in itertools.combinations(keys, 2):
        qa = torch.tensor([a[0] / VGS_SCALE, a[1] / VDS_SCALE])
        qb = torch.tensor([b[0] / VGS_SCALE, b[1] / VDS_SCALE])
        qdist = (qa - qb).norm().item()
        tdist = (thetas[a] - thetas[b]).norm().item()
        ratio = tdist / theta_scale
        rows.append((qdist, tdist, ratio, a, b))
    rows.sort()
    for qdist, tdist, ratio, a, b in rows:
        print(f"{str(a)+' <-> '+str(b):45s} {qdist:20.3f} {tdist:16.3f} {ratio:20.3f}")

    print("\n--- 'slope' (theta-distance / qpoint-distance) for the 3 CLOSEST pairs ---")
    for qdist, tdist, ratio, a, b in rows[:3]:
        print(f"  {a} <-> {b}: slope = {tdist/qdist:.2f}")
    print("--- and the 3 FARTHEST pairs ---")
    for qdist, tdist, ratio, a, b in rows[-3:]:
        print(f"  {a} <-> {b}: slope = {tdist/qdist:.2f}")


if __name__ == "__main__":
    main()
