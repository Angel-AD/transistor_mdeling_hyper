"""
Train the hypernetwork jointly on the 6 real measured quiescent-point datasets and
run leave-one-out validation: for each held-out quiescent point, train H (+ the
weightless main-net) on the OTHER 5 points' real data, then generate theta for the
held-out point's (Vgsq, Vdsq) and evaluate against that point's own real measured
(Vgs, Vds, Ids) rows. This is compared against the earlier weight-interpolation
baselines (raw-weight polynomial: ~39-90% rel.RMSE; Angelov-parameter-augmented:
much worse; permutation-aligned: much worse).

Usage:
    python train_loo.py --csv_dir "C:\\Users\\acost\\repos\\csvs" --epochs 4000
"""
import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from data_loader import load_all
from model import HyperNetwork, main_net_forward, main_net_n_params, ARCHITECTURES
from physics_features import extract_angelov_features, predict_physics_features
from plotting import qtag, build_iv_curves, plot_iv_grid, build_html_report
from signal_utils import smooth_derivative

VGS_SCALE = 4.0
VDS_SCALE = 45.0
# Normalization scales for the extra Angelov-style physics features when
# --h_physics is enabled -- brings each feature to roughly O(1), matching the
# qpoint's own /VGS_SCALE, /VDS_SCALE normalization.
IPK_SCALE = 3.0
ALPHA_SCALE = 0.3
LAMBDA_SCALE = 0.01
N_PHYSICS_FEATURES = 4


def normalize(vgsq, vdsq, vgs, vds):
    return (vgsq / VGS_SCALE, vdsq / VDS_SCALE,
            vgs / VGS_SCALE, vds / VDS_SCALE)


def rel_rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.abs(true).max() + 1e-30))


def physics_feature_vector(feats):
    return [feats["Vpk"] / VGS_SCALE, feats["Ipk"] / IPK_SCALE,
            feats["alpha"] / ALPHA_SCALE, feats["lambda_"] / LAMBDA_SCALE]


def build_qpoint_tensors(data, train_keys, held_out, device, use_physics, oracle_held_out=False):
    """Returns {key: qpoint tensor} for every key in train_keys plus held_out (if given).
    Without --h_physics: the usual 2-vector [Vgsq/4, Vdsq/45].
    With --h_physics: that 2-vector plus 4 Angelov features (Vpk,Ipk,alpha,lambda) --
    REAL (extracted from that point's own data) for train_keys. For held_out: PREDICTED
    (inverse-distance interpolation from train_keys' real features) by default -- the
    honest generalization test, since held_out's own curve is what's being evaluated and
    must not leak in. oracle_held_out=True instead extracts held_out's features from its
    OWN real data too (--h_physics_oracle): a diagnostic ablation, NOT a blind LOO test
    anymore -- it isolates whether poor interpolation (vs. the physics-augmentation
    approach itself) is what hurts LOO."""
    out = {}
    if not use_physics:
        for k in list(train_keys) + ([held_out] if held_out is not None else []):
            vgsq, vdsq = k
            out[k] = torch.tensor([vgsq / VGS_SCALE, vdsq / VDS_SCALE], dtype=torch.float32, device=device)
        return out

    known_feats = {k: extract_angelov_features(data[k]) for k in train_keys}
    for k in train_keys:
        vgsq, vdsq = k
        vec = [vgsq / VGS_SCALE, vdsq / VDS_SCALE] + physics_feature_vector(known_feats[k])
        out[k] = torch.tensor(vec, dtype=torch.float32, device=device)
    if held_out is not None:
        if oracle_held_out:
            feats = extract_angelov_features(data[held_out])
        else:
            feats = predict_physics_features(held_out, known_feats)
        vgsq, vdsq = held_out
        vec = [vgsq / VGS_SCALE, vdsq / VDS_SCALE] + physics_feature_vector(feats)
        out[held_out] = torch.tensor(vec, dtype=torch.float32, device=device)
    return out


def build_gm_targets(df, n_vds_targets=4):
    """Ground-truth gm1 = dIds/dVgs, estimated the same way the base transistor_modeling
    repo does it (create_gms_for_train): build several real transfer curves (Ids vs Vgs
    at ~fixed Vds) and take smooth_derivative (Savitzky-Golay smoothed, not a raw
    np.gradient) along Vgs. Our raw data has no ready-made transfer curves (each TN group
    is the OPPOSITE: ~fixed Vgs, Vds ramping) -- reusing plotting.py's build_iv_curves
    technique, one transfer curve is assembled per target Vds by taking each TN group's
    nearest-real-point match, exactly as already done there for the Ids-Vgs plot.
    Returns (vgs_pts, vds_pts, gm1_true) raw 1-D numpy arrays, all real measured values."""
    groups = [g.sort_values("Vds") for _, g in df.groupby("TN")]
    vds_lo, vds_hi = df["Vds"].min(), df["Vds"].max()
    target_vds_list = np.linspace(vds_lo, vds_hi, n_vds_targets)

    vgs_all, vds_all, gm1_all = [], [], []
    for t_vds in target_vds_list:
        vgs_pts, ids_pts, vds_pts = [], [], []
        for g in groups:
            idx = (g["Vds"] - t_vds).abs().idxmin()
            vgs_pts.append(g.loc[idx, "Vgs"])
            ids_pts.append(g.loc[idx, "Ids"])
            vds_pts.append(g.loc[idx, "Vds"])
        vgs_pts, ids_pts, vds_pts = np.array(vgs_pts), np.array(ids_pts), np.array(vds_pts)
        order = np.argsort(vgs_pts)
        vgs_pts, ids_pts, vds_pts = vgs_pts[order], ids_pts[order], vds_pts[order]
        if len(vgs_pts) < 3:
            continue
        gm1 = smooth_derivative(vgs_pts, ids_pts, order=1)
        vgs_all.append(vgs_pts)
        vds_all.append(vds_pts)
        gm1_all.append(gm1)
    return np.concatenate(vgs_all), np.concatenate(vds_all), np.concatenate(gm1_all)


def build_tensors(data, device, gm_n_vds_targets=4, ids_region_frac=0.05,
                   gm_vgs_min=None, gm_vds_min=0.0):
    """data: dict (Vgsq,Vdsq)->DataFrame -> dict (Vgsq,Vdsq)-> tensors.

    `low_current_mask` flags points where |Ids| < ids_region_frac * max(|Ids|) for that
    quiescent point -- i.e. the deep-cutoff region, defined by the data itself (not a
    hand-picked Vgs threshold) so it generalizes across quiescent points automatically.
    Used by train_hypernet's --ids_region_weight to up-weight that region in the loss.

    gm_vgs_min/gm_vds_min restrict which points enter the gm1 loss at all (the Ids loss is
    NEVER affected) -- same convention as the base transistor_modeling repo's
    per_neuron_simple_angelov_nn_test.py --gm_vgs_min/--gm_vds_min: points are EXCLUDED
    (not just down-weighted) where Vgs < gm_vgs_min or Vds < gm_vds_min, combined via
    logical AND. gm_vds_min=0.0 is "off" (Vds >= 0 always holds); gm_vgs_min=None is "off"
    (0.0 would be a real Vgs threshold, since Vgs spans negative values here). Filtering
    happens AFTER build_gm_targets computes gm1 over the full transfer curve -- the
    Savitzky-Golay derivative needs the full curve to be accurate; only the LOSS should
    ignore the excluded points, not the derivative estimate itself."""
    out = {}
    for (vgsq, vdsq), df in data.items():
        vgsq_n = vgsq / VGS_SCALE
        vdsq_n = vdsq / VDS_SCALE
        vgs_t = torch.tensor(df["Vgs"].values / VGS_SCALE, dtype=torch.float32, device=device)
        vds_t = torch.tensor(df["Vds"].values / VDS_SCALE, dtype=torch.float32, device=device)
        ids_np = df["Ids"].values.astype(float)
        ids_t = torch.tensor(ids_np, dtype=torch.float32, device=device)
        qpoint = torch.tensor([vgsq_n, vdsq_n], dtype=torch.float32, device=device)

        low_current_np = (np.abs(ids_np) < ids_region_frac * np.abs(ids_np).max()).astype(np.float32)
        low_current_mask = torch.tensor(low_current_np, dtype=torch.float32, device=device)

        gm_vgs_np, gm_vds_np, gm1_true_np = build_gm_targets(df, gm_n_vds_targets)
        if gm_vgs_min is not None or gm_vds_min > 0.0:
            window_mask = np.ones_like(gm_vgs_np, dtype=bool)
            if gm_vgs_min is not None:
                window_mask &= (gm_vgs_np >= gm_vgs_min)
            if gm_vds_min > 0.0:
                window_mask &= (gm_vds_np >= gm_vds_min)
            if window_mask.any():
                gm_vgs_np = gm_vgs_np[window_mask]
                gm_vds_np = gm_vds_np[window_mask]
                gm1_true_np = gm1_true_np[window_mask]
            else:
                # This quiescent point's whole gm curve falls outside the window (e.g. a
                # point with Vdsq/Vds range entirely below --gm_vds_min) -- fall back to
                # keeping all its points rather than crashing on an empty gm1 loss later.
                print(f"  [gm window] Vgsq={vgsq:.1f} Vdsq={vdsq:.1f}: window excludes ALL "
                      f"{len(gm_vgs_np)} gm points -- keeping them unfiltered for this point.")
        gm_vgs_t = torch.tensor(gm_vgs_np, dtype=torch.float32, device=device)  # raw scale (needs grad later)
        gm_vds_t = torch.tensor(gm_vds_np / VDS_SCALE, dtype=torch.float32, device=device)
        gm1_true_t = torch.tensor(gm1_true_np, dtype=torch.float32, device=device)

        out[(vgsq, vdsq)] = dict(qpoint=qpoint, vgs=vgs_t, vds=vds_t, ids=ids_t,
                                  ids_np=ids_np, low_current_mask=low_current_mask,
                                  gm_vgs=gm_vgs_t, gm_vds=gm_vds_t, gm1_true=gm1_true_t)
    return out


def relative_sq_err(pred, true, floor_frac=0.1):
    """Pointwise (pred-true)^2 / max(|true_i|, floor_frac * max(|true|))^2 -- unlike dividing
    by a single per-curve scalar (norm[k]/gm1_norm[k] below), this equalizes RELATIVE error
    along the curve itself: a point near cutoff (true~0) gets the same relative weight as a
    point at peak current/gm, instead of its tiny absolute error being drowned out by the
    curve's own max. floor_frac keeps the denominator from blowing up where true is exactly
    0 (and where the numerically-differentiated gm1 target is noisiest)."""
    floor = floor_frac * true.abs().max().clamp_min(1e-12)
    scale = torch.maximum(true.abs(), floor)
    return ((pred - true) / scale) ** 2


def smoothness_penalty(hyper, qpoint):
    """Hutchinson-trick estimate of ||d theta/d qpoint||_F^2 (the squared Frobenius norm of
    H's own Jacobian at this qpoint) -- ONE extra backward pass instead of one per theta
    entry (which would cost n_params passes for an exact Jacobian, too slow to use as a loss
    term for architectures with dozens of params). Encodes directly the assumption LOO
    generalization actually needs: a held-out qpoint sits near the training ones, so if H
    varies smoothly, its predicted theta there should be close to its smooth-neighbor
    predictions too -- unlike gm1_weight/weight_decay, which regularize the CURVE SHAPE or
    the WEIGHT MAGNITUDES respectively, this regularizes H's own qpoint->theta mapping.
    Normalized by n_params so the penalty's scale doesn't grow with the main-net's size.
    See Hoffman et al. 2019, "Robust Learning with Jacobian Regularization"."""
    qpoint_req = qpoint.clone().requires_grad_(True)
    theta = hyper(qpoint_req)
    v = torch.randn_like(theta)
    proj = (theta * v).sum()
    jac_grad = torch.autograd.grad(proj, qpoint_req, create_graph=True)[0]
    return (jac_grad ** 2).sum() / theta.shape[0]


def _total_loss(hyper, train_keys, tensors, architecture, norm, gm1_weight, gm1_norm, ids_region_weight,
                 ids_relative_norm=False, gm1_relative_norm=False, relative_norm_floor_frac=0.1,
                 smoothness_weight=0.0):
    """Sum over train_keys of the (optionally gm1- and cutoff-region-weighted) per-point
    normalized MSE. Shared by the Adam loop and the L-BFGS closure so both optimize the
    exact same objective."""
    total_loss = 0.0
    for k in train_keys:
        t = tensors[k]
        theta = hyper(t["qpoint"])
        pred = main_net_forward(theta, t["vgs"], t["vds"], architecture)
        if ids_relative_norm:
            sq_err = relative_sq_err(pred, t["ids"], relative_norm_floor_frac)
        else:
            sq_err = (pred - t["ids"]) ** 2
        if ids_region_weight > 0:
            # Plain MSE(Ids) in absolute units barely penalizes the deep-cutoff region
            # (true Ids~0 there), since its absolute errors are tiny next to the
            # saturation-region ones -- so a non-monotonic wiggle there costs the
            # optimizer almost nothing. Up-weighting those low-current points (see
            # build_tensors' low_current_mask) directly counteracts that.
            sq_err = sq_err * (1.0 + ids_region_weight * t["low_current_mask"])
        loss = torch.mean(sq_err) if ids_relative_norm else torch.mean(sq_err) / norm[k]

        if gm1_weight > 0:
            # gm1 = dIds/dVgs, matched against the real (measured) transconductance --
            # penalizing this shapes the WHOLE Ids(Vgs) curve, not just its value, which
            # discourages the kind of non-monotonic wiggle a mixed tanh/swish net can
            # otherwise settle into (e.g. near cutoff, where Ids itself gives almost no
            # gradient signal but the true gm1 there is still ~0).
            gm_vgs_req = t["gm_vgs"].clone().requires_grad_(True)
            ids_gm = main_net_forward(theta, gm_vgs_req / VGS_SCALE, t["gm_vds"], architecture)
            gm1_pred = torch.autograd.grad(ids_gm, gm_vgs_req, grad_outputs=torch.ones_like(ids_gm),
                                            create_graph=True)[0]
            if gm1_relative_norm:
                loss_gm1 = torch.mean(relative_sq_err(gm1_pred, t["gm1_true"], relative_norm_floor_frac))
            else:
                loss_gm1 = torch.mean((gm1_pred - t["gm1_true"]) ** 2) / gm1_norm[k]
            loss = loss + gm1_weight * loss_gm1

        if smoothness_weight > 0:
            loss = loss + smoothness_weight * smoothness_penalty(hyper, t["qpoint"])

        total_loss = total_loss + loss
    return total_loss


def train_hypernet(train_keys, tensors, n_params, architecture, epochs, lr, device,
                    gm1_weight=0.0, ids_region_weight=0.0, lbfgs_epochs=5, lbfgs_max_iter=200,
                    h_hidden=32, h_layers=2, n_in=2, log_every=500, seed=27,
                    ids_relative_norm=False, gm1_relative_norm=False, relative_norm_floor_frac=0.1,
                    weight_decay=0.0, smoothness_weight=0.0):
    torch.manual_seed(seed)
    hyper = HyperNetwork(n_params, hidden=h_hidden, n_hidden_layers=h_layers, n_in=n_in).to(device)
    # AdamW, not Adam -- with weight_decay=0.0 (the default) its decoupled-decay term vanishes
    # and it's identical to plain Adam, so this is a no-op for every existing caller that
    # doesn't pass weight_decay. Passing a nonzero value regularizes H's own weights directly
    # (unlike gm1_weight/ids_region_weight, which shape the loss, not the weights themselves) --
    # aimed at the generalization gap (full-fit vs. LOO) this project has repeatedly run into,
    # not at "escaping local minima" (the loss surface here is small, smooth, full-batch, and
    # already gets an L-BFGS polish -- optimization difficulty was never the bottleneck).
    opt = torch.optim.AdamW(hyper.parameters(), lr=lr, weight_decay=weight_decay)

    # per-point normalization so no single quiescent point's current (or gm1) scale dominates the
    # loss -- unused when ids_relative_norm/gm1_relative_norm is on (relative_sq_err normalizes
    # pointwise instead), still computed cheaply either way to keep the call sites simple.
    norm = {k: (np.abs(tensors[k]["ids_np"]).max() + 1e-6) ** 2 for k in train_keys}
    gm1_norm = {}
    if gm1_weight > 0:
        gm1_norm = {k: (tensors[k]["gm1_true"].abs().max().item() + 1e-6) ** 2 for k in train_keys}

    for epoch in range(epochs):
        opt.zero_grad()
        total_loss = _total_loss(hyper, train_keys, tensors, architecture, norm,
                                  gm1_weight, gm1_norm, ids_region_weight,
                                  ids_relative_norm, gm1_relative_norm, relative_norm_floor_frac,
                                  smoothness_weight)
        total_loss.backward()
        opt.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:5d}  train_loss(sum of per-point rel-MSE) = {total_loss.item():.5f}")

    # L-BFGS polishing: Adam's stochastic-ish updates plateau short of a local optimum on a
    # loss this smooth (small net, full-batch, deterministic) -- a few L-BFGS steps with a
    # strong-Wolfe line search reliably sharpen the last stretch, same as the base
    # transistor_modeling repo's per-CSV pipeline (--lbfgs_epochs/--lbfgs_max_iter).
    if lbfgs_epochs > 0:
        lbfgs_opt = torch.optim.LBFGS(hyper.parameters(), max_iter=lbfgs_max_iter,
                                       line_search_fn="strong_wolfe")

        def closure():
            lbfgs_opt.zero_grad()
            loss = _total_loss(hyper, train_keys, tensors, architecture, norm,
                                gm1_weight, gm1_norm, ids_region_weight,
                                ids_relative_norm, gm1_relative_norm, relative_norm_floor_frac,
                                smoothness_weight)
            loss.backward()
            return loss

        for step in range(lbfgs_epochs):
            loss = lbfgs_opt.step(closure)
            if log_every:
                print(f"    L-BFGS step {step + 1}/{lbfgs_epochs}  loss = {loss.item():.5f}")

    return hyper


def evaluate(hyper, key, tensors, architecture, device):
    """Returns (rel_rmse, pred_np) for `key` under `hyper`."""
    t = tensors[key]
    with torch.no_grad():
        theta = hyper(t["qpoint"])
        pred = main_net_forward(theta, t["vgs"], t["vds"], architecture).cpu().numpy()
    return rel_rmse(pred, t["ids_np"]), pred


def make_predict_fn(hyper, qpoint_tensor, architecture, device):
    """Ids(Vgs, Vds) at this hyper's fixed theta=hyper(qpoint), for arbitrary Vgs/Vds arrays
    (used to draw the smooth model curves in the IV plots)."""
    def predict_fn(vgs_arr, vds_arr):
        with torch.no_grad():
            theta = hyper(qpoint_tensor)
            vgs_t = torch.tensor(vgs_arr / VGS_SCALE, dtype=torch.float32, device=device)
            vds_t = torch.tensor(vds_arr / VDS_SCALE, dtype=torch.float32, device=device)
            return main_net_forward(theta, vgs_t, vds_t, architecture).cpu().numpy()
    return predict_fn


def run_full_fit_job(data, n_params, architecture, epochs, lr, gm1_weight,
                      ids_region_weight, ids_region_frac, lbfgs_epochs, lbfgs_max_iter,
                      h_hidden, h_layers, h_physics, seed, models_dir, plots_dir,
                      ids_relative_norm=False, gm1_relative_norm=False, relative_norm_floor_frac=0.1,
                      gm_vgs_min=None, gm_vds_min=0.0, weight_decay=0.0, smoothness_weight=0.0):
    """Runs in its own process: train jointly on all quiescent points, evaluate + plot
    each in-sample, save the model. Independent of the LOO jobs -> safe to parallelize."""
    torch.set_num_threads(1)  # tiny model/data -> intra-op threading only adds overhead;
                               # parallelism instead comes from running jobs as separate processes
    device = torch.device("cpu")
    keys = sorted(data.keys())
    tensors = build_tensors(data, device, ids_region_frac=ids_region_frac,
                             gm_vgs_min=gm_vgs_min, gm_vds_min=gm_vds_min)
    qpoints = build_qpoint_tensors(data, keys, None, device, h_physics)
    for k in keys:
        tensors[k]["qpoint"] = qpoints[k]
    n_in = 2 + (N_PHYSICS_FEATURES if h_physics else 0)
    hyper = train_hypernet(keys, tensors, n_params, architecture, epochs, lr, device,
                            gm1_weight=gm1_weight, ids_region_weight=ids_region_weight,
                            lbfgs_epochs=lbfgs_epochs, lbfgs_max_iter=lbfgs_max_iter,
                            h_hidden=h_hidden, h_layers=h_layers, n_in=n_in, log_every=0, seed=seed,
                            ids_relative_norm=ids_relative_norm, gm1_relative_norm=gm1_relative_norm,
                            relative_norm_floor_frac=relative_norm_floor_frac,
                            weight_decay=weight_decay, smoothness_weight=smoothness_weight)

    errs = {}
    for k in keys:
        err, pred = evaluate(hyper, k, tensors, architecture, device)
        errs[k] = err
        predict_fn = make_predict_fn(hyper, tensors[k]["qpoint"], architecture, device)
        iv_data = build_iv_curves(data[k], predict_fn)
        plot_iv_grid(iv_data, k[0], k[1], err,
                     os.path.join(plots_dir, f"infit_{qtag(*k)}.png"),
                     title_suffix="(in-sample, full-fit)")
    torch.save(hyper.state_dict(), os.path.join(models_dir, "hyper_full.pt"))
    return errs


def run_loo_job(held_out, data, n_params, architecture, epochs, lr, gm1_weight,
                 ids_region_weight, ids_region_frac, lbfgs_epochs, lbfgs_max_iter,
                 h_hidden, h_layers, h_physics, h_physics_oracle, seed, models_dir, plots_dir,
                 ids_relative_norm=False, gm1_relative_norm=False, relative_norm_floor_frac=0.1,
                 gm_vgs_min=None, gm_vds_min=0.0, weight_decay=0.0, smoothness_weight=0.0):
    """Runs in its own process: train on the other 5 quiescent points, evaluate + plot the
    held-out one, save the model. Each held_out fold is fully independent of the others."""
    torch.set_num_threads(1)
    device = torch.device("cpu")
    keys = sorted(data.keys())
    train_keys = [k for k in keys if k != held_out]
    tensors = build_tensors(data, device, ids_region_frac=ids_region_frac,
                             gm_vgs_min=gm_vgs_min, gm_vds_min=gm_vds_min)
    qpoints = build_qpoint_tensors(data, train_keys, held_out, device, h_physics,
                                    oracle_held_out=h_physics_oracle)
    for k, qp in qpoints.items():
        tensors[k]["qpoint"] = qp
    n_in = 2 + (N_PHYSICS_FEATURES if h_physics else 0)
    hyper = train_hypernet(train_keys, tensors, n_params, architecture, epochs, lr, device,
                            gm1_weight=gm1_weight, ids_region_weight=ids_region_weight,
                            lbfgs_epochs=lbfgs_epochs, lbfgs_max_iter=lbfgs_max_iter,
                            h_hidden=h_hidden, h_layers=h_layers, n_in=n_in, log_every=0, seed=seed,
                            ids_relative_norm=ids_relative_norm, gm1_relative_norm=gm1_relative_norm,
                            relative_norm_floor_frac=relative_norm_floor_frac,
                            weight_decay=weight_decay, smoothness_weight=smoothness_weight)

    err, pred = evaluate(hyper, held_out, tensors, architecture, device)
    tag = qtag(*held_out)
    torch.save(hyper.state_dict(), os.path.join(models_dir, f"hyper_loo_{tag}.pt"))
    predict_fn = make_predict_fn(hyper, tensors[held_out]["qpoint"], architecture, device)
    iv_data = build_iv_curves(data[held_out], predict_fn)
    plot_iv_grid(iv_data, held_out[0], held_out[1], err,
                 os.path.join(plots_dir, f"loo_{tag}.png"),
                 title_suffix="(held-out, LOO)")
    return held_out, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=r"C:\Users\acost\repos\csvs")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--architecture", default="arch89",
                     help="Main-net per-neuron mixed-activation architecture: either a preset name "
                          f"from model.ARCHITECTURES ({sorted(ARCHITECTURES.keys())}), or a raw JSON "
                          "string of per-layer activation lists, e.g. '[[\"tanh\",\"swish\"],[\"tanh\"]]' "
                          "(same format architecture_search.py emits for its winning candidates).")
    ap.add_argument("--gm1_weight", type=float, default=0.0,
                     help="Weight of a gm1 (dIds/dVgs) matching loss added to the Ids MSE, as done "
                          "in the base transistor_modeling repo's per-CSV models. gm1 targets are "
                          "estimated from real transfer curves assembled via nearest-point matching "
                          "per TN group (see build_gm_targets); gm1_pred comes from autograd through "
                          "the hypernetwork-generated main-net. 0.0 (default) = off, matches the "
                          "original Ids-only loss. Try e.g. 0.1 (the base repo's default gm1 weight).")
    ap.add_argument("--ids_region_weight", type=float, default=0.0,
                     help="Extra weight on the deep-cutoff region of the Ids MSE loss (points where "
                          "|Ids| < ids_region_frac * max(|Ids|) for that quiescent point): squared "
                          "error there is multiplied by (1 + ids_region_weight). Plain absolute-unit "
                          "MSE barely penalizes that region (its errors are tiny next to the "
                          "saturation-region ones), which is why the main-net can settle into a "
                          "non-monotonic wiggle there at zero cost. 0.0 (default) = off.")
    ap.add_argument("--ids_region_frac", type=float, default=0.05,
                     help="Threshold (fraction of that quiescent point's max |Ids|) defining the "
                          "'low-current'/cutoff region up-weighted by --ids_region_weight.")
    ap.add_argument("--ids_relative_norm", action="store_true",
                     help="Replace the Ids loss's single per-quiescent-point normalization "
                          "(divide by max|Ids|^2, same for every sample in that curve) with a "
                          "POINTWISE relative normalization: each sample's squared error is "
                          "divided by max(|Ids_i|, --relative_norm_floor_frac * max|Ids|)^2. This "
                          "equalizes RELATIVE error along the curve (a tiny absolute error near "
                          "cutoff counts as much as the same relative error at peak current), "
                          "instead of --ids_region_weight's binary mask + fixed multiplier. Off by "
                          "default (matches the original absolute-unit MSE).")
    ap.add_argument("--gm1_relative_norm", action="store_true",
                     help="Same pointwise relative normalization as --ids_relative_norm, but for "
                          "the gm1 loss (only has an effect when --gm1_weight > 0). Replaces "
                          "gm1's single per-quiescent-point normalization (divide by max|gm1|^2) "
                          "with dividing each sample by max(|gm1_i|, --relative_norm_floor_frac * "
                          "max|gm1|)^2. Off by default.")
    ap.add_argument("--relative_norm_floor_frac", type=float, default=0.1,
                     help="Floor for --ids_relative_norm/--gm1_relative_norm's per-sample "
                          "denominator, as a fraction of that curve's own max|value| -- keeps the "
                          "denominator from blowing up where the true value is exactly (or very "
                          "near) 0, which would otherwise make the loss unstable.")
    ap.add_argument("--gm_vgs_min", type=float, default=None,
                     help="Restrict the gm1 loss to points with Vgs >= this value [V] -- points "
                          "outside are EXCLUDED entirely (not down-weighted), same convention as "
                          "the base transistor_modeling repo's --gm_vgs_min. The Ids loss is never "
                          "affected. None (default) = off; 0.0 is a real threshold here (Vgs spans "
                          "negative values), so 'off' can't be 0.0. E.g. --gm_vgs_min -3 keeps only "
                          "the gm1 loss where Vgs >= -3V, dropping the noisiest deep-cutoff samples.")
    ap.add_argument("--gm_vds_min", type=float, default=0.0,
                     help="Restrict the gm1 loss to points with Vds >= this value [V] -- points "
                          "outside are EXCLUDED entirely, same convention as the base repo's "
                          "--gm_vds_min. The Ids loss is never affected. 0.0 (default) = off "
                          "(Vds >= 0 always holds). Combined with --gm_vgs_min via logical AND, "
                          "e.g. --gm_vgs_min -3 --gm_vds_min 4 keeps only the gm1 window "
                          "Vgs >= -3V AND Vds >= 4V.")
    ap.add_argument("--weight_decay", type=float, default=0.0,
                     help="AdamW weight decay on H's own weights. 0.0 (default) makes AdamW "
                          "identical to plain Adam (fully backward compatible). Regularizes the "
                          "weights directly, aimed at the LOO generalization gap.")
    ap.add_argument("--smoothness_weight", type=float, default=0.0,
                     help="Weight of a Jacobian-norm penalty on H's own qpoint->theta mapping "
                          "(Hutchinson-trick estimate of ||d theta/d qpoint||_F^2, one extra "
                          "backward pass per point). Encourages H to vary smoothly with "
                          "(Vgsq,Vdsq), which is exactly the property LOO extrapolation to a "
                          "nearby held-out qpoint needs. 0.0 (default) = off.")
    ap.add_argument("--lbfgs_epochs", type=int, default=5,
                     help="L-BFGS polishing steps run after the Adam phase (outer steps; each runs "
                          "up to --lbfgs_max_iter iterations with a strong-Wolfe line search), same "
                          "as the base transistor_modeling repo's per-CSV pipeline. 0 = skip polishing.")
    ap.add_argument("--lbfgs_max_iter", type=int, default=200,
                     help="Max L-BFGS iterations per polishing step.")
    ap.add_argument("--seed", type=int, default=27,
                     help="torch.manual_seed for the HyperNetwork's weight init. Fixed by default "
                          "(27, matching the base repo's default) so runs are reproducible.")
    ap.add_argument("--h_hidden", type=int, default=32,
                     help="H's hidden-layer width. H's own size is ~1152 + 33*n_params(f) at the "
                          "default 32 -- the LAST layer (hidden -> n_params) dominates H's total "
                          "param count, so shrinking this directly shrinks H.")
    ap.add_argument("--h_layers", type=int, default=2,
                     help="Number of hidden layers in H (each `--h_hidden` wide). 1 removes the "
                          "middle hidden->hidden layer entirely.")
    ap.add_argument("--h_physics", action="store_true",
                     help="Feed H 4 extra Angelov-style physics features (Vpk, Ipk, alpha, lambda) "
                          "alongside (Vgsq, Vdsq): 6 inputs instead of 2. Extracted from each "
                          "training point's own real data (physics_features.extract_angelov_features); "
                          "for the LOO held-out point, PREDICTED via inverse-distance interpolation "
                          "from the other points' real features (never extracted from its own curve, "
                          "which would leak the answer). Off by default (2 inputs, qpoint only).")
    ap.add_argument("--h_physics_oracle", action="store_true",
                     help="Only relevant with --h_physics: for the LOO held-out point, extract "
                          "its physics features from its OWN real data instead of predicting them "
                          "via interpolation. This is a diagnostic ablation, NOT a blind LOO test "
                          "anymore -- it isolates whether bad interpolation (vs. the physics-"
                          "augmentation approach itself) is what hurts LOO. Off by default.")
    ap.add_argument("--output_dir", default=r"C:\Users\acost\repos\transistor_modeling_hyper_outputs\results",
                     help="Where to write saved models (models/), diagnostic plots (plots/), "
                          "and the metrics summary (run_info.json). Defaults OUTSIDE this repo "
                          "(sibling folder) so training runs never create untracked files inside "
                          "the git repo -- pass an absolute path to write anywhere else.")
    ap.add_argument("--workers", type=int, default=None,
                     help="Parallel processes for the 7 independent trainings (full-fit + 6 "
                          "LOO folds). Default: min(7, cpu_count()). Each fold trains its own "
                          "HyperNetwork from scratch with no shared state, so this is embarrassingly "
                          "parallel -- and since the model/data are tiny, per-process work is "
                          "overhead-bound rather than FLOP-bound, so running folds concurrently "
                          "as separate single-threaded processes is far more effective than "
                          "PyTorch's default intra-op thread pool on a model this small.")
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
        architecture = json.loads(args.architecture)  # raw '[["tanh","swish"],...]' string
    n_params = main_net_n_params(architecture)
    print(f"Main-net architecture={args.architecture} ({architecture}), n_params={n_params}")

    n_jobs = 1 + len(keys)  # full-fit + one LOO fold per quiescent point
    workers = args.workers or min(n_jobs, os.cpu_count() or 1)
    print(f"\nRunning {n_jobs} independent trainings ({args.epochs} epochs each) "
          f"across {workers} worker processes ...")

    full_errs_by_key = {}
    loo_errs_by_key = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        full_future = ex.submit(run_full_fit_job, data, n_params, architecture, args.epochs, args.lr,
                                 args.gm1_weight, args.ids_region_weight, args.ids_region_frac,
                                 args.lbfgs_epochs, args.lbfgs_max_iter, args.h_hidden, args.h_layers,
                                 args.h_physics, args.seed, models_dir, plots_dir,
                                 args.ids_relative_norm, args.gm1_relative_norm, args.relative_norm_floor_frac,
                                 args.gm_vgs_min, args.gm_vds_min, args.weight_decay, args.smoothness_weight)
        loo_futures = {ex.submit(run_loo_job, k, data, n_params, architecture, args.epochs, args.lr,
                                  args.gm1_weight, args.ids_region_weight, args.ids_region_frac,
                                  args.lbfgs_epochs, args.lbfgs_max_iter, args.h_hidden, args.h_layers,
                                  args.h_physics, args.h_physics_oracle,
                                  args.seed, models_dir, plots_dir,
                                  args.ids_relative_norm, args.gm1_relative_norm,
                                  args.relative_norm_floor_frac,
                                  args.gm_vgs_min, args.gm_vds_min,
                                  args.weight_decay, args.smoothness_weight): k for k in keys}

        full_errs_by_key = full_future.result()
        print("\n=== Full-fit sanity check (train on all 6, evaluate in-sample) ===")
        for k in keys:
            print(f"  in-sample Vgsq={k[0]:.1f} Vdsq={k[1]:.1f}: rel.RMSE = {full_errs_by_key[k]*100:.2f}%")
        print(f"  --> mean in-sample rel.RMSE: {np.mean(list(full_errs_by_key.values()))*100:.2f}%")

        print("\n=== Leave-one-out (train on 5, predict the 6th via H(Vgsq,Vdsq)) ===")
        for fut in as_completed(loo_futures):
            held_out, err = fut.result()
            loo_errs_by_key[held_out] = err
            print(f"  held out Vgsq={held_out[0]:.1f} Vdsq={held_out[1]:.1f}: rel.RMSE(Ids) = {err*100:.2f}%")

    full_errs = [full_errs_by_key[k] for k in keys]
    loo_errs = [loo_errs_by_key[k] for k in keys]
    print(f"\n  --> LOO mean rel.RMSE across 6 held-out points: {np.mean(loo_errs)*100:.2f}%")
    print("\n  (compare: raw-weight polynomial interpolation of 6 independently-trained "
          "nets gave ~39-90% mean rel.RMSE)")

    run_info = {
        "csv_dir": args.csv_dir,
        "epochs": args.epochs,
        "lr": args.lr,
        "architecture_name": args.architecture,
        "architecture": architecture,
        "n_params": n_params,
        "gm1_weight": args.gm1_weight,
        "ids_region_weight": args.ids_region_weight,
        "ids_region_frac": args.ids_region_frac,
        "ids_relative_norm": args.ids_relative_norm,
        "gm1_relative_norm": args.gm1_relative_norm,
        "relative_norm_floor_frac": args.relative_norm_floor_frac,
        "gm_vgs_min": args.gm_vgs_min,
        "gm_vds_min": args.gm_vds_min,
        "weight_decay": args.weight_decay,
        "smoothness_weight": args.smoothness_weight,
        "lbfgs_epochs": args.lbfgs_epochs,
        "lbfgs_max_iter": args.lbfgs_max_iter,
        "h_hidden": args.h_hidden,
        "h_layers": args.h_layers,
        "h_physics": args.h_physics,
        "h_physics_oracle": args.h_physics_oracle,
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
                                     args.epochs, args.lr, n_params, args.architecture)
    print(f"\nSaved models to {models_dir}")
    print(f"Saved plots to {plots_dir}")
    print(f"Saved metrics to {os.path.join(args.output_dir, 'run_info.json')}")
    print(f"Saved HTML report to {report_path}")


if __name__ == "__main__":
    main()
