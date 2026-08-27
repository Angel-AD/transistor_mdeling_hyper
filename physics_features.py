"""
Angelov-style physics features (Vpk, Ipk, alpha, lambda), extracted from one quiescent
point's real pulsed-IV data, for use as EXTRA inputs to the hypernetwork H (in addition
to the raw (Vgsq, Vdsq) it already takes).

Classic Angelov large-signal model (the transconductance-shape term (1+tanh(psi)) is
exactly 1 at Vgs=Vpk by construction, so at that one Vgs slice the output equation
reduces to):
    Ids(Vds) = Ipk * tanh(alpha*Vds) * (1 + lambda*Vds)      at Vgs = Vpk

Extraction, per quiescent point:
  1. Build a transfer curve (Ids vs Vgs) at a representative (high) Vds via nearest-point
     matching per TN group (same technique as build_gm_targets/build_iv_curves).
  2. Vpk = Vgs at max gm1 = d(Ids)/d(Vgs); Ipk = Ids there.
  3. Build the output curve (Ids vs Vds) from the TN group whose own mean Vgs is closest
     to Vpk (nearest REAL sweep, no interpolation).
  4. Fit alpha, lambda (Ipk held fixed from step 2) via least squares.

IMPORTANT (leakage): these are only valid to use directly for a quiescent point whose
OWN measured data you have. For a leave-one-out held-out point, extracting from its own
curve would leak the answer -- use predict_physics_features() instead, which interpolates
from the OTHER (known) points' real extracted features.
"""
import numpy as np
from scipy.optimize import curve_fit


def _transfer_curve(df, target_vds):
    groups = [g.sort_values("Vds") for _, g in df.groupby("TN")]
    vgs_pts, ids_pts = [], []
    for g in groups:
        idx = (g["Vds"] - target_vds).abs().idxmin()
        vgs_pts.append(g.loc[idx, "Vgs"])
        ids_pts.append(g.loc[idx, "Ids"])
    vgs_pts, ids_pts = np.array(vgs_pts), np.array(ids_pts)
    order = np.argsort(vgs_pts)
    return vgs_pts[order], ids_pts[order]


def _output_curve_nearest_vgs(df, target_vgs):
    groups = [(g["Vgs"].mean(), g.sort_values("Vds")) for _, g in df.groupby("TN")]
    _, best_g = min(groups, key=lambda t: abs(t[0] - target_vgs))
    return best_g["Vds"].values, best_g["Ids"].values


def extract_angelov_features(df):
    """df: raw DataFrame (Vgs, Vds, Ids, TN) for ONE quiescent point. Returns
    dict(Vpk=..., Ipk=..., alpha=..., lambda_=...), all real numbers from real data."""
    vds_hi = df["Vds"].max()
    vgs_t, ids_t = _transfer_curve(df, vds_hi)
    gm1 = np.gradient(ids_t, vgs_t)
    i_pk = int(np.argmax(gm1))
    Vpk, Ipk = float(vgs_t[i_pk]), float(ids_t[i_pk])

    vds_o, ids_o = _output_curve_nearest_vgs(df, Vpk)

    def model(vds, alpha, lam):
        return Ipk * np.tanh(alpha * vds) * (1.0 + lam * vds)

    try:
        popt, _ = curve_fit(model, vds_o, ids_o, p0=[0.2, 0.005],
                             bounds=([1e-4, -0.1], [10.0, 0.5]), maxfev=5000)
        alpha, lam = float(popt[0]), float(popt[1])
    except Exception:
        alpha, lam = 0.2, 0.005  # fallback to the initial guess if the fit fails

    return dict(Vpk=Vpk, Ipk=Ipk, alpha=alpha, lambda_=lam)


def predict_physics_features(qpoint, known_features):
    """qpoint: (Vgsq, Vdsq). known_features: dict {(Vgsq,Vdsq): feature_dict} for the
    OTHER (known) quiescent points -- inverse-distance-weighted interpolation, robust
    with very few (~5) points and always defined (no extrapolation blowup like a fitted
    polynomial could have). NEVER call this with qpoint's own real data in known_features
    for a leave-one-out evaluation -- that would leak the held-out point's answer."""
    keys = list(known_features.keys())
    vgsq, vdsq = qpoint
    # normalize the two axes before computing distance so Vdsq's larger raw scale doesn't dominate
    dists = []
    for k in keys:
        d = np.hypot((k[0] - vgsq) / 4.0, (k[1] - vdsq) / 45.0)
        dists.append(d)
    dists = np.array(dists)
    if np.any(dists < 1e-9):
        return known_features[keys[int(np.argmin(dists))]]
    w = 1.0 / dists**2
    w /= w.sum()
    out = {}
    for feat in ["Vpk", "Ipk", "alpha", "lambda_"]:
        out[feat] = float(sum(wi * known_features[k][feat] for wi, k in zip(w, keys)))
    return out


if __name__ == "__main__":
    from data_loader import load_all
    data = load_all(r"C:\Users\acost\repos\csvs")
    for k in sorted(data.keys()):
        feats = extract_angelov_features(data[k])
        print(f"Vgsq={k[0]:.1f} Vdsq={k[1]:.1f}: Vpk={feats['Vpk']:.3f}  Ipk={feats['Ipk']:.3f}  "
              f"alpha={feats['alpha']:.4f}  lambda={feats['lambda_']:.5f}")
