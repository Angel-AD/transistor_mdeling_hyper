"""
Diagnostic plots for a trained hypernetwork fold (full-fit or one LOO held-out point),
in the same style as the base transistor_modeling repo's IV plots: Ids-Vds curves for
several Vgs values, and Ids-Vgs curves for several Vds values.

Each quiescent-point CSV's TN column groups one pulsed sweep (~fixed Vgs, Vds ramping),
exactly like the Auriga-format TN groups the base repo's plotting already relies on --
so each TN group IS one natural Ids-vs-Vds-at-fixed-Vgs curve, and picking one matched
point per TN group at a target Vds gives one Ids-vs-Vgs-at-fixed-Vds curve.
"""
import html as _html
import os

import matplotlib
matplotlib.use("Agg")  # no GUI backend -> avoids blocking/hanging on Windows
import matplotlib.pyplot as plt
import numpy as np

from signal_utils import smooth_derivative


def qtag(vgsq, vdsq):
    """(-2.9, 0.0) -> 'Vgsq-2.9_Vdsq0'"""
    return f"Vgsq{vgsq:g}_Vdsq{vdsq:g}"


def build_iv_curves(df, predict_fn, n_vgs_curves=5, n_vds_curves=4, n_fine=200):
    """df: DataFrame with Vgs, Vds, Ids, TN for ONE quiescent point (raw measured rows).
    predict_fn(vgs_array, vds_array) -> ids_array: model prediction (theta already fixed
    for this quiescent point via closure).

    Returns dict(vgs_sweeps=[...], vds_sweeps=[...]):
      vgs_sweeps: Ids vs Vds curves, one per selected (near-fixed) Vgs value (from TN groups).
      vds_sweeps: Ids vs Vgs curves, one per selected fixed-Vds target (nearest real point
                  per TN group). Each also carries gm1/gm2/gm3 (d/dVgs, d2/dVgs2, d3/dVgs3
                  of Ids), true (from the measured points) and pred (from the fine model
                  curve), via the same Savitzky-Golay smooth_derivative used for the gm1
                  training target (see train_loo.py's build_gm_targets).
    """
    groups = [g.sort_values("Vds") for _, g in df.groupby("TN")]
    groups = sorted(groups, key=lambda g: g["Vgs"].mean())
    n = len(groups)

    # --- Ids vs Vds, for several Vgs values (each TN group = one real sweep) ---
    if n <= n_vgs_curves:
        picks = groups
    else:
        idxs = sorted(set(np.linspace(0, n - 1, n_vgs_curves).round().astype(int)))
        picks = [groups[i] for i in idxs]

    vgs_sweeps = []
    for g in picks:
        vgs_mean = float(g["Vgs"].mean())
        vds_true = g["Vds"].values
        ids_true = g["Ids"].values
        vds_fine = np.linspace(vds_true.min(), vds_true.max(), n_fine)
        vgs_fine = np.full_like(vds_fine, vgs_mean)
        ids_pred = predict_fn(vgs_fine, vds_fine)
        vgs_sweeps.append(dict(vgs=vgs_mean, vds_true=vds_true, ids_true=ids_true,
                                vds_pred=vds_fine, ids_pred=ids_pred))

    # --- Ids vs Vgs, for several fixed-Vds targets (nearest real point per TN group) ---
    vds_lo, vds_hi = df["Vds"].min(), df["Vds"].max()
    target_vds_list = (np.linspace(vds_lo, vds_hi, n_vds_curves) if n_vds_curves > 1
                        else [0.5 * (vds_lo + vds_hi)])

    vds_sweeps = []
    for t_vds in target_vds_list:
        vgs_pts, ids_pts, matched_vds = [], [], []
        for g in groups:
            idx = (g["Vds"] - t_vds).abs().idxmin()
            vgs_pts.append(g.loc[idx, "Vgs"])
            ids_pts.append(g.loc[idx, "Ids"])
            matched_vds.append(g.loc[idx, "Vds"])
        vgs_pts, ids_pts = np.array(vgs_pts), np.array(ids_pts)
        order = np.argsort(vgs_pts)
        vgs_pts, ids_pts = vgs_pts[order], ids_pts[order]
        actual_vds = float(np.mean(matched_vds))
        vgs_fine = np.linspace(vgs_pts.min(), vgs_pts.max(), n_fine)
        vds_fine = np.full_like(vgs_fine, actual_vds)
        ids_pred = predict_fn(vgs_fine, vds_fine)

        gm_true = [smooth_derivative(vgs_pts, ids_pts, order=o) for o in (1, 2, 3)]
        gm_pred = [smooth_derivative(vgs_fine, ids_pred, order=o) for o in (1, 2, 3)]

        vds_sweeps.append(dict(vds=actual_vds, vgs_true=vgs_pts, ids_true=ids_pts,
                                vgs_pred=vgs_fine, ids_pred=ids_pred,
                                gm1_true=gm_true[0], gm2_true=gm_true[1], gm3_true=gm_true[2],
                                gm1_pred=gm_pred[0], gm2_pred=gm_pred[1], gm3_pred=gm_pred[2]))

    return dict(vgs_sweeps=vgs_sweeps, vds_sweeps=vds_sweeps)


def plot_iv_grid(iv_data, vgsq, vdsq, rel_rmse_val, save_path, title_suffix=""):
    """Points = measured, lines = model. 2x3 grid, same layout convention as the base
    transistor_modeling repo's plot_grid: Ids-Vds, Ids-Vgs, gm1-Vgs (top row), gm2-Vgs,
    gm3-Vgs (bottom row). gm1/gm2/gm3 = d/dVgs, d2/dVgs2, d3/dVgs3 of Ids, at the same
    fixed-Vds curves as the Ids-Vgs panel (see build_iv_curves)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    cmap = plt.cm.viridis

    ax = axes[0, 0]
    sweeps = iv_data["vgs_sweeps"]
    for i, s in enumerate(sweeps):
        c = cmap(i / max(len(sweeps) - 1, 1))
        ax.plot(s["vds_true"], s["ids_true"], "o", ms=3, color=c, alpha=0.6)
        ax.plot(s["vds_pred"], s["ids_pred"], "-", color=c, lw=1.5, label=f"Vgs={s['vgs']:.2f}V")
    ax.set_xlabel("Vds [V]")
    ax.set_ylabel("Ids [A]")
    ax.set_title("Ids-Vds (points: measured, lines: model)")
    ax.legend(fontsize=7)

    sweeps2 = iv_data["vds_sweeps"]
    panels = [
        (axes[0, 1], "ids", "Ids [A]", "Ids-Vgs"),
        (axes[0, 2], "gm1", "gm1 = dIds/dVgs [S]", "gm1-Vgs"),
        (axes[1, 0], "gm2", "gm2 = d2Ids/dVgs2 [S/V]", "gm2-Vgs"),
        (axes[1, 1], "gm3", "gm3 = d3Ids/dVgs3 [S/V2]", "gm3-Vgs"),
    ]
    for ax2, key, ylabel, title in panels:
        for i, s in enumerate(sweeps2):
            c = cmap(i / max(len(sweeps2) - 1, 1))
            ax2.plot(s["vgs_true"], s[f"{key}_true"], "o", ms=3, color=c, alpha=0.6)
            ax2.plot(s["vgs_pred"], s[f"{key}_pred"], "-", color=c, lw=1.5, label=f"Vds={s['vds']:.1f}V")
        ax2.set_xlabel("Vgs [V]")
        ax2.set_ylabel(ylabel)
        ax2.set_title(f"{title} (points: measured, lines: model)")
        ax2.legend(fontsize=7)

    axes[1, 2].set_visible(False)

    fig.suptitle(f"Vgsq={vgsq:.1f} Vdsq={vdsq:.1f} {title_suffix} rel.RMSE={rel_rmse_val*100:.2f}%")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: -apple-system, "Segoe UI", sans-serif; max-width: 1400px; margin: 2rem auto;
        padding: 0 1rem; line-height: 1.5; }}
h1 {{ border-bottom: 1px solid #88888844; padding-bottom: 0.3rem; }}
h2 {{ border-bottom: 1px solid #88888844; padding-bottom: 0.3rem; margin-top: 2.5rem; }}
table.summary {{ border-collapse: collapse; margin: 1rem 0; }}
table.summary th, table.summary td {{ border: 1px solid #88888844; padding: 0.4rem 0.7rem; text-align: right; }}
table.summary th {{ background: #80808022; }}
table.summary td:first-child, table.summary th:first-child {{ text-align: left; }}
.plot-row {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.5rem 0 1rem; }}
.plot-row figure {{ margin: 0; flex: 1 1 480px; }}
.plot-row img {{ max-width: 100%; height: auto; display: block; border: 1px solid #88888833; }}
figcaption {{ font-size: 0.85rem; opacity: 0.75; margin-top: 0.3rem; }}
code {{ font-family: ui-monospace, Consolas, monospace; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def build_html_report(output_dir, keys, full_errs, loo_errs, epochs, lr, n_params, architecture_name=""):
    """Writes output_dir/report.html gathering plots/infit_*.png and plots/loo_*.png
    (as written by plot_iv_grid) into one page, one section per quiescent point."""
    rows = []
    for k, e in zip(keys, full_errs):
        rows.append(f"<tr><td>Vgsq={k[0]:.1f} Vdsq={k[1]:.1f}</td><td>{e*100:.2f}%</td></tr>")
    summary_full = "\n".join(rows)
    rows = []
    for k, e in zip(keys, loo_errs):
        rows.append(f"<tr><td>Vgsq={k[0]:.1f} Vdsq={k[1]:.1f}</td><td>{e*100:.2f}%</td></tr>")
    summary_loo = "\n".join(rows)

    title = f"Hypernetwork LOO report ({architecture_name})" if architecture_name else "Hypernetwork LOO report"
    body = [
        f"<h1>{_html.escape(title)}</h1>",
        f"<p>epochs={epochs}, lr={lr}, architecture={_html.escape(architecture_name)}, "
        f"main-net params={n_params}</p>",
        "<div style='display:flex; gap:2rem; flex-wrap:wrap;'>",
        "<div><h3>Full-fit (in-sample)</h3><table class='summary'>"
        f"<tr><th>Quiescent point</th><th>rel.RMSE</th></tr>{summary_full}"
        f"<tr><th>mean</th><th>{np.mean(full_errs)*100:.2f}%</th></tr></table></div>",
        "<div><h3>Leave-one-out (held-out)</h3><table class='summary'>"
        f"<tr><th>Quiescent point</th><th>rel.RMSE</th></tr>{summary_loo}"
        f"<tr><th>mean</th><th>{np.mean(loo_errs)*100:.2f}%</th></tr></table></div>",
        "</div>",
    ]

    for k in keys:
        tag = qtag(*k)
        vgsq, vdsq = k
        body.append(f"<h2>Vgsq={vgsq:.1f}V  Vdsq={vdsq:.1f}V</h2>")
        body.append("<div class='plot-row'>")
        body.append(f"<figure><img src='plots/infit_{_html.escape(tag)}.png' alt='in-sample {tag}'>"
                    f"<figcaption>Full-fit, in-sample</figcaption></figure>")
        body.append(f"<figure><img src='plots/loo_{_html.escape(tag)}.png' alt='LOO {tag}'>"
                    f"<figcaption>Leave-one-out, held-out</figcaption></figure>")
        body.append("</div>")

    html_text = _PAGE_TEMPLATE.format(title=title, body="\n".join(body))
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return report_path
