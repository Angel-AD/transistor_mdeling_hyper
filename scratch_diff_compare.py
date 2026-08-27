"""One-off analysis: compare diff = Ids - f_base for the 2 problematic LOO points
(Vdsq=28, Vdsq=20) against the best-performing one (Vdsq=0), both as Ids-vs-Vgs (fixed Vds)
and Ids-vs-Vds (fixed Vgs) style curves, to understand WHY the extreme points fail."""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_all

VGS_SCALE = 4.0
VDS_SCALE = 45.0
BASE_KEY = (-2.9, 0.0)


class FBase(nn.Module):
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


def transfer_curves(df, n_vds_curves=4):
    groups = [g.sort_values("Vds") for _, g in df.groupby("TN")]
    vds_lo, vds_hi = df["Vds"].min(), df["Vds"].max()
    targets = np.linspace(vds_lo, vds_hi, n_vds_curves)
    curves = []
    for t_vds in targets:
        vgs_pts, ids_pts, vds_pts = [], [], []
        for g in groups:
            idx = (g["Vds"] - t_vds).abs().idxmin()
            vgs_pts.append(g.loc[idx, "Vgs"]); ids_pts.append(g.loc[idx, "Ids"]); vds_pts.append(g.loc[idx, "Vds"])
        vgs_pts, ids_pts, vds_pts = np.array(vgs_pts), np.array(ids_pts), np.array(vds_pts)
        order = np.argsort(vgs_pts)
        curves.append(dict(vgs=vgs_pts[order], ids=ids_pts[order], vds=vds_pts[order]))
    return curves


def output_curves(df, n_vgs_curves=5):
    groups = [g.sort_values("Vds") for _, g in df.groupby("TN")]
    groups = sorted(groups, key=lambda g: g["Vgs"].mean())
    n = len(groups)
    if n <= n_vgs_curves:
        picks = groups
    else:
        idxs = sorted(set(np.linspace(0, n - 1, n_vgs_curves).round().astype(int)))
        picks = [groups[i] for i in idxs]
    curves = []
    for g in picks:
        curves.append(dict(vgs=g["Vgs"].mean(), vds=g["Vds"].values, ids=g["Ids"].values))
    return curves


def main():
    data = load_all(r"C:\Users\acost\repos\csvs")
    f_base = FBase(hidden=8)
    f_base.load_state_dict(torch.load("results_basediff_arch37/models/f_base.pt"))
    f_base.eval()

    points = [
        ("PROBLEMATICO: Vdsq=28 (LOO 42-52%)", (-2.9, 28.0)),
        ("PROBLEMATICO: Vdsq=20 (LOO 26% en rand032)", (-2.5, 20.0)),
        ("MEJOR: Vdsq=0 (LOO 1.3-1.9%)", (-2.5, 0.0)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    cmap = plt.cm.viridis

    for col, (label, k) in enumerate(points):
        df = data[k]

        # Row 0: diff vs Vgs at fixed Vds (transfer curves)
        ax = axes[0, col]
        curves = transfer_curves(df)
        for i, c in enumerate(curves):
            color = cmap(i / max(len(curves) - 1, 1))
            vgs_t = torch.tensor(c["vgs"] / VGS_SCALE, dtype=torch.float32)
            vds_t = torch.tensor(c["vds"] / VDS_SCALE, dtype=torch.float32)
            with torch.no_grad():
                base_pred = f_base(vgs_t, vds_t).numpy()
            diff = c["ids"] - base_pred
            ax.plot(c["vgs"], diff, "o-", ms=3, color=color, label=f"Vds={c['vds'].mean():.1f}V")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{label}\ndiff vs Vgs (Vds fijo)")
        ax.set_xlabel("Vgs [V]"); ax.set_ylabel("diff [A]")
        ax.legend(fontsize=7)

        # Row 1: diff vs Vds at fixed Vgs (output curves)
        ax2 = axes[1, col]
        ocurves = output_curves(df)
        for i, c in enumerate(ocurves):
            color = cmap(i / max(len(ocurves) - 1, 1))
            vgs_t = torch.full_like(torch.tensor(c["vds"], dtype=torch.float32), c["vgs"] / VGS_SCALE)
            vds_t = torch.tensor(c["vds"] / VDS_SCALE, dtype=torch.float32)
            with torch.no_grad():
                base_pred = f_base(vgs_t, vds_t).numpy()
            diff = c["ids"] - base_pred
            ax2.plot(c["vds"], diff, "o-", ms=3, color=color, label=f"Vgs={c['vgs']:.2f}V")
        ax2.axhline(0, color="gray", lw=0.8, ls="--")
        ax2.set_title("diff vs Vds (Vgs fijo)")
        ax2.set_xlabel("Vds [V]"); ax2.set_ylabel("diff [A]")
        ax2.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig("scratch_diff_compare.png", dpi=120)
    print("Saved scratch_diff_compare.png")

    # quantitative summary
    print("\n--- resumen ---")
    for label, k in points:
        df = data[k]
        vgs_t = torch.tensor(df["Vgs"].values / VGS_SCALE, dtype=torch.float32)
        vds_t = torch.tensor(df["Vds"].values / VDS_SCALE, dtype=torch.float32)
        ids = df["Ids"].values
        with torch.no_grad():
            base_pred = f_base(vgs_t, vds_t).numpy()
        diff = ids - base_pred
        print(f"{label}: diff range=[{diff.min():.3f}, {diff.max():.3f}]  "
              f"|diff|max/|Ids|max={np.abs(diff).max()/np.abs(ids).max()*100:.1f}%  "
              f"std(diff)={diff.std():.3f}")


if __name__ == "__main__":
    main()
