"""
Loads the 6 real pulsed-IV Auriga CSVs (one per quiescent bias point) into a single
table with columns: Vgsq, Vdsq, Vgs, Vds, Ids. Each row is a real measured point
(pulsed Vgs/Vds around that file's quiescent bias), matching exactly what the
per-quiescent-point .va NN models were trained on -- so the hypernetwork trains
directly on real measurements, not on a synthetic grid re-derived from a trained NN.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

CSV_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "csvs_link")

AURIGA_COLS = ["Item", "TN", "PT", "VoutQ", "IoutQ", "Vds", "Ids", "VinQ", "IinQ", "Vgs", "Igs"]


def parse_qpoint(fname):
    """cg2h40010_new_2.2_5_2_70W_center9.csv -> (Vgsq=-2.2, Vdsq=5.0)"""
    m = re.search(r"new_(\d+)\.(\d+)_(\d+)_", os.path.basename(fname))
    vgsq = -float(f"{m.group(1)}.{m.group(2)}")
    vdsq = float(m.group(3))
    return vgsq, vdsq


def load_one_csv(path):
    df = pd.read_csv(path, skiprows=106, header=None)
    df.columns = AURIGA_COLS
    out = df[["Vgs", "Vds", "Ids", "TN"]].astype(float).reset_index(drop=True)
    out["TN"] = out["TN"].astype(int)  # TN groups one pulsed sweep (~fixed Vgs, Vds ramps) -- used for IV-curve plots
    return out


def load_all(csv_dir):
    """Returns a dict: (Vgsq, Vdsq) -> DataFrame[Vgs, Vds, Ids]."""
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no CSVs found in {csv_dir}")
    out = {}
    for f in files:
        vgsq, vdsq = parse_qpoint(f)
        out[(vgsq, vdsq)] = load_one_csv(f)
    return out


if __name__ == "__main__":
    import sys
    d = load_all(sys.argv[1] if len(sys.argv) > 1 else CSV_DIR_DEFAULT)
    for k, v in d.items():
        print(k, len(v), "rows  Vgs[", v.Vgs.min(), v.Vgs.max(), "] Vds[", v.Vds.min(), v.Vds.max(),
              "] Ids[", v.Ids.min(), v.Ids.max(), "]")
