"""Shared numerical-derivative helpers, used by both train_loo.py (gm1 loss target)
and plotting.py (gm1/gm2/gm3 diagnostic plots) -- kept in its own module so plotting.py
doesn't need to import train_loo.py (which itself imports plotting.py).

Three ways to turn a measured transfer curve (Ids vs Vgs) into a gm derivative are
offered, selected by train_loo.py's --gm_smoothing flag via `gm_derivative(...)`:

  "savgol"  -> smooth_derivative          (default; the base transistor_modeling port)
  "cascade" -> smooth_derivative_cascade  (lighter: median-3 despike + SG(5,3) per order;
                                           tracks the real gm2/gm3 peak far better, at the
                                           cost of a little more ripple in flat regions)
  "none"    -> raw_derivative             (plain iterated np.gradient, no filtering)
"""
import numpy as np
from scipy.signal import savgol_filter, medfilt

GM_SMOOTHING_METHODS = ("savgol", "cascade", "none")


def _odd_win(win, n, floor=3):
    """Largest odd window <= min(win, n), never below `floor`."""
    win = min(int(win), n if n % 2 == 1 else n - 1)
    if win % 2 == 0:
        win += 1
    return max(win, floor)


def smooth_derivative(x, y, order=1, win_pre=11, win_post=13, poly=2):
    """Verbatim port of the base transistor_modeling repo's per_neuron_plotting.py
    smooth_derivative: Savitzky-Golay pre-smooth of y, then np.gradient, then a second
    Savitzky-Golay pass on the derivative -- NOT a plain np.gradient. Windows are clamped
    (and forced odd) to fit however many points x/y actually have, since our transfer
    curves (one matched point per TN group, ~35 pts) are shorter than the dense sweeps
    the base repo differentiates."""
    n = len(x)
    win_pre = min(win_pre, n if n % 2 == 1 else n - 1)
    win_post = min(win_post, n if n % 2 == 1 else n - 1)
    if win_pre % 2 == 0:
        win_pre += 1
    if win_post % 2 == 0:
        win_post += 1
    poly_pre = min(poly, win_pre - 1)
    poly_post = min(poly, win_post - 1)

    y_s = savgol_filter(y, win_pre, poly_pre)
    for _ in range(order):
        y_s = np.gradient(y_s, x)
        y_s = savgol_filter(y_s, win_post, poly_post)
    return y_s


def smooth_derivative_cascade(x, y, order=1, win=5, med=3):
    """Lighter derivative cascade -- keeps the real gm2/gm3 peak amplitude and location
    where smooth_derivative flattens/shifts them (its poly=2 SG cannot carry 2nd/3rd
    derivative curvature, and its wide 11/13 windows average the peak down).

    Per derivative order: np.gradient -> width-`med` median filter (removes lone noise
    spikes, leaves a real 2-3 sample peak almost untouched) -> Savitzky-Golay(win, poly 3).
    Ids is pre-smoothed once with SG(win, poly 2). Narrow fixed windows (5 pts) -> the
    peaks survive; cost is a little extra ripple in the near-zero parts of the curve.

    No edge protection: the narrow median+SG(5) windows do not blow up at the deep-cutoff
    knee on our curves (verified on all 6 real CSVs), unlike wide-window / spline schemes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    w = _odd_win(win, n)
    poly_pre = min(2, w - 1)
    poly_post = min(3, w - 1)
    kmed = _odd_win(med, n)

    d = savgol_filter(y, w, poly_pre)
    for _ in range(order):
        d = np.gradient(d, x)
        d = medfilt(d, kmed)
        d = savgol_filter(d, w, poly_post)
    return d


def raw_derivative(x, y, order=1):
    """Plain iterated central difference, no filtering ("none")."""
    x = np.asarray(x, dtype=float)
    d = np.asarray(y, dtype=float)
    for _ in range(order):
        d = np.gradient(d, x)
    return d


def gm_derivative(x, y, order=1, method="savgol"):
    """Dispatch to the gm-derivative estimator named by train_loo.py's --gm_smoothing.
    See module docstring for the three methods."""
    if method == "savgol":
        return smooth_derivative(x, y, order=order)
    if method == "cascade":
        return smooth_derivative_cascade(x, y, order=order)
    if method == "none":
        return raw_derivative(x, y, order=order)
    raise ValueError(
        f"unknown gm_smoothing method {method!r}; choose from {GM_SMOOTHING_METHODS}")
