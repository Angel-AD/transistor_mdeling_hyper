"""Shared numerical-derivative helper, used by both train_loo.py (gm1 loss target)
and plotting.py (gm1/gm2/gm3 diagnostic plots) -- kept in its own module so plotting.py
doesn't need to import train_loo.py (which itself imports plotting.py)."""
import numpy as np
from scipy.signal import savgol_filter


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
