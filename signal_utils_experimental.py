"""EXPERIMENTAL copy of signal_utils.smooth_derivative -- NOT wired into training.

signal_utils.smooth_derivative (the one train_loo.py uses for the gm1 loss target) is
left completely untouched. This module keeps a verbatim copy of it plus improved
smoothers that track the real *shape* of gm1/gm2/gm3 on our short (~35-point),
non-uniformly-spaced transfer curves:

    smooth_derivative_v2  : PCHIP-uniform-grid + analytic SG derivative. Smoothest in
                            the flat parts, but still attenuates the gm2/gm3 peak
                            ~20% and can shift it ~0.3 V.
    smooth_derivative_v3  : minimal-bias derivative cascade (recommended). Keeps ~92%
                            of the gm2 peak and ~88% of the gm3 dip with ZERO peak
                            shift; costs a little extra ripple in near-zero regions.
    smooth_derivative_spline : quintic smoothing spline; good peak amplitude, but more
                            end-overshoot and a ~0.1 V shift.

Extrema-conservation medians over the 6 real CSVs (ratio = |smoothed| / |reference|,
reference = the near-raw cascade; see scratchpad/check_gm_extrema.py):

    gm         original      v2       v3    spline
    gm1 peak     1.02       1.02     1.00     1.01     (all fine; gm1 is not the problem)
    gm2 +peak    0.84       0.79     0.92     0.89
    gm3 +lobe   -0.06      -0.01     0.43     0.57     (near noise floor -- unreliable)
    gm3 -lobe    0.39       0.75     0.88     0.67
  peak shift dVgs: original/v2 move gm2&gm3 ~0.3 V toward threshold; v3 ~0 V.

Why the original loses the shape past gm1
-----------------------------------------
* poly=2 everywhere: a local quadratic has zero 3rd derivative, so every SG pass after
  the first actively flattens gm2/gm3 curvature.
* np.gradient is a 2-point central difference; iterating it 3x compounds its bias and
  amplifies noise (~1/dx per pass).
* SG assumes uniform sampling, but our Vgs points (nearest real point per TN group) are
  not uniform, so the effective window wobbles along the curve.

What smooth_derivative_v2 does
------------------------------
1. sort + de-duplicate Vgs.
2. PCHIP-resample onto a dense *uniform* Vgs grid (shape-preserving, no overshoot).
3. Savitzky-Golay with deriv=order, delta=dx, poly>=3: one polynomial fit per window
   gives the k-th derivative analytically -- no iterated np.gradient, no poly=2 ceiling.
4. clip to a robust band taken from a median-filtered raw finite difference, so ringing
   spikes near the deep-cutoff knee can't escape the physically observed range.
5. taper the outermost few points into the ORIGINAL smoother (which is stable at the
   ends) so v2 is never worse than the training filter at the boundaries.

Same signature as the original:  smooth_derivative_v2(x, y, order=1|2|3) -> ndarray on x.
"""
import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy.interpolate import PchipInterpolator, UnivariateSpline


# --------------------------------------------------------------------------------------
# verbatim copy of the production smoother (keep in sync with signal_utils.py by hand)
# --------------------------------------------------------------------------------------
def smooth_derivative_original(x, y, order=1, win_pre=11, win_post=13, poly=2):
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


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _sorted_unique(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    xu, inv = np.unique(x, return_inverse=True)
    if len(xu) == len(x):
        return x, y
    yu = np.zeros_like(xu)
    np.add.at(yu, inv, y)
    yu /= np.bincount(inv)
    return xu, yu


def _odd(w, hi):
    w = int(round(w))
    w = min(w, hi if hi % 2 == 1 else hi - 1)
    if w % 2 == 0:
        w += 1
    return max(w, 3)


def _raw_deriv(x, y, order):
    d = np.asarray(y, float)
    for _ in range(order):
        d = np.gradient(d, x)
    return d


# --------------------------------------------------------------------------------------
# improved smoother
# --------------------------------------------------------------------------------------
def smooth_derivative_v2(x, y, order=1,
                         win_frac=None, poly=3, oversample=8,
                         pre_win=7, clip_pad_frac=0.6,
                         edge_taper=6):
    """Shape-tracking replacement for signal_utils.smooth_derivative. See module docstring.

    win_frac     : SG window as a fraction of the Vgs span (physical, curve-independent).
                   None -> order-dependent default (0.42 / 0.30 / 0.30 for gm1/gm2/gm3):
                   a wider window for gm1 so its near-zero (deep-cutoff / low-Vds) region
                   is not fitted to noise, tighter for gm2/gm3 so the peak/trough survive.
    poly         : SG polynomial order (>=3 so gm2/gm3 curvature survives).
    oversample   : uniform-grid density = oversample * n_points.
    pre_win      : light SG pre-smooth of Ids before resampling (noise knock-down).
    clip_pad_frac: only gross blow-ups are clipped -- the allowed band is
                   [min, max] of a median-filtered raw finite difference, widened by
                   clip_pad_frac * (max - min) on each side.
    edge_taper   : blend this many end points on each side into smooth_derivative_original.
    """
    if win_frac is None:
        win_frac = {1: 0.42, 2: 0.30, 3: 0.30}.get(order, 0.30)

    x_in = np.asarray(x, float)
    xs, ys = _sorted_unique(x_in, y)
    n = len(xs)
    if n < 5:
        return np.zeros_like(x_in)

    # 1. light pre-smooth of Ids
    wpre = _odd(pre_win, n)
    ys_pre = savgol_filter(ys, wpre, min(2, wpre - 1))

    # 2. uniform grid via shape-preserving interpolation
    m = max(int(oversample * n), 128)
    xu = np.linspace(xs[0], xs[-1], m)
    yu = PchipInterpolator(xs, ys_pre)(xu)
    dx = xu[1] - xu[0]

    # 3. analytic SG derivative on the uniform grid
    win = _odd(win_frac * m, m)
    p = min(poly, win - 1)
    du = savgol_filter(yu, win, p, deriv=order, delta=dx, mode="interp")
    d_on_xs = np.interp(xs, xu, du)

    # 4. clip only gross blow-ups: band = [min,max] of median-filtered raw diff, padded
    raw = _raw_deriv(xs, ys_pre, order)
    raw_med = medfilt(raw, _odd(5, n))
    lo, hi = float(np.min(raw_med)), float(np.max(raw_med))
    pad = max(hi - lo, 1e-12) * clip_pad_frac
    d_on_xs = np.clip(d_on_xs, lo - pad, hi + pad)

    # 5. taper the ends into the stable original smoother
    if edge_taper and n > 2 * edge_taper + 2:
        d_orig = smooth_derivative_original(xs, ys, order=order)
        w = np.ones(n)
        ramp = np.linspace(0.0, 1.0, edge_taper + 2)[1:-1]
        w[:edge_taper] = ramp
        w[-edge_taper:] = ramp[::-1]
        d_on_xs = w * d_on_xs + (1.0 - w) * d_orig

    return np.interp(x_in, xs, d_on_xs)


# --------------------------------------------------------------------------------------
# secondary candidate: quintic smoothing spline (interior only, edges tapered)
# --------------------------------------------------------------------------------------
def smooth_derivative_spline(x, y, order=1, k=5, s_factor=3.0,
                             pre_win=7, edge_taper=5):
    x_in = np.asarray(x, float)
    xs, ys = _sorted_unique(x_in, y)
    n = len(xs)
    kk = min(k, n - 1)
    if kk < 2:
        return np.zeros_like(x_in)

    wpre = _odd(pre_win, n)
    ys_pre = savgol_filter(ys, wpre, min(2, wpre - 1))
    resid = ys - ys_pre
    mad = np.median(np.abs(resid - np.median(resid)))
    sigma = max(mad / 0.6745 if mad > 0 else float(np.std(resid)), 1e-12)
    s = s_factor * n * sigma ** 2

    spl = UnivariateSpline(xs, ys_pre, k=kk, s=s)
    d = spl.derivative(min(order, kk))(xs)

    if edge_taper and n > 2 * edge_taper + 2:
        d_orig = smooth_derivative_original(xs, ys, order=order)
        w = np.ones(n)
        ramp = np.linspace(0.0, 1.0, edge_taper + 2)[1:-1]
        w[:edge_taper] = ramp
        w[-edge_taper:] = ramp[::-1]
        d = w * d + (1.0 - w) * d_orig

    return np.interp(x_in, xs, d)


# --------------------------------------------------------------------------------------
# v3: minimal-bias derivative cascade -- keeps gm2/gm3 peak amplitude, spike-guarded
# --------------------------------------------------------------------------------------
def smooth_derivative_v3(x, y, order=1, pre_win=5, sg_win=7, sg_poly=3,
                         med=3, edge_taper=6, clip_pad_frac=0.4):
    """Same 3-stage shape as the original (pre-smooth Ids / differentiate / smooth the
    derivative) but tuned so the gm2/gm3 peaks are NOT flattened:

    * one SG stage per derivative order (like the original), each pass
        np.gradient  ->  width-`med` median filter (kills lone spikes, keeps a real
        2-3 sample peak)  ->  SG(sg_win, sg_poly>=3).
    * sg_poly>=3 so a local cubic can carry gm2/gm3 curvature (the original's poly=2
      cannot -- that is what shrinks the peaks).
    * narrow, fixed windows (7 pts) instead of 11/13 -> far less amplitude loss.
    * edge taper into the original + gross-blow-up clip, same guard rails as v2.

    This tracks the raw data more tightly than v2 at the cost of a little more ripple
    in the flat (near-zero) parts of the curve.
    """
    x_in = np.asarray(x, float)
    xs, ys = _sorted_unique(x_in, y)
    n = len(xs)
    if n < 6:
        return np.zeros_like(x_in)

    d = savgol_filter(ys, _odd(pre_win, n), min(2, _odd(pre_win, n) - 1))
    w = _odd(sg_win, n)
    p = min(sg_poly, w - 1)
    for _ in range(order):
        d = np.gradient(d, xs)
        d = medfilt(d, min(med if med % 2 else med + 1, _odd(n, n)))
        d = savgol_filter(d, w, p)

    # gross-blow-up clip from the median-filtered raw finite difference
    raw = _raw_deriv(xs, savgol_filter(ys, _odd(pre_win, n), 2), order)
    raw_med = medfilt(raw, _odd(5, n))
    lo, hi = float(np.min(raw_med)), float(np.max(raw_med))
    pad = max(hi - lo, 1e-12) * clip_pad_frac
    d = np.clip(d, lo - pad, hi + pad)

    if edge_taper and n > 2 * edge_taper + 2:
        d_orig = smooth_derivative_original(xs, ys, order=order)
        wt = np.ones(n)
        ramp = np.linspace(0.0, 1.0, edge_taper + 2)[1:-1]
        wt[:edge_taper] = ramp
        wt[-edge_taper:] = ramp[::-1]
        d = wt * d + (1.0 - wt) * d_orig

    return np.interp(x_in, xs, d)


SMOOTHERS = {
    "original": smooth_derivative_original,
    "v2": smooth_derivative_v2,
    "v3": smooth_derivative_v3,
    "spline_k5": smooth_derivative_spline,
}
