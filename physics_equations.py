"""
Angelov-family compact-model equations as an alternative main-net `f`, instead of a
generic MLP. H is UNCHANGED here -- it still takes only (Vgsq, Vdsq) (2 inputs) and
outputs theta; the only difference is what theta MEANS: instead of NN weight matrices,
each entry of theta is one NAMED PHYSICAL PARAMETER (Ipk, Vpk, alpha, ...) plugged
straight into a closed-form Angelov equation. This is a different way of shrinking
f (and therefore H's output layer, which scales with n_params(f)) than a smaller MLP --
and unlike feeding physics features INTO H (tried and confirmed worse in every variant,
see train_loo.py's --h_physics), this keeps H's input at 2 -- no extra input dimensions
for it to overfit on with only 5-6 training curves.

Equations transcribed from the base transistor_modeling repo's
optim_utils/per_neuron_noNN.py (classic_angelov, modern_angelov, mod1_angelov), with
Vgs/Vds passed directly (not stacked as X columns) since main_net_forward already has
them separately. A handful of parameters that must stay positive for the equation to
make physical sense (peak current, saturation steepness, the mod1 smoothness exponent)
are passed through softplus -- the base repo's per_neuron_noNN.py enforced this via
box-constrained (SLSQP) optimization instead, which isn't available here since theta
comes unconstrained out of a plain Linear layer.
"""
import torch
import torch.nn.functional as F

VGS_SCALE = 4.0
VDS_SCALE = 45.0


def modern_angelov(Vgs, Vds, Ipk, Vpk, P1, P2, P3, P4, alpha, alpha_s, lambda_):
    Ipk = F.softplus(Ipk)
    alpha = F.softplus(alpha)
    dv = Vgs - Vpk
    psi = P1 * dv + P2 * dv**2 + P3 * dv**3 + P4 * dv**4
    alpha_eff = alpha + alpha_s * (1 + torch.tanh(psi))
    return Ipk * (1 + torch.tanh(psi)) * torch.tanh(alpha_eff * Vds) * (1 + lambda_ * Vds)


def mod1_angelov(Vgs, Vds, Ipk, Ipk1, P1, P21, P22, P31, P32, Vpks, deltVpks,
                  alphaR, alphaS, lambda_, n, Vgsf, deltP1, deltP21, deltP22, deltP31, deltP32):
    Ipk = F.softplus(Ipk)
    Ipk1 = F.softplus(Ipk1)
    alphaR = F.softplus(alphaR)
    n = F.softplus(n) + 0.05  # keep away from 0 (divides Vgspa below)

    Vpk = Vpks - deltVpks + deltVpks * torch.tanh(alphaS * Vds)
    Vgsp = Vgs - Vpk
    x = n * Vgsp
    Vgspa = (1.0 / n) * (torch.logaddexp(x, -x) - torch.log(torch.tensor(2.0, device=Vgs.device)))

    Veffp1 = 0.5 * (Vgsp - Vgspa)
    Veffp2 = 0.5 * (Vgsf + Vgspa)

    common = 1 + torch.tanh(alphaS * Vds)
    P1m = P1 * (1 + deltP1) * common
    P21m = P21 * (1 + deltP21) * common
    P22m = P22 * (1 + deltP22) * common
    P31m = P31 * (1 + deltP31) * common
    P32m = P32 * (1 + deltP32) * common

    P111 = P1 * Ipk / (Ipk1 + 1e-6)

    ph1 = P1m * Veffp1 + P21m * Veffp1**2 + P31m * Veffp1**3
    ph2 = P111 * Veffp2 + P22m * Veffp2**2 + P32m * Veffp2**3

    Ids1 = Ipk * (1 + torch.tanh(ph1)) + Ipk1 * torch.tanh(ph2)
    alpha_eff = alphaR + alphaS * (1 + torch.tanh(ph1))
    Ids2 = torch.tanh(alpha_eff * Vds) * (1 + lambda_ * Vds)
    return Ids1 * Ids2


def mod3_angelov(Vgs, Vds, I0, I1, I2, I3, Mpk0, MpkA, Vgm0, Vpk, P1m, P2, P3, Pz0, Pz1, alphaZ, alpha):
    """Ipk is a POLYNOMIAL in Vds (I0 + I1*Vds + I2*Vds^2 + I3*Vds^3) instead of a
    constant -- with the right signs on I2/I3 this alone can rise, overshoot, and settle,
    which is the knee/kink shape a single tanh(alpha*Vds) term (mod1/modern) can't produce."""
    I0 = F.softplus(I0)
    Mpk0 = F.softplus(Mpk0)
    alphaZ = F.softplus(alphaZ)
    alpha = F.softplus(alpha)

    Ipk = I0 + I1 * Vds + I2 * Vds**2 + I3 * Vds**3
    Zm = (Pz0 + Pz1 * Vds) * torch.tanh(alphaZ * Vds) + Pz0
    PhiM = Zm * (Vgs - Vgm0)
    Mpk = Mpk0 + MpkA * torch.tanh(PhiM)
    Vgsp = Vgs - Vpk
    PhiP = P1m * Vgsp + P2 * Vgsp**2 + P3 * Vgsp**3

    Ids1 = Ipk * (1 + Mpk * torch.tanh(PhiP))
    Ids2 = torch.tanh(alpha * Vds)
    return Ids1 * Ids2


def dual_knee_angelov(Vgs, Vds, Ipk, Vpk, P1, P2, P3, alpha_sharp_raw, alpha_soft, lambda_, mix_logit):
    """classic_angelov's Vgs-gate (Ipk, Vpk, P1-P3) unchanged; the Vds-knee term is
    replaced by a MIX of two tanh knees -- one deliberately sharper (steeper initial
    rise) than the other (broader, slower saturation) -- so the blend can reproduce a
    fast initial rise that then eases into a different saturation slope, instead of one
    single smooth tanh S-curve. alpha_sharp is defined as alpha_soft + softplus(raw) so
    it's always >= alpha_soft by construction (keeps the two knees from swapping
    identities during training, which would make the fit unidentifiable)."""
    Ipk = F.softplus(Ipk)
    alpha_soft = F.softplus(alpha_soft)
    alpha_sharp = alpha_soft + F.softplus(alpha_sharp_raw)
    mix = torch.sigmoid(mix_logit)

    dv = Vgs - Vpk
    psi = P1 * dv + P2 * dv**2 + P3 * dv**3
    gate = Ipk * (1 + torch.tanh(psi))

    knee_sharp = torch.tanh(alpha_sharp * Vds)
    knee_soft = torch.tanh(alpha_soft * Vds) * (1 + lambda_ * Vds)
    drain = mix * knee_sharp + (1 - mix) * knee_soft
    return gate * drain


def bai_kink_angelov(Vgs, Vds, a0, b0, c0,
                      Ipk1, Vpk1, P1_1, P2_1, P3_1, a1_1, b1_1, c1_1, a2_1, b2_1,
                      Ipk2, Vpk2, P1_2, P2_2, P3_2, a1_2, b1_2, c1_2, a2_2, b2_2):
    """Adapted from Bai, Zhang & Gao (2024) -- a GENUINELY piecewise kink model (Vds <=
    Vkink(Vgs) vs Vds > Vkink(Vgs), literal torch.where switch, not a soft blend), which
    that paper's own comparison (see kink docx) found beats every smooth/continuous
    'effective voltage' formulation (Jarndal, Somerville, Siligaris) by ~2-4x lower error.
    Each region has its OWN full Angelov-style gate (Ipk,Vpk,P1-P3) and drain term
    (Vsat(Vgs), kappa(Vgs)); Idso^comp's exact EEHEMT form wasn't reproducible from the
    source doc, so each region's gate reuses our own validated classic_angelov gate term
    instead of a different formula. Vkink is a Vgs-only boundary, so Ids1/Ids2 need only
    be evaluated where each is actually selected -- torch.where handles the split cleanly."""
    Ipk1 = F.softplus(Ipk1)
    Ipk2 = F.softplus(Ipk2)
    Vkink = a0 * Vgs**2 + b0 * Vgs + c0

    def branch(Ipk, Vpk, P1, P2, P3, a1, b1, c1, a2, b2):
        dv = Vgs - Vpk
        psi = P1 * dv + P2 * dv**2 + P3 * dv**3
        gate = Ipk * (1 + torch.tanh(psi))
        Vsat = F.softplus(a1 * Vgs**2 + b1 * Vgs + c1) + 0.5  # keep denominator away from 0
        kappa = a2 + b2 * Vgs
        return gate * (1 + kappa * Vds) * torch.tanh(3.0 * Vds / Vsat)

    Ids1 = branch(Ipk1, Vpk1, P1_1, P2_1, P3_1, a1_1, b1_1, c1_1, a2_1, b2_1)
    Ids2 = branch(Ipk2, Vpk2, P1_2, P2_2, P3_2, a1_2, b1_2, c1_2, a2_2, b2_2)
    return torch.where(Vds <= Vkink, Ids1, Ids2)


# Fixed parameter order per equation -- theta[i] is named PARAM_ORDER[eq][i], in this order.
PARAM_ORDER = {
    "modern_angelov": ["Ipk", "Vpk", "P1", "P2", "P3", "P4", "alpha", "alpha_s", "lambda_"],
    "mod1_angelov": ["Ipk", "Ipk1", "P1", "P21", "P22", "P31", "P32", "Vpks", "deltVpks",
                     "alphaR", "alphaS", "lambda_", "n", "Vgsf",
                     "deltP1", "deltP21", "deltP22", "deltP31", "deltP32"],
    "mod3_angelov": ["I0", "I1", "I2", "I3", "Mpk0", "MpkA", "Vgm0", "Vpk",
                      "P1m", "P2", "P3", "Pz0", "Pz1", "alphaZ", "alpha"],
    "dual_knee_angelov": ["Ipk", "Vpk", "P1", "P2", "P3", "alpha_sharp_raw", "alpha_soft",
                           "lambda_", "mix_logit"],
    "bai_kink_angelov": ["a0", "b0", "c0",
                          "Ipk1", "Vpk1", "P1_1", "P2_1", "P3_1", "a1_1", "b1_1", "c1_1", "a2_1", "b2_1",
                          "Ipk2", "Vpk2", "P1_2", "P2_2", "P3_2", "a1_2", "b1_2", "c1_2", "a2_2", "b2_2"],
}

EQUATIONS = {
    "modern_angelov": modern_angelov,
    "mod1_angelov": mod1_angelov,
    "mod3_angelov": mod3_angelov,
    "dual_knee_angelov": dual_knee_angelov,
    "bai_kink_angelov": bai_kink_angelov,
}


def physics_n_params(eq_name):
    return len(PARAM_ORDER[eq_name])


def physics_forward(theta, Vgs_norm, Vds_norm, eq_name):
    """Vgs_norm, Vds_norm: (batch,) tensors normalized the same way as the NN main-net's
    (/VGS_SCALE, /VDS_SCALE) -- un-normalized back to volts here, since the Angelov
    equations' calibration (Vpk range, alpha scale, ...) assumes real Vgs/Vds in volts."""
    Vgs = Vgs_norm * VGS_SCALE
    Vds = Vds_norm * VDS_SCALE
    names = PARAM_ORDER[eq_name]
    params = {name: theta[i] for i, name in enumerate(names)}
    return EQUATIONS[eq_name](Vgs, Vds, **params)
