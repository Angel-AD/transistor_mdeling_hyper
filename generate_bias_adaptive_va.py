"""
Generates ONE Verilog-A file that is valid for ANY quiescent point, instead of the
static per-point .va convention (fixed baked-in coefficients per bias point, as in
all_vas_5334ca16/*.va). No coefficient here is tied to a specific (Vgsq, Vdsq):

  1. An RC low-pass network senses (Vgsq, Vdsq) directly from the fast-pulsed
     (Vgs, Vds) node voltages -- tau = R*C, chosen slower than the measurement
     pulse period so it tracks the DC bias and rejects the pulse ripple.
  2. H (the trained HyperNetwork, 2->32->32->87) is written out in full using its
     OWN trained, fixed weights (loaded from results_rand032/models/hyper_full.pt)
     -- these ~4000 numbers are the only baked constants in the file.
  3. H's output theta (87 runtime expressions, not constants) becomes f's weights,
     evaluated on the real fast (Vgs, Vds) to produce Ids.

Usage:
    python generate_bias_adaptive_va.py
"""
import torch

from model import ARCHITECTURES, layer_sizes_for, main_net_n_params

VGS_SCALE = 4.0
VDS_SCALE = 45.0
TAU = 10e-6  # 10us, per user request (period at 100kHz = 10us; ripple attenuation is limited -- see chat)
R_BIAS = 1e3
C_BIAS = TAU / R_BIAS

ARCH_NAME = "rand032"
ARCHITECTURE = ARCHITECTURES[ARCH_NAME]
LAYER_SIZES = layer_sizes_for(ARCHITECTURE)  # [2, 8, 3, 7, 1]
N_PARAMS = main_net_n_params(ARCHITECTURE)  # 87

MODULE_NAME = "hnn_bias_adaptive"


def fmt(x):
    return f"{x:+.8e}"


def act_expr(kind, x_expr):
    if kind == "tanh":
        return f"tanh({x_expr})"
    if kind == "swish":
        return f"(({x_expr})/(1.0+exp(-({x_expr}))))"
    raise ValueError(kind)


def build_H_lines(sd):
    """H: 2 -> 32 -> 32 -> 87, tanh hidden, linear output. Fixed trained weights."""
    W0 = sd["net.0.weight"].tolist()  # (32,2)
    b0 = sd["net.0.bias"].tolist()
    W2 = sd["net.2.weight"].tolist()  # (32,32)
    b2 = sd["net.2.bias"].tolist()
    W4 = sd["net.4.weight"].tolist()  # (87,32)
    b4 = sd["net.4.bias"].tolist()

    lines = []
    decls = []

    # layer 0: hq0_i = tanh(W0[i,0]*vgsq_n + W0[i,1]*vdsq_n + b0[i])
    n0 = len(W0)
    decls.append("real " + ", ".join(f"hq0_{i}" for i in range(n0)) + ";")
    lines.append("    // H layer 0 (%d neurons) -- fixed trained weights" % n0)
    for i in range(n0):
        expr = f"({fmt(W0[i][0])})*vgsq_n + ({fmt(W0[i][1])})*vdsq_n + ({fmt(b0[i])})"
        lines.append(f"    hq0_{i} = tanh({expr});")

    # layer 1: hq1_i = tanh(sum_j W2[i,j]*hq0_j + b2[i])
    n1 = len(W2)
    decls.append("real " + ", ".join(f"hq1_{i}" for i in range(n1)) + ";")
    lines.append("")
    lines.append("    // H layer 1 (%d neurons)" % n1)
    for i in range(n1):
        terms = " + ".join(f"({fmt(W2[i][j])})*hq0_{j}" for j in range(n0))
        lines.append(f"    hq1_{i} = tanh({terms} + ({fmt(b2[i])}));")

    # output: th_i = sum_j W4[i,j]*hq1_j + b4[i]   (theta, linear -- these become f's weights)
    n_out = len(W4)
    decls.append("real " + ", ".join(f"th_{i}" for i in range(n_out)) + ";")
    lines.append("")
    lines.append("    // H output layer: theta (%d numbers) -- NOT physical, these become f's weights below" % n_out)
    for i in range(n_out):
        terms = " + ".join(f"({fmt(W4[i][j])})*hq1_{j}" for j in range(n1))
        lines.append(f"    th_{i} = {terms} + ({fmt(b4[i])});")

    return decls, lines


def build_f_lines():
    """f: unflatten theta (th_0..th_86) into (W,b) per layer_sizes, apply architecture's
    per-neuron activations, output linear. Weights are RUNTIME expressions (th_*), not
    baked constants -- this is what makes the file valid for any (Vgsq, Vdsq)."""
    decls = []
    lines = []
    idx = 0
    x_names = ["Vgs_n", "Vds_n"]

    for li in range(len(LAYER_SIZES) - 1):
        fan_in, fan_out = LAYER_SIZES[li], LAYER_SIZES[li + 1]
        is_last = (li == len(LAYER_SIZES) - 2)
        out_names = ["Ids"] if is_last else [f"hf{li}_{i}" for i in range(fan_out)]
        if not is_last:
            decls.append("real " + ", ".join(out_names) + ";")
        lines.append("")
        lines.append(f"    // f layer {li}: fan_in={fan_in} fan_out={fan_out}"
                      + ("  (output, linear)" if is_last else ""))
        # unflatten_theta lays out theta as [ALL of W, row-major fan_out x fan_in] then
        # [ALL of b, fan_out] -- NOT interleaved per neuron. Match that exactly here.
        w_base = idx
        b_base = idx + fan_out * fan_in
        idx = b_base + fan_out
        for i in range(fan_out):
            w_terms = [f"th_{w_base + i*fan_in + j}*{x_names[j]}" for j in range(fan_in)]
            bias_term = f"th_{b_base + i}"
            pre_act = " + ".join(w_terms) + f" + {bias_term}"
            if is_last:
                lines.append(f"    Ids = {pre_act};")
            else:
                act = ARCHITECTURE[li][i]
                lines.append(f"    hf{li}_{i} = {act_expr(act, pre_act)};")
        x_names = out_names

    assert idx == N_PARAMS, f"consumed {idx} of {N_PARAMS} theta entries"
    return decls, lines


def main():
    sd = torch.load("results_rand032/models/hyper_full.pt", map_location="cpu")
    h_decls, h_lines = build_H_lines(sd)
    f_decls, f_lines = build_f_lines()

    all_decls = h_decls + f_decls
    n_H_params = sum(t.numel() for t in sd.values())

    src = f'''// Verilog-A GaN HEMT NN model -- BIAS-ADAPTIVE (single file, any quiescent point)
// Architecture "{ARCH_NAME}": f = {"->".join(str(s) for s in LAYER_SIZES)}, H = 2->32->32->{N_PARAMS}
// H's {n_H_params} weights are the ONLY baked constants (trained once, results_rand032/models/hyper_full.pt).
// The quiescent point (Vgsq, Vdsq) is NOT an input pin and NOT baked in -- it is sensed
// on-line from the (Vgs, Vds) node voltages via an internal RC low-pass (tau = R_bias*C_bias),
// then fed through H to generate theta = f's weights for whatever bias point the circuit
// is actually sitting at. f then evaluates on the real (fast) Vgs, Vds using that theta.
//
// tau = {TAU*1e6:.1f} us for a {1/TAU/1000:.0f}kHz-ish pulse rate (R_bias={R_BIAS:g}, C_bias={C_BIAS:g}).
// NOTE: tau == the pulse period here, not >>period -- pulse ripple is only mildly
// attenuated (~9dB) in the sensed bias point at this tau (chosen deliberately; see chat).

`include "disciplines.vams"
`include "constants.vams"

module {MODULE_NAME}(d, g, s);

  inout d, g, s;
  electrical d, g, s;
  electrical vgsq_node, vdsq_node;

  parameter real R_bias = {R_BIAS:g};
  parameter real C_bias = {C_BIAS:g};   // tau = R_bias*C_bias = {TAU:g} s

  real Vgs, Vds, Ids;
  real vgsq_n, vdsq_n;   // sensed quiescent point, normalized (/{VGS_SCALE:g}V, /{VDS_SCALE:g}V)
  real Vgs_n, Vds_n;     // fast signal, normalized

  {chr(10).join(f"  {d}" for d in all_decls)}

  analog begin

    Vgs = V(g,s);
    Vds = V(d,s);

    // --- RC bias sensing: vgsq_node/vdsq_node track the slow/DC part of Vgs/Vds ---
    I(vgsq_node, s) <+ C_bias*ddt(V(vgsq_node, s)) + (V(vgsq_node, s) - Vgs)/R_bias;
    I(vdsq_node, s) <+ C_bias*ddt(V(vdsq_node, s)) + (V(vdsq_node, s) - Vds)/R_bias;

    vgsq_n = V(vgsq_node, s) / {VGS_SCALE:g};
    vdsq_n = V(vdsq_node, s) / {VDS_SCALE:g};
    Vgs_n  = Vgs / {VGS_SCALE:g};
    Vds_n  = Vds / {VDS_SCALE:g};

{chr(10).join(h_lines)}

{chr(10).join(f_lines)}

    I(d,s) <+ Ids;
    I(g,s) <+ 0.0;

  end

endmodule
'''

    out_path = "results_rand032/hnn_bias_adaptive.va"
    with open(out_path, "w") as fh:
        fh.write(src)
    print(f"wrote {out_path}")
    print(f"H terms (fixed, baked): {n_H_params}")
    print(f"f terms (runtime, = theta): {N_PARAMS}")
    print(f"tau = R_bias*C_bias = {R_BIAS*C_BIAS:g} s")


if __name__ == "__main__":
    main()
