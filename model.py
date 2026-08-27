"""
Hypernetwork H(Vgsq, Vdsq) -> theta (all weights/biases of a small MLP main-net),
main-net f(Vgs, Vds; theta) -> Ids. The main-net has NO parameters of its own --
every number it uses comes from H's output. Both are trained jointly, end-to-end,
on all quiescent points at once (or on a leave-one-out subset), so there is a
single optimization and therefore no neuron-permutation ambiguity between points
(unlike interpolating weights of 6 separately-trained networks).

The main-net's architecture is a list of per-layer activation lists (one entry per
hidden neuron in that layer), e.g. [["tanh","swish"], ["swish","tanh"]] -- the same
per-neuron mixed-activation format used by the base transistor_modeling repo's
PerNeuronLinear/DynamicNN for its individually-trained per-quiescent-point .va models.
The output layer is always linear (no activation).
"""
import torch
import torch.nn as nn

from physics_equations import physics_n_params, physics_forward, PARAM_ORDER as PHYSICS_PARAM_ORDER

N_IN = 2
N_OUT = 1

ACTIVATION_FNS = {
    "tanh": torch.tanh,
    "swish": lambda x: x * torch.sigmoid(x),
    "linear": lambda x: x,
}

# Two per-neuron mixed-activation architectures found (in the base transistor_modeling
# repo) to fit each quiescent point's own individually-trained .va model well -- 37 and
# 89 params respectively, matching the exact per-.va parameter counts this project's
# docstrings reference.
ARCHITECTURES = {
    "arch37": [["swish", "tanh", "tanh", "swish", "tanh"], ["swish", "swish", "swish"]],
    "arch89": [["tanh", "swish"],
               ["swish", "tanh", "swish", "swish", "tanh", "swish"],
               ["tanh", "tanh", "tanh", "swish", "swish", "tanh", "swish", "swish"]],
    # All-tanh variants at the SAME layer sizes as arch37/arch89 (isolates whether the
    # non-monotonic cutoff-region wiggle seen with arch37/arch89 comes from swish itself).
    "tanh37": [["tanh"] * 5, ["tanh"] * 3],
    "tanh89": [["tanh"] * 2, ["tanh"] * 6, ["tanh"] * 8],
    # Smaller than either -- tests whether shrinking further (fewer params for H to
    # generate) keeps improving LOO generalization the way arch37 (37) beat arch89 (89).
    "tanh20": [["tanh"] * 3, ["tanh"] * 2],
    # Original uniform-tanh baseline (2->8->8->1, 105 params), re-added here so it can be
    # run through the exact same code path as the others for an apples-to-apples comparison.
    "tanh105": [["tanh"] * 8, ["tanh"] * 8],
    # Winner of architecture_search.py's 100-candidate random search (rand032): confirmed
    # best LOO (2.33% with L-BFGS, seed=27) among everything tried so far.
    "rand032": [["tanh", "swish", "tanh", "tanh", "swish", "swish", "swish", "tanh"],
                ["tanh", "tanh", "swish"],
                ["tanh", "swish", "tanh", "swish", "tanh", "tanh", "swish"]],
    # Angelov-family compact-model equations (physics_equations.py) as f, instead of a
    # generic MLP -- H is unchanged (still 2 inputs), theta's entries are named physical
    # parameters (Ipk, Vpk, alpha, ...) plugged into a closed-form equation rather than
    # NN weight matrices. See main_net_n_params/main_net_forward's "physics:" dispatch.
    "modern_angelov": "physics:modern_angelov",
    "mod1_angelov": "physics:mod1_angelov",
    "mod3_angelov": "physics:mod3_angelov",
    "dual_knee_angelov": "physics:dual_knee_angelov",
    "bai_kink_angelov": "physics:bai_kink_angelov",
}


def layer_sizes_for(architecture, n_in=N_IN, n_out=N_OUT):
    return [n_in] + [len(layer) for layer in architecture] + [n_out]


def main_net_n_params(architecture, n_in=N_IN, n_out=N_OUT):
    if isinstance(architecture, str) and architecture.startswith("physics:"):
        return physics_n_params(architecture.split(":", 1)[1])
    sizes = layer_sizes_for(architecture, n_in, n_out)
    n = 0
    for i in range(len(sizes) - 1):
        n += sizes[i + 1] * sizes[i] + sizes[i + 1]  # weights + biases
    return n


def unflatten_theta(theta, architecture, n_in=N_IN, n_out=N_OUT):
    """theta: (n_params,) 1D tensor -> list of (W, b) per layer."""
    sizes = layer_sizes_for(architecture, n_in, n_out)
    idx = 0
    layers = []
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        w_n = fan_out * fan_in
        W = theta[idx:idx + w_n].reshape(fan_out, fan_in)
        idx += w_n
        b = theta[idx:idx + fan_out]
        idx += fan_out
        layers.append((W, b))
    assert idx == theta.shape[0], f"unflatten mismatch: used {idx} of {theta.shape[0]}"
    return layers


def _apply_layer_activation(x, act_list):
    """x: (batch, n_neurons). Per-neuron activation, matching act_list column-by-column
    (mirrors PerNeuronLinear's fast path for homogeneous layers / safe path for mixed)."""
    if len(set(act_list)) == 1:
        return ACTIVATION_FNS[act_list[0]](x)
    return torch.cat([ACTIVATION_FNS[a](x[:, i:i + 1]) for i, a in enumerate(act_list)], dim=1)


def main_net_forward(theta, Vgs, Vds, architecture):
    """Vgs, Vds: (batch,) tensors (already normalized). theta: (n_params,) for
    ONE quiescent point, broadcast across the whole batch. Hidden layers use
    `architecture`'s per-neuron activations; the output layer is always linear.
    Returns Ids: (batch,)."""
    if isinstance(architecture, str) and architecture.startswith("physics:"):
        return physics_forward(theta, Vgs, Vds, architecture.split(":", 1)[1])
    x = torch.stack([Vgs, Vds], dim=-1)  # (batch, 2)
    layers = unflatten_theta(theta, architecture)
    for li, (W, b) in enumerate(layers[:-1]):
        x = torch.nn.functional.linear(x, W, b)
        x = _apply_layer_activation(x, architecture[li])
    W, b = layers[-1]
    x = torch.nn.functional.linear(x, W, b)  # output layer: linear
    return x.squeeze(-1)  # (batch,)


class HyperNetwork(nn.Module):
    """H(Vgsq_norm, Vdsq_norm) -> theta (n_params,) for the main net."""

    def __init__(self, n_params, hidden=32, n_hidden_layers=2, n_in=2):
        super().__init__()
        layers = [nn.Linear(n_in, hidden), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, n_params)]
        self.net = nn.Sequential(*layers)
        # small init on the last layer so the main-net starts near-linear/stable
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.mul_(0.1)

    def forward(self, qpoint):
        """qpoint: (n_in,) tensor, e.g. [Vgsq_norm, Vdsq_norm] or that plus extra
        physics features -> (n_params,)"""
        return self.net(qpoint)
