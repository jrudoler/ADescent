"""
Does dropout make the per-layer kernel Φ more diagonal, and does that improve
r(ΔA, -∂ℒ/∂A) — the testable correlation from the paper?

Self-contained dropout-aware simulation with forward/backward passes and
Jacobian assembly. Normally run through Snakemake.

Outputs:
  - data/generated/dropout_results.json
  - results/figures/fig_dropout_sweep.{pdf,png}
  - results/figures/fig_dropout_phi_heatmaps.{pdf,png}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================== CONFIG ========================

DEPTH = 3
N_STEPS = 500
DIAG_EVERY = 50
SEEDS = [0, 1]
ETA = 0.005

# Conditions for the proof-of-concept sweep.
# Effective-width control: (w=48, p=0.5) and (w=24, p=0) have the same expected
# number of active hidden units per layer (~24); the comparison isolates the
# decorrelation effect of dropout from the smaller-layer effect.
CONDITIONS = [
    {"width": 16, "p": 0.0, "label": "w=16, p=0\n(small baseline)"},
    {"width": 24, "p": 0.0, "label": "w=24, p=0\n(eff-width ctrl for 48,0.5)"},
    {"width": 32, "p": 0.0, "label": "w=32, p=0\n(eff-width ctrl for 48,0.33)"},
    {"width": 48, "p": 0.0, "label": "w=48, p=0\n(full baseline)"},
    {"width": 48, "p": 0.33, "label": "w=48, p=0.33"},
    {"width": 48, "p": 0.5, "label": "w=48, p=0.5"},
]

DATA_OUTPUT = Path("dropout_results.json")
FIGURE_OUTPUT_DIR = Path(".")


# ======================== NETWORK ========================


def create_network(layer_sizes, rng):
    weight_matrices = []
    for layer_index in range(len(layer_sizes) - 1):
        scale = np.sqrt(2.0 / layer_sizes[layer_index])
        layer_weights = (
            rng.standard_normal(
                (layer_sizes[layer_index + 1], layer_sizes[layer_index] + 1)
            )
            * scale
        )
        layer_weights[:, -1] = 0
        weight_matrices.append(layer_weights)
    return weight_matrices


def sample_dropout_masks(layer_sizes, p, rng):
    """Bernoulli(1-p) masks for hidden layers only. Returns list of length
    num_layers+1 with None for input (index 0) and output (index num_layers)."""
    num_layers = len(layer_sizes) - 1
    masks = [None] * (num_layers + 1)
    if p <= 0.0:
        return masks
    for layer_index in range(1, num_layers):  # hidden layers only
        width = layer_sizes[layer_index]
        masks[layer_index] = (rng.random(width) >= p).astype(float)
    return masks


def forward(weight_matrices, input_vector, dropout_masks=None, p=0.0):
    """Forward pass with optional inverted dropout on hidden activations.

    activities_by_layer[ℓ] is the activity actually propagating to layer ℓ+1:
      ℓ=0: raw input
      ℓ hidden: relu(z^(ℓ)) ⊙ mask^(ℓ) / (1-p)
      ℓ=output: linear z^(L)
    """
    num_layers = len(weight_matrices)
    activities_by_layer = [input_vector.copy()]
    pre_activations_by_layer = [None]
    scale = 1.0 / (1.0 - p) if p > 0.0 else 1.0
    for layer_index in range(num_layers):
        augmented = np.append(activities_by_layer[-1], 1.0)
        pre_activations = weight_matrices[layer_index] @ augmented
        pre_activations_by_layer.append(pre_activations.copy())
        if layer_index < num_layers - 1:
            relu_out = np.maximum(0, pre_activations)
            if dropout_masks is not None and dropout_masks[layer_index + 1] is not None:
                masked = relu_out * dropout_masks[layer_index + 1] * scale
            else:
                masked = relu_out
            activities_by_layer.append(masked)
        else:
            activities_by_layer.append(pre_activations.copy())
    return activities_by_layer, pre_activations_by_layer


def backprop(
    weight_matrices,
    activities_by_layer,
    pre_activations_by_layer,
    target_vector,
    dropout_masks=None,
    p=0.0,
):
    """Backprop matching the dropout-aware forward.

    ∂L/∂A^(ℓ)_masked is what we store in loss_gradient_by_activity[ℓ].
    For hidden layers the chain to z^(ℓ) picks up an extra mask⊙scale factor:
      ∂L/∂z^(ℓ) = ∂L/∂A^(ℓ)_masked · mask^(ℓ)/(1-p) · 1{z^(ℓ)>0}
    """
    num_layers = len(weight_matrices)
    loss_gradient_by_activity = [None] * (num_layers + 1)
    loss_gradient_by_activity[num_layers] = (
        activities_by_layer[num_layers] - target_vector
    )
    loss_gradient_by_weights = [None] * num_layers
    scale = 1.0 / (1.0 - p) if p > 0.0 else 1.0

    for layer_index in range(num_layers - 1, -1, -1):
        if layer_index == num_layers - 1:
            local_gradient = loss_gradient_by_activity[layer_index + 1].copy()
        else:
            relu_deriv = (pre_activations_by_layer[layer_index + 1] > 0).astype(float)
            if (
                dropout_masks is not None
                and dropout_masks[layer_index + 1] is not None
            ):
                mask = dropout_masks[layer_index + 1] * scale
            else:
                mask = 1.0
            local_gradient = (
                loss_gradient_by_activity[layer_index + 1] * mask * relu_deriv
            )

        augmented = np.append(activities_by_layer[layer_index], 1.0)
        loss_gradient_by_weights[layer_index] = np.outer(local_gradient, augmented)
        if layer_index > 0:
            loss_gradient_by_activity[layer_index] = (
                weight_matrices[layer_index][:, :-1].T @ local_gradient
            )
    return loss_gradient_by_activity, loss_gradient_by_weights


def make_bar_images(n_per_class, rng):
    inputs, targets = [], []
    for _ in range(n_per_class):
        h = np.zeros(16)
        row = rng.integers(4)
        h[row * 4 : (row + 1) * 4] = 1.0
        h += rng.standard_normal(16) * 0.15
        inputs.append(h)
        targets.append([1, 0])
        v = np.zeros(16)
        col = rng.integers(4)
        for r in range(4):
            v[r * 4 + col] = 1.0
        v += rng.standard_normal(16) * 0.15
        inputs.append(v)
        targets.append([0, 1])
    return np.array(inputs), np.array(targets)


# ======================== JACOBIAN ========================


def compute_jacobian_and_predictions(
    layer_sizes,
    weight_matrices,
    activities_by_layer,
    pre_activations_by_layer,
    loss_gradient_by_weights,
    loss_gradient_by_activity,
    learning_rate,
    dropout_masks=None,
    p=0.0,
):
    """Build the explicit activity Jacobian and predictions, accounting for the
    dropout mask. When dropout_masks is None or all-ones, this matches
    gen_figures.compute_jacobian_and_predictions exactly.

    The activity-derivative at each hidden layer is
      mask^(ℓ)/(1-p) · 1{z^(ℓ)>0},
    so the Jacobian row for any unit i in a hidden layer ℓ picks up that factor.
    For dropped units (mask=0) the Jacobian row is all zero, so they cleanly
    drop out of every prediction.
    """
    num_layers = len(weight_matrices)
    neuron_counts = layer_sizes[1:]
    total_neurons = sum(neuron_counts)
    scale = 1.0 / (1.0 - p) if p > 0.0 else 1.0

    layer_offsets = []
    offset = 0
    for layer_index in range(num_layers):
        layer_offsets.append(offset)
        offset += layer_sizes[layer_index + 1]

    def hidden_factor(layer_index_one_based, neuron_index):
        """mask·scale·relu' for a hidden layer; 1 for the output layer."""
        if layer_index_one_based == num_layers:
            return 1.0
        pre = pre_activations_by_layer[layer_index_one_based][neuron_index]
        if pre <= 0:
            return 0.0
        if dropout_masks is not None and dropout_masks[layer_index_one_based] is not None:
            return dropout_masks[layer_index_one_based][neuron_index] * scale
        return 1.0

    backprop_activity_gradient = np.zeros(total_neurons)
    for layer_index in range(1, num_layers + 1):
        if loss_gradient_by_activity[layer_index] is not None:
            start = layer_offsets[layer_index - 1]
            backprop_activity_gradient[
                start : start + len(loss_gradient_by_activity[layer_index])
            ] = loss_gradient_by_activity[layer_index]

    total_params = sum(
        (layer_sizes[layer_index] + 1) * layer_sizes[layer_index + 1]
        for layer_index in range(num_layers)
    )

    flat_weight_update = np.zeros(total_params)
    param_offset = 0
    for layer_index in range(num_layers):
        size = loss_gradient_by_weights[layer_index].size
        flat_weight_update[param_offset : param_offset + size] = (
            -learning_rate * loss_gradient_by_weights[layer_index].ravel()
        )
        param_offset += size

    full_jacobian = np.zeros((total_neurons, total_params))
    param_offset = 0
    for source in range(num_layers):
        src_in = layer_sizes[source]
        src_out = layer_sizes[source + 1]
        n_src_params = src_out * (src_in + 1)
        augmented = np.append(activities_by_layer[source], 1.0)

        direct_block = np.zeros((src_out, n_src_params))
        for i in range(src_out):
            factor = hidden_factor(source + 1, i)
            ps = i * (src_in + 1)
            direct_block[i, ps : ps + (src_in + 1)] = factor * augmented

        start = layer_offsets[source]
        full_jacobian[
            start : start + src_out, param_offset : param_offset + n_src_params
        ] = direct_block

        propagated = direct_block
        for down in range(source + 1, num_layers):
            d_in = layer_sizes[down]
            d_out = layer_sizes[down + 1]
            inter = np.zeros((d_out, d_in))
            for i in range(d_out):
                factor = hidden_factor(down + 1, i)
                inter[i] = factor * weight_matrices[down][i, :-1]
            propagated = inter @ propagated
            d_start = layer_offsets[down]
            full_jacobian[
                d_start : d_start + d_out,
                param_offset : param_offset + n_src_params,
            ] = propagated

        param_offset += n_src_params

    exact_change = full_jacobian @ flat_weight_update

    phi_diagonal_per_neuron = np.sum(full_jacobian**2, axis=1)
    diagonal_prediction = (
        -learning_rate * phi_diagonal_per_neuron * backprop_activity_gradient
    )

    # Kernel prediction (Eq. 3): ΔA = -η Φ ∂ℒ/∂A computed per-layer.
    kernel_prediction = np.zeros(total_neurons)
    layerwise_phi = np.zeros((total_neurons, total_neurons))
    for layer_index in range(num_layers):
        start = layer_offsets[layer_index]
        width = layer_sizes[layer_index + 1]
        J_ell = full_jacobian[start : start + width, :]
        Phi_ell = J_ell @ J_ell.T
        layerwise_phi[start : start + width, start : start + width] = Phi_ell
        kernel_prediction[start : start + width] = (
            -learning_rate * Phi_ell @ backprop_activity_gradient[start : start + width]
        )

    # Active set: alive-and-not-dropped for hidden, always-on for output.
    active_mask = np.zeros(total_neurons)
    for layer_index in range(1, num_layers + 1):
        start = layer_offsets[layer_index - 1]
        width = layer_sizes[layer_index]
        if layer_index < num_layers:
            for i in range(width):
                alive = pre_activations_by_layer[layer_index][i] > 0
                dropped = (
                    dropout_masks is not None
                    and dropout_masks[layer_index] is not None
                    and dropout_masks[layer_index][i] == 0
                )
                active_mask[start + i] = 1.0 if (alive and not dropped) else 0.0
        else:
            active_mask[start : start + width] = 1.0

    raw_negative_gradient = -backprop_activity_gradient * active_mask

    return {
        "exact_change": exact_change,
        "kernel_prediction": kernel_prediction,
        "diagonal_prediction": diagonal_prediction,
        "raw_negative_gradient": raw_negative_gradient,
        "layerwise_phi": layerwise_phi,
        "active_mask": active_mask,
        "neuron_counts": neuron_counts,
    }


def corr(x, y):
    if len(x) < 2:
        return 0.0
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    denom = np.sqrt(np.sum(xc**2) * np.sum(yc**2))
    return float(np.sum(xc * yc) / denom) if denom > 1e-30 else 0.0


def phi_diagonality(Phi_active):
    """Fraction of Φ's Frobenius energy on the diagonal. 1 = perfectly diagonal."""
    diag = np.diag(Phi_active)
    diag_norm_sq = float(np.sum(diag**2))
    total_norm_sq = float(np.sum(Phi_active**2))
    return diag_norm_sq / total_norm_sq if total_norm_sq > 1e-30 else 1.0


def phi_ii_cv(Phi_active):
    """Coefficient of variation of the diagonal entries across active units."""
    diag = np.diag(Phi_active)
    mu = float(np.mean(diag))
    sigma = float(np.std(diag))
    return sigma / mu if abs(mu) > 1e-30 else 0.0


# ======================== RUN CONDITION ========================


def run_condition(width, depth, dropout_p, seed, n_steps, diag_every, eta):
    rng = np.random.default_rng(seed)
    layer_sizes = [16] + [width] * depth + [2]
    weight_matrices = create_network(layer_sizes, rng)
    inputs, targets = make_bar_images(20, rng)
    num_examples = len(inputs)

    history = {
        "step": [],
        "loss": [],
        # Train-mode Φ (same per-step mask as the SGD step)
        "train_corr_kernel": [],
        "train_corr_diagonal": [],
        "train_corr_raw": [],
        "train_D_phi": [],
        "train_cv_phi_ii": [],
        # Inference-mode Φ (mask off) compared to the same actual ΔA
        "infer_corr_kernel": [],
        "infer_corr_diagonal": [],
        "infer_corr_raw": [],
        "infer_D_phi": [],
        "infer_cv_phi_ii": [],
    }
    last_phi_train = None
    last_phi_infer = None

    for step in range(n_steps):
        sampled = int(rng.integers(num_examples))
        masks = sample_dropout_masks(layer_sizes, dropout_p, rng)

        acts, pre_acts = forward(
            weight_matrices, inputs[sampled], dropout_masks=masks, p=dropout_p
        )
        grads_A, grads_W = backprop(
            weight_matrices,
            acts,
            pre_acts,
            targets[sampled],
            dropout_masks=masks,
            p=dropout_p,
        )

        do_diag = (step % diag_every == 0) or (step == n_steps - 1)
        if do_diag:
            acts_before = np.concatenate(
                [acts[layer_index] for layer_index in range(1, len(weight_matrices) + 1)]
            )

            # Train-mode kernel: mask = the step's mask
            preds_train = compute_jacobian_and_predictions(
                layer_sizes,
                weight_matrices,
                acts,
                pre_acts,
                grads_W,
                grads_A,
                eta,
                dropout_masks=masks,
                p=dropout_p,
            )

            # Inference-mode kernel: same weights+input, mask off. Use a deterministic
            # forward and backward to recover the right activations/pre-activations
            # for the inference setting (no scaling, no mask).
            acts_inf, pre_acts_inf = forward(
                weight_matrices, inputs[sampled], dropout_masks=None, p=0.0
            )
            grads_A_inf, grads_W_inf = backprop(
                weight_matrices,
                acts_inf,
                pre_acts_inf,
                targets[sampled],
                dropout_masks=None,
                p=0.0,
            )
            preds_infer = compute_jacobian_and_predictions(
                layer_sizes,
                weight_matrices,
                acts_inf,
                pre_acts_inf,
                grads_W_inf,
                grads_A_inf,
                eta,
                dropout_masks=None,
                p=0.0,
            )

            # Apply the real SGD step using the per-step (dropout-masked) gradients
            for layer_index in range(len(weight_matrices)):
                weight_matrices[layer_index] -= eta * grads_W[layer_index]

            # Re-run forward with the SAME mask to measure actual ΔA
            acts_after, _ = forward(
                weight_matrices, inputs[sampled], dropout_masks=masks, p=dropout_p
            )
            acts_after_flat = np.concatenate(
                [
                    acts_after[layer_index]
                    for layer_index in range(1, len(weight_matrices) + 1)
                ]
            )
            actual_change_full = acts_after_flat - acts_before

            train_active = preds_train["active_mask"] > 0
            actual_change = actual_change_full[train_active]
            history["train_corr_kernel"].append(
                corr(actual_change, preds_train["kernel_prediction"][train_active])
            )
            history["train_corr_diagonal"].append(
                corr(actual_change, preds_train["diagonal_prediction"][train_active])
            )
            history["train_corr_raw"].append(
                corr(actual_change, preds_train["raw_negative_gradient"][train_active])
            )
            Phi_train_active = preds_train["layerwise_phi"][
                np.ix_(np.where(train_active)[0], np.where(train_active)[0])
            ]
            history["train_D_phi"].append(phi_diagonality(Phi_train_active))
            history["train_cv_phi_ii"].append(phi_ii_cv(Phi_train_active))

            # For inference-mode Φ, use the inference active set; ΔA is the same
            # measured actual change but indexed by the inference active mask. Since
            # the inference active set is a superset of the train active set
            # (it doesn't exclude dropped units), we use it to evaluate the inference
            # kernel on its own native indexing.
            infer_active = preds_infer["active_mask"] > 0
            actual_change_inf = actual_change_full[infer_active]
            history["infer_corr_kernel"].append(
                corr(actual_change_inf, preds_infer["kernel_prediction"][infer_active])
            )
            history["infer_corr_diagonal"].append(
                corr(
                    actual_change_inf, preds_infer["diagonal_prediction"][infer_active]
                )
            )
            history["infer_corr_raw"].append(
                corr(
                    actual_change_inf,
                    preds_infer["raw_negative_gradient"][infer_active],
                )
            )
            Phi_infer_active = preds_infer["layerwise_phi"][
                np.ix_(np.where(infer_active)[0], np.where(infer_active)[0])
            ]
            history["infer_D_phi"].append(phi_diagonality(Phi_infer_active))
            history["infer_cv_phi_ii"].append(phi_ii_cv(Phi_infer_active))

            last_phi_train = Phi_train_active
            last_phi_infer = Phi_infer_active

            avg_loss = 0.0
            for ex in range(num_examples):
                # Loss is reported in inference mode so it's comparable across p.
                a, _ = forward(weight_matrices, inputs[ex], dropout_masks=None, p=0.0)
                avg_loss += 0.5 * np.sum((a[-1] - targets[ex]) ** 2) / num_examples

            history["step"].append(step)
            history["loss"].append(float(avg_loss))
        else:
            for layer_index in range(len(weight_matrices)):
                weight_matrices[layer_index] -= eta * grads_W[layer_index]

    return history, last_phi_train, last_phi_infer


# ======================== SWEEP & PLOTS ========================


def main():
    DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    last_phis = {}  # keyed by (width, p) for heatmap figure
    for cond in CONDITIONS:
        for seed in SEEDS:
            print(
                f"Running width={cond['width']}, p={cond['p']}, seed={seed} ...",
                flush=True,
            )
            history, phi_train, phi_infer = run_condition(
                cond["width"], DEPTH, cond["p"], seed, N_STEPS, DIAG_EVERY, ETA
            )
            results.append(
                {
                    "width": cond["width"],
                    "p": cond["p"],
                    "label": cond["label"],
                    "seed": seed,
                    "history": history,
                }
            )
            # Keep the last seed's Φ for the heatmap figure (one snapshot per condition).
            last_phis[(cond["width"], cond["p"])] = {
                "train": phi_train.tolist() if phi_train is not None else None,
                "infer": phi_infer.tolist() if phi_infer is not None else None,
            }

    with DATA_OUTPUT.open("w") as f:
        json.dump(
            {
                "config": {
                    "depth": DEPTH,
                    "n_steps": N_STEPS,
                    "diag_every": DIAG_EVERY,
                    "seeds": SEEDS,
                    "eta": ETA,
                    "conditions": CONDITIONS,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved {DATA_OUTPUT}")

    # Aggregate: take mean of the last 3 diagnostic steps per (condition, seed)
    # to summarise the trained state, then mean ± std over seeds.
    def trained_mean(values):
        return float(np.mean(values[-3:])) if values else float("nan")

    summary = {}
    for cond in CONDITIONS:
        key = (cond["width"], cond["p"])
        summary[key] = {"label": cond["label"]}
        for metric in [
            "train_corr_kernel",
            "train_corr_diagonal",
            "train_corr_raw",
            "train_D_phi",
            "train_cv_phi_ii",
            "infer_corr_kernel",
            "infer_corr_diagonal",
            "infer_corr_raw",
            "infer_D_phi",
            "infer_cv_phi_ii",
        ]:
            per_seed = [
                trained_mean(r["history"][metric])
                for r in results
                if (r["width"], r["p"]) == key
            ]
            summary[key][metric + "_mean"] = float(np.mean(per_seed))
            summary[key][metric + "_std"] = float(np.std(per_seed))

    plot_sweep(summary)
    plot_phi_heatmaps(last_phis)


def plot_sweep(summary):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.5), dpi=200, sharex=True)
    fig.patch.set_facecolor("#faf9f6")
    cond_order = [(c["width"], c["p"]) for c in CONDITIONS]
    labels = [summary[k]["label"] for k in cond_order]
    x = np.arange(len(cond_order))

    panel_metrics = [
        ("corr_raw", "$r(\\Delta A, -\\partial\\mathcal{L}/\\partial A)$", (0.0, 1.0)),
        (
            "corr_diagonal",
            "$r(\\Delta A, -\\Phi_{ii}\\,\\partial\\mathcal{L}/\\partial A_i)$",
            (0.0, 1.05),
        ),
        ("D_phi", r"$\|\mathrm{diag}\Phi\|_F^2 / \|\Phi\|_F^2$", (0.0, 1.05)),
    ]

    for col, (metric, ylabel, ylim) in enumerate(panel_metrics):
        for row, mode in enumerate(["train", "infer"]):
            ax = axes[row, col]
            means = [summary[k][f"{mode}_{metric}_mean"] for k in cond_order]
            stds = [summary[k][f"{mode}_{metric}_std"] for k in cond_order]
            colors = [
                "#888" if p == 0 else "#2563eb" for (_, p) in cond_order
            ]
            ax.bar(x, means, yerr=stds, color=colors, capsize=3, alpha=0.85)
            ax.set_xticks(x)
            ax.set_ylim(*ylim)
            ax.tick_params(labelsize=7)
            if row == 1:
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6.5)
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(ylabel + ("\n(train Φ)" if row == 0 else "\n(infer Φ)"), fontsize=8)
            ax.axhline(0, color="#ccc", linewidth=0.5)

    # Annotate the key comparison: dropout (48, 0.5) vs effective-width control (24, 0)
    ctrl_idx = cond_order.index((24, 0.0))
    drop_idx = cond_order.index((48, 0.5))
    for col, (metric, _, _) in enumerate(panel_metrics):
        for row, mode in enumerate(["train", "infer"]):
            ax = axes[row, col]
            ctrl = summary[(24, 0.0)][f"{mode}_{metric}_mean"]
            drop = summary[(48, 0.5)][f"{mode}_{metric}_mean"]
            ax.annotate(
                "",
                xy=(drop_idx, drop),
                xytext=(ctrl_idx, ctrl),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#d97706",
                    lw=1.2,
                    shrinkA=4,
                    shrinkB=4,
                ),
            )

    fig.suptitle(
        f"Dropout × diagonality (depth={DEPTH}, {N_STEPS} steps, {len(SEEDS)} seeds; "
        "orange arrow: eff-width-controlled effect of dropout)",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIGURE_OUTPUT_DIR / "fig_dropout_sweep.pdf", bbox_inches="tight", facecolor="#faf9f6")
    plt.savefig(FIGURE_OUTPUT_DIR / "fig_dropout_sweep.png", bbox_inches="tight", facecolor="#faf9f6")
    print("Saved fig_dropout_sweep.{pdf,png}")
    plt.close(fig)


def plot_phi_heatmaps(last_phis):
    pairs = [
        ((48, 0.0), "baseline (w=48, p=0)"),
        ((48, 0.5), "dropout (w=48, p=0.5)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7), dpi=200)
    fig.patch.set_facecolor("#faf9f6")
    for row, (key, label) in enumerate(pairs):
        for col, mode in enumerate(["train", "infer"]):
            ax = axes[row, col]
            mat = last_phis.get(key, {}).get(mode)
            if mat is None:
                ax.set_axis_off()
                continue
            mat = np.asarray(mat)
            v = float(np.max(np.abs(mat)))
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_title(f"{label}\nΦ {mode}", fontsize=9)
            ax.tick_params(labelsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / "fig_dropout_phi_heatmaps.pdf", bbox_inches="tight", facecolor="#faf9f6")
    plt.savefig(FIGURE_OUTPUT_DIR / "fig_dropout_phi_heatmaps.png", bbox_inches="tight", facecolor="#faf9f6")
    print("Saved fig_dropout_phi_heatmaps.{pdf,png}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run dropout simulations and render their diagnostic figures."
    )
    parser.add_argument(
        "--data-output",
        required=True,
        type=Path,
        help="Path for the generated JSON intermediate.",
    )
    parser.add_argument(
        "--figure-output-dir",
        required=True,
        type=Path,
        help="Directory for publication-ready PDF figures and PNG previews.",
    )
    args = parser.parse_args()
    DATA_OUTPUT = args.data_output
    FIGURE_OUTPUT_DIR = args.figure_output_dir
    main()
