"""
Does training with dropout make the per-layer kernel Φ more diagonal, and does
that improve the diagonal approximation to the activity update?

Self-contained dropout-aware simulation with forward/backward passes and
Jacobian assembly. Normally run through Snakemake.

Outputs:
  - data/generated/dropout_results.json
  - results/data/dropout_summary.csv
  - results/figures/fig_dropout_sweep.{pdf,png}
  - results/figures/fig_dropout_phi_heatmaps.{pdf,png}
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================== CONFIG ========================

DEPTHS = [3, 4]
WIDTH = 48
N_STEPS = 10_000
DIAG_EVERY = 200
SEEDS = list(range(10))
ETA = 0.005

# Standard range: rates above 0.5 leave too few active units at width 48 and
# increasingly test capacity loss rather than regularisation against coadaptation.
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
CONDITIONS = [
    {
        "width": WIDTH,
        "depth": depth,
        "p": dropout_p,
        "label": f"depth={depth}, p={dropout_p:.1f}",
    }
    for depth in DEPTHS
    for dropout_p in DROPOUT_RATES
]

DATA_OUTPUT = Path("dropout_results.json")
SUMMARY_OUTPUT = Path("dropout_summary.csv")
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
            if dropout_masks is not None and dropout_masks[layer_index + 1] is not None:
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
        if (
            dropout_masks is not None
            and dropout_masks[layer_index_one_based] is not None
        ):
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
    # Separate streams make conditions paired across dropout rates: every p
    # uses the same initial weights, dataset, and sampled-example order. Common
    # mask randomness also makes higher-p masks nested versions of lower-p masks.
    initialization_rng = np.random.default_rng(seed)
    data_rng = np.random.default_rng(10_000 + seed)
    sampling_rng = np.random.default_rng(20_000 + seed)
    mask_rng = np.random.default_rng(30_000 + seed)
    layer_sizes = [16] + [width] * depth + [2]
    weight_matrices = create_network(layer_sizes, initialization_rng)
    inputs, targets = make_bar_images(20, data_rng)
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
        # Counterfactual inference-mode diagnostic (mask off). This is the
        # primary test of learned regularisation rather than acute sparsity.
        "infer_corr_kernel": [],
        "infer_corr_diagonal": [],
        "infer_corr_raw": [],
        "infer_D_phi": [],
        "infer_cv_phi_ii": [],
    }
    last_phi_train = None
    last_phi_infer = None

    for step in range(n_steps):
        sampled = int(sampling_rng.integers(num_examples))
        masks = sample_dropout_masks(layer_sizes, dropout_p, mask_rng)

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
                [
                    acts[layer_index]
                    for layer_index in range(1, len(weight_matrices) + 1)
                ]
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

            # Inference-mode counterfactual: same learned weights and input, but
            # no acute mask. Measure the actual activity change after a temporary
            # standard-GD step that never changes the dropout training path.
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
            acts_inf_before = np.concatenate(
                [
                    acts_inf[layer_index]
                    for layer_index in range(1, len(weight_matrices) + 1)
                ]
            )
            counterfactual_weights = [
                weights - eta * gradient
                for weights, gradient in zip(weight_matrices, grads_W_inf, strict=True)
            ]
            acts_inf_after, _ = forward(
                counterfactual_weights,
                inputs[sampled],
                dropout_masks=None,
                p=0.0,
            )
            actual_change_inf_full = (
                np.concatenate(
                    [
                        acts_inf_after[layer_index]
                        for layer_index in range(1, len(weight_matrices) + 1)
                    ]
                )
                - acts_inf_before
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

            # Evaluate the counterfactual no-dropout step on its native active set.
            infer_active = preds_infer["active_mask"] > 0
            actual_change_inf = actual_change_inf_full[infer_active]
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


SUMMARY_METRICS = [
    "loss",
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
]


def trained_mean(values):
    """Mean over the final five diagnostics for one seed and condition."""
    return float(np.mean(values[-5:])) if values else float("nan")


def bootstrap_mean_ci(values, seed, n_bootstrap=10_000):
    """Deterministic percentile bootstrap interval for a paired mean."""
    values = np.asarray(values, dtype=float)
    if np.all(values == values[0]):
        constant = float(values[0])
        return constant, constant
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    bootstrap_means = np.mean(values[sample_indices], axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def summarise_results(results):
    per_seed = {}
    for result in results:
        key = (result["depth"], result["p"], result["seed"])
        per_seed[key] = {
            metric: trained_mean(result["history"][metric])
            for metric in SUMMARY_METRICS
        }

    summary_rows = []
    for depth in DEPTHS:
        baseline_by_seed = {seed: per_seed[(depth, 0.0, seed)] for seed in SEEDS}
        for dropout_p in DROPOUT_RATES:
            row = {
                "width": WIDTH,
                "depth": depth,
                "dropout_p": dropout_p,
                "n_seeds": len(SEEDS),
                "n_steps": N_STEPS,
            }
            for metric in SUMMARY_METRICS:
                values = np.array(
                    [per_seed[(depth, dropout_p, seed)][metric] for seed in SEEDS]
                )
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1))
                row[f"{metric}_sem"] = float(
                    np.std(values, ddof=1) / np.sqrt(len(values))
                )

            for metric in (
                "infer_corr_diagonal",
                "train_corr_diagonal",
                "infer_D_phi",
                "train_D_phi",
                "loss",
            ):
                paired_differences = np.array(
                    [
                        per_seed[(depth, dropout_p, seed)][metric]
                        - baseline_by_seed[seed][metric]
                        for seed in SEEDS
                    ]
                )
                ci_low, ci_high = bootstrap_mean_ci(
                    paired_differences,
                    seed=(
                        40_000
                        + 1_000 * depth
                        + int(round(100 * dropout_p))
                        + len(metric)
                    ),
                )
                row[f"delta_{metric}_vs_p0_mean"] = float(np.mean(paired_differences))
                row[f"delta_{metric}_vs_p0_ci_low"] = ci_low
                row[f"delta_{metric}_vs_p0_ci_high"] = ci_high

            summary_rows.append(row)

    return per_seed, summary_rows


def main():
    DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    last_phis = {}  # keyed by (depth, width, p) for heatmap figure
    for cond in CONDITIONS:
        for seed in SEEDS:
            print(
                f"Running depth={cond['depth']}, width={cond['width']}, "
                f"p={cond['p']}, seed={seed} ...",
                flush=True,
            )
            history, phi_train, phi_infer = run_condition(
                cond["width"],
                cond["depth"],
                cond["p"],
                seed,
                N_STEPS,
                DIAG_EVERY,
                ETA,
            )
            results.append(
                {
                    "width": cond["width"],
                    "depth": cond["depth"],
                    "p": cond["p"],
                    "label": cond["label"],
                    "seed": seed,
                    "history": history,
                }
            )
            # Keep the last seed's Φ for the heatmap figure (one snapshot per condition).
            last_phis[(cond["depth"], cond["width"], cond["p"])] = {
                "train": phi_train.tolist() if phi_train is not None else None,
                "infer": phi_infer.tolist() if phi_infer is not None else None,
            }

    _, summary_rows = summarise_results(results)

    with DATA_OUTPUT.open("w") as f:
        json.dump(
            {
                "config": {
                    "width": WIDTH,
                    "depths": DEPTHS,
                    "n_steps": N_STEPS,
                    "diag_every": DIAG_EVERY,
                    "seeds": SEEDS,
                    "eta": ETA,
                    "dropout_rates": DROPOUT_RATES,
                    "conditions": CONDITIONS,
                    "primary_test": (
                        "counterfactual no-dropout activity update at weights "
                        "learned with each dropout rate"
                    ),
                },
                "summary": summary_rows,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved {DATA_OUTPUT}")

    with SUMMARY_OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved {SUMMARY_OUTPUT}")

    plot_sweep(summary_rows)
    plot_phi_heatmaps(last_phis)


def plot_sweep(summary_rows):
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.8), dpi=200, sharex=True)
    fig.patch.set_facecolor("#faf9f6")
    rows_by_depth = {
        depth: [row for row in summary_rows if row["depth"] == depth]
        for depth in DEPTHS
    }
    depth_colors = {DEPTHS[0]: "#059669", DEPTHS[1]: "#7c3aed"}

    def plot_depth_comparison(ax, metric, ylabel, ylim=None):
        for depth in DEPTHS:
            depth_rows = rows_by_depth[depth]
            dropout_rates = np.array([row["dropout_p"] for row in depth_rows])
            means = np.array([row[f"{metric}_mean"] for row in depth_rows])
            stds = np.array([row[f"{metric}_std"] for row in depth_rows])
            ax.errorbar(
                dropout_rates,
                means,
                yerr=stds,
                marker="o",
                linewidth=1.5,
                capsize=3,
                color=depth_colors[depth],
                label=f"depth {depth}",
            )
        ax.set_ylabel(ylabel, fontsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=6.5, framealpha=0.85)

    plot_depth_comparison(
        axes[0, 0],
        "infer_corr_diagonal",
        r"$r(\Delta A,-\Phi_{ii}\,\partial\mathcal{L}/\partial A_i)$",
        (0.5, 1.02),
    )
    axes[0, 0].set_title("(a) Diagonal-approximation accuracy", fontsize=9)

    plot_depth_comparison(
        axes[0, 1],
        "infer_D_phi",
        r"$\|\mathrm{diag}\Phi\|_F^2/\|\Phi\|_F^2$",
        (0.0, 1.02),
    )
    axes[0, 1].set_title("(b) Kernel diagonal energy", fontsize=9)

    plot_depth_comparison(axes[1, 0], "loss", "inference-mode loss")
    axes[1, 0].set_title("(c) Task performance", fontsize=9)

    for depth in DEPTHS:
        depth_rows = rows_by_depth[depth]
        dropout_rates = np.array([row["dropout_p"] for row in depth_rows])
        delta_means = np.array(
            [row["delta_infer_corr_diagonal_vs_p0_mean"] for row in depth_rows]
        )
        ci_low = np.array(
            [row["delta_infer_corr_diagonal_vs_p0_ci_low"] for row in depth_rows]
        )
        ci_high = np.array(
            [row["delta_infer_corr_diagonal_vs_p0_ci_high"] for row in depth_rows]
        )
        axes[1, 1].errorbar(
            dropout_rates,
            delta_means,
            yerr=np.vstack((delta_means - ci_low, ci_high - delta_means)),
            marker="o",
            linewidth=1.5,
            capsize=3,
            color=depth_colors[depth],
            label=f"depth {depth}",
        )
    axes[1, 1].axhline(0, color="#9ca3af", linestyle="--", linewidth=0.8)
    axes[1, 1].set_ylabel(r"paired $\Delta r$ vs. $p=0$ (95\% CI)", fontsize=8)
    axes[1, 1].set_title("(d) Primary paired test", fontsize=9)
    axes[1, 1].legend(fontsize=6.5, framealpha=0.85)

    for ax in axes.flat:
        ax.set_xticks(DROPOUT_RATES)
        ax.set_xlabel("dropout probability $p$", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Dropout and depth (width={WIDTH}, {N_STEPS} steps, "
        f"{len(SEEDS)} paired seeds; inference-mode evaluation)",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        FIGURE_OUTPUT_DIR / "fig_dropout_sweep.pdf",
        bbox_inches="tight",
        facecolor="#faf9f6",
    )
    plt.savefig(
        FIGURE_OUTPUT_DIR / "fig_dropout_sweep.png",
        bbox_inches="tight",
        facecolor="#faf9f6",
    )
    print("Saved fig_dropout_sweep.{pdf,png}")
    plt.close(fig)


def plot_phi_heatmaps(last_phis):
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7), dpi=200)
    fig.patch.set_facecolor("#faf9f6")
    for row, depth in enumerate(DEPTHS):
        for col, dropout_p in enumerate((0.0, 0.5)):
            ax = axes[row, col]
            key = (depth, WIDTH, dropout_p)
            mat = last_phis.get(key, {}).get("infer")
            if mat is None:
                ax.set_axis_off()
                continue
            mat = np.asarray(mat)
            v = float(np.max(np.abs(mat)))
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_title(
                f"depth={depth}, w={WIDTH}, p={dropout_p:.1f}\n"
                r"inference-mode $\Phi$",
                fontsize=9,
            )
            ax.tick_params(labelsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        FIGURE_OUTPUT_DIR / "fig_dropout_phi_heatmaps.pdf",
        bbox_inches="tight",
        facecolor="#faf9f6",
    )
    plt.savefig(
        FIGURE_OUTPUT_DIR / "fig_dropout_phi_heatmaps.png",
        bbox_inches="tight",
        facecolor="#faf9f6",
    )
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
    parser.add_argument(
        "--summary-output",
        required=True,
        type=Path,
        help="Path for the final CSV summary and paired statistical contrasts.",
    )
    args = parser.parse_args()
    DATA_OUTPUT = args.data_output
    SUMMARY_OUTPUT = args.summary_output
    FIGURE_OUTPUT_DIR = args.figure_output_dir
    main()
