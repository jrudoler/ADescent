"""
Generate figures for the activity-space NTK paper.
Same computation as the interactive demo, frozen into publication-quality plots.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(42)

# ======================== NETWORK ========================


def create_network(layer_sizes):
    weight_matrices = []
    for layer_index in range(len(layer_sizes) - 1):
        # Extra column for bias (input augmented with 1)
        scale = np.sqrt(2.0 / layer_sizes[layer_index])
        layer_weights = (
            np.random.randn(layer_sizes[layer_index + 1], layer_sizes[layer_index] + 1)
            * scale
        )
        layer_weights[:, -1] = 0  # init biases to zero
        weight_matrices.append(layer_weights)
    return weight_matrices


def forward(weight_matrices, input_vector):
    activities_by_layer = [input_vector.copy()]
    pre_activations_by_layer = [None]
    for layer_index in range(len(weight_matrices)):
        augmented_activities = np.append(
            activities_by_layer[-1], 1.0
        )  # append 1 for bias
        pre_activations = weight_matrices[layer_index] @ augmented_activities
        pre_activations_by_layer.append(pre_activations.copy())
        if layer_index < len(weight_matrices) - 1:
            activities_by_layer.append(np.maximum(0, pre_activations))
        else:
            activities_by_layer.append(pre_activations.copy())
    return activities_by_layer, pre_activations_by_layer


def backprop(
    weight_matrices,
    activities_by_layer,
    pre_activations_by_layer,
    target_vector,
):
    num_layers = len(weight_matrices)
    loss_gradient_by_activity = [None] * (num_layers + 1)
    loss_gradient_by_activity[num_layers] = (
        activities_by_layer[num_layers] - target_vector
    )
    loss_gradient_by_weights = [None] * num_layers
    for layer_index in range(num_layers - 1, -1, -1):
        if layer_index == num_layers - 1:
            local_gradient = loss_gradient_by_activity[layer_index + 1].copy()
        else:
            local_gradient = loss_gradient_by_activity[layer_index + 1] * (
                pre_activations_by_layer[layer_index + 1] > 0
            ).astype(float)
        augmented_activities = np.append(activities_by_layer[layer_index], 1.0)
        loss_gradient_by_weights[layer_index] = np.outer(
            local_gradient, augmented_activities
        )
        if layer_index > 0:
            loss_gradient_by_activity[layer_index] = (
                weight_matrices[layer_index][:, :-1].T @ local_gradient
            )  # exclude bias column
    return loss_gradient_by_activity, loss_gradient_by_weights


def make_bar_images(n_per_class=20, noise=0.15):
    input_examples, target_vectors = [], []
    for _ in range(n_per_class):
        # Horizontal bar: random row bright
        horizontal_image = np.zeros(16)
        row = np.random.randint(4)
        horizontal_image[row * 4 : (row + 1) * 4] = 1.0
        horizontal_image += np.random.randn(16) * noise
        input_examples.append(horizontal_image)
        target_vectors.append([1, 0])
        # Vertical bar: random column bright
        vertical_image = np.zeros(16)
        col = np.random.randint(4)
        for r in range(4):
            vertical_image[r * 4 + col] = 1.0
        vertical_image += np.random.randn(16) * noise
        input_examples.append(vertical_image)
        target_vectors.append([0, 1])
    return np.array(input_examples), np.array(target_vectors)


# ======================== JACOBIAN ========================


def compute_layer_local_prediction(
    layer_sizes,
    weight_matrices,
    activities_by_layer,
    pre_activations_by_layer,
    loss_gradient_by_activity,
    learning_rate,
):
    """Layer-local kernel prediction: ΔA^(ℓ) = -η D_ℓ ∇L + J_ℓ ΔA^(ℓ-1)."""
    num_layers = len(weight_matrices)
    total_neurons = sum(layer_sizes[1:])
    predicted_activity_change = np.zeros(total_neurons)
    previous_layer_prediction = None
    layer_offset = 0

    for layer_index in range(num_layers):
        output_width = layer_sizes[layer_index + 1]
        augmented_activity_norm_sq = np.sum(activities_by_layer[layer_index] ** 2) + 1.0

        if layer_index < num_layers - 1:
            activation_derivative = (
                pre_activations_by_layer[layer_index + 1] > 0
            ).astype(float)
        else:
            activation_derivative = np.ones(output_width)

        activity_gradient = loss_gradient_by_activity[layer_index + 1]
        current_layer_prediction = (
            -learning_rate
            * activation_derivative
            * activation_derivative
            * augmented_activity_norm_sq
            * activity_gradient
        )

        if previous_layer_prediction is not None:
            current_layer_prediction += activation_derivative * (
                weight_matrices[layer_index][:, :-1] @ previous_layer_prediction
            )

        predicted_activity_change[layer_offset : layer_offset + output_width] = (
            current_layer_prediction
        )
        layer_offset += output_width
        previous_layer_prediction = current_layer_prediction

    return predicted_activity_change


def compute_jacobian_and_predictions(
    layer_sizes,
    weight_matrices,
    activities_by_layer,
    pre_activations_by_layer,
    loss_gradient_by_weights,
    loss_gradient_by_activity,
    learning_rate,
):
    """
    Build the exact activity Jacobian for the sampled example and compare
    several activity-space predictions for a single SGD step.

    Returns flattened vectors over all non-input neurons:
      exact_activity_change: first-order activity change J @ ΔW from the actual SGD step
      kernel_prediction: per-layer kernel prediction from Eq. 3
      diagonal_prediction: diagonal approximation from Eq. 5, i.e. -η Φ_ii dℒ/dA_i
      raw_negative_gradient: raw -dℒ/dA baseline (with dead hidden ReLUs masked out)
      layerwise_kernel_matrix: block-diagonal matrix of per-layer kernels Φ^(ℓ)
    """
    num_layers = len(weight_matrices)
    neuron_counts = layer_sizes[1:]
    total_neurons = sum(neuron_counts)

    layer_offsets = []
    layer_offset = 0
    for layer_index in range(num_layers):
        layer_offsets.append(layer_offset)
        layer_offset += layer_sizes[layer_index + 1]

    backprop_activity_gradient = np.zeros(total_neurons)
    for layer_index in range(1, num_layers + 1):
        if loss_gradient_by_activity[layer_index] is not None:
            layer_start = layer_offsets[layer_index - 1]
            backprop_activity_gradient[
                layer_start : layer_start + len(loss_gradient_by_activity[layer_index])
            ] = loss_gradient_by_activity[layer_index]

    total_parameters = sum(
        (layer_sizes[layer_index] + 1) * layer_sizes[layer_index + 1]
        for layer_index in range(num_layers)
    )

    flat_weight_update = np.zeros(total_parameters)
    parameter_offset = 0
    for layer_index in range(num_layers):
        num_layer_parameters = loss_gradient_by_weights[layer_index].size
        flat_weight_update[
            parameter_offset : parameter_offset + num_layer_parameters
        ] = -learning_rate * loss_gradient_by_weights[layer_index].ravel()
        parameter_offset += num_layer_parameters

    full_jacobian = np.zeros((total_neurons, total_parameters))
    parameter_offset = 0
    for source_layer_index in range(num_layers):
        source_input_width = layer_sizes[source_layer_index]
        source_output_width = layer_sizes[source_layer_index + 1]
        num_source_parameters = source_output_width * (source_input_width + 1)
        augmented_activities = np.append(activities_by_layer[source_layer_index], 1.0)

        direct_sensitivity_block = np.zeros(
            (source_output_width, num_source_parameters)
        )
        for neuron_index in range(source_output_width):
            activation_derivative = (
                1.0
                if source_layer_index == num_layers - 1
                else (
                    1.0
                    if pre_activations_by_layer[source_layer_index + 1][neuron_index]
                    > 0
                    else 0.0
                )
            )
            parameter_slice_start = neuron_index * (source_input_width + 1)
            parameter_slice_end = (neuron_index + 1) * (source_input_width + 1)
            direct_sensitivity_block[
                neuron_index, parameter_slice_start:parameter_slice_end
            ] = activation_derivative * augmented_activities

        layer_start = layer_offsets[source_layer_index]
        full_jacobian[
            layer_start : layer_start + source_output_width,
            parameter_offset : parameter_offset + num_source_parameters,
        ] = direct_sensitivity_block

        propagated_sensitivity_block = direct_sensitivity_block
        for downstream_layer_index in range(source_layer_index + 1, num_layers):
            downstream_input_width = layer_sizes[downstream_layer_index]
            downstream_output_width = layer_sizes[downstream_layer_index + 1]
            inter_layer_jacobian = np.zeros(
                (downstream_output_width, downstream_input_width)
            )

            for downstream_neuron_index in range(downstream_output_width):
                activation_derivative = (
                    1.0
                    if downstream_layer_index == num_layers - 1
                    else (
                        1.0
                        if pre_activations_by_layer[downstream_layer_index + 1][
                            downstream_neuron_index
                        ]
                        > 0
                        else 0.0
                    )
                )
                inter_layer_jacobian[downstream_neuron_index] = (
                    activation_derivative
                    * weight_matrices[downstream_layer_index][
                        downstream_neuron_index, :-1
                    ]
                )  # exclude bias column

            propagated_sensitivity_block = (
                inter_layer_jacobian @ propagated_sensitivity_block
            )
            downstream_layer_start = layer_offsets[downstream_layer_index]
            full_jacobian[
                downstream_layer_start : downstream_layer_start
                + downstream_output_width,
                parameter_offset : parameter_offset + num_source_parameters,
            ] = propagated_sensitivity_block

        parameter_offset += num_source_parameters

    exact_activity_change = full_jacobian @ flat_weight_update

    phi_diagonal = np.sum(full_jacobian**2, axis=1)
    diagonal_prediction = -learning_rate * phi_diagonal * backprop_activity_gradient

    kernel_prediction = compute_layer_local_prediction(
        layer_sizes,
        weight_matrices,
        activities_by_layer,
        pre_activations_by_layer,
        loss_gradient_by_activity,
        learning_rate,
    )

    active_neuron_mask = np.zeros(total_neurons)
    for layer_index in range(1, num_layers + 1):
        layer_start = layer_offsets[layer_index - 1]
        layer_width = layer_sizes[layer_index]
        if layer_index < num_layers:
            for neuron_index in range(layer_width):
                active_neuron_mask[layer_start + neuron_index] = (
                    1.0
                    if pre_activations_by_layer[layer_index][neuron_index] > 0
                    else 0.0
                )
        else:
            active_neuron_mask[layer_start : layer_start + layer_width] = 1.0

    raw_negative_gradient = -backprop_activity_gradient * active_neuron_mask

    layerwise_kernel_matrix = np.zeros((total_neurons, total_neurons))
    for layer_index in range(num_layers):
        layer_start = layer_offsets[layer_index]
        layer_width = layer_sizes[layer_index + 1]
        layer_jacobian = full_jacobian[layer_start : layer_start + layer_width, :]
        layerwise_kernel_matrix[
            layer_start : layer_start + layer_width,
            layer_start : layer_start + layer_width,
        ] = layer_jacobian @ layer_jacobian.T

    return (
        exact_activity_change,
        kernel_prediction,
        diagonal_prediction,
        raw_negative_gradient,
        layerwise_kernel_matrix,
        neuron_counts,
        active_neuron_mask,
    )


def compute_kernel_diagonal_predictions(
    layer_sizes,
    weight_matrices,
    activities_by_layer,
    pre_activations_by_layer,
    loss_gradient_by_activity,
    learning_rate,
):
    """Compute sweep diagnostics without materializing the full Jacobian.

    For layer ``ell``, the activity kernel obeys

        Phi_ell = q_ell D_ell^2 + B_ell Phi_{ell-1} B_ell.T,

    where ``q_ell`` is the squared norm of the bias-augmented presynaptic
    activity and ``B_ell`` is the inter-layer activity Jacobian. Keeping the
    current layer kernel is sufficient to recover every diagonal entry used by
    Eq. 5. This is algebraically equivalent to taking row norms of the full
    activity Jacobian, but uses O(width^2) rather than O(width^3) memory.
    """
    num_layers = len(weight_matrices)
    neuron_counts = layer_sizes[1:]
    total_neurons = sum(neuron_counts)
    phi_diagonal = np.zeros(total_neurons)
    backprop_activity_gradient = np.zeros(total_neurons)
    active_neuron_mask = np.zeros(total_neurons)

    previous_layer_kernel = None
    layer_offset = 0
    for layer_index in range(num_layers):
        output_width = layer_sizes[layer_index + 1]
        if layer_index < num_layers - 1:
            activation_derivative = (
                pre_activations_by_layer[layer_index + 1] > 0
            ).astype(float)
        else:
            activation_derivative = np.ones(output_width)

        augmented_activity_norm_sq = np.sum(activities_by_layer[layer_index] ** 2) + 1.0
        current_layer_kernel = np.diag(
            augmented_activity_norm_sq * activation_derivative**2
        )

        if previous_layer_kernel is not None:
            inter_layer_jacobian = (
                activation_derivative[:, None] * weight_matrices[layer_index][:, :-1]
            )
            propagated_kernel = inter_layer_jacobian @ previous_layer_kernel
            current_layer_kernel += propagated_kernel @ inter_layer_jacobian.T

        layer_slice = slice(layer_offset, layer_offset + output_width)
        phi_diagonal[layer_slice] = np.diag(current_layer_kernel)
        backprop_activity_gradient[layer_slice] = loss_gradient_by_activity[
            layer_index + 1
        ]
        active_neuron_mask[layer_slice] = activation_derivative

        previous_layer_kernel = current_layer_kernel
        layer_offset += output_width

    diagonal_prediction = -learning_rate * phi_diagonal * backprop_activity_gradient
    kernel_prediction = compute_layer_local_prediction(
        layer_sizes,
        weight_matrices,
        activities_by_layer,
        pre_activations_by_layer,
        loss_gradient_by_activity,
        learning_rate,
    )
    raw_negative_gradient = -backprop_activity_gradient * active_neuron_mask

    return (
        kernel_prediction,
        diagonal_prediction,
        raw_negative_gradient,
        neuron_counts,
        active_neuron_mask,
    )


def corr(actual_values, predicted_values):
    # Pearson correlation over neuron indices. This removes the mean first, so
    # it measures centered linear alignment, not cosine similarity.
    if len(actual_values) < 2:
        return 0.0
    centered_actual = actual_values - np.mean(actual_values)
    centered_prediction = predicted_values - np.mean(predicted_values)
    denominator = np.sqrt(np.sum(centered_actual**2) * np.sum(centered_prediction**2))
    return (
        np.sum(centered_actual * centered_prediction) / denominator
        if denominator > 1e-30
        else 0.0
    )


# ======================== RUN EXPERIMENT ========================


def run_experiment(
    width,
    depth,
    eta=0.005,
    n_steps=2000,
    diag_every=50,
    full_jacobian=True,
):
    """
    Train one random MLP with online SGD and periodically compare the observed
    single-step activity change against several predictions.

    Most SGD steps only update weights. Every `diag_every` steps we also:
      1. sample one training example,
      2. compute the requested kernel diagnostics for that sample,
      3. apply the real SGD step on that same sample,
      4. measure the resulting activity change ΔA on that same sample.

    The returned history contains:
      corr_exact: r(actual ΔA, JΔW), or NaN when full_jacobian=False
      corr_kernel: r(actual ΔA, full kernel Eq. 3)
      corr_diagonal: r(actual ΔA, diagonal Eq. 5)
      corr_raw_gradient: r(actual ΔA, raw -dℒ/dA)
    """
    layer_sizes = [16] + [width] * depth + [2]
    weight_matrices = create_network(layer_sizes)
    training_inputs, training_targets = make_bar_images(20)
    num_examples = len(training_inputs)

    history = {
        "step": [],
        "loss": [],
        "corr_exact": [],
        "corr_kernel": [],
        "corr_diagonal": [],
        "corr_raw_gradient": [],
    }
    latest_snapshot = None

    for step in range(n_steps):
        sampled_example_index = np.random.randint(num_examples)
        activities_by_layer, pre_activations_by_layer = forward(
            weight_matrices, training_inputs[sampled_example_index]
        )
        loss_gradient_by_activity, loss_gradient_by_weights = backprop(
            weight_matrices,
            activities_by_layer,
            pre_activations_by_layer,
            training_targets[sampled_example_index],
        )

        # Expensive diagnostics are only done periodically because building the
        # full Jacobian scales poorly with width/depth.
        should_compute_diagnostics = (step % diag_every == 0) or (step == n_steps - 1)

        if should_compute_diagnostics:
            # Activities before the SGD step, flattened across all hidden/output
            # layers so they can be compared directly to the flattened
            # predictions returned by compute_jacobian_and_predictions().
            activities_before_update = np.concatenate(
                [
                    activities_by_layer[layer_index]
                    for layer_index in range(1, len(weight_matrices) + 1)
                ]
            )
            if full_jacobian:
                (
                    exact_prediction,
                    kernel_prediction,
                    diagonal_prediction,
                    raw_negative_gradient,
                    layerwise_kernel_matrix,
                    neuron_counts,
                    active_neuron_mask,
                ) = compute_jacobian_and_predictions(
                    layer_sizes,
                    weight_matrices,
                    activities_by_layer,
                    pre_activations_by_layer,
                    loss_gradient_by_weights,
                    loss_gradient_by_activity,
                    eta,
                )
            else:
                (
                    kernel_prediction,
                    diagonal_prediction,
                    raw_negative_gradient,
                    neuron_counts,
                    active_neuron_mask,
                ) = compute_kernel_diagonal_predictions(
                    layer_sizes,
                    weight_matrices,
                    activities_by_layer,
                    pre_activations_by_layer,
                    loss_gradient_by_activity,
                    eta,
                )
                exact_prediction = None
                layerwise_kernel_matrix = None

            # Apply the real SGD step on this exact sampled example.
            for layer_index in range(len(weight_matrices)):
                weight_matrices[layer_index] -= (
                    eta * loss_gradient_by_weights[layer_index]
                )

            updated_activities_by_layer, _ = forward(
                weight_matrices, training_inputs[sampled_example_index]
            )
            activities_after_update = np.concatenate(
                [
                    updated_activities_by_layer[layer_index]
                    for layer_index in range(1, len(weight_matrices) + 1)
                ]
            )
            full_activity_change = activities_after_update - activities_before_update

            # Compare diagnostics only on active neurons. For dead hidden ReLUs,
            # both the theory and the manuscript exclude them on that sample.
            active_neuron_indices = active_neuron_mask > 0
            actual_activity_change = full_activity_change[active_neuron_indices]
            kernel_prediction = kernel_prediction[active_neuron_indices]
            diagonal_prediction = diagonal_prediction[active_neuron_indices]
            raw_negative_gradient = raw_negative_gradient[active_neuron_indices]
            if exact_prediction is not None:
                exact_prediction = exact_prediction[active_neuron_indices]

            # These are the core diagnostics:
            #   corr_exact ~= 1 checks the Jacobian bookkeeping
            #   corr_kernel ~= 1 checks the exact kernel recursion
            #   corr_diagonal measures how good Eq. 5 is
            #   corr_raw_gradient measures alignment with the raw activity
            #     gradient only and is stricter because Eq. 5 still allows
            #     neuron-specific Φ_ii
            exact_correlation = (
                corr(actual_activity_change, exact_prediction)
                if exact_prediction is not None
                else np.nan
            )
            kernel_correlation = corr(actual_activity_change, kernel_prediction)
            diagonal_correlation = corr(actual_activity_change, diagonal_prediction)
            raw_gradient_correlation = corr(
                actual_activity_change, raw_negative_gradient
            )

            # Full-dataset loss after the step, used only for plotting dynamics.
            average_loss = 0
            for example_index in range(num_examples):
                forward_activities, _ = forward(
                    weight_matrices, training_inputs[example_index]
                )
                average_loss += (
                    0.5
                    * np.sum(
                        (forward_activities[-1] - training_targets[example_index]) ** 2
                    )
                    / num_examples
                )

            history["step"].append(step)
            history["loss"].append(average_loss)
            history["corr_exact"].append(exact_correlation)
            history["corr_kernel"].append(kernel_correlation)
            history["corr_diagonal"].append(diagonal_correlation)
            history["corr_raw_gradient"].append(raw_gradient_correlation)

            # Keep a filtered snapshot for the figure panels: only active neurons
            # remain in the heatmap and scatter plots.
            if full_jacobian:
                active_neuron_indices_flat = np.where(active_neuron_indices)[0]
                filtered_kernel_matrix = layerwise_kernel_matrix[
                    np.ix_(active_neuron_indices_flat, active_neuron_indices_flat)
                ]
                filtered_neuron_counts = []
                neuron_offset = 0
                for layer_neuron_count in neuron_counts:
                    filtered_neuron_counts.append(
                        int(
                            np.sum(
                                active_neuron_indices[
                                    neuron_offset : neuron_offset + layer_neuron_count
                                ]
                            )
                        )
                    )
                    neuron_offset += layer_neuron_count

                latest_snapshot = {
                    "actual_activity_change": actual_activity_change,
                    "exact_prediction": exact_prediction,
                    "kernel_prediction": kernel_prediction,
                    "diagonal_prediction": diagonal_prediction,
                    "raw_negative_gradient": raw_negative_gradient,
                    "Phi": filtered_kernel_matrix,
                    "neuron_counts": filtered_neuron_counts,
                    "step": step,
                }
        else:
            # Cheap path: ordinary SGD with no Jacobian diagnostics.
            for layer_index in range(len(weight_matrices)):
                weight_matrices[layer_index] -= (
                    eta * loss_gradient_by_weights[layer_index]
                )

    return history, latest_snapshot, layer_sizes


# ======================== PLOTTING ========================

LAYER_COLORS = ["#2563eb", "#0891b2", "#059669", "#d97706", "#dc2626"]


def plot_phi_heatmap(
    ax,
    layerwise_kernel_matrix,
    neuron_counts,
    title_suffix="",
    color_limit=None,
):
    if color_limit is None:
        color_limit = np.max(np.abs(layerwise_kernel_matrix))
    color_limit = max(color_limit, 1e-12)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rbu", [(1, 0.22, 0.22), (1, 1, 1), (0.22, 0.22, 1)]
    )
    heatmap_image = ax.imshow(
        layerwise_kernel_matrix,
        cmap=cmap,
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="nearest",
        aspect="equal",
    )

    boundary_offset = 0
    for layer_index in range(len(neuron_counts) - 1):
        boundary_offset += neuron_counts[layer_index]
        ax.axhline(boundary_offset - 0.5, color="k", linewidth=0.5, alpha=0.4)
        ax.axvline(boundary_offset - 0.5, color="k", linewidth=0.5, alpha=0.4)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        r"raw $\Phi^{(\ell)}_{ik}$" + title_suffix,
        fontsize=9,
        fontweight="bold",
    )
    return heatmap_image


def plot_scatter(
    ax,
    predicted_activity_change,
    actual_activity_change,
    neuron_counts,
    prediction_label,
    panel_title,
):
    max_predicted_magnitude = max(np.max(np.abs(predicted_activity_change)), 1e-10)
    max_actual_magnitude = max(np.max(np.abs(actual_activity_change)), 1e-10)
    x_limit = max_predicted_magnitude * 1.15
    y_limit = max_actual_magnitude * 1.15

    # Fit line through origin for correlation display
    ax.axhline(0, color="#ddd8cc", linewidth=0.3)
    ax.axvline(0, color="#ddd8cc", linewidth=0.3)

    # Best-fit line through origin
    squared_prediction_norm = np.dot(
        predicted_activity_change, predicted_activity_change
    )
    if squared_prediction_norm > 1e-30:
        best_fit_slope = (
            np.dot(predicted_activity_change, actual_activity_change)
            / squared_prediction_norm
        )
        ax.plot(
            [-x_limit, x_limit],
            [-x_limit * best_fit_slope, x_limit * best_fit_slope],
            "--",
            color="#b0a890",
            linewidth=1,
            zorder=1,
        )

    neuron_offset = 0
    for layer_index, layer_neuron_count in enumerate(neuron_counts):
        ax.scatter(
            predicted_activity_change[
                neuron_offset : neuron_offset + layer_neuron_count
            ],
            actual_activity_change[neuron_offset : neuron_offset + layer_neuron_count],
            c=LAYER_COLORS[layer_index % len(LAYER_COLORS)],
            s=8,
            alpha=0.6,
            edgecolors="none",
            zorder=2,
            label=f"L{layer_index + 1} ({layer_neuron_count})",
        )
        neuron_offset += layer_neuron_count

    correlation_value = corr(actual_activity_change, predicted_activity_change)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel(f"predicted ({prediction_label})", fontsize=6)
    ax.set_ylabel(r"actual $\Delta A$", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.set_title(
        f"{panel_title}\n$r = {correlation_value:.3f}$",
        fontsize=7,
        fontweight="bold",
        pad=3,
    )


def plot_dynamics(ax, history, show_loss_label=True):
    diagnostic_steps = history["step"]
    # The dynamics panel tracks the diagonal-approximation metric directly:
    # corr_diagonal = r(actual ΔA, Eq. 5 prediction).
    ax.plot(
        diagnostic_steps,
        history["corr_diagonal"],
        "-",
        color="#d97706",
        linewidth=1.2,
        label=r"$r(\Delta A,\;-\Phi_{ii}\,\partial \mathcal{L}/\partial A_i)$",
    )

    ax2 = ax.twinx()
    ax2.plot(
        diagnostic_steps,
        history["loss"],
        "--",
        color="#d44a",
        linewidth=0.8,
        label="loss",
    )
    if show_loss_label:
        ax2.set_ylabel("loss", fontsize=7, color="#d44a")
        ax2.tick_params(labelsize=5, colors="#d44a")
    else:
        ax2.set_yticklabels([])
        ax2.tick_params(right=False)

    ax.axhline(1, color="#e8e5dd", linestyle="--", linewidth=0.5)
    ax.axhline(0, color="#e8e5dd", linewidth=0.5)
    ax.set_ylim(-0.5, 1.1)
    ax.set_xlabel("SGD step", fontsize=7)
    ax.set_ylabel(r"$r$", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, loc="lower left", framealpha=0.8)


# ======================== MAIN ========================

parser = argparse.ArgumentParser(
    description="Run the main activity-kernel simulations and render all figures."
)
parser.add_argument(
    "--output-dir",
    required=True,
    type=Path,
    help="Directory for publication-ready PDF figures and PNG previews.",
)
args = parser.parse_args()
output_dir = args.output_dir.resolve()
output_dir.mkdir(parents=True, exist_ok=True)
os.chdir(output_dir)

print("Running width=8 experiment...")
history_width_8, snapshot_width_8, _ = run_experiment(
    width=8, depth=3, eta=0.005, n_steps=3000, diag_every=30
)

print("Running width=48 experiment...")
history_width_48, snapshot_width_48, _ = run_experiment(
    width=48, depth=3, eta=0.005, n_steps=3000, diag_every=30
)

# ---- FIGURE: 2-row comparison ----
figure_one = plt.figure(figsize=(7.0, 6.2), dpi=200)
figure_one.patch.set_facecolor("#faf9f6")

grid_spec = GridSpec(
    3,
    4,
    figure=figure_one,
    hspace=0.65,
    wspace=0.55,
    left=0.07,
    right=0.97,
    top=0.95,
    bottom=0.07,
)

# Row 1: width=8
heatmap_ax_width_8 = figure_one.add_subplot(grid_spec[0, 0])
heatmap_image_width_8 = plot_phi_heatmap(
    heatmap_ax_width_8,
    snapshot_width_8["Phi"],
    snapshot_width_8["neuron_counts"],
    " (width=8)",
)
phi_colorbar_width_8 = figure_one.colorbar(
    heatmap_image_width_8,
    ax=heatmap_ax_width_8,
    orientation="horizontal",
    fraction=0.08,
    pad=0.12,
)
phi_colorbar_width_8.set_label(r"$\Phi^{(\ell)}_{ik}$", fontsize=6)
phi_colorbar_width_8.ax.tick_params(labelsize=5)

kernel_scatter_ax_width_8 = figure_one.add_subplot(grid_spec[0, 1])
plot_scatter(
    kernel_scatter_ax_width_8,
    snapshot_width_8["kernel_prediction"],
    snapshot_width_8["actual_activity_change"],
    snapshot_width_8["neuron_counts"],
    r"$\Phi\cdot\nabla \mathcal{L}$",
    "Eq. 3 (kernel)",
)

diagonal_scatter_ax_width_8 = figure_one.add_subplot(grid_spec[0, 2])
plot_scatter(
    diagonal_scatter_ax_width_8,
    snapshot_width_8["diagonal_prediction"],
    snapshot_width_8["actual_activity_change"],
    snapshot_width_8["neuron_counts"],
    r"$-\Phi_{ii}\,\partial \mathcal{L}/\partial A_i$",
    r"$-\Phi_{ii}\,\partial \mathcal{L}/\partial A_i$",
)

raw_gradient_scatter_ax_width_8 = figure_one.add_subplot(grid_spec[0, 3])
plot_scatter(
    raw_gradient_scatter_ax_width_8,
    snapshot_width_8["raw_negative_gradient"],
    snapshot_width_8["actual_activity_change"],
    snapshot_width_8["neuron_counts"],
    r"$-\partial \mathcal{L}/\partial A$",
    r"$-d\mathcal{L}/dA$ (raw)",
)
raw_gradient_scatter_ax_width_8.legend(
    fontsize=4.5, loc="lower right", framealpha=0.8, markerscale=0.8
)

# Row 2: width=48
heatmap_ax_width_48 = figure_one.add_subplot(grid_spec[1, 0])
heatmap_image_width_48 = plot_phi_heatmap(
    heatmap_ax_width_48,
    snapshot_width_48["Phi"],
    snapshot_width_48["neuron_counts"],
    " (width=48)",
)
phi_colorbar_width_48 = figure_one.colorbar(
    heatmap_image_width_48,
    ax=heatmap_ax_width_48,
    orientation="horizontal",
    fraction=0.08,
    pad=0.12,
)
phi_colorbar_width_48.set_label(r"$\Phi^{(\ell)}_{ik}$", fontsize=6)
phi_colorbar_width_48.ax.tick_params(labelsize=5)

kernel_scatter_ax_width_48 = figure_one.add_subplot(grid_spec[1, 1])
plot_scatter(
    kernel_scatter_ax_width_48,
    snapshot_width_48["kernel_prediction"],
    snapshot_width_48["actual_activity_change"],
    snapshot_width_48["neuron_counts"],
    r"$\Phi\cdot\nabla \mathcal{L}$",
    "Eq. 3 (kernel)",
)

diagonal_scatter_ax_width_48 = figure_one.add_subplot(grid_spec[1, 2])
plot_scatter(
    diagonal_scatter_ax_width_48,
    snapshot_width_48["diagonal_prediction"],
    snapshot_width_48["actual_activity_change"],
    snapshot_width_48["neuron_counts"],
    r"$-\Phi_{ii}\,\partial \mathcal{L}/\partial A_i$",
    r"$-\Phi_{ii}\,\partial \mathcal{L}/\partial A_i$",
)

raw_gradient_scatter_ax_width_48 = figure_one.add_subplot(grid_spec[1, 3])
plot_scatter(
    raw_gradient_scatter_ax_width_48,
    snapshot_width_48["raw_negative_gradient"],
    snapshot_width_48["actual_activity_change"],
    snapshot_width_48["neuron_counts"],
    r"$-\partial \mathcal{L}/\partial A$",
    r"$-d\mathcal{L}/dA$ (raw)",
)

# Row 3: Training dynamics side by side
dynamics_ax_width_8 = figure_one.add_subplot(grid_spec[2, :2])
plot_dynamics(dynamics_ax_width_8, history_width_8, show_loss_label=False)
dynamics_ax_width_8.set_title(
    "Training dynamics (width=8)", fontsize=8, fontweight="bold"
)

dynamics_ax_width_48 = figure_one.add_subplot(grid_spec[2, 2:])
plot_dynamics(dynamics_ax_width_48, history_width_48, show_loss_label=True)
dynamics_ax_width_48.set_title(
    "Training dynamics (width=48)", fontsize=8, fontweight="bold"
)

# Save into the current project directory instead of an author-local path.
plt.savefig("fig_ntk.pdf", bbox_inches="tight", facecolor="#faf9f6")
plt.savefig("fig_ntk.png", bbox_inches="tight", facecolor="#faf9f6")
print("Figure 1 saved.")

# ======================== WIDTH SWEEP ========================

widths = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
n_seeds = 3
n_steps_sweep = 2000
diag_every_sweep = 40

# For each width, collect summary statistics of the diagonal-approximation
# correlation over training.
width_diagonal_corr_median = {width_value: [] for width_value in widths}
width_diagonal_corr_final = {width_value: [] for width_value in widths}

for width_value in widths:
    print(f"  Width sweep: w={width_value} ...", flush=True)
    for seed in range(n_seeds):
        np.random.seed(1000 + seed * 100 + width_value)
        history, _, _ = run_experiment(
            width=width_value,
            depth=3,
            eta=0.005,
            n_steps=n_steps_sweep,
            diag_every=diag_every_sweep,
            full_jacobian=False,
        )
        # If the diagonal approximation itself improves with width, this is the
        # directly relevant metric to inspect.
        diagonal_corr_values = history["corr_diagonal"]
        width_diagonal_corr_median[width_value].append(np.median(diagonal_corr_values))
        width_diagonal_corr_final[width_value].append(
            np.mean(diagonal_corr_values[-5:])
        )

# Compute mean and std across seeds
width_array = np.array(widths)
median_diagonal_corr_mean = np.array(
    [np.mean(width_diagonal_corr_median[width_value]) for width_value in widths]
)
median_diagonal_corr_std = np.array(
    [np.std(width_diagonal_corr_median[width_value]) for width_value in widths]
)
late_diagonal_corr_mean = np.array(
    [np.mean(width_diagonal_corr_final[width_value]) for width_value in widths]
)
late_diagonal_corr_std = np.array(
    [np.std(width_diagonal_corr_final[width_value]) for width_value in widths]
)

width_sweep_figure, width_sweep_ax = plt.subplots(figsize=(3.8, 2.8), dpi=200)
width_sweep_figure.patch.set_facecolor("#faf9f6")

width_sweep_ax.fill_between(
    width_array,
    late_diagonal_corr_mean - late_diagonal_corr_std,
    late_diagonal_corr_mean + late_diagonal_corr_std,
    color="#d97706",
    alpha=0.15,
)
width_sweep_ax.plot(
    width_array,
    late_diagonal_corr_mean,
    "o-",
    color="#d97706",
    linewidth=1.5,
    markersize=4,
    label=r"late training $r$",
)

width_sweep_ax.fill_between(
    width_array,
    median_diagonal_corr_mean - median_diagonal_corr_std,
    median_diagonal_corr_mean + median_diagonal_corr_std,
    color="#2563eb",
    alpha=0.15,
)
width_sweep_ax.plot(
    width_array,
    median_diagonal_corr_mean,
    "s--",
    color="#2563eb",
    linewidth=1.2,
    markersize=3.5,
    label=r"median $r$",
)

width_sweep_ax.axhline(1, color="#e8e5dd", linestyle="--", linewidth=0.5)
width_sweep_ax.axhline(0, color="#e8e5dd", linewidth=0.5)
width_sweep_ax.set_xlabel("hidden layer width", fontsize=8)
width_sweep_ax.set_ylabel(
    r"$r(\Delta A,\;-\Phi_{ii}\,\partial \mathcal{L}/\partial A_i)$", fontsize=8
)
width_sweep_ax.tick_params(labelsize=7)
width_sweep_ax.legend(fontsize=7, loc="lower right", framealpha=0.8)
width_sweep_ax.set_ylim(0.5, 1.05)
width_sweep_ax.set_xscale("log", base=2)
width_sweep_ax.set_xticks([4, 8, 16, 32, 64, 128, 256, 512])
width_sweep_ax.set_xticklabels(["4", "8", "16", "32", "64", "128", "256", "512"])
width_sweep_ax.set_xlim(widths[0] / np.sqrt(2), widths[-1] * np.sqrt(2))

# Save into the current project directory instead of an author-local path.
plt.savefig("fig_width_sweep.pdf", bbox_inches="tight", facecolor="#faf9f6")
plt.savefig("fig_width_sweep.png", bbox_inches="tight", facecolor="#faf9f6")
print("Figure 2 (width sweep) saved.")

# ======================== DEPTH SWEEP ========================
# Tests the appendix prediction: under He init, r(ΔA, -∂ℒ/∂A) is approximately
# depth-independent (~√3/2), while r(ΔA, -Φ_ii ∂ℒ/∂A) approaches 1 with depth.

depths = [2, 3, 4, 5, 6, 7, 8]
depth_width = 32
n_seeds_d = 3
n_steps_depth = 1500
diag_every_depth = 40

depth_neg_init = {d: [] for d in depths}
depth_diag_init = {d: [] for d in depths}
depth_neg_late = {d: [] for d in depths}
depth_diag_late = {d: [] for d in depths}

for d in depths:
    print(f"  Depth sweep: d={d} ...", flush=True)
    for seed in range(n_seeds_d):
        np.random.seed(5000 + seed * 100 + d)
        h, _, _ = run_experiment(
            width=depth_width,
            depth=d,
            eta=0.005,
            n_steps=n_steps_depth,
            diag_every=diag_every_depth,
            full_jacobian=False,
        )
        neg_vals = h["corr_raw_gradient"]
        diag_vals = h["corr_diagonal"]
        # Init/early-training values (first 3 diagnostics) test the analytical prediction
        depth_neg_init[d].append(np.mean(neg_vals[:3]))
        depth_diag_init[d].append(np.mean(diag_vals[:3]))
        # Late training values
        depth_neg_late[d].append(np.mean(neg_vals[-5:]))
        depth_diag_late[d].append(np.mean(diag_vals[-5:]))

d_arr = np.array(depths)
neg_init_mean = np.array([np.mean(depth_neg_init[d]) for d in depths])
neg_init_std = np.array([np.std(depth_neg_init[d]) for d in depths])
diag_init_mean = np.array([np.mean(depth_diag_init[d]) for d in depths])
diag_init_std = np.array([np.std(depth_diag_init[d]) for d in depths])
neg_late_mean = np.array([np.mean(depth_neg_late[d]) for d in depths])
neg_late_std = np.array([np.std(depth_neg_late[d]) for d in depths])
diag_late_mean = np.array([np.mean(depth_diag_late[d]) for d in depths])
diag_late_std = np.array([np.std(depth_diag_late[d]) for d in depths])

fig3, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=200, sharey=True)
fig3.patch.set_facecolor("#faf9f6")


def plot_depth_panel(ax, neg_mean, neg_std, diag_mean, diag_std, title):
    ax.fill_between(
        d_arr, neg_mean - neg_std, neg_mean + neg_std, color="#d97706", alpha=0.15
    )
    ax.plot(
        d_arr,
        neg_mean,
        "o-",
        color="#d97706",
        linewidth=1.5,
        markersize=4,
        label=r"$r(\Delta A,\;-\partial \mathcal{L}/\partial A)$ (raw)",
    )
    ax.fill_between(
        d_arr, diag_mean - diag_std, diag_mean + diag_std, color="#059669", alpha=0.15
    )
    ax.plot(
        d_arr,
        diag_mean,
        "s-",
        color="#059669",
        linewidth=1.5,
        markersize=4,
        label=r"$r(\Delta A,\;-\Phi_{ii}\,\partial \mathcal{L}/\partial A)$ (scaled)",
    )
    # Leading-order raw-gradient prediction with Phi_ii ~ rho + ell - 1,
    # rho = d/n = 16/32 = 0.5 for the depth sweep (d=16 input, n=32 width).
    d_fine = np.linspace(d_arr[0], d_arr[-1], 200)
    rho_depth = 16.0 / depth_width
    E_c = rho_depth + (d_fine - 1) / 2
    E_c2 = rho_depth**2 + rho_depth * (d_fine - 1) + (d_fine - 1) * (2 * d_fine - 1) / 6
    r_pred = E_c / np.sqrt(E_c2)
    ax.plot(d_fine, r_pred, color="#d97706", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(
        d_arr[-1] - 0.1,
        r_pred[-1] - 0.05,
        rf"raw pred. ($\rho{{=}}d/n{{=}}{rho_depth:g}$)",
        fontsize=7,
        color="#a45a04",
        ha="right",
    )
    # Scaled-gradient prediction is r = 1 under the diagonal approximation.
    ax.axhline(1.0, color="#059669", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(
        d_arr[-1] - 0.1,
        1.0 - 0.04,
        "scaled pred. = 1",
        fontsize=7,
        color="#047857",
        ha="right",
    )
    ax.axhline(0, color="#e8e5dd", linewidth=0.5)
    ax.set_xlabel("depth (number of hidden layers)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(depths[0] - 0.3, depths[-1] + 0.3)
    ax.set_title(title, fontsize=9, fontweight="bold")


plot_depth_panel(
    axes[0],
    neg_init_mean,
    neg_init_std,
    diag_init_mean,
    diag_init_std,
    "At initialisation (theory regime)",
)
plot_depth_panel(
    axes[1],
    neg_late_mean,
    neg_late_std,
    diag_late_mean,
    diag_late_std,
    "After training (1500 SGD steps)",
)
axes[0].set_ylabel("Pearson $r$", fontsize=8)
axes[1].legend(fontsize=6.5, loc="lower right", framealpha=0.8)

plt.savefig("fig_depth_sweep.pdf", bbox_inches="tight", facecolor="#faf9f6")
plt.savefig("fig_depth_sweep.png", bbox_inches="tight", facecolor="#faf9f6")
print("Figure 3 (depth sweep) saved.")

# ======================== APPENDIX WIDTH SWEEP ========================
# Companion to the depth sweep: holds depth fixed at D=3 and sweeps width,
# reporting raw and diagonally-scaled correlations at initialisation and after
# training. Tests the finite-width assumption underlying the leading-order
# r ~ sqrt(3(D+1)/(2(2D+1))) prediction.

appendix_widths = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
appendix_depth = 3
n_seeds_w = 3
n_steps_width = 1500
diag_every_width = 40

width_neg_init = {w: [] for w in appendix_widths}
width_diag_init = {w: [] for w in appendix_widths}
width_neg_late = {w: [] for w in appendix_widths}
width_diag_late = {w: [] for w in appendix_widths}

for w in appendix_widths:
    print(f"  Appendix width sweep: w={w} ...", flush=True)
    for seed in range(n_seeds_w):
        np.random.seed(7000 + seed * 100 + w)
        h, _, _ = run_experiment(
            width=w,
            depth=appendix_depth,
            eta=0.005,
            n_steps=n_steps_width,
            diag_every=diag_every_width,
            full_jacobian=False,
        )
        neg_vals = h["corr_raw_gradient"]
        diag_vals = h["corr_diagonal"]
        width_neg_init[w].append(np.mean(neg_vals[:3]))
        width_diag_init[w].append(np.mean(diag_vals[:3]))
        width_neg_late[w].append(np.mean(neg_vals[-5:]))
        width_diag_late[w].append(np.mean(diag_vals[-5:]))

w_arr = np.array(appendix_widths)
w_neg_init_mean = np.array([np.mean(width_neg_init[w]) for w in appendix_widths])
w_neg_init_std = np.array([np.std(width_neg_init[w]) for w in appendix_widths])
w_diag_init_mean = np.array([np.mean(width_diag_init[w]) for w in appendix_widths])
w_diag_init_std = np.array([np.std(width_diag_init[w]) for w in appendix_widths])
w_neg_late_mean = np.array([np.mean(width_neg_late[w]) for w in appendix_widths])
w_neg_late_std = np.array([np.std(width_neg_late[w]) for w in appendix_widths])
w_diag_late_mean = np.array([np.mean(width_diag_late[w]) for w in appendix_widths])
w_diag_late_std = np.array([np.std(width_diag_late[w]) for w in appendix_widths])

# Leading-order prediction for the raw correlation at fixed depth D=3:
#   Phi_ii^(ell) ~ d + (ell-1)*n  (own-layer contribution is O(d) at layer 1,
#   O(n) at layers >= 2), so up to a common scale c_ell ~ rho + ell - 1 with
#   rho = d/n. Pooling uniformly over layers gives
#     r ~ E[c] / sqrt(E[c^2])
#       = (rho + (D-1)/2) / sqrt(rho^2 + rho(D-1) + (D-1)(2D-1)/6).
D_app = appendix_depth
INPUT_DIM = 16  # 4x4 bar images


def r_pred_raw(D, rho):
    E_c = rho + (D - 1) / 2.0
    E_c2 = rho**2 + rho * (D - 1) + (D - 1) * (2 * D - 1) / 6.0
    return E_c / np.sqrt(E_c2)


# Smooth curve over the swept widths.
n_fine = np.logspace(np.log10(appendix_widths[0]), np.log10(appendix_widths[-1]), 200)
r_pred_curve = r_pred_raw(D_app, INPUT_DIM / n_fine)
r_pred_wide_limit = r_pred_raw(D_app, 0.0)  # n -> infinity at fixed d

fig4, axesW = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=200, sharey=True)
fig4.patch.set_facecolor("#faf9f6")


def plot_width_panel(ax, neg_mean, neg_std, diag_mean, diag_std, title):
    ax.fill_between(
        w_arr, neg_mean - neg_std, neg_mean + neg_std, color="#d97706", alpha=0.15
    )
    ax.plot(
        w_arr,
        neg_mean,
        "o-",
        color="#d97706",
        linewidth=1.5,
        markersize=4,
        label=r"$r(\Delta A,\;-\partial \mathcal{L}/\partial A)$ (raw)",
    )
    ax.fill_between(
        w_arr, diag_mean - diag_std, diag_mean + diag_std, color="#059669", alpha=0.15
    )
    ax.plot(
        w_arr,
        diag_mean,
        "s-",
        color="#059669",
        linewidth=1.5,
        markersize=4,
        label=r"$r(\Delta A,\;-\Phi_{ii}\,\partial \mathcal{L}/\partial A)$ (scaled)",
    )
    # Raw-gradient prediction: r(D=3, rho=d/n) — decreases as n grows.
    ax.plot(
        n_fine, r_pred_curve, color="#d97706", linestyle=":", linewidth=1.2, alpha=0.7
    )
    ax.text(
        w_arr[-1],
        r_pred_curve[-1] - 0.06,
        rf"raw pred. ($\rho{{=}}d/n$, $n{{\to}}\infty$: $\sqrt{{3/5}}{{\approx}}{r_pred_wide_limit:.3f}$)",
        fontsize=7,
        color="#a45a04",
        ha="right",
    )
    # Scaled-gradient prediction is r = 1 under the diagonal approximation.
    ax.axhline(1.0, color="#059669", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(
        w_arr[-1],
        1.0 - 0.04,
        "scaled pred. = 1",
        fontsize=7,
        color="#047857",
        ha="right",
    )
    ax.axhline(0, color="#e8e5dd", linewidth=0.5)
    ax.set_xlabel("hidden layer width $n$", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(w_arr)
    ax.set_xticklabels([str(w) for w in appendix_widths], fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=9, fontweight="bold")


plot_width_panel(
    axesW[0],
    w_neg_init_mean,
    w_neg_init_std,
    w_diag_init_mean,
    w_diag_init_std,
    "At initialisation (theory regime)",
)
plot_width_panel(
    axesW[1],
    w_neg_late_mean,
    w_neg_late_std,
    w_diag_late_mean,
    w_diag_late_std,
    "After training (1500 SGD steps)",
)
axesW[0].set_ylabel("Pearson $r$", fontsize=8)
axesW[1].legend(fontsize=6.5, loc="lower right", framealpha=0.8)

plt.savefig("fig_width_sweep_appendix.pdf", bbox_inches="tight", facecolor="#faf9f6")
plt.savefig("fig_width_sweep_appendix.png", bbox_inches="tight", facecolor="#faf9f6")
print("Figure 4 (appendix width sweep) saved.")
