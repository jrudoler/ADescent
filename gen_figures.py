"""
Generate figures for the activity-space NTK paper.
Same computation as the interactive demo, frozen into publication-quality plots.
"""

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
      diagonal_prediction: diagonal approximation from Eq. 5, i.e. -η Φ_ii dL/dA_i
      raw_negative_gradient: raw -dL/dA baseline (with dead hidden ReLUs masked out)
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


def run_experiment(width, depth, eta=0.005, n_steps=2000, diag_every=50):
    """
    Train one random MLP with online SGD and periodically compare the observed
    single-step activity change against several predictions.

    Most SGD steps only update weights. Every `diag_every` steps we also:
      1. sample one training example,
      2. compute J, Φ, and the predicted ΔA for that sample,
      3. apply the real SGD step on that same sample,
      4. measure the resulting activity change ΔA on that same sample.

    The returned history contains:
      corr_exact: r(actual ΔA, JΔW)
      corr_kernel: r(actual ΔA, full kernel Eq. 3)
      corr_diagonal: r(actual ΔA, diagonal Eq. 5)
      corr_raw_gradient: r(actual ΔA, raw -dL/dA)
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
            exact_prediction = exact_prediction[active_neuron_indices]
            kernel_prediction = kernel_prediction[active_neuron_indices]
            diagonal_prediction = diagonal_prediction[active_neuron_indices]
            raw_negative_gradient = raw_negative_gradient[active_neuron_indices]

            # These are the core diagnostics:
            #   corr_exact ~= 1 checks the Jacobian bookkeeping
            #   corr_kernel ~= 1 checks the exact kernel recursion
            #   corr_diagonal measures how good Eq. 5 is
            #   corr_raw_gradient measures alignment with the raw activity
            #     gradient only and is stricter because Eq. 5 still allows
            #     neuron-specific Φ_ii
            exact_correlation = corr(actual_activity_change, exact_prediction)
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


def plot_phi_heatmap(ax, layerwise_kernel_matrix, neuron_counts, title_suffix=""):
    total_neurons = layerwise_kernel_matrix.shape[0]
    diagonal_norm = np.sqrt(np.diag(layerwise_kernel_matrix))
    diagonal_norm[diagonal_norm == 0] = 1
    correlation_matrix = layerwise_kernel_matrix / np.outer(
        diagonal_norm, diagonal_norm
    )
    correlation_matrix = np.clip(correlation_matrix, -1, 1)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rbu", [(1, 0.22, 0.22), (1, 1, 1), (0.22, 0.22, 1)]
    )
    ax.imshow(
        correlation_matrix,
        cmap=cmap,
        vmin=-1,
        vmax=1,
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
        r"$\Phi^{(\ell)}_{ik}$ correlation" + title_suffix,
        fontsize=9,
        fontweight="bold",
    )


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
        label=r"$r(\Delta A,\;-\Phi_{ii}\,\partial L/\partial A_i)$",
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
plot_phi_heatmap(
    heatmap_ax_width_8,
    snapshot_width_8["Phi"],
    snapshot_width_8["neuron_counts"],
    " (width=8)",
)

kernel_scatter_ax_width_8 = figure_one.add_subplot(grid_spec[0, 1])
plot_scatter(
    kernel_scatter_ax_width_8,
    snapshot_width_8["kernel_prediction"],
    snapshot_width_8["actual_activity_change"],
    snapshot_width_8["neuron_counts"],
    r"$\Phi\cdot\nabla L$",
    "Eq. 3 (kernel)",
)

diagonal_scatter_ax_width_8 = figure_one.add_subplot(grid_spec[0, 2])
plot_scatter(
    diagonal_scatter_ax_width_8,
    snapshot_width_8["diagonal_prediction"],
    snapshot_width_8["actual_activity_change"],
    snapshot_width_8["neuron_counts"],
    r"$-\Phi_{ii}\,\partial L/\partial A_i$",
    r"$-\Phi_{ii}\,\partial L/\partial A_i$",
)

raw_gradient_scatter_ax_width_8 = figure_one.add_subplot(grid_spec[0, 3])
plot_scatter(
    raw_gradient_scatter_ax_width_8,
    snapshot_width_8["raw_negative_gradient"],
    snapshot_width_8["actual_activity_change"],
    snapshot_width_8["neuron_counts"],
    r"$-\partial L/\partial A$",
    r"$-dL/dA$ (raw)",
)
raw_gradient_scatter_ax_width_8.legend(
    fontsize=4.5, loc="lower right", framealpha=0.8, markerscale=0.8
)

# Row 2: width=48
heatmap_ax_width_48 = figure_one.add_subplot(grid_spec[1, 0])
plot_phi_heatmap(
    heatmap_ax_width_48,
    snapshot_width_48["Phi"],
    snapshot_width_48["neuron_counts"],
    " (width=48)",
)

kernel_scatter_ax_width_48 = figure_one.add_subplot(grid_spec[1, 1])
plot_scatter(
    kernel_scatter_ax_width_48,
    snapshot_width_48["kernel_prediction"],
    snapshot_width_48["actual_activity_change"],
    snapshot_width_48["neuron_counts"],
    r"$\Phi\cdot\nabla L$",
    "Eq. 3 (kernel)",
)

diagonal_scatter_ax_width_48 = figure_one.add_subplot(grid_spec[1, 2])
plot_scatter(
    diagonal_scatter_ax_width_48,
    snapshot_width_48["diagonal_prediction"],
    snapshot_width_48["actual_activity_change"],
    snapshot_width_48["neuron_counts"],
    r"$-\Phi_{ii}\,\partial L/\partial A_i$",
    r"$-\Phi_{ii}\,\partial L/\partial A_i$",
)

raw_gradient_scatter_ax_width_48 = figure_one.add_subplot(grid_spec[1, 3])
plot_scatter(
    raw_gradient_scatter_ax_width_48,
    snapshot_width_48["raw_negative_gradient"],
    snapshot_width_48["actual_activity_change"],
    snapshot_width_48["neuron_counts"],
    r"$-\partial L/\partial A$",
    r"$-dL/dA$ (raw)",
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

widths = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
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
    r"$r(\Delta A,\;-\Phi_{ii}\,\partial L/\partial A_i)$", fontsize=8
)
width_sweep_ax.tick_params(labelsize=7)
width_sweep_ax.legend(fontsize=7, loc="lower right", framealpha=0.8)
width_sweep_ax.set_ylim(0.5, 1.05)
width_sweep_ax.set_xlim(0, widths[-1] + 4)

# Save into the current project directory instead of an author-local path.
plt.savefig("fig_width_sweep.pdf", bbox_inches="tight", facecolor="#faf9f6")
plt.savefig("fig_width_sweep.png", bbox_inches="tight", facecolor="#faf9f6")
print("Figure 2 (width sweep) saved.")
