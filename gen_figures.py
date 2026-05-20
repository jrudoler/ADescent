"""
Generate figures for the activity-space NTK paper.
Same computation as the interactive demo, frozen into publication-quality plots.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

np.random.seed(42)

# ======================== NETWORK ========================

def create_network(sizes):
    W = []
    for i in range(len(sizes)-1):
        # Extra column for bias (input augmented with 1)
        scale = np.sqrt(2.0 / sizes[i])
        w = np.random.randn(sizes[i+1], sizes[i] + 1) * scale
        w[:, -1] = 0  # init biases to zero
        W.append(w)
    return sizes, W

def forward(sizes, W, x):
    a = [x.copy()]
    h = [None]
    for l in range(len(W)):
        a_aug = np.append(a[-1], 1.0)  # append 1 for bias
        hl = W[l] @ a_aug
        h.append(hl.copy())
        if l < len(W) - 1:
            a.append(np.maximum(0, hl))
        else:
            a.append(hl.copy())
    return a, h

def backprop(sizes, W, a, h, target):
    L = len(W)
    dLdA = [None] * (L + 1)
    dLdA[L] = a[L] - target
    dLdW = [None] * L
    for l in range(L-1, -1, -1):
        if l == L - 1:
            delta = dLdA[l+1].copy()
        else:
            delta = dLdA[l+1] * (h[l+1] > 0).astype(float)
        a_aug = np.append(a[l], 1.0)
        dLdW[l] = np.outer(delta, a_aug)
        if l > 0:
            dLdA[l] = W[l][:, :-1].T @ delta  # exclude bias column
    return dLdA, dLdW

def make_bar_images(n_per_class=20, noise=0.15):
    X, Y = [], []
    for _ in range(n_per_class):
        # Horizontal bar: random row bright
        h_img = np.zeros(16)
        row = np.random.randint(4)
        h_img[row*4:(row+1)*4] = 1.0
        h_img += np.random.randn(16) * noise
        X.append(h_img)
        Y.append([1, 0])
        # Vertical bar: random column bright
        v_img = np.zeros(16)
        col = np.random.randint(4)
        for r in range(4):
            v_img[r*4 + col] = 1.0
        v_img += np.random.randn(16) * noise
        X.append(v_img)
        Y.append([0, 1])
    return np.array(X), np.array(Y)

# ======================== JACOBIAN ========================

def compute_layer_local_pred(sizes, W, a, h, dLdA, eta):
    """Layer-local kernel prediction: ΔA^(ℓ) = -η D_ℓ ∇L + J_ℓ ΔA^(ℓ-1)"""
    L = len(W)
    totalN = sum(sizes[1:])
    result = np.zeros(totalN)
    prevDeltaA = None
    off = 0
    for l in range(L):
        nin, nout = sizes[l], sizes[l+1]
        # ||a_aug||^2 (includes bias=1)
        normSq = np.sum(a[l]**2) + 1.0
        # sigma'(z_i)
        if l < L - 1:
            fp = (h[l+1] > 0).astype(float)
        else:
            fp = np.ones(nout)
        gradA = dLdA[l+1]
        # Own-layer: -eta * fp^2 * normSq * dL/dA
        deltaA = -eta * fp * fp * normSq * gradA
        # Propagated: fp[i] * sum_j W[l][i,j] * prevDeltaA[j] (weight cols only, no bias)
        if prevDeltaA is not None:
            deltaA += fp * (W[l][:, :-1] @ prevDeltaA)
        result[off:off+nout] = deltaA
        off += nout
        prevDeltaA = deltaA
    return result

def compute_jacobian_and_predictions(sizes, W, a, h, dLdW, dLdA, target, eta):
    L = len(W)
    neuron_counts = sizes[1:]
    totalN = sum(neuron_counts)

    layer_offsets = []
    off = 0
    for l in range(L):
        layer_offsets.append(off)
        off += sizes[l+1]

    # Backprop gradient (total derivative)
    gradA_bp = np.zeros(totalN)
    for l in range(1, L+1):
        if dLdA[l] is not None:
            go = layer_offsets[l-1]
            gradA_bp[go:go+len(dLdA[l])] = dLdA[l]

    # Each layer's weight matrix is nout x (nin+1) due to bias augmentation
    totalParams = sum((sizes[i]+1)*sizes[i+1] for i in range(L))

    deltaW_flat = np.zeros(totalParams)
    poff = 0
    for l in range(L):
        n = dLdW[l].size
        deltaW_flat[poff:poff+n] = -eta * dLdW[l].ravel()
        poff += n

    # Build full Jacobian
    J = np.zeros((totalN, totalParams))
    param_offset = 0
    for m in range(L):
        nin_m, nout_m = sizes[m], sizes[m+1]
        P = nout_m * (nin_m + 1)  # +1 for bias
        a_aug = np.append(a[m], 1.0)

        J_cur = np.zeros((nout_m, P))
        for j in range(nout_m):
            fp = 1.0 if m == L-1 else (1.0 if h[m+1][j] > 0 else 0.0)
            J_cur[j, j*(nin_m+1):(j+1)*(nin_m+1)] = fp * a_aug

        go = layer_offsets[m]
        J[go:go+nout_m, param_offset:param_offset+P] = J_cur

        J_prev = J_cur
        for l in range(m+1, L):
            nin_l, nout_l = sizes[l], sizes[l+1]
            DW = np.zeros((nout_l, nin_l))
            for i in range(nout_l):
                fp = 1.0 if l == L-1 else (1.0 if h[l+1][i] > 0 else 0.0)
                DW[i] = fp * W[l][i, :-1]  # exclude bias column
            J_new = DW @ J_prev
            go2 = layer_offsets[l]
            J[go2:go2+nout_l, param_offset:param_offset+P] = J_new
            J_prev = J_new

        param_offset += P

    groundTruth = J @ deltaW_flat

    theta_diag = np.sum(J**2, axis=1)
    diagPred = -eta * theta_diag * gradA_bp

    # Layer-local kernel prediction
    kernelPred = compute_layer_local_pred(sizes, W, a, h, dLdA, eta)

    # Active mask: hidden neurons active iff pre-activation > 0; output neurons always active
    active = np.zeros(totalN)
    for l in range(1, L+1):
        go = layer_offsets[l-1]
        n = sizes[l]
        if l < L:  # hidden layer
            for i in range(n):
                active[go + i] = 1.0 if h[l][i] > 0 else 0.0
        else:  # output layer
            active[go:go+n] = 1.0
    negGradA = -gradA_bp * active

    # Per-layer kernel Phi (block-diagonal)
    Phi = np.zeros((totalN, totalN))
    for l in range(L):
        go = layer_offsets[l]
        n = sizes[l+1]
        Jl = J[go:go+n, :]
        Phi[go:go+n, go:go+n] = Jl @ Jl.T

    return groundTruth, kernelPred, diagPred, negGradA, Phi, neuron_counts, layer_offsets, active

def corr(actual, pred):
    if len(actual) < 2:
        return 0.0
    a_m = actual - np.mean(actual)
    p_m = pred - np.mean(pred)
    denom = np.sqrt(np.sum(a_m**2) * np.sum(p_m**2))
    return np.sum(a_m * p_m) / denom if denom > 1e-30 else 0.0

# ======================== RUN EXPERIMENT ========================

def run_experiment(width, depth, eta=0.005, n_steps=2000, diag_every=50):
    sizes = [16] + [width]*depth + [2]
    sizes_tuple, W = create_network(sizes)
    X, Y = make_bar_images(20)
    B = len(X)
    
    history = {'step': [], 'loss': [], 'corr_ground': [], 'corr_kernel': [], 'corr_diag': [], 'corr_neg': []}
    last_snap = None

    for step in range(n_steps):
        bi = np.random.randint(B)
        a, h = forward(sizes, W, X[bi])
        dLdA, dLdW = backprop(sizes, W, a, h, Y[bi])

        do_diag = (step % diag_every == 0) or (step == n_steps - 1)

        if do_diag:
            a_before = np.concatenate([a[l] for l in range(1, len(W)+1)])
            gt, kp, dp, ng, Phi, nc, lo, active = compute_jacobian_and_predictions(
                sizes, W, a, h, dLdW, dLdA, Y[bi], eta)

            for l in range(len(W)):
                W[l] -= eta * dLdW[l]

            a2, _ = forward(sizes, W, X[bi])
            a_after = np.concatenate([a2[l] for l in range(1, len(W)+1)])
            deltaA_full = a_after - a_before

            # Filter to active neurons only
            mask = active > 0
            deltaA = deltaA_full[mask]
            gt = gt[mask]; kp = kp[mask]; dp = dp[mask]; ng = ng[mask]

            cg = corr(deltaA, gt)
            ck = corr(deltaA, kp)
            cd = corr(deltaA, dp)
            cn = corr(deltaA, ng)

            loss = 0
            for b in range(B):
                af, _ = forward(sizes, W, X[b])
                loss += 0.5 * np.sum((af[-1] - Y[b])**2) / B

            history['step'].append(step)
            history['loss'].append(loss)
            history['corr_ground'].append(cg)
            history['corr_kernel'].append(ck)
            history['corr_diag'].append(cd)
            history['corr_neg'].append(cn)

            # Filter Phi and neuron counts for snapshot
            active_idx = np.where(mask)[0]
            Phi_filt = Phi[np.ix_(active_idx, active_idx)]
            nc_filt = []
            idx = 0
            for count in nc:
                nc_filt.append(int(np.sum(mask[idx:idx+count])))
                idx += count

            last_snap = {
                'deltaA': deltaA, 'gt': gt, 'kp': kp, 'dp': dp, 'ng': ng,
                'Phi': Phi_filt, 'nc': nc_filt, 'lo': lo, 'step': step
            }
        else:
            for l in range(len(W)):
                W[l] -= eta * dLdW[l]

    return history, last_snap, sizes

# ======================== PLOTTING ========================

LAYER_COLORS = ['#2563eb', '#0891b2', '#059669', '#d97706', '#dc2626']

def plot_theta_heatmap(ax, Theta, neuron_counts, title_suffix=''):
    N = Theta.shape[0]
    diag = np.sqrt(np.diag(Theta))
    diag[diag == 0] = 1
    corr = Theta / np.outer(diag, diag)
    corr = np.clip(corr, -1, 1)
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'rbu', [(1, 0.22, 0.22), (1, 1, 1), (0.22, 0.22, 1)])
    ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest', aspect='equal')
    
    acc = 0
    for l in range(len(neuron_counts)-1):
        acc += neuron_counts[l]
        ax.axhline(acc-0.5, color='k', linewidth=0.5, alpha=0.4)
        ax.axvline(acc-0.5, color='k', linewidth=0.5, alpha=0.4)
    
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r'$\Phi^{(\ell)}_{ik}$ correlation' + title_suffix, fontsize=9, fontweight='bold')

def plot_scatter(ax, pred, actual, neuron_counts, label, eq_label):
    xmax = max(np.max(np.abs(pred)), 1e-10) * 1.15
    ymax = max(np.max(np.abs(actual)), 1e-10) * 1.15

    # Fit line through origin for correlation display
    ax.axhline(0, color='#ddd8cc', linewidth=0.3)
    ax.axvline(0, color='#ddd8cc', linewidth=0.3)

    # Best-fit line through origin
    denom = np.dot(pred, pred)
    if denom > 1e-30:
        slope = np.dot(pred, actual) / denom
        ax.plot([-xmax, xmax], [-xmax * slope, xmax * slope], '--', color='#b0a890', linewidth=1, zorder=1)

    idx = 0
    for l, nc in enumerate(neuron_counts):
        ax.scatter(pred[idx:idx+nc], actual[idx:idx+nc],
                   c=LAYER_COLORS[l % len(LAYER_COLORS)], s=8, alpha=0.6,
                   edgecolors='none', zorder=2, label=f'L{l+1} ({nc})')
        idx += nc

    rv = corr(actual, pred)
    ax.set_xlim(-xmax, xmax); ax.set_ylim(-ymax, ymax)
    ax.set_xlabel(f'predicted ({label})', fontsize=6)
    ax.set_ylabel(r'actual $\Delta A$', fontsize=6)
    ax.tick_params(labelsize=5)
    ax.set_title(f'{eq_label}\n$r = {rv:.3f}$', fontsize=7, fontweight='bold', pad=3)

def plot_dynamics(ax, history, show_loss_label=True):
    steps = history['step']
    ax.plot(steps, history['corr_neg'], '-', color='#d97706', linewidth=1.2, label=r'$r(\Delta A,\;-dL/dA)$')

    ax2 = ax.twinx()
    ax2.plot(steps, history['loss'], '--', color='#d44a', linewidth=0.8, label='loss')
    if show_loss_label:
        ax2.set_ylabel('loss', fontsize=7, color='#d44a')
        ax2.tick_params(labelsize=5, colors='#d44a')
    else:
        ax2.set_yticklabels([])
        ax2.tick_params(right=False)

    ax.axhline(1, color='#e8e5dd', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='#e8e5dd', linewidth=0.5)
    ax.set_ylim(-0.5, 1.1)
    ax.set_xlabel('SGD step', fontsize=7)
    ax.set_ylabel(r'$r$', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, loc='lower left', framealpha=0.8)

# ======================== MAIN ========================

print("Running width=8 experiment...")
h8, snap8, sizes8 = run_experiment(width=8, depth=3, eta=0.005, n_steps=3000, diag_every=30)

print("Running width=48 experiment...")
h48, snap48, sizes48 = run_experiment(width=48, depth=3, eta=0.005, n_steps=3000, diag_every=30)

# ---- FIGURE: 2-row comparison ----
fig = plt.figure(figsize=(7.0, 6.2), dpi=200)
fig.patch.set_facecolor('#faf9f6')

gs = GridSpec(3, 4, figure=fig, hspace=0.65, wspace=0.55,
              left=0.07, right=0.97, top=0.95, bottom=0.07)


# Row 1: width=8
ax1 = fig.add_subplot(gs[0, 0])
plot_theta_heatmap(ax1, snap8['Phi'], snap8['nc'], ' (width=8)')

ax3 = fig.add_subplot(gs[0, 1])
plot_scatter(ax3, snap8['kp'], snap8['deltaA'], snap8['nc'],
             r'$\Phi\cdot\nabla L$', 'Eq. 3 (kernel)')

ax4 = fig.add_subplot(gs[0, 2])
plot_scatter(ax4, snap8['dp'], snap8['deltaA'], snap8['nc'],
             r'diag $\Phi$', 'Eq. 5 (diagonal)')

ax4b = fig.add_subplot(gs[0, 3])
plot_scatter(ax4b, snap8['ng'], snap8['deltaA'], snap8['nc'],
             r'$-\partial L/\partial A$', r'$-dL/dA$ (raw)')
ax4b.legend(fontsize=4.5, loc='lower right', framealpha=0.8, markerscale=0.8)

# Row 2: width=48
ax5 = fig.add_subplot(gs[1, 0])
plot_theta_heatmap(ax5, snap48['Phi'], snap48['nc'], ' (width=48)')

ax7 = fig.add_subplot(gs[1, 1])
plot_scatter(ax7, snap48['kp'], snap48['deltaA'], snap48['nc'],
             r'$\Phi\cdot\nabla L$', 'Eq. 3 (kernel)')

ax8 = fig.add_subplot(gs[1, 2])
plot_scatter(ax8, snap48['dp'], snap48['deltaA'], snap48['nc'],
             r'diag $\Phi$', 'Eq. 5 (diagonal)')

ax8b = fig.add_subplot(gs[1, 3])
plot_scatter(ax8b, snap48['ng'], snap48['deltaA'], snap48['nc'],
             r'$-\partial L/\partial A$', r'$-dL/dA$ (raw)')

# Row 3: Training dynamics side by side
ax9 = fig.add_subplot(gs[2, :2])
plot_dynamics(ax9, h8, show_loss_label=False)
ax9.set_title('Training dynamics (width=8)', fontsize=8, fontweight='bold')

ax10 = fig.add_subplot(gs[2, 2:])
plot_dynamics(ax10, h48, show_loss_label=True)
ax10.set_title('Training dynamics (width=48)', fontsize=8, fontweight='bold')

plt.savefig('/Users/konrad_1/AvsWdescent/fig_ntk.pdf', bbox_inches='tight', facecolor='#faf9f6')
plt.savefig('/Users/konrad_1/AvsWdescent/fig_ntk.png', bbox_inches='tight', facecolor='#faf9f6')
print("Figure 1 saved.")

# ======================== WIDTH SWEEP ========================

widths = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
n_seeds = 3
n_steps_sweep = 2000
diag_every_sweep = 40

# For each width, collect median corr of diagonal approx over training, per seed
width_corr_median = {w: [] for w in widths}
width_corr_final = {w: [] for w in widths}
width_neg_median = {w: [] for w in widths}
width_neg_final = {w: [] for w in widths}

for w in widths:
    print(f"  Width sweep: w={w} ...", flush=True)
    for seed in range(n_seeds):
        np.random.seed(1000 + seed * 100 + w)
        h, snap, _ = run_experiment(width=w, depth=3, eta=0.005,
                                     n_steps=n_steps_sweep, diag_every=diag_every_sweep)
        corr_vals = h['corr_diag']
        width_corr_median[w].append(np.median(corr_vals))
        width_corr_final[w].append(np.mean(corr_vals[-5:]))
        neg_vals = h['corr_neg']
        width_neg_median[w].append(np.median(neg_vals))
        width_neg_final[w].append(np.mean(neg_vals[-5:]))

# Compute mean and std across seeds
w_arr = np.array(widths)
median_mean = np.array([np.mean(width_corr_median[w]) for w in widths])
median_std = np.array([np.std(width_corr_median[w]) for w in widths])
final_mean = np.array([np.mean(width_corr_final[w]) for w in widths])
final_std = np.array([np.std(width_corr_final[w]) for w in widths])

fig2, ax = plt.subplots(figsize=(3.8, 2.8), dpi=200)
fig2.patch.set_facecolor('#faf9f6')

neg_final_mean = np.array([np.mean(width_neg_final[w]) for w in widths])
neg_final_std = np.array([np.std(width_neg_final[w]) for w in widths])
neg_median_mean = np.array([np.mean(width_neg_median[w]) for w in widths])
neg_median_std = np.array([np.std(width_neg_median[w]) for w in widths])

ax.fill_between(w_arr, neg_final_mean - neg_final_std, neg_final_mean + neg_final_std,
                color='#d97706', alpha=0.15)
ax.plot(w_arr, neg_final_mean, 'o-', color='#d97706', linewidth=1.5, markersize=4,
        label=r'late training $r$')

ax.fill_between(w_arr, neg_median_mean - neg_median_std, neg_median_mean + neg_median_std,
                color='#2563eb', alpha=0.15)
ax.plot(w_arr, neg_median_mean, 's--', color='#2563eb', linewidth=1.2, markersize=3.5,
        label=r'median $r$')

ax.axhline(1, color='#e8e5dd', linestyle='--', linewidth=0.5)
ax.axhline(0, color='#e8e5dd', linewidth=0.5)
ax.set_xlabel('hidden layer width', fontsize=8)
ax.set_ylabel(r'$r(\Delta A,\; -\partial L/\partial A)$', fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, loc='lower right', framealpha=0.8)
ax.set_ylim(-0.5, 1.1)
ax.set_xlim(0, widths[-1] + 4)

plt.savefig('/Users/konrad_1/AvsWdescent/fig_width_sweep.pdf', bbox_inches='tight', facecolor='#faf9f6')
plt.savefig('/Users/konrad_1/AvsWdescent/fig_width_sweep.png', bbox_inches='tight', facecolor='#faf9f6')
print("Figure 2 (width sweep) saved.")

# ======================== DEPTH SWEEP ========================
# Tests the appendix prediction: under He init, r(ΔA, -∂L/∂A) is approximately
# depth-independent (~√3/2), while r(ΔA, -Φ_ii ∂L/∂A) approaches 1 with depth.

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
        h, _, _ = run_experiment(width=depth_width, depth=d, eta=0.005,
                                 n_steps=n_steps_depth, diag_every=diag_every_depth)
        neg_vals = h['corr_neg']
        diag_vals = h['corr_diag']
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
fig3.patch.set_facecolor('#faf9f6')

def plot_depth_panel(ax, neg_mean, neg_std, diag_mean, diag_std, title):
    ax.fill_between(d_arr, neg_mean - neg_std, neg_mean + neg_std,
                    color='#d97706', alpha=0.15)
    ax.plot(d_arr, neg_mean, 'o-', color='#d97706', linewidth=1.5, markersize=4,
            label=r'$r(\Delta A,\;-\partial L/\partial A)$ (raw)')
    ax.fill_between(d_arr, diag_mean - diag_std, diag_mean + diag_std,
                    color='#059669', alpha=0.15)
    ax.plot(d_arr, diag_mean, 's-', color='#059669', linewidth=1.5, markersize=4,
            label=r'$r(\Delta A,\;-\Phi_{ii}\,\partial L/\partial A)$ (scaled)')
    ax.axhline(np.sqrt(3)/2, color='#b0a890', linestyle=':', linewidth=1.0)
    ax.text(d_arr[-1] - 0.1, np.sqrt(3)/2 + 0.015, r'$\sqrt{3}/2$',
            fontsize=7, color='#8a7d4a', ha='right')
    ax.axhline(1, color='#e8e5dd', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='#e8e5dd', linewidth=0.5)
    ax.set_xlabel('depth (number of hidden layers)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(depths[0] - 0.3, depths[-1] + 0.3)
    ax.set_title(title, fontsize=9, fontweight='bold')

plot_depth_panel(axes[0], neg_init_mean, neg_init_std, diag_init_mean, diag_init_std,
                 'At initialisation (theory regime)')
plot_depth_panel(axes[1], neg_late_mean, neg_late_std, diag_late_mean, diag_late_std,
                 'After training (1500 SGD steps)')
axes[0].set_ylabel('Pearson $r$', fontsize=8)
axes[1].legend(fontsize=6.5, loc='lower right', framealpha=0.8)

plt.savefig('/Users/konrad_1/AvsWdescent/fig_depth_sweep.pdf', bbox_inches='tight', facecolor='#faf9f6')
plt.savefig('/Users/konrad_1/AvsWdescent/fig_depth_sweep.png', bbox_inches='tight', facecolor='#faf9f6')
print("Figure 3 (depth sweep) saved.")
