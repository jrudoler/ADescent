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

def compute_jacobian_and_predictions(sizes, W, a, h, dLdW, dLdA, target, eta):
    L = len(W)
    neuron_counts = sizes[1:]
    totalN = sum(neuron_counts)
    
    layer_offsets = []
    off = 0
    for l in range(L):
        layer_offsets.append(off)
        off += sizes[l+1]
    
    # Partial gradient (zero for hidden, output error for output)
    gradA_partial = np.zeros(totalN)
    out_off = layer_offsets[-1]
    nout = sizes[-1]
    gradA_partial[out_off:out_off+nout] = a[L] - target
    
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
    fullThetaPred = -eta * J @ (J.T @ gradA_partial)
    
    theta_diag = np.sum(J**2, axis=1)
    diagPred = -eta * theta_diag * gradA_bp
    
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

    Theta = J @ J.T

    return groundTruth, fullThetaPred, diagPred, negGradA, Theta, neuron_counts, layer_offsets, active

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
    
    history = {'step': [], 'loss': [], 'corr_ground': [], 'corr_full': [], 'corr_diag': [], 'corr_neg': []}
    last_snap = None

    for step in range(n_steps):
        bi = np.random.randint(B)
        a, h = forward(sizes, W, X[bi])
        dLdA, dLdW = backprop(sizes, W, a, h, Y[bi])

        do_diag = (step % diag_every == 0) or (step == n_steps - 1)

        if do_diag:
            a_before = np.concatenate([a[l] for l in range(1, len(W)+1)])
            gt, fp, dp, ng, Theta, nc, lo, active = compute_jacobian_and_predictions(
                sizes, W, a, h, dLdW, dLdA, Y[bi], eta)

            for l in range(len(W)):
                W[l] -= eta * dLdW[l]

            a2, _ = forward(sizes, W, X[bi])
            a_after = np.concatenate([a2[l] for l in range(1, len(W)+1)])
            deltaA_full = a_after - a_before

            # Filter to active neurons only
            mask = active > 0
            deltaA = deltaA_full[mask]
            gt = gt[mask]; fp = fp[mask]; dp = dp[mask]; ng = ng[mask]

            cg = corr(deltaA, gt)
            cf = corr(deltaA, fp)
            cd = corr(deltaA, dp)
            cn = corr(deltaA, ng)

            loss = 0
            for b in range(B):
                af, _ = forward(sizes, W, X[b])
                loss += 0.5 * np.sum((af[-1] - Y[b])**2) / B

            history['step'].append(step)
            history['loss'].append(loss)
            history['corr_ground'].append(cg)
            history['corr_full'].append(cf)
            history['corr_diag'].append(cd)
            history['corr_neg'].append(cn)

            # Filter Theta and neuron counts for snapshot
            active_idx = np.where(mask)[0]
            Theta_filt = Theta[np.ix_(active_idx, active_idx)]
            nc_filt = []
            idx = 0
            for count in nc:
                nc_filt.append(int(np.sum(mask[idx:idx+count])))
                idx += count

            last_snap = {
                'deltaA': deltaA, 'gt': gt, 'fp': fp, 'dp': dp, 'ng': ng,
                'Theta': Theta_filt, 'nc': nc_filt, 'lo': lo, 'step': step
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
    ax.set_title(r'$\Theta_{ik}$ correlation' + title_suffix, fontsize=9, fontweight='bold')

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
fig = plt.figure(figsize=(8.5, 6.2), dpi=200)
fig.patch.set_facecolor('#faf9f6')

gs = GridSpec(3, 5, figure=fig, hspace=0.65, wspace=0.55,
              left=0.06, right=0.97, top=0.95, bottom=0.07)


# Row 1: width=8
ax1 = fig.add_subplot(gs[0, 0])
plot_theta_heatmap(ax1, snap8['Theta'], snap8['nc'], ' (width=8)')

ax2 = fig.add_subplot(gs[0, 1])
plot_scatter(ax2, snap8['gt'], snap8['deltaA'], snap8['nc'],
             r'$J\cdot\Delta W$', 'Eq. 2 (identity)')

ax3 = fig.add_subplot(gs[0, 2])
plot_scatter(ax3, snap8['fp'], snap8['deltaA'], snap8['nc'],
             r'$\Theta\cdot\partial L$', 'Eq. 3 (kernel)')

ax4 = fig.add_subplot(gs[0, 3])
plot_scatter(ax4, snap8['dp'], snap8['deltaA'], snap8['nc'],
             r'diag $\Theta$', 'Eq. 5 (diagonal)')

ax4b = fig.add_subplot(gs[0, 4])
plot_scatter(ax4b, snap8['ng'], snap8['deltaA'], snap8['nc'],
             r'$-\partial L/\partial A$', r'$-dL/dA$ (raw)')
ax4b.legend(fontsize=4.5, loc='lower right', framealpha=0.8, markerscale=0.8)

# Row 2: width=48
ax5 = fig.add_subplot(gs[1, 0])
plot_theta_heatmap(ax5, snap48['Theta'], snap48['nc'], ' (width=48)')

ax6 = fig.add_subplot(gs[1, 1])
plot_scatter(ax6, snap48['gt'], snap48['deltaA'], snap48['nc'],
             r'$J\cdot\Delta W$', 'Eq. 2 (identity)')

ax7 = fig.add_subplot(gs[1, 2])
plot_scatter(ax7, snap48['fp'], snap48['deltaA'], snap48['nc'],
             r'$\Theta\cdot\partial L$', 'Eq. 3 (kernel)')

ax8 = fig.add_subplot(gs[1, 3])
plot_scatter(ax8, snap48['dp'], snap48['deltaA'], snap48['nc'],
             r'diag $\Theta$', 'Eq. 5 (diagonal)')

ax8b = fig.add_subplot(gs[1, 4])
plot_scatter(ax8b, snap48['ng'], snap48['deltaA'], snap48['nc'],
             r'$-\partial L/\partial A$', r'$-dL/dA$ (raw)')

# Row 3: Training dynamics side by side
ax9 = fig.add_subplot(gs[2, :2])
plot_dynamics(ax9, h8, show_loss_label=False)
ax9.set_title('Training dynamics (width=8)', fontsize=8, fontweight='bold')

ax10 = fig.add_subplot(gs[2, 3:])
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
