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
        scale = np.sqrt(2.0 / sizes[i])
        W.append(np.random.randn(sizes[i+1], sizes[i]) * scale)
    return sizes, W

def forward(sizes, W, x):
    a = [x.copy()]
    h = [None]
    for l in range(len(W)):
        hl = W[l] @ a[-1]
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
        dLdW[l] = np.outer(delta, a[l])
        if l > 0:
            dLdA[l] = W[l].T @ delta
    return dLdA, dLdW

def make_spirals(n_per_class=20):
    X, Y = [], []
    for i in range(n_per_class):
        t = (i / n_per_class) * 3 * np.pi + np.pi / 2
        r = (i / n_per_class) * 0.8 + 0.2
        X.append([r*np.cos(t) + np.random.randn()*0.06, r*np.sin(t) + np.random.randn()*0.06])
        Y.append([1, 0])
        X.append([-r*np.cos(t) + np.random.randn()*0.06, -r*np.sin(t) + np.random.randn()*0.06])
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
    
    totalParams = sum(sizes[i]*sizes[i+1] for i in range(L))
    
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
        P = nout_m * nin_m
        
        J_cur = np.zeros((nout_m, P))
        for j in range(nout_m):
            fp = 1.0 if m == L-1 else (1.0 if h[m+1][j] > 0 else 0.0)
            J_cur[j, j*nin_m:(j+1)*nin_m] = fp * a[m]
        
        go = layer_offsets[m]
        J[go:go+nout_m, param_offset:param_offset+P] = J_cur
        
        J_prev = J_cur
        for l in range(m+1, L):
            nin_l, nout_l = sizes[l], sizes[l+1]
            DW = np.zeros((nout_l, nin_l))
            for i in range(nout_l):
                fp = 1.0 if l == L-1 else (1.0 if h[l+1][i] > 0 else 0.0)
                DW[i] = fp * W[l][i]
            J_new = DW @ J_prev
            go2 = layer_offsets[l]
            J[go2:go2+nout_l, param_offset:param_offset+P] = J_new
            J_prev = J_new
        
        param_offset += P
    
    groundTruth = J @ deltaW_flat
    fullThetaPred = -eta * J @ (J.T @ gradA_partial)
    
    theta_diag = np.sum(J**2, axis=1)
    diagPred = -eta * theta_diag * gradA_bp
    
    Theta = J @ J.T
    
    return groundTruth, fullThetaPred, diagPred, Theta, neuron_counts, layer_offsets

def r2(actual, pred):
    ss_tot = np.sum((actual - np.mean(actual))**2)
    ss_res = np.sum((actual - pred)**2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0

# ======================== RUN EXPERIMENT ========================

def run_experiment(width, depth, eta=0.005, n_steps=2000, diag_every=50):
    sizes = [2] + [width]*depth + [2]
    sizes_tuple, W = create_network(sizes)
    X, Y = make_spirals(20)
    B = len(X)
    
    history = {'step': [], 'loss': [], 'r2_ground': [], 'r2_full': [], 'r2_diag': []}
    last_snap = None
    
    for step in range(n_steps):
        bi = np.random.randint(B)
        a, h = forward(sizes, W, X[bi])
        dLdA, dLdW = backprop(sizes, W, a, h, Y[bi])
        
        do_diag = (step % diag_every == 0) or (step == n_steps - 1)
        
        if do_diag:
            a_before = np.concatenate([a[l] for l in range(1, len(W)+1)])
            gt, fp, dp, Theta, nc, lo = compute_jacobian_and_predictions(
                sizes, W, a, h, dLdW, dLdA, Y[bi], eta)
            
            for l in range(len(W)):
                W[l] -= eta * dLdW[l]
            
            a2, _ = forward(sizes, W, X[bi])
            a_after = np.concatenate([a2[l] for l in range(1, len(W)+1)])
            deltaA = a_after - a_before
            
            r2g = r2(deltaA, gt)
            r2f = r2(deltaA, fp)
            r2d = r2(deltaA, dp)
            
            loss = 0
            for b in range(B):
                af, _ = forward(sizes, W, X[b])
                loss += 0.5 * np.sum((af[-1] - Y[b])**2) / B
            
            history['step'].append(step)
            history['loss'].append(loss)
            history['r2_ground'].append(r2g)
            history['r2_full'].append(r2f)
            history['r2_diag'].append(r2d)
            
            last_snap = {
                'deltaA': deltaA, 'gt': gt, 'fp': fp, 'dp': dp,
                'Theta': Theta, 'nc': nc, 'lo': lo, 'step': step
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
    all_vals = np.concatenate([pred, actual])
    maxabs = max(np.max(np.abs(all_vals)), 1e-10) * 1.15
    
    ax.plot([-maxabs, maxabs], [-maxabs, maxabs], '--', color='#b0a890', linewidth=1, zorder=1)
    ax.axhline(0, color='#ddd8cc', linewidth=0.3)
    ax.axvline(0, color='#ddd8cc', linewidth=0.3)
    
    idx = 0
    for l, nc in enumerate(neuron_counts):
        ax.scatter(pred[idx:idx+nc], actual[idx:idx+nc],
                   c=LAYER_COLORS[l % len(LAYER_COLORS)], s=8, alpha=0.6, 
                   edgecolors='none', zorder=2, label=f'L{l+1} ({nc})')
        idx += nc
    
    r2v = r2(actual, pred)
    ax.set_xlim(-maxabs, maxabs); ax.set_ylim(-maxabs, maxabs)
    ax.set_aspect('equal')
    ax.set_xlabel(f'predicted ({label})', fontsize=6)
    ax.set_ylabel(r'actual $\Delta A$', fontsize=6)
    ax.tick_params(labelsize=5)
    ax.set_title(f'{eq_label}\n$R^2 = {r2v:.3f}$', fontsize=7, fontweight='bold', pad=3)

def plot_dynamics(ax, history, show_loss_label=True):
    steps = history['step']
    ax.plot(steps, history['r2_ground'], '-', color='#aaa', linewidth=0.8, label=r'$R^2$ ($J\cdot\Delta W$)', alpha=0.7)
    ax.plot(steps, history['r2_full'], '-', color='#2563eb', linewidth=1.2, label=r'$R^2$ ($\Theta\cdot\partial L$)')
    ax.plot(steps, history['r2_diag'], '-', color='#059669', linewidth=1.2, label=r'$R^2$ (diag $\Theta$)')

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
    ax.set_ylabel(r'$R^2$', fontsize=7)
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

gs = GridSpec(3, 4, figure=fig, hspace=0.65, wspace=0.60,
              left=0.08, right=0.96, top=0.95, bottom=0.07)


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
ax4.legend(fontsize=4.5, loc='lower right', framealpha=0.8, markerscale=0.8)

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

# For each width, collect median R² of diagonal approx over training, per seed
width_r2_median = {w: [] for w in widths}
width_r2_final = {w: [] for w in widths}

for w in widths:
    print(f"  Width sweep: w={w} ...", flush=True)
    for seed in range(n_seeds):
        np.random.seed(1000 + seed * 100 + w)
        h, snap, _ = run_experiment(width=w, depth=3, eta=0.005,
                                     n_steps=n_steps_sweep, diag_every=diag_every_sweep)
        r2_vals = h['r2_diag']
        width_r2_median[w].append(np.median(r2_vals))
        # Use mean of last 5 measurements as "final" R²
        width_r2_final[w].append(np.mean(r2_vals[-5:]))

# Compute mean and std across seeds
w_arr = np.array(widths)
median_mean = np.array([np.mean(width_r2_median[w]) for w in widths])
median_std = np.array([np.std(width_r2_median[w]) for w in widths])
final_mean = np.array([np.mean(width_r2_final[w]) for w in widths])
final_std = np.array([np.std(width_r2_final[w]) for w in widths])

fig2, ax = plt.subplots(figsize=(3.8, 2.8), dpi=200)
fig2.patch.set_facecolor('#faf9f6')

ax.fill_between(w_arr, final_mean - final_std, final_mean + final_std,
                color='#059669', alpha=0.15)
ax.plot(w_arr, final_mean, 'o-', color='#059669', linewidth=1.5, markersize=4,
        label=r'late training $R^2$')

ax.fill_between(w_arr, median_mean - median_std, median_mean + median_std,
                color='#2563eb', alpha=0.15)
ax.plot(w_arr, median_mean, 's--', color='#2563eb', linewidth=1.2, markersize=3.5,
        label=r'median $R^2$')

ax.axhline(1, color='#e8e5dd', linestyle='--', linewidth=0.5)
ax.axhline(0, color='#e8e5dd', linewidth=0.5)
ax.set_xlabel('hidden layer width', fontsize=8)
ax.set_ylabel(r'$R^2$ (diagonal $\Theta$ approximation)', fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, loc='lower right', framealpha=0.8)
ax.set_ylim(-0.5, 1.1)
ax.set_xlim(0, widths[-1] + 4)

plt.savefig('/Users/konrad_1/AvsWdescent/fig_width_sweep.pdf', bbox_inches='tight', facecolor='#faf9f6')
plt.savefig('/Users/konrad_1/AvsWdescent/fig_width_sweep.png', bbox_inches='tight', facecolor='#faf9f6')
print("Figure 2 (width sweep) saved.")
