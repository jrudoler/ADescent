import { useState, useRef, useEffect, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid } from "recharts";

// ======================== LINEAR ALGEBRA ========================

function matVec(A, rows, cols, v) {
  const out = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    let s = 0, off = i * cols;
    for (let j = 0; j < cols; j++) s += A[off + j] * v[j];
    out[i] = s;
  }
  return out;
}

function matmul(A, ar, ac, B, br, bc) {
  const C = new Float64Array(ar * bc);
  for (let i = 0; i < ar; i++)
    for (let k = 0; k < ac; k++) {
      const a = A[i * ac + k];
      if (a === 0) continue;
      for (let j = 0; j < bc; j++) C[i * bc + j] += a * B[k * bc + j];
    }
  return C;
}

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ======================== NETWORK ========================

function createNetwork(sizes) {
  const W = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const nin = sizes[l], nout = sizes[l + 1];
    const scale = Math.sqrt(2 / nin);
    const w = new Float64Array(nout * nin);
    for (let i = 0; i < w.length; i++) w[i] = randn() * scale;
    W.push(w);
  }
  return { sizes, W };
}

function forward(net, x) {
  const { sizes, W } = net;
  const L = W.length;
  const a = [Float64Array.from(x)];
  const h = [null];
  for (let l = 0; l < L; l++) {
    const nin = sizes[l], nout = sizes[l + 1];
    const hl = new Float64Array(nout);
    for (let i = 0; i < nout; i++) {
      let s = 0;
      for (let j = 0; j < nin; j++) s += W[l][i * nin + j] * a[l][j];
      hl[i] = s;
    }
    h.push(hl);
    if (l < L - 1) {
      const al = new Float64Array(nout);
      for (let i = 0; i < nout; i++) al[i] = hl[i] > 0 ? hl[i] : 0;
      a.push(al);
    } else {
      a.push(Float64Array.from(hl));
    }
  }
  return { a, h };
}

function backprop(net, fwd, target) {
  const { sizes, W } = net;
  const { a, h } = fwd;
  const L = W.length;
  const dLdA = new Array(L + 1).fill(null);
  const dLdW = [];
  const nout = sizes[L];
  dLdA[L] = new Float64Array(nout);
  for (let i = 0; i < nout; i++) dLdA[L][i] = a[L][i] - target[i];
  for (let l = L - 1; l >= 0; l--) {
    const nin = sizes[l], no = sizes[l + 1];
    const delta = new Float64Array(no);
    if (l === L - 1) { for (let i = 0; i < no; i++) delta[i] = dLdA[l + 1][i]; }
    else { for (let i = 0; i < no; i++) delta[i] = h[l + 1][i] > 0 ? dLdA[l + 1][i] : 0; }
    const dw = new Float64Array(no * nin);
    for (let i = 0; i < no; i++)
      for (let j = 0; j < nin; j++) dw[i * nin + j] = delta[i] * a[l][j];
    dLdW.unshift(dw);
    if (l > 0) {
      dLdA[l] = new Float64Array(nin);
      for (let j = 0; j < nin; j++) {
        let s = 0;
        for (let i = 0; i < no; i++) s += W[l][i * nin + j] * delta[i];
        dLdA[l][j] = s;
      }
    }
  }
  return { dLdA, dLdW };
}

// ======================== EXPENSIVE: JACOBIAN & PREDICTIONS ========================

function computeAllPredictions(net, fwd, dLdW, dLdA, target, eta) {
  const { sizes, W } = net;
  const { a, h } = fwd;
  const L = W.length;
  const neuronCounts = sizes.slice(1);
  const totalN = neuronCounts.reduce((s, n) => s + n, 0);

  const layerOffsets = [];
  let off = 0;
  for (let l = 0; l < L; l++) { layerOffsets.push(off); off += sizes[l + 1]; }

  // Partial gradient: zero for hidden, output error for output
  const gradA_partial = new Float64Array(totalN);
  const outputOffset = layerOffsets[L - 1];
  const nout = sizes[L];
  for (let i = 0; i < nout; i++) gradA_partial[outputOffset + i] = a[L][i] - target[i];

  // Backprop gradient (total derivative)
  const gradA_bp = new Float64Array(totalN);
  for (let l = 1; l <= L; l++) {
    if (dLdA[l]) {
      const gOff = layerOffsets[l - 1];
      for (let i = 0; i < dLdA[l].length; i++) gradA_bp[gOff + i] = dLdA[l][i];
    }
  }

  let totalParams = 0;
  for (let l = 0; l < L; l++) totalParams += sizes[l] * sizes[l + 1];

  const deltaW = new Float64Array(totalParams);
  let pOff = 0;
  for (let l = 0; l < L; l++) {
    for (let i = 0; i < dLdW[l].length; i++) deltaW[pOff + i] = -eta * dLdW[l][i];
    pOff += dLdW[l].length;
  }

  // Build full Jacobian
  const J = new Float64Array(totalN * totalParams);
  let paramOffset = 0;
  for (let m = 0; m < L; m++) {
    const nin_m = sizes[m], nout_m = sizes[m + 1];
    const P = nout_m * nin_m;

    let J_cur = new Float64Array(nout_m * P);
    for (let j = 0; j < nout_m; j++) {
      const fp = (m < L - 1) ? (h[m + 1][j] > 0 ? 1 : 0) : 1;
      for (let k = 0; k < nin_m; k++)
        J_cur[j * P + j * nin_m + k] = fp * a[m][k];
    }

    const gOff = layerOffsets[m];
    for (let i = 0; i < nout_m; i++)
      for (let j = 0; j < P; j++)
        J[(gOff + i) * totalParams + paramOffset + j] = J_cur[i * P + j];

    let J_prev = J_cur, prevRows = nout_m;
    for (let l = m + 1; l < L; l++) {
      const nin_l = sizes[l], nout_l = sizes[l + 1];
      const DW = new Float64Array(nout_l * nin_l);
      for (let i = 0; i < nout_l; i++) {
        const fp = (l < L - 1) ? (h[l + 1][i] > 0 ? 1 : 0) : 1;
        for (let j = 0; j < nin_l; j++) DW[i * nin_l + j] = fp * W[l][i * nin_l + j];
      }
      const J_new = matmul(DW, nout_l, nin_l, J_prev, prevRows, P);
      const gOff2 = layerOffsets[l];
      for (let i = 0; i < nout_l; i++)
        for (let j = 0; j < P; j++)
          J[(gOff2 + i) * totalParams + paramOffset + j] = J_new[i * P + j];
      J_prev = J_new; prevRows = nout_l;
    }
    paramOffset += P;
  }

  // Ground truth: J @ deltaW
  const groundTruth = matVec(J, totalN, totalParams, deltaW);

  // Full Phi: -eta * J @ (J^T @ gradA_partial)
  const JtGp = new Float64Array(totalParams);
  for (let alpha = 0; alpha < totalParams; alpha++) {
    let s = 0;
    for (let i = 0; i < totalN; i++) s += J[i * totalParams + alpha] * gradA_partial[i];
    JtGp[alpha] = s;
  }
  const phiGradP = matVec(J, totalN, totalParams, JtGp);
  const fullPhiPred = new Float64Array(totalN);
  for (let i = 0; i < totalN; i++) fullPhiPred[i] = -eta * phiGradP[i];

  // Diagonal: -eta * ||J_row_i||^2 * gradA_bp[i]
  const diagPred = new Float64Array(totalN);
  for (let i = 0; i < totalN; i++) {
    let rowNorm = 0;
    const rOff = i * totalParams;
    for (let j = 0; j < totalParams; j++) rowNorm += J[rOff + j] ** 2;
    diagPred[i] = -eta * rowNorm * gradA_bp[i];
  }

  // Phi for heatmap
  let Phi = null;
  if (totalN <= 200) {
    Phi = new Float64Array(totalN * totalN);
    for (let i = 0; i < totalN; i++) {
      for (let j = i; j < totalN; j++) {
        let s = 0;
        const ri = i * totalParams, rj = j * totalParams;
        for (let k = 0; k < totalParams; k++) s += J[ri + k] * J[rj + k];
        Phi[i * totalN + j] = s;
        if (i !== j) Phi[j * totalN + i] = s;
      }
    }
  }

  return { groundTruth, fullPhiPred, diagPred, Phi, totalN, neuronCounts };
}

// ======================== DATA ========================

function makeSpirals(nPerClass) {
  const X = [], Y = [];
  for (let i = 0; i < nPerClass; i++) {
    const t = (i / nPerClass) * 3 * Math.PI + Math.PI / 2;
    const r = (i / nPerClass) * 0.8 + 0.2;
    X.push([r * Math.cos(t) + randn() * 0.06, r * Math.sin(t) + randn() * 0.06]);
    Y.push([1, 0]);
    X.push([-r * Math.cos(t) + randn() * 0.06, -r * Math.sin(t) + randn() * 0.06]);
    Y.push([0, 1]);
  }
  return { X, Y };
}

// ======================== CANVAS ========================

function drawHeatmap(canvas, Phi, totalN, neuronCounts) {
  if (!canvas || !Phi) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const corr = new Float64Array(totalN * totalN);
  for (let i = 0; i < totalN; i++)
    for (let j = 0; j < totalN; j++) {
      const dii = Phi[i * totalN + i], djj = Phi[j * totalN + j];
      corr[i * totalN + j] = (dii > 0 && djj > 0) ? Phi[i * totalN + j] / Math.sqrt(dii * djj) : 0;
    }
  const cW = W / totalN, cH = H / totalN;
  for (let i = 0; i < totalN; i++)
    for (let j = 0; j < totalN; j++) {
      const v = Math.max(-1, Math.min(1, corr[i * totalN + j]));
      if (v >= 0) ctx.fillStyle = `rgb(${255 - Math.round(v * 200)},${255 - Math.round(v * 200)},255)`;
      else ctx.fillStyle = `rgb(255,${255 + Math.round(v * 200)},${255 + Math.round(v * 200)})`;
      ctx.fillRect(Math.floor(j * cW), Math.floor(i * cH), Math.ceil(cW) + 1, Math.ceil(cH) + 1);
    }
  ctx.strokeStyle = "rgba(0,0,0,0.35)"; ctx.lineWidth = 1.5;
  let acc = 0;
  for (let l = 0; l < neuronCounts.length - 1; l++) {
    acc += neuronCounts[l];
    const px = acc * cW, py = acc * cH;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(W, py); ctx.stroke();
  }
}

function drawScatter(canvas, pred, actual, neuronCounts, label) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const pad = 32;
  ctx.clearRect(0, 0, W, H);
  if (!actual || actual.length === 0) return;
  const allVals = [...pred, ...actual];
  let maxAbs = 0;
  for (const v of allVals) if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
  if (maxAbs === 0) maxAbs = 1;
  maxAbs *= 1.15;
  const toX = v => pad + (v / maxAbs + 1) / 2 * (W - 2 * pad);
  const toY = v => (H - pad) - (v / maxAbs + 1) / 2 * (H - 2 * pad);
  ctx.strokeStyle = "#ddd8cc"; ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(toX(0), pad); ctx.lineTo(toX(0), H - pad); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad, toY(0)); ctx.lineTo(W - pad, toY(0)); ctx.stroke();
  ctx.strokeStyle = "#b0a890"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(toX(-maxAbs), toY(-maxAbs)); ctx.lineTo(toX(maxAbs), toY(maxAbs)); ctx.stroke();
  ctx.setLineDash([]);
  const colors = ["#2563eb", "#0891b2", "#059669", "#d97706"];
  let idx = 0;
  for (let l = 0; l < neuronCounts.length; l++) {
    ctx.fillStyle = colors[l % 4];
    for (let i = 0; i < neuronCounts[l]; i++) {
      ctx.globalAlpha = 0.6; ctx.beginPath();
      ctx.arc(toX(pred[idx]), toY(actual[idx]), 2.5, 0, Math.PI * 2); ctx.fill(); idx++;
    }
  }
  ctx.globalAlpha = 1; ctx.fillStyle = "#7a7568"; ctx.font = "11px monospace"; ctx.textAlign = "center";
  ctx.fillText("predicted (" + label + ")", W / 2, H - 6);
  ctx.save(); ctx.translate(11, H / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("actual \u0394A", 0, 0); ctx.restore();
}

function r2fn(actual, pred) {
  const n = actual.length;
  let mu = 0; for (let i = 0; i < n; i++) mu += actual[i]; mu /= n;
  let ssTot = 0, ssRes = 0;
  for (let i = 0; i < n; i++) { ssTot += (actual[i] - mu) ** 2; ssRes += (actual[i] - pred[i]) ** 2; }
  return ssTot > 1e-30 ? 1 - ssRes / ssTot : 0;
}

// ======================== COMPONENT ========================

const LCOLORS = ["#2563eb", "#0891b2", "#059669", "#d97706"];
const MODES = ["ground", "fullPhi", "diagBP"];
const MODE_LABELS = {
  ground: "Eq.2: J\u00B7\u0394W",
  fullPhi: "Eq.3: \u03A6\u00B7\u2202L",
  diagBP: "Eq.5: \u03A6\u1D62\u1D62\u00B7\u2207L",
};

export default function App() {
  const [width, setWidth] = useState(32);
  const [depth, setDepth] = useState(3);
  const [lr, setLr] = useState(0.005);
  const [diagEvery, setDiagEvery] = useState(50);
  const [step, setStep] = useState(0);
  const [metrics, setMetrics] = useState({});
  const [history, setHistory] = useState([]);
  const [running, setRunning] = useState(false);
  const [scatterMode, setScatterMode] = useState("fullPhi");

  const netRef = useRef(null);
  const dataRef = useRef(null);
  const runRef = useRef(false);
  const stepRef = useRef(0);
  const heatRef = useRef(null);
  const scatRef = useRef(null);
  const lastData = useRef(null);

  const initNetwork = useCallback(() => {
    const sizes = [2];
    for (let i = 0; i < depth; i++) sizes.push(width);
    sizes.push(2);
    netRef.current = createNetwork(sizes);
    dataRef.current = makeSpirals(20);
    stepRef.current = 0;
    setStep(0); setMetrics({}); setHistory([]);
    lastData.current = null;
    if (heatRef.current) heatRef.current.getContext("2d").clearRect(0, 0, 340, 340);
    if (scatRef.current) scatRef.current.getContext("2d").clearRect(0, 0, 340, 340);
  }, [width, depth]);

  useEffect(() => { initNetwork(); }, [initNetwork]);

  // Cheap SGD step — just forward, backprop, update weights, compute loss
  const cheapStep = useCallback(() => {
    const net = netRef.current;
    const data = dataRef.current;
    if (!net || !data) return;
    const { X, Y } = data;
    const B = X.length;
    const L = net.W.length;

    const bi = Math.floor(Math.random() * B);
    const fwd = forward(net, X[bi]);
    const { dLdW } = backprop(net, fwd, Y[bi]);

    for (let l = 0; l < L; l++)
      for (let i = 0; i < net.W[l].length; i++) net.W[l][i] -= lr * dLdW[l][i];

    // Compute loss over full dataset
    let totalLoss = 0;
    for (let b = 0; b < B; b++) {
      const f = forward(net, X[b]);
      const out = f.a[L];
      for (let i = 0; i < out.length; i++) totalLoss += 0.5 * (out[i] - Y[b][i]) ** 2 / B;
    }

    stepRef.current++;
    return totalLoss;
  }, [lr]);

  // Expensive diagnostic step — builds full Jacobian, computes Phi, predictions
  const expensiveStep = useCallback(() => {
    const net = netRef.current;
    const data = dataRef.current;
    if (!net || !data) return null;
    const { X, Y } = data;
    const B = X.length;
    const L = net.W.length;

    const bi = Math.floor(Math.random() * B);
    const fwd = forward(net, X[bi]);
    const { dLdA, dLdW } = backprop(net, fwd, Y[bi]);

    // Activities BEFORE this update
    const aBefore = [];
    for (let l = 1; l <= L; l++)
      for (let i = 0; i < fwd.a[l].length; i++) aBefore.push(fwd.a[l][i]);

    const { groundTruth, fullPhiPred, diagPred, Phi, totalN, neuronCounts } =
      computeAllPredictions(net, fwd, dLdW, dLdA, Y[bi], lr);

    // Apply this step's SGD update
    for (let l = 0; l < L; l++)
      for (let i = 0; i < net.W[l].length; i++) net.W[l][i] -= lr * dLdW[l][i];

    const fwd2 = forward(net, X[bi]);
    const aAfter = [];
    for (let l = 1; l <= L; l++)
      for (let i = 0; i < fwd2.a[l].length; i++) aAfter.push(fwd2.a[l][i]);

    const deltaA = aBefore.map((v, i) => aAfter[i] - v);

    // Loss
    let totalLoss = 0;
    for (let b = 0; b < B; b++) {
      const f = forward(net, X[b]);
      const out = f.a[L];
      for (let i = 0; i < out.length; i++) totalLoss += 0.5 * (out[i] - Y[b][i]) ** 2 / B;
    }

    const r2G = r2fn(deltaA, Array.from(groundTruth));
    const r2F = r2fn(deltaA, Array.from(fullPhiPred));
    const r2D = r2fn(deltaA, Array.from(diagPred));

    stepRef.current++;

    return {
      preds: {
        ground: Array.from(groundTruth),
        fullPhi: Array.from(fullPhiPred),
        diagBP: Array.from(diagPred),
      },
      actual: deltaA, neuronCounts, Phi, totalN,
      totalLoss, r2G, r2F, r2D,
    };
  }, [lr]);

  // Combined step: cheap most of the time, expensive every diagEvery steps
  const doStep = useCallback((forceExpensive = false) => {
    const curStep = stepRef.current;
    const doExpensive = forceExpensive || (curStep % diagEvery === 0);

    if (doExpensive) {
      const result = expensiveStep();
      if (!result) return;
      const { preds, actual, neuronCounts, Phi, totalN, totalLoss, r2G, r2F, r2D } = result;

      if (Phi) drawHeatmap(heatRef.current, Phi, totalN, neuronCounts);
      lastData.current = { preds, actual, neuronCounts };
      drawScatter(scatRef.current, preds[scatterMode], actual, neuronCounts, MODE_LABELS[scatterMode]);

      setStep(stepRef.current);
      setMetrics({ loss: totalLoss, r2G, r2F, r2D });
      setHistory(h => {
        return [...h, {
          step: stepRef.current, loss: +totalLoss.toFixed(5),
          "R² Eq.2": +r2G.toFixed(4),
          "R² Eq.3": +r2F.toFixed(4),
          "R² Eq.5": +r2D.toFixed(4),
        }];
      });
    } else {
      const totalLoss = cheapStep();
      setStep(stepRef.current);
      setMetrics(m => ({ ...m, loss: totalLoss }));
      // Update loss in history too
      setHistory(h => {
        if (h.length === 0) return h;
        // Just update the latest entry's loss or add a loss-only point
        const last = h[h.length - 1];
        return [...h.slice(0, -1), { ...last, loss: +totalLoss.toFixed(5) }];
      });
    }
  }, [cheapStep, expensiveStep, diagEvery, scatterMode]);

  // Redraw scatter on mode toggle
  useEffect(() => {
    if (lastData.current) {
      const { preds, actual, neuronCounts } = lastData.current;
      drawScatter(scatRef.current, preds[scatterMode], actual, neuronCounts, MODE_LABELS[scatterMode]);
    }
  }, [scatterMode]);

  // Training loop
  useEffect(() => {
    runRef.current = running;
    if (!running) return;
    let frame;
    const loop = () => {
      if (!runRef.current) return;
      // Run a batch of cheap steps, then one expensive
      const batchSize = Math.max(1, diagEvery);
      for (let i = 0; i < batchSize - 1; i++) {
        cheapStep();
        stepRef.current; // just advance
      }
      doStep(true); // expensive step with visuals
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [running, doStep, cheapStep, diagEvery]);

  const fmt = (v, d = 3) => v != null ? v.toFixed(d) : "\u2014";
  const sizes = [2, ...Array(depth).fill(width), 2];
  const totalN = sizes.slice(1).reduce((s, n) => s + n, 0);
  const m = metrics;

  // Thin history for chart: show at most ~200 points
  const maxPts = 200;
  const chartData = history.length <= maxPts
    ? history
    : history.filter((_, i) => {
        const k = Math.ceil(history.length / maxPts);
        return i % k === 0 || i === history.length - 1;
      });

  return (
    <div style={{ fontFamily: "'Georgia', serif", background: "#faf9f6", minHeight: "100vh", color: "#2a2a2e" }}>
      <div style={{ maxWidth: 940, margin: "0 auto", padding: "20px 20px" }}>

        <div style={{ borderBottom: "2px solid #2a2a2e", paddingBottom: 8, marginBottom: 14 }}>
          <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>
            Weight-space GD is kernel descent in activity space
          </h1>
          <p style={{ margin: "4px 0 0", fontSize: 12, color: "#6b6860", fontFamily: "monospace" }}>
            {"ΔA"}<sub>i</sub>{" = −η Σ"}<sub>k∈out</sub>{" Θ"}<sub>ik</sub>{" ∂L/∂A"}<sub>k</sub>
            {"  (partial, Eq.3)   ·   ΔA"}<sub>i</sub>{" ≈ −ηΘ"}<sub>ii</sub>{"·dL/dA"}<sub>i</sub>{" (backprop, Eq.5)"}
          </p>
        </div>

        <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap", marginBottom: 10, fontFamily: "monospace", fontSize: 11 }}>
          <label style={{ display: "flex", alignItems: "center", gap: 4 }}>
            width
            <input type="range" min={8} max={64} step={4} value={width}
              onChange={e => { setRunning(false); setWidth(+e.target.value); }} style={{ width: 70 }} />
            <b style={{ width: 22, textAlign: "right" }}>{width}</b>
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 4 }}>
            depth
            <input type="range" min={2} max={5} step={1} value={depth}
              onChange={e => { setRunning(false); setDepth(+e.target.value); }} style={{ width: 50 }} />
            <b>{depth}</b>
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 4 }}>
            η
            <input type="range" min={-4} max={0} step={0.05}
              value={Math.log10(lr)}
              onChange={e => setLr(+(10 ** +e.target.value).toPrecision(2))}
              style={{ width: 65 }} />
            <span>{lr < 0.001 ? lr.toExponential(0) : lr < 0.01 ? lr.toFixed(4) : lr < 0.1 ? lr.toFixed(3) : lr.toFixed(2)}</span>
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 4 }}>
            diag every
            <input type="range" min={10} max={200} step={10} value={diagEvery}
              onChange={e => setDiagEvery(+e.target.value)} style={{ width: 55 }} />
            <span>{diagEvery}</span>
          </label>
          {[
            { label: "Step", fn: () => doStep(true), bg: "#fff", fg: "#2a2a2e", bd: "#2a2a2e" },
            { label: running ? "Stop" : "Train", fn: () => setRunning(r => !r),
              bg: running ? "#2a2a2e" : "#fff", fg: running ? "#faf9f6" : "#2a2a2e", bd: "#2a2a2e" },
            { label: "Reset", fn: () => { setRunning(false); initNetwork(); },
              bg: "#fff", fg: "#888", bd: "#bbb" },
          ].map(b => (
            <button key={b.label} onClick={b.fn}
              style={{ padding: "3px 12px", border: `1.5px solid ${b.bd}`, borderRadius: 3,
                background: b.bg, color: b.fg, cursor: "pointer", fontFamily: "monospace", fontSize: 11 }}>
              {b.label}
            </button>
          ))}
          <span style={{ color: "#999", fontSize: 10 }}>
            [{sizes.join(",")}] &middot; {totalN}n &middot; step {step}
          </span>
        </div>

        <div style={{ display: "flex", gap: 18, marginBottom: 12, fontFamily: "monospace", fontSize: 11, flexWrap: "wrap" }}>
          <span>loss: <b>{fmt(m.loss, 4)}</b></span>
          <span style={{ color: "#888" }}>R²<sub style={{fontSize:9}}>Eq.2</sub>: <b>{fmt(m.r2G)}</b></span>
          <span style={{ color: "#2563eb" }}>R²<sub style={{fontSize:9}}>Eq.3</sub>: <b>{fmt(m.r2F)}</b></span>
          <span style={{ color: "#059669" }}>R²<sub style={{fontSize:9}}>Eq.5</sub>: <b>{fmt(m.r2D)}</b></span>
        </div>

        <div style={{ background: "#fff", border: "1px solid #e0ddd5", borderRadius: 4, padding: 14, marginBottom: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
            <div style={{ fontSize: 15, fontWeight: 700 }}>Does GD on weights look like GD on activities?</div>
            <div style={{ display: "flex", gap: 0, fontFamily: "monospace", fontSize: 10 }}>
              {MODES.map((mode, mi) => (
                <button key={mode} onClick={() => setScatterMode(mode)}
                  style={{
                    padding: "3px 8px", border: "1px solid #ccc",
                    borderLeft: mi > 0 ? "none" : undefined,
                    borderRadius: mi === 0 ? "3px 0 0 3px" : mi === 2 ? "0 3px 3px 0" : "0",
                    background: scatterMode === mode ? "#2a2a2e" : "#fff",
                    color: scatterMode === mode ? "#fff" : "#666",
                    cursor: "pointer", fontFamily: "monospace", fontSize: 10,
                  }}>
                  {mode === "ground" ? "Eq.2" : mode === "fullPhi" ? "Eq.3" : "Eq.5"}
                </button>
              ))}
            </div>
          </div>
          <div style={{ fontSize: 11, color: "#888", marginBottom: 8, fontFamily: "monospace" }}>
            {scatterMode === "ground" && "Eq.2: J·ΔW — first-order Taylor identity (sanity check)"}
            {scatterMode === "fullPhi" && "Eq.3: Φ·∂L — kernel with partial derivs, sum over outputs only"}
            {scatterMode === "diagBP" && "Eq.5: Φᵢᵢ × backprop gradient — diagonal approximation"}
          </div>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <canvas ref={scatRef} width={540} height={540}
              style={{ width: "100%", maxWidth: 540, height: "auto", aspectRatio: "1", border: "1px solid #eee" }} />
          </div>
          <div style={{ display: "flex", gap: 12, justifyContent: "center", marginTop: 6, fontSize: 10, fontFamily: "monospace" }}>
            {sizes.slice(1).map((n, i) => (
              <span key={i} style={{ display: "flex", alignItems: "center", gap: 3, color: LCOLORS[i % 4] }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: LCOLORS[i % 4], display: "inline-block" }} />
                L{i + 1} ({n})
              </span>
            ))}
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <div style={{ background: "#fff", border: "1px solid #e0ddd5", borderRadius: 4, padding: 10 }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{"Θ"}<sub>ik</sub> correlation</div>
            <div style={{ fontSize: 10, color: "#888", marginBottom: 6, fontFamily: "monospace" }}>
              {"Θ"}<sub>ik</sub>{"/√(Θ"}<sub>ii</sub>{"Θ"}<sub>kk</sub>{") · lines = layers"}
            </div>
            <canvas ref={heatRef} width={340} height={340}
              style={{ width: "100%", maxWidth: 340, height: "auto", aspectRatio: "1", imageRendering: "pixelated", border: "1px solid #eee" }} />
            <div style={{ display: "flex", justifyContent: "center", gap: 10, marginTop: 5, fontSize: 9, fontFamily: "monospace", color: "#888" }}>
              {[["rgb(255,55,55)", "−1"], ["#fff", "0"], ["rgb(55,55,255)", "+1"]].map(([c, l]) => (
                <span key={l} style={{ display: "flex", alignItems: "center", gap: 2 }}>
                  <span style={{ width: 9, height: 9, background: c, border: "1px solid #ccc", display: "inline-block" }} />{l}
                </span>
              ))}
            </div>
          </div>

          <div style={{ background: "#fff", border: "1px solid #e0ddd5", borderRadius: 4, padding: 10 }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>Training dynamics</div>
            <div style={{ fontSize: 10, color: "#888", marginBottom: 6, fontFamily: "monospace" }}>
              R² at diagnostic checkpoints. Gap between Eq.3 and Eq.5 = cost of diagonal approximation.
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData} margin={{ top: 2, right: 40, bottom: 2, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="step" tick={{ fontSize: 9, fontFamily: "monospace" }} />
                <YAxis yAxisId="left" domain={[-0.5, 1.05]} tick={{ fontSize: 9, fontFamily: "monospace" }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 9, fontFamily: "monospace" }} />
                <Tooltip contentStyle={{ fontSize: 10, fontFamily: "monospace" }} />
                <ReferenceLine yAxisId="left" y={1} stroke="#e8e5dd" strokeDasharray="4 4" />
                <ReferenceLine yAxisId="left" y={0} stroke="#e8e5dd" />
                <Line yAxisId="left" type="monotone" dataKey="R² Eq.2" stroke="#aaa" strokeWidth={1} dot={false} isAnimationActive={false} strokeDasharray="3 2" />
                <Line yAxisId="left" type="monotone" dataKey="R² Eq.3" stroke="#2563eb" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                <Line yAxisId="left" type="monotone" dataKey="R² Eq.5" stroke="#059669" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                <Line yAxisId="right" type="monotone" dataKey="loss" stroke="#d4a" strokeWidth={1} dot={false} isAnimationActive={false} strokeDasharray="4 3" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div style={{ marginTop: 14, padding: "10px 12px", background: "#f5f4f0", borderRadius: 4,
          fontSize: 11, color: "#6b6860", lineHeight: 1.7, fontFamily: "monospace" }}>
          <b>Equations from the paper:</b> SGD runs every step (fast). Jacobian/Θ computed every {diagEvery} steps.<br/>
          <b>Eq.2</b> (J·ΔW): first-order Taylor, exact by construction.
          <b>Eq.3</b> (Θ·∂L): kernel prediction using partial derivatives (sum over outputs only). Should match Eq.2.
          <b>Eq.5</b> (Θ<sub>ii</sub>·dL/dA<sub>i</sub>): diagonal approx with backprop gradient. Tightens with width.
        </div>
      </div>
    </div>
  );
}
