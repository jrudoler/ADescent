# Gradient Descent in Weight Space Is Kernel Descent in Activity Space

A short paper showing that gradient descent on network weights induces **kernel descent** in the space of neural activities, governed by a neural-tangent-kernel-style Gram matrix on internal neurons.

**Key result:** When the kernel is diagonally dominant (wide networks), each neuron's activity change is approximately proportional to the negative loss gradient with respect to that neuron's activity — converting untestable claims about synaptic learning rules into testable predictions about observable activity changes.

## Files

- [`activity_dynamics.tex`](activity_dynamics.tex) — Paper source
- [`activity_dynamics.pdf`](https://github.com/koerding/ADescent/raw/main/activity_dynamics.pdf) — Compiled PDF
- [`gen_figures.py`](gen_figures.py) — Python script to generate figures
- [`activity_ntk_demo.jsx`](activity_ntk_demo.jsx) — Interactive demo (React/JSX)

## Interactive Demo

See [`activity_ntk_demo.jsx`](activity_ntk_demo.jsx) for a React component that lets you explore the kernel and activity dynamics interactively.
