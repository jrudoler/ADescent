# Gradient Descent in Weight Space Is Kernel Descent in Activity Space

A short paper showing that gradient descent on network weights induces **kernel descent** in the space of neural activities, governed by a neural-tangent-kernel-style Gram matrix on internal neurons.

**Key result:** When the kernel is diagonally dominant (wide networks), each neuron's activity change is approximately proportional to the negative loss gradient with respect to that neuron's activity — converting untestable claims about synaptic learning rules into testable predictions about observable activity changes.

## Reproduce the results

Install `uv`, then run the complete dependency graph from the repository root:

```bash
uv run snakemake --cores 1
```

This regenerates seeded simulation outputs, publication-ready figures,
manuscript assets, and the compiled paper. The Snakemake dependency graph is
the authoritative provenance record.
To rebuild one step, use `uv run snakemake --cores 1 --forcerun <rule_name>`.

## Repository layout

- [`Snakefile`](Snakefile) — the executable workflow and artifact lineage
- [`analysis/`](analysis/) — one explicit entrypoint per workflow step
- [`data/generated/`](data/generated/) — machine-generated intermediates
- [`results/`](results/) — final scientific tables and figures
- [`paper/activity_dynamics.tex`](paper/activity_dynamics.tex) — manuscript source
- [`paper/activity_dynamics.pdf`](https://github.com/koerding/ADescent/raw/main/paper/activity_dynamics.pdf) — compiled paper
- [`index.html`](index.html) — static interactive demo for GitHub Pages

## Interactive Demo

**[Launch the interactive demo](https://koerding.github.io/ADescent/)** — the runnable demo is the static browser page in [`index.html`](index.html), served directly by GitHub Pages with no install or build step. Adjust width, depth, and learning rate to see how the kernel and diagonal approximation behave.
