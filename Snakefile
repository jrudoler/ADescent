"""Reproducible workflow for simulations, figures, and manuscript."""

MAIN_FIGURE_NAMES = [
    "fig_ntk",
    "fig_width_sweep",
    "fig_depth_sweep",
    "fig_width_sweep_appendix",
]
MAIN_FIGURES = [
    f"results/figures/{name}.{suffix}"
    for name in MAIN_FIGURE_NAMES
    for suffix in ("pdf", "png")
]
PAPER_FIGURES = [
    f"paper/generated/figures/{name}.pdf" for name in MAIN_FIGURE_NAMES
]
DROPOUT_FIGURES = [
    f"results/figures/{name}.{suffix}"
    for name in ("fig_dropout_sweep", "fig_dropout_phi_heatmaps")
    for suffix in ("pdf", "png")
]
rule all:
    input:
        MAIN_FIGURES,
        DROPOUT_FIGURES,
        "data/generated/dropout_results.json",
        PAPER_FIGURES,
        "paper/activity_dynamics.pdf",


rule generate_main_figures:
    input:
        script="analysis/generate_main_figures/run.py",
        lockfile="uv.lock",
    output:
        MAIN_FIGURES,
    shell:
        "PYTHONPATH=src uv run python {input.script} --output-dir results/figures"


rule run_dropout:
    input:
        script="analysis/run_dropout/run.py",
        lockfile="uv.lock",
    output:
        data="data/generated/dropout_results.json",
        figures=DROPOUT_FIGURES,
    shell:
        """
        PYTHONPATH=src uv run python {input.script} \
          --data-output {output.data} \
          --figure-output-dir results/figures
        """


rule collect_paper_assets:
    input:
        figures=[f"results/figures/{name}.pdf" for name in MAIN_FIGURE_NAMES],
    output:
        figures=PAPER_FIGURES,
    shell:
        """
        mkdir -p paper/generated/figures
        cp {input.figures} paper/generated/figures/
        """


rule compile_paper:
    input:
        manuscript="paper/activity_dynamics.tex",
        references="paper/activity_dynamics.bib",
        figures=PAPER_FIGURES,
        template=["paper/jmlr.cls", "paper/jmlrutils.sty"],
    output:
        "paper/activity_dynamics.pdf",
    shell:
        "latexmk -pdf -interaction=nonstopmode -halt-on-error -cd {input.manuscript}"
