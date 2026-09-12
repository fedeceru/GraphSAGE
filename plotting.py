"""
All matplotlib figure-generation code used by ``notebook.ipynb``'s §8/§8.1/
§8.2/§8.3/§9.1 analysis cells, kept out of the notebook itself so its
narrative reads as data loading + analysis choices + a call into this
module, not inline chart-drawing boilerplate.

Every ``plot_*`` function here takes already-prepared data (a DataFrame, a
dict loaded from one of ``results/*.json``, or a map of already-resolved
log directories) and produces one matplotlib figure: it draws the figure,
optionally saves it to ``save_path`` (the caller decides *where*, e.g.
under ``results/``, since this module has no opinion on the repo's
directory layout), displays it inline via ``plt.show()``, and returns the
``Figure`` object. A couple of pure data-shaping helpers
(``extract_last_avg_time``, ``_read_metrics_csv``) live here too: their only
reason to exist is to feed one of the plotting functions below, not to
serve any other part of the reproduction pipeline.

House style: a single ``plt.rcParams`` block below (top/right spines off,
a subtle opt-in grid, frameless legends) applies to every figure in this
module, so the whole notebook has one consistent, low-clutter visual
language instead of each function repeating its own styling. Individual
functions still choose *whether* a grid helps their specific chart (a
sparse bar/line chart reads more precisely with one; a dense multi-panel
sheet or a 2D embedding scatter would only get noisier).

This module is imported as a plain top-level module (``import plotting``)
from ``notebook.ipynb``, which inserts the repo root onto ``sys.path``
before importing it -- see notebook.ipynb §8's first code cell.
"""
from __future__ import annotations

import csv
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4a4a4a",
    "axes.linewidth": 0.8,
    "axes.grid": False,          # grids are opt-in per axis, via _light_grid()
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

# Shared across every plot in the notebook that breaks results down by
# aggregator, so "GraphSAGE-pool" (say) is always the same color everywhere
# -- results tables' bar charts, loss curves, the embedding-snapshot titles.
MODEL_COLORS = {
    "GraphSAGE-GCN": "#eb6834",
    "GraphSAGE-mean": "#2a78d6",
    "GraphSAGE-LSTM": "#4a3aa7",
    "GraphSAGE-pool": "#1baf7a",
}
# Neutral color for rows/series that aren't a GraphSAGE variant (the
# Random/Raw-features baselines) -- reused everywhere a baseline needs a
# color, so "grey = not a GraphSAGE variant" reads consistently.
BASELINE_COLOR = "#888888"
# "Cost" color (training/inference time), reused wherever a chart shows a
# runtime rather than an accuracy quantity.
COST_COLOR = "#c05621"


def _bar_color(display_name):
    return MODEL_COLORS.get(display_name, BASELINE_COLOR)


def _light_grid(ax, axis="y"):
    """Opt-in, deliberately subtle gridline (uses the rcParams grid style
    above) drawn behind the data -- for charts where reading an exact value
    matters (bar/line charts with few series), not dense multi-panel sheets
    or 2D scatter plots where it would only add noise."""
    ax.set_axisbelow(True)
    ax.grid(axis=axis)


def _ema(values, alpha=0.85):
    """TensorBoard-style exponential moving average -- the standard fix for
    a noisy, high-frequency per-step training scalar. ``alpha`` is the
    smoothing weight (closer to 1 = smoother); returns a same-length list,
    seeded at the first value so it doesn't drift on short series."""
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * out[-1] + (1 - alpha) * v)
    return out


def _darken(hex_color, factor=0.55):
    """Scale a color toward black by ``factor`` (0 = unchanged, 1 = black)
    -- used to give validation series a distinct, darker tone of the same
    model color rather than an unrelated second hue."""
    r, g, b = to_rgb(hex_color)
    return (r * (1 - factor), g * (1 - factor), b * (1 - factor))


def plot_f1_comparison(df, save_path=None):
    """§8 bar chart: reproduced vs. paper Micro F1, one panel each for the
    unsupervised and supervised settings. Bars are colored by aggregator
    using the same ``MODEL_COLORS`` as every other chart in the notebook
    (baselines get the neutral ``BASELINE_COLOR``); "paper" vs. "reproduced"
    is distinguished by alpha on that same color, not an unrelated second
    color pair.

    ``df`` must have the columns produced by notebook.ipynb's results-table
    cell: ``"Name"``, ``"Unsup. F1 (reproduced)"``, ``"Unsup. F1 (paper)"``,
    ``"Sup. F1 (reproduced)"``, ``"Sup. F1 (paper)"``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    x = np.arange(len(df))
    width = 0.35
    colors = [_bar_color(n) for n in df["Name"]]

    for i, (ax, col_repro, col_paper, title) in enumerate([
        (axes[0], "Unsup. F1 (reproduced)", "Unsup. F1 (paper)", "Unsupervised F1 — PPI"),
        (axes[1], "Sup. F1 (reproduced)", "Sup. F1 (paper)", "Supervised F1 — PPI"),
    ]):
        ax.bar(x - width / 2, df[col_paper], width, color=colors, alpha=0.4)
        ax.bar(x + width / 2, df[col_repro], width, color=colors, alpha=0.95)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Name"], rotation=40, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Micro F1")
        _light_grid(ax)
        if i == 0:
            handles = [Patch(facecolor="#4a4a4a", alpha=0.4, label="Paper"),
                       Patch(facecolor="#4a4a4a", alpha=0.95, label="Reproduced")]
            ax.legend(handles=handles, loc="upper left")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    plt.show()
    return fig


def extract_last_avg_time(log_path):
    """Extract the last `time=` value (average seconds per iteration)
    printed by the training script in its console log."""
    if not os.path.exists(log_path):
        return None
    last = None
    with open(log_path, encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            m = re.search(r"time=\s*([\d.]+)\s*$", line.strip())
            if m:
                last = float(m.group(1))
    return last


def plot_timing(timing_df, save_path=None):
    """§8.1 bar chart: average seconds/iteration per aggregator, supervised
    vs. unsupervised side by side.

    ``timing_df`` needs columns ``"Name"``, ``"Sup. sec/iter"``,
    ``"Unsup. sec/iter"`` (see ``extract_last_avg_time`` above).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(timing_df))
    width = 0.35
    ax.bar(x - width / 2, timing_df["Sup. sec/iter"], width, label="Supervised", color="#2b6cb0")
    ax.bar(x + width / 2, timing_df["Unsup. sec/iter"], width, label="Unsupervised", color=COST_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(timing_df["Name"], rotation=30, ha="right")
    ax.set_ylabel("seconds / iteration (average, GPU)")
    ax.set_title("Training time per variant — PPI (RTX 2060)")
    ax.legend()
    _light_grid(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    plt.show()
    return fig


def _read_metrics_csv(csv_path):
    """Load a supervised_train.py / unsupervised_train.py ``metrics.csv``
    (step, epoch, train/val loss, and train/val F1 or MRR -- written
    specifically so this kind of programmatic plotting doesn't need to
    parse printed console text) into a dict of column name -> list of
    float values. Returns ``{}`` if the file doesn't exist (e.g. a
    candidate that hasn't been trained under this exact config)."""
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        cols = {name: [] for name in (reader.fieldnames or [])}
        for row in reader:
            for k, v in row.items():
                cols[k].append(float(v))
    return cols


def _plot_curve_row(axes_row, models, log_dirs, train_col, val_col, ylabel,
                     xlabel=None, show_legend=False):
    """Draw one row of a training-curve sheet: one panel per (model_flag,
    display_name) in ``models``, each plotting ``train_col``/``val_col``
    from that model's ``metrics.csv``, with a faint vertical line at each
    new epoch boundary. Shared by ``plot_loss_curves`` and
    ``plot_performance_curves`` -- they differ only in which columns they
    read.

    Train and val are genuinely different kinds of signal here -- train is
    logged every step (dense, noisy), val only refreshes once per epoch
    (sparse, discrete) -- so they're drawn with different encodings, not
    just a color/dash variant of the same line style, so they stay legible
    even where the two are numerically close or cross: train is a faint
    raw trace plus a bold EMA-smoothed line (the noise-reduction trick
    TensorBoard's own smoothing slider uses); val is plotted only at its
    actual change-points, as sparse hollow markers in a darkened tone of
    the same model color, connected by a thin dotted line -- reading as
    "discrete measurement" rather than a second continuous curve.
    """
    for i, (ax, (model_flag, display_name)) in enumerate(zip(axes_row, models)):
        metrics = _read_metrics_csv(os.path.join(log_dirs[model_flag], "metrics.csv"))
        steps = metrics.get("step", [])
        epochs = metrics.get("epoch", [])
        color = MODEL_COLORS[display_name]
        dark = _darken(color)

        seen_epochs = set()
        for step, epoch in zip(steps, epochs):
            if epoch not in seen_epochs:
                seen_epochs.add(epoch)
                if len(seen_epochs) > 1:
                    ax.axvline(step, color="#e1e0d9", lw=0.8, zorder=0)

        train_vals = metrics.get(train_col, [])
        ax.plot(steps, train_vals, color=color, lw=0.6, alpha=0.25, zorder=1)
        ax.plot(steps, _ema(train_vals), color=color, lw=1.8, zorder=3)

        val_vals = metrics.get(val_col, [])
        val_change_steps, val_change_vals = [], []
        for step, v in zip(steps, val_vals):
            if not val_change_vals or v != val_change_vals[-1]:
                val_change_steps.append(step)
                val_change_vals.append(v)
        if val_change_steps and val_change_steps[-1] != steps[-1]:
            # extend the last known value flat to the right edge -- it's
            # still in effect, just not re-measured again after this point
            line_steps = val_change_steps + [steps[-1]]
            line_vals = val_change_vals + [val_change_vals[-1]]
        else:
            line_steps, line_vals = val_change_steps, val_change_vals
        ax.plot(line_steps, line_vals, color=dark, lw=1.0, ls=":", drawstyle="steps-post", zorder=4)
        ax.plot(val_change_steps, val_change_vals, lw=0, marker="o", ms=4.5,
                 mfc="white", mec=dark, mew=1.2, zorder=5)

        ax.set_title(display_name, color=color, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=8)
        if show_legend and i == 0:
            ax.legend([Line2D([0], [0], color="#4a4a4a", lw=1.8),
                       Line2D([0], [0], color="#4a4a4a", lw=1.0, ls=":", marker="o", ms=4.5,
                              mfc="white", mec="#4a4a4a", mew=1.2)],
                      ["train (smoothed)", "val"], loc="upper right", fontsize=7)


def plot_loss_curves(sup_log_dirs, unsup_log_dirs, models, save_path=None):
    """§8.1 training-loss sheet: one row for supervised, one for
    unsupervised, one column per aggregator in ``models``, read from each
    run's ``metrics.csv``.

    Train (dense, logged every step) and val (sparse, refreshed once per
    epoch) are genuinely different kinds of signal, so they're drawn with
    different encodings rather than a same-color solid/dashed pair -- see
    ``_plot_curve_row`` for the reasoning and rendering.

    ``sup_log_dirs``/``unsup_log_dirs`` map ``model_flag -> log_dir`` for
    the run that actually produced this repo's reported result (the
    Appendix-C-sweep winner when one exists, else the code's default) --
    the notebook resolves these paths, since that requires knowing about
    ``results/best_hparams_ppi.json``, not something this module has an
    opinion on.
    """
    fig, axes = plt.subplots(2, len(models), figsize=(14, 7.2), squeeze=False)
    _plot_curve_row(axes[0], models, sup_log_dirs, "train_loss", "val_loss",
                     "Supervised loss", show_legend=True)
    _plot_curve_row(axes[1], models, unsup_log_dirs, "train_loss", "val_loss",
                     "Unsupervised loss", xlabel="logged step (one point per --print_every steps)")
    fig.suptitle("Training loss — PPI (faint vertical lines mark epoch boundaries)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    plt.show()
    return fig


def plot_performance_curves(sup_log_dirs, unsup_log_dirs, models, save_path=None):
    """Companion to ``plot_loss_curves``, same layout, data source, and
    train/val encoding (see ``_plot_curve_row``): the metric each setting is
    actually judged on (F1 micro for supervised, MRR for unsupervised) over
    training, instead of the loss. Loss shows whether optimization is
    converging; this shows whether that convergence is translating into the
    task performance reported elsewhere in this notebook.
    """
    fig, axes = plt.subplots(2, len(models), figsize=(14, 7.2), squeeze=False)
    _plot_curve_row(axes[0], models, sup_log_dirs, "train_f1_micro", "val_f1_micro",
                     "Supervised F1 (micro)", show_legend=True)
    _plot_curve_row(axes[1], models, unsup_log_dirs, "train_mrr", "val_mrr",
                     "Unsupervised MRR", xlabel="logged step (one point per --print_every steps)")
    fig.suptitle("Training performance — PPI (the metric each setting is actually judged on)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    plt.show()
    return fig


def plot_stage_sheet(coords_by_step, steps, split_colors, sub_split, n_nodes,
                      title, xlabel, ylabel, n_traj=35, rng_seed=7,
                      sub_degree=None, save_path=None):
    """§8.2 one consolidated sheet per projection method (PCA or t-SNE): the
    small-multiples "stages" row (one panel per snapshot step) on top, and
    the "trajectories" panel (a subset of nodes traced across every stage)
    below -- both are the same underlying (coords_by_step, embeddings)
    data, just two views of it, so they read as one figure instead of two.

    ``coords_by_step`` maps each of ``steps`` to an ``[n_nodes, 2]``
    projected-coordinate array; ``split_colors`` maps split name -> color,
    and ``sub_split`` is the split label ("train"/"val"/"test") of each of
    the ``n_nodes`` tracked nodes, in the same row order as
    ``coords_by_step``.

    If ``sub_degree`` (that same node ordering's graph degree) is given, a
    third panel is added: the final stage colored by degree instead of
    split -- Theorem 1 is specifically about the pooling aggregator
    exploiting structural (degree/clustering-related) signal, so this is
    the most direct visual check of that claim available from data this
    repo already collects.
    """
    n_steps = len(steps)
    has_degree = sub_degree is not None
    n_rows = 3 if has_degree else 2
    height_ratios = [1, 1.9, 1.3] if has_degree else [1, 1.9]
    fig = plt.figure(figsize=(2.6 * n_steps, 9.5 + (2.7 if has_degree else 0)))
    gs = GridSpec(n_rows, n_steps, height_ratios=height_ratios, hspace=0.6, figure=fig)

    stage_axes = [fig.add_subplot(gs[0, i]) for i in range(n_steps)]
    for ax, step in zip(stage_axes, steps):
        coords = coords_by_step[step]
        for split_name, color in split_colors.items():
            mask = sub_split == split_name
            ax.scatter(coords[mask, 0], coords[mask, 1], s=9, alpha=0.45, color=color, linewidths=0)
        ax.set_title(f"step {step}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():  # tick-less panels: a box outline is pure frame, no information
            spine.set_visible(False)
    stage_axes[0].set_ylabel(ylabel, fontsize=10)
    stage_axes[n_steps // 2].set_xlabel(xlabel, fontsize=10)

    span = max(2, n_steps // 3)
    start_col = (n_steps - span) // 2
    traj_ax = fig.add_subplot(gs[1, start_col:start_col + span])
    rng_traj = np.random.RandomState(rng_seed)
    traj_idx = rng_traj.choice(n_nodes, size=min(n_traj, n_nodes), replace=False)
    for k in traj_idx:
        xs_ = [coords_by_step[step][k, 0] for step in steps]
        ys_ = [coords_by_step[step][k, 1] for step in steps]
        color = split_colors[sub_split[k]]
        traj_ax.plot(xs_, ys_, color="#898781", alpha=0.45, lw=1.0, zorder=1)
        traj_ax.scatter(xs_[0], ys_[0], s=30, color=color, marker="o", edgecolors="#0b0b0b", linewidths=0.5, zorder=2)
        traj_ax.scatter(xs_[-1], ys_[-1], s=110, color=color, marker="*", edgecolors="#0b0b0b", linewidths=0.6, zorder=3)
    traj_ax.set_xlabel(xlabel, fontsize=10)
    traj_ax.set_ylabel(ylabel, fontsize=10)
    traj_ax.set_title(
        f"individual trajectories (circle = step {steps[0]}, star = step {steps[-1]}, n={len(traj_idx)} nodes)",
        fontsize=11)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="#0b0b0b", markersize=9, label=s)
               for s, c in split_colors.items()]
    traj_ax.legend(handles=handles, loc="best", fontsize=9)

    if has_degree:
        deg_span = max(2, n_steps // 3)
        deg_start = (n_steps - deg_span) // 2
        deg_ax = fig.add_subplot(gs[2, deg_start:deg_start + deg_span])
        final_coords = coords_by_step[steps[-1]]
        log_degree = np.log10(np.asarray(sub_degree) + 1)
        sc = deg_ax.scatter(final_coords[:, 0], final_coords[:, 1], s=14, c=log_degree,
                             cmap="viridis", linewidths=0)
        deg_ax.set_xlabel(xlabel, fontsize=10)
        deg_ax.set_ylabel(ylabel, fontsize=10)
        deg_ax.set_title(f"final stage (step {steps[-1]}), colored by degree "
                          "(structural signal Theorem 1 argues pooling can exploit)", fontsize=10)
        cbar = fig.colorbar(sc, ax=deg_ax, fraction=0.03, pad=0.02)
        cbar.set_label("log10(degree + 1)", fontsize=8)

    fig.suptitle(title, y=0.98, fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()
    return fig


def plot_k_sensitivity(k_df, save_path=None):
    """§8.3 both halves of Section 4.3's headline K claim in one panel: F1
    (left axis) and average training seconds/iteration (right axis, where
    available -- the K=2 point typically reuses the canonical run, which
    wasn't separately timed) against K. The paper's own claim is explicitly
    about this tradeoff (a 10-15% F1 gain from K=1->2 against a 10-100x
    runtime cost beyond K=2), so both halves belong in one panel rather
    than two unrelated charts.

    ``k_df`` needs columns ``"K"``, ``"test_f1_micro"``, and (optionally,
    may contain missing/NaN entries) ``"sec_per_iter"``.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(k_df["K"], k_df["test_f1_micro"], marker="o", color="#2a78d6", label="Test F1 (micro)")
    ax.set_xlabel("K (search depth)")
    ax.set_ylabel("Test Micro F1", color="#2a78d6")
    ax.tick_params(axis="y", labelcolor="#2a78d6")
    ax.set_xticks(k_df["K"])
    _light_grid(ax)

    lines = ax.get_lines()
    if "sec_per_iter" in k_df:
        timing = k_df.dropna(subset=["sec_per_iter"])
        if len(timing):
            ax2 = ax.twinx()
            ax2.plot(timing["K"], timing["sec_per_iter"], marker="s", color=COST_COLOR, label="sec / iteration")
            ax2.set_ylabel("sec / iteration", color=COST_COLOR)
            ax2.tick_params(axis="y", labelcolor=COST_COLOR)
            lines = lines + ax2.get_lines()

    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
    ax.set_title("GraphSAGE-mean (supervised): F1 & runtime vs. K, on PPI")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    plt.show()
    return fig


def plot_sample_size_sensitivity(ss_df, save_path=None):
    """Analogue of ``plot_k_sensitivity`` for the paper's other Section 4.3
    claim -- "diminishing returns for sampling large neighborhoods"
    (Figure 2B): F1 vs. neighborhood sample size S, at fixed K=2. Only
    populated when ``sensitivity_ppi_experiments.py`` is run without
    ``--k_only`` (the scaled run in this repo skips it to fit the time
    budget, so this plot is here for whenever that fuller sweep is run).

    ``ss_df`` needs columns ``"sample_size"`` and ``"test_f1_micro"``.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ss_df["sample_size"], ss_df["test_f1_micro"], marker="o", color="#2a78d6")
    ax.set_xlabel("Neighborhood sample size S (S1 = S2 = S, K=2)")
    ax.set_ylabel("Test Micro F1")
    ax.set_title("GraphSAGE-mean (supervised): F1 vs. sample size, on PPI")
    _light_grid(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    plt.show()
    return fig


def plot_noise_robustness(noise_results, save_path=None):
    """§9.1 F1-vs-feature-noise-proportion line plot (PPI analogue of the
    paper's Figure 3). Reuses the same ``MODEL_COLORS``/``BASELINE_COLOR``
    convention as every other chart, instead of its own one-off palette.

    ``noise_results`` is the dict loaded from
    ``results/noise_robustness_ppi.json``.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    series_colors = {
        "GraphSAGE-pool": MODEL_COLORS["GraphSAGE-pool"],
        "GraphSAGE-GCN": MODEL_COLORS["GraphSAGE-GCN"],
        "Raw features": BASELINE_COLOR,
    }
    for name, color in series_colors.items():
        f1s = noise_results.get(name, {}).get("test_f1_micro", [])
        if f1s:
            ax.plot(noise_results["noise_props"][:len(f1s)], f1s, marker="o", color=color, label=name)
    ax.set_xlabel("Feature noise proportion")
    ax.set_ylabel("Test Micro F1")
    ax.set_title("PPI analogue of Figure 3: F1 vs. feature noise proportion")
    ax.legend()
    _light_grid(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    plt.show()
    return fig
