import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, ranksums, wilcoxon
from matplotlib.lines import Line2D
from plotting.plot_utils import _plot_single_polar_tuning
from utils import (
    load_config,
    load_inc,
    load_pat,
    sort_dates_and_make_session_titles,
    set_plot_defaults,
    load_video,
)

def make_example_polar_plot(outpath, tuning, color, r_gv=None):
    fig, ax = plt.subplots(figsize=(1.2, 1.2))
    scale = np.nanmax(tuning)
    
    _plot_single_polar_tuning(
        ax=ax,
        tuning=tuning,
        tuning_sem=np.zeros_like(tuning),
        baseline_rate_hz=0.0,
        scale=scale,
        color=color,
        ls="-",
        add_error=False,
        add_baseline=False,
        size_mult=0.4,
    )

    if r_gv is not None:
        ax.text(
            -1.20,
            1.12,
            f"$R_{{GV}}$ = {r_gv:.2f}",
            fontsize=6,
            ha="left",
            va="top",
        )

    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def make_combined_polar_plot(outpath, gr_tuning, vid_tuning, color, r_gv=None):
    fig, ax = plt.subplots(figsize=(1.2, 1.2))
    scale = max(np.nanmax(gr_tuning), np.nanmax(vid_tuning))

    # Grating tuning curve (dashed line)
    _plot_single_polar_tuning(
        ax=ax,
        tuning=gr_tuning,
        tuning_sem=np.zeros_like(gr_tuning),
        baseline_rate_hz=0.0,
        scale=scale,
        color=color,
        ls="--",
        add_error=False,
        add_baseline=False,
        size_mult=0.001,
    )

    # Video tuning curve (solid line)
    _plot_single_polar_tuning(
        ax=ax,
        tuning=vid_tuning,
        tuning_sem=np.zeros_like(vid_tuning),
        baseline_rate_hz=0.0,
        scale=scale,
        color=color,
        ls="-",
        add_error=False,
        add_baseline=False,
        size_mult=0.4,
    )

    legend_handles = [
        Line2D([0], [0], color=color, ls="--", lw=0.5),  # Grating
        Line2D([0], [0], color=color, ls="-", lw=0.5),   # Video
    ]
    ax.legend(
        handles=legend_handles,
        labels=["", ""],
        loc="lower left",
        frameon=False,
        handletextpad=0.0,
        handlelength=1.5,
    )

    if r_gv is not None:
        ax.text(
            -1.20,
            1.12,
            f"$R_{{GV}}$ = {r_gv:.2f}",
            fontsize=6,
            ha="left",
            va="top",
        )
    
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def get_inc_mask(inc_dict, date):
    if date in inc_dict:
        return np.asarray(inc_dict[date], bool)
    elif str(date) in inc_dict:
        return np.asarray(inc_dict[str(date)], bool)
    return None


def get_asterisks(p_val):
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    return ""

def plot_single_hist(data, xlabel, color="#AA0B07"):
    fig, ax = plt.subplots(figsize=(2.8, 2.75))
    bins = np.linspace(-1.0, 1.0, 21)
    
    clean_data = data[~np.isnan(data)]
    weights = np.ones_like(clean_data) / len(clean_data) if len(clean_data) > 0 else None

    ax.hist(
        clean_data,
        bins=bins,
        weights=weights,
        color=color,
        edgecolor="k",
        linewidth=0.8,
        alpha=0.85,
    )
    
    med_val = np.median(clean_data)
    ax.axvline(
        med_val, color="k", linestyle="--", linewidth=1.2, zorder=3
    )

    # One-sample Wilcoxon signed-rank test against 0 (greater)
    stat, p_val = wilcoxon(clean_data, alternative="greater")
    print(f"[{xlabel}] One-sample Wilcoxon vs 0: Median = {med_val:.3f}, p = {p_val:.4e}")

    ast_text = get_asterisks(p_val)
    if ast_text:
        ax.text(
            0.95,
            0.95,
            ast_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=14,
            color="k",
        )

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_xlim(-1.05, 1.05)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.05,
        0.92,
        "$n_{neurons}$" + f"$N$ = {len(clean_data):,}\nMedian = {med_val:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    plt.tight_layout()
    return fig, ax

def plot_overlay_hist(data_pat, data_comp, xlabel, pattern_color, component_color):
    fig, ax = plt.subplots(figsize=(2.8, 2.3))
    bins = np.linspace(-1.0, 1.0, 21)
    weights_pat = np.ones_like(data_pat) / len(data_pat) if len(data_pat) > 0 else None
    weights_comp = (
        np.ones_like(data_comp) / len(data_comp) if len(data_comp) > 0 else None
    )

    sns.kdeplot(
        x=data_comp,
        ax=ax,
        color=component_color,
        fill=True,
        alpha=0.35,
        linewidth=1,
        label="Component",
        clip=(-1, 1),
    )

    sns.kdeplot(
        x=data_pat,
        ax=ax,
        color=pattern_color,
        fill=True,
        alpha=0.35,
        linewidth=1,
        label="Pattern",
        clip=(-1, 1),
    )

    ax.axvline(
        np.nanmedian(data_pat),
        color=pattern_color,
        linestyle="--",
        linewidth=1.2,
        zorder=3,
    )
    ax.axvline(
        np.nanmedian(data_comp),
        color=component_color,
        linestyle="--",
        linewidth=1.2,
        zorder=3,
    )

    print(np.nanmedian(data_pat))
    print(np.nanmedian(data_comp))

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_xlim(-1.05, 1.05)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")

    leg.get_texts()[0].set_color(component_color)
    leg.get_texts()[1].set_color(pattern_color)

    _, p_val = ranksums(data_pat, data_comp)
    print(f"Pattern vs Component Wilcoxon rank-sum test p-value: {p_val:.4e}")
    ast_text = get_asterisks(p_val)
    if ast_text:
        ax.text(
            0.825,
            1.05,
            ast_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=14,
            color="k",
            fontweight='bold',
        )

    plt.tight_layout()
    return fig, ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml", type=str)
    parser.add_argument("--split", default="all", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    ordered_dates, _ = sort_dates_and_make_session_titles(config, args.split)

    pattern_color, component_color, unclass_color = set_plot_defaults()

    pat = load_pat()
    inc = load_inc()
    video_metrics = load_video()

    corr_all = []
    corr_pattern = []
    corr_component = []

    gr_pattern_all = []
    vid_pattern_all = []
    gr_component_all = []
    vid_component_all = []

    inc_plaid_dict = inc.get("rf_plaid", {})
    inc_video_dict = inc.get("video", {})

    for date in ordered_dates:
        if date not in pat["main"] or date not in video_metrics:
            continue

        pl_inc = get_inc_mask(inc_plaid_dict, date)
        vid_inc = get_inc_mask(inc_video_dict, date)

        if pl_inc is None or vid_inc is None:
            continue

        both_inc = pl_inc & vid_inc
        if not np.any(both_inc):
            continue

        pat_main = pat["main"][date]
        vid_main = video_metrics[date]

        gr_tun = pat_main["grating_tuning"]
        vid_tun = vid_main["video_dir_tuning"]

        gr_sub = gr_tun[both_inc[pl_inc]]


        vid_sub = vid_tun[both_inc[vid_inc]]

        is_pat = np.asarray(pat_main["is_pds"], dtype=bool)
        is_comp = np.asarray(pat_main["is_cds"], dtype=bool)

        is_pat_sub = is_pat[both_inc[pl_inc]]
        is_comp_sub = is_comp[both_inc[pl_inc]]

        # Accumulate pattern and component tuning curves
        gr_pattern_all.append(gr_sub[is_pat_sub])
        vid_pattern_all.append(vid_sub[is_pat_sub])
        gr_component_all.append(gr_sub[is_comp_sub])
        vid_component_all.append(vid_sub[is_comp_sub])

        corrs = np.array(
            [pearsonr(gr_sub[i], vid_sub[i])[0] for i in range(len(gr_sub))]
        )

        valid_mask = ~np.isnan(corrs)

        corr_all.extend(corrs[valid_mask])
        corr_pattern.extend(corrs[valid_mask & is_pat_sub])
        corr_component.extend(corrs[valid_mask & is_comp_sub])

    corr_all = np.array(corr_all)
    corr_pattern = np.array(corr_pattern)
    corr_component = np.array(corr_component)

    out_dir = f"figures/general/{args.split}/all_sessions/"
    os.makedirs(out_dir, exist_ok=True)

    fig_corr_all, _ = plot_single_hist(
        corr_all, xlabel="Video-Grating Dir. Tuning Corr."
    )
    fig_corr_all.savefig(
        os.path.join(out_dir, "video_vs_grating_tuning_corr_all.pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig_corr_all)

    fig_corr_overlay, _ = plot_overlay_hist(
        corr_pattern,
        corr_component,
        xlabel="Video-Grating Dir. Tuning Corr",
        pattern_color=pattern_color,
        component_color=component_color,
    )
    fig_corr_overlay.savefig(
        os.path.join(out_dir, "video_vs_grating_tuning_corr_pattern_vs_component.pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig_corr_overlay)

    
    def norm(pool):
        pool = pool - np.min(pool, axis=1, keepdims=True)
        pool = pool / np.max(pool, axis=1, keepdims=True)
        return pool 
    
    gr_pat_pool = norm(np.vstack(gr_pattern_all))
    vid_pat_pool = norm(np.vstack(vid_pattern_all))
    gr_comp_pool = norm(np.vstack(gr_component_all))
    vid_comp_pool = norm(np.vstack(vid_component_all))


    print(gr_pat_pool.shape)
    print(gr_comp_pool.shape)

    # Set random seed for reproducible example selection
    np.random.seed(1)

    pat_idx = np.random.choice(len(gr_pat_pool))
    comp_idx = np.random.choice(len(gr_comp_pool))

    r_gv_pat = pearsonr(gr_pat_pool[pat_idx], vid_pat_pool[pat_idx])[0]
    r_gv_comp = pearsonr(gr_comp_pool[comp_idx], vid_comp_pool[comp_idx])[0]

    make_example_polar_plot(
        os.path.join(out_dir, "example_pattern_grating.pdf"),
        gr_pat_pool[pat_idx],
        pattern_color,
        r_gv=None,
    )
    make_example_polar_plot(
        os.path.join(out_dir, "example_pattern_video.pdf"),
        vid_pat_pool[pat_idx],
        pattern_color,
        r_gv=r_gv_pat,
    )

    # Component Neuron Examples (Component Color)
    make_example_polar_plot(
        os.path.join(out_dir, "example_component_grating.pdf"),
        gr_comp_pool[comp_idx],
        component_color,
        r_gv=None,
    )
    make_example_polar_plot(
        os.path.join(out_dir, "example_component_video.pdf"),
        vid_comp_pool[comp_idx],
        component_color,
        r_gv=r_gv_comp,
    )

    # Pattern Neuron Combined (Grating dashed, Video solid)
    make_combined_polar_plot(
        os.path.join(out_dir, "example_pattern_combined.pdf"),
        gr_pat_pool[pat_idx],
        vid_pat_pool[pat_idx],
        pattern_color,
        r_gv=r_gv_pat,
    )

    # Component Neuron Combined (Grating dashed, Video solid)
    make_combined_polar_plot(
        os.path.join(out_dir, "example_component_combined.pdf"),
        gr_comp_pool[comp_idx],
        vid_comp_pool[comp_idx],
        component_color,
        r_gv=r_gv_comp,
    )


if __name__ == "__main__":
    main()