import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import CubicSpline
from scipy.stats import wilcoxon
from scipy.stats import binomtest

from utils import load_config, load_fractures, load_session, load_inc, load_pat, sort_dates_and_make_session_titles, load_ccgs, set_plot_defaults, load_rfs

from plotting.plot_utils import _plot_single_polar_tuning

from scipy.optimize import curve_fit

pattern_color, component_color, unclass_color = set_plot_defaults()

def half_gaussian(x, bias, amp, sigma):
    return bias + amp * np.exp(-(x ** 2) / (2 * sigma ** 2))

def fit_half_gaussian_to_hist(counts, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])

    bias0 = np.min(counts)
    amp0 = np.max(counts) - bias0
    sigma0 = 45.0

    popt, pcov = curve_fit(
        half_gaussian,
        centers,
        counts,
        p0=[bias0, amp0, sigma0],
        bounds=([0.0, 0.0, 1e-3], [np.inf, np.inf, np.inf]),
        maxfev=20000,
    )
    se_sigma = np.sqrt(pcov[2, 2])
    return centers, popt, se_sigma

def plot_sig_ccg_probe_graph(
    ccg_met,
    neuron_x,
    neuron_y,
    pi_vals,
    is_cds,
    is_pds,
    pi_cls,
    patcomp_split,
    ymin=0,
    ymax=None,
    n_show=4,
    seed=0,
    jitter_width=0.22,
    outpath=None,
):
    rng = np.random.default_rng(int(seed))

    neuron_y = np.asarray(neuron_y, float)
    neuron_x = np.asarray(neuron_x, float)
    pi_vals = np.asarray(pi_vals, float)

    sig = np.asarray(ccg_met["sig_indices_excit"], bool)

    y_mask = (neuron_y >= float(ymin)) & (neuron_y <= float(ymax))

    cds_to_pds_sig = sig & is_cds[:, None] & is_pds[None, :]
    cds_input_counts_targ = np.sum(cds_to_pds_sig, axis=0)

    valid_targ_mask = is_pds & y_mask

    if np.any(valid_targ_mask):
        valid_idx = np.where(valid_targ_mask)[0]
        valid_counts = cds_input_counts_targ[valid_idx]

        max_count = np.max(valid_counts)
        tied = valid_idx[valid_counts == max_count]

        if tied.size == 1:
            seed_neuron = int(tied[0])
        else:
            tied_pi = pi_vals[tied]
            seed_neuron = int(tied[np.argmax(tied_pi)])
    else:
        seed_neuron = int(np.argmax(cds_input_counts_targ))

    cds_mask = is_cds
    partners = np.where(sig[:, seed_neuron] & cds_mask)[0]
    partners = partners[partners != seed_neuron]

    if partners.size > n_show:
        partners = rng.choice(partners, size=n_show, replace=False)

    partners = partners.astype(int)

    if partners.size > 0:
        partners = partners[np.argsort(neuron_y[partners])[::-1]]

    x = rng.uniform(-jitter_width, jitter_width, size=len(neuron_y))

    fig, ax = plt.subplots(figsize=(2.3, 6.0))

    valid = (neuron_y >= ymin) & (neuron_y <= ymax)

    src_all, targ_all = np.where(sig)
    pair_mask = valid[src_all] & valid[targ_all] & (src_all != targ_all)
    src_all = src_all[pair_mask]
    targ_all = targ_all[pair_mask]

    for i, j in zip(src_all, targ_all):
        ax.annotate(
            "",
            xy=(x[j], neuron_y[j]),
            xytext=(x[i], neuron_y[i]),
            arrowprops=dict(
                arrowstyle="->",
                color="k",
                lw=0.8,
                alpha=0.35,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=8,
            ),
            zorder=0,
        )

    ax.scatter(x[valid], neuron_y[valid], c="k", s=10, zorder=1)

    sm = _pi_sm_class(pi_cap=patcomp_split)

    def _class_color(cls_val):
        if cls_val == 1:
            return pattern_color
        elif cls_val == -1:
            return component_color
        return "k"

    text_dx = jitter_width * 0.6

    seed_color = sm.to_rgba(pi_vals[seed_neuron])
    seed_text_color = _class_color(pi_cls[seed_neuron])

    ax.scatter(x[seed_neuron], neuron_y[seed_neuron], c=[seed_color], s=40, zorder=3)
    ax.text(
        x[seed_neuron] + text_dx,
        neuron_y[seed_neuron],
        "Target",
        fontsize=7,
        ha="left",
        va="center",
        color=seed_text_color,
        zorder=4,
    )

    for k, j in enumerate(partners, start=1):
        target_color = sm.to_rgba(pi_vals[j])
        target_text_color = _class_color(pi_cls[j])

        ax.scatter(x[j], neuron_y[j], c=[target_color], s=35, zorder=3)

        ax.text(
            x[j] + text_dx,
            neuron_y[j],
            f"Source {k}",
            fontsize=7,
            ha="left",
            va="center",
            color=target_color,
            zorder=4,
        )

        ax.annotate(
            "",
            xy=(x[seed_neuron], neuron_y[seed_neuron]),
            xytext=(x[j], neuron_y[j]),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.2,
                color=target_color,
                alpha=0.95,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=8,
            ),
            zorder=2,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_xlim(-jitter_width * 1.2, jitter_width * 1.2)
    ax.set_ylim(-50, float(ymax) + 10)

    xmin, xmax = ax.get_xlim()
    ymin_ax, ymax_ax = ax.get_ylim()

    xr = xmax - xmin
    yr = ymax_ax - ymin_ax

    x0 = xmin + 0.05 * xr
    y0 = ymin_ax + 0.05 * yr

    depth_bar_len_um = 200.0

    ax.annotate(
        "",
        xy=(x0, y0 + depth_bar_len_um),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="k", shrinkA=0, shrinkB=0),
        zorder=5,
    )

    ax.text(
        x0 - 0.04 * xr,
        y0 + 0.5 * depth_bar_len_um,
        f"200 µm",
        ha="right",
        va="center",
        rotation=90,
        fontsize=8,
        color="k",
        zorder=5,
    )

    if outpath is not None:
        plt.savefig(outpath, bbox_inches="tight", pad_inches=0.02, dpi=400)
        plt.close(fig)

def p_to_ast(p):
    if p < 0.001:
        p_txt = "***"
    elif p < 0.01:
        p_txt = "**"
    elif p < 0.05:
        p_txt = "*"
    else:
        p_txt = np.around(p, 2)
    return p_txt

def _pi_sm_class(pi_cap):
    pi_cmap = mpl.colors.ListedColormap([component_color, "#dddddd", pattern_color])
    pi_norm = mpl.colors.BoundaryNorm([-100, -pi_cap, pi_cap, 100], pi_cmap.N)
    sm = mpl.cm.ScalarMappable(norm=pi_norm, cmap=pi_cmap)
    sm.set_array([])
    return sm

def ccg_neuron_gallery_excit_targ(
    ccg_met,
    neuron_x,
    neuron_y,
    pi_vals,
    is_cds,
    is_pds,
    patcomp_split,
    gr_tun_pol,
    gr_sem_pol,
    pl_tun_pol,
    pl_sem_pol,
    base,
    uq_dirs,
    outpath,
    seed_neuron=None,
    n_show=6,
    ymin=0,
    ymax=None,
    gallery_seed=0,
):
    ccg_true = np.asarray(ccg_met["ccg_true"], float)
    ccg_slow = np.asarray(ccg_met["ccg_slow"], float)
    maxlag = int(ccg_true.shape[0] // 2)
    lags = np.arange(-maxlag, maxlag + 1)

    sig = np.asarray(ccg_met["sig_indices_excit"], bool)

    if seed_neuron is None:
        pi_arr = np.asarray(pi_vals, float)
        y_arr = np.asarray(neuron_y, float)

        cds_mask = is_cds
        pds_mask = is_pds

        y_mask = (y_arr >= float(ymin)) & (y_arr <= float(ymax))

        cds_to_pds_sig = sig & cds_mask[:, None] & pds_mask[None, :]
        cds_input_counts_targ = np.sum(cds_to_pds_sig, axis=0)

        valid_targ_mask = pds_mask & y_mask

        if np.any(valid_targ_mask):
            valid_idx = np.where(valid_targ_mask)[0]
            valid_counts = cds_input_counts_targ[valid_idx]

            max_count = np.max(valid_counts)
            tied = valid_idx[valid_counts == max_count]

            if tied.size == 1:
                seed_neuron = int(tied[0])
            else:
                tied_pi = pi_arr[tied]
                seed_neuron = int(tied[np.argmax(tied_pi)])
        else:
            seed_neuron = int(np.argmax(cds_input_counts_targ))

    seed_neuron = int(seed_neuron)

    n_slots = int(n_show)

    cds_mask = is_cds
    partners = np.where(sig[:, seed_neuron] & cds_mask)[0]
    partners = partners[partners != seed_neuron]

    rng = np.random.default_rng(int(gallery_seed))
    if partners.size > n_slots:
        partners = rng.choice(partners, size=n_slots, replace=False)

    partners = partners.astype(int)
    if partners.size < n_slots:
        partners = np.concatenate([partners, -np.ones(n_slots - partners.size, dtype=int)], axis=0)

    yy = np.asarray(neuron_y, float)
    valid_mask = partners >= 0
    partners_valid = partners[valid_mask]
    partners_blank = partners[~valid_mask]
    if partners_valid.size > 0:
        partners_valid = partners_valid[np.argsort(yy[partners_valid])]
        partners_valid = partners_valid[::-1]
    partners = np.concatenate([partners_valid, partners_blank], axis=0)

    n_valid = int(np.sum(partners >= 0))
    last_filled_row = n_valid

    n_rows = n_slots + 1

    fig = plt.figure(figsize=(9.3, 6.5))
    gs = fig.add_gridspec(nrows=n_rows, ncols=4, width_ratios=[1.5, 0.75, 0.5, 0.5], wspace=0.01, hspace=0.35)
    fig.subplots_adjust(left=0.06, right=0.93, bottom=0.06, top=0.96)

    sm = _pi_sm_class(pi_cap=patcomp_split)

    seed_color = sm.to_rgba(pi_vals[seed_neuron])

    label_targets = [int(j) for j in partners if int(j) >= 0]

    ax_gr0 = fig.add_subplot(gs[0, 2])
    ax_pl0 = fig.add_subplot(gs[0, 3])

    _plot_single_polar_tuning(ax_gr0, gr_tun_pol[seed_neuron, :], gr_sem_pol[seed_neuron, :], baseline_rate_hz=base[seed_neuron], scale=np.nanmax(gr_tun_pol[seed_neuron, :]), color=seed_color, add_baseline=True)
    _plot_single_polar_tuning(ax_pl0, pl_tun_pol[seed_neuron, :], pl_sem_pol[seed_neuron, :], baseline_rate_hz=base[seed_neuron], scale=np.nanmax(gr_tun_pol[seed_neuron, :]), color=seed_color, add_baseline=True)

    ax_pl0.text(0.70, 0.98, f"Target", transform=ax_pl0.transAxes, ha="left", va="top", fontsize=7, color="k", clip_on=False)

    mask_10 = (lags >= -10) & (lags <= 10)
    l10 = lags[mask_10]
    target_to_num = {j: k for k, j in enumerate(label_targets, start=1)}

    for r in range(1, n_rows):
        j = int(partners[r - 1])

        ax_ccg = fig.add_subplot(gs[r, 1])
        ax_gr = fig.add_subplot(gs[r, 2])
        ax_pl = fig.add_subplot(gs[r, 3])

        if j < 0:
            ax_ccg.axis("off")
            ax_gr.axis("off")
            ax_pl.axis("off")
            continue

        col_pi = sm.to_rgba(pi_vals[j])

        ax_ccg.spines["top"].set_visible(False)
        ax_ccg.spines["right"].set_visible(False)

        y_true = ccg_true[:, j, seed_neuron][mask_10]
        y_base = ccg_slow[:, j, seed_neuron][mask_10]

        ax_ccg.bar(l10, y_true, width=1.0, align="center", color=col_pi, edgecolor=col_pi, linewidth=0.0)
        ax_ccg.plot(l10, y_base, "--", color="#888888", lw=1.2)
        ax_ccg.axvline(0, color="k", lw=0.8, alpha=0.5)
        ax_ccg.set_xlim(-10, 10)
        if r == last_filled_row:
            ax_ccg.set_xlabel("Lag (ms)", fontsize=11, fontweight="regular")
        else:
            ax_ccg.set_xticks([])
        ax_ccg.set_ylabel("Coinc.", fontsize=11, fontweight="regular")

        num = target_to_num[j]
        ax_ccg.text(0.80, 0.92, f"Source {num}", transform=ax_ccg.transAxes, ha="left", va="top", fontsize=7, color="k", clip_on=False)

        _plot_single_polar_tuning(ax_gr, gr_tun_pol[j, :], gr_sem_pol[j, :], baseline_rate_hz=base[j], scale=np.nanmax(gr_tun_pol[j, :]), add_baseline=True, color=col_pi)
        _plot_single_polar_tuning(ax_pl, pl_tun_pol[j, :], pl_sem_pol[j, :], baseline_rate_hz=base[j], scale=np.nanmax(pl_tun_pol[j, :]), add_baseline=True, color=col_pi)
        ax_pl.text(0.70, 0.98, f"Source {num}", transform=ax_pl.transAxes, ha="left", va="top", fontsize=7, color="k", clip_on=False)

    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)

def plot_lags_mirrored(cds_to_pds_lags, pds_to_cds_lags, ccg_config, split, date=""):
    cds_to_pds_lags = np.abs(np.asarray(cds_to_pds_lags, float))
    pds_to_cds_lags = np.abs(np.asarray(pds_to_cds_lags, float))

    k = len(cds_to_pds_lags)
    n = len(cds_to_pds_lags) + len(pds_to_cds_lags)
    p = binomtest(k, n, p=0.5, alternative='two-sided').pvalue 

    lag_max = int(ccg_config["sig_range"][1])
    x = np.arange(1, lag_max + 1)
    bins = np.arange(0.5, lag_max + 1.5, 1.0)

    cds_counts, _ = np.histogram(cds_to_pds_lags, bins=bins)
    pds_counts, _ = np.histogram(pds_to_cds_lags, bins=bins)

    ymax = max(
        1,
        np.max(cds_counts) if cds_counts.size else 0,
        np.max(pds_counts) if pds_counts.size else 0,
    )
    # Added padding to top limit so "Comp. -> Patt." label sits cleanly above bars
    ymax_padded = int(np.ceil((ymax * 1.25) / 10.0) * 10) if ymax > 0 else 10

    fig = plt.figure(figsize=(2.35, 3.2))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1.0, 1.0],
        hspace=0.15,  # b) Slightly increased gap between top and bottom subplots
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    # --- TOP HISTOGRAM ---
    ax_top.bar(
        x,
        cds_counts,
        width=0.9,
        color="#dddddd",
        edgecolor="k",
        linewidth=1.0,
        align="center",
    )

    ax_top.set_xlim(0.5, lag_max + 0.5)
    ax_top.set_ylim(0, ymax_padded)

    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(True)

    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)

    # d) Centered annotation positioned higher to avoid overlapping bars
    ax_top.text(
        0.42, 0.95, "Comp.",
        transform=ax_top.transAxes,
        ha="right", va="top",
        fontsize=10, color=component_color,
    )
    ax_top.text(
        0.50, 0.95, r"$\rightarrow$",
        transform=ax_top.transAxes,
        ha="center", va="top",
        fontsize=10, color="k",
    )
    ax_top.text(
        0.58, 0.95, "Patt.",
        transform=ax_top.transAxes,
        ha="left", va="top",
        fontsize=10, color=pattern_color,
    )

    # e) Slightly larger & bold asterisks
    ax_top.text(
        0.98, 1.02, p_to_ast(p),
        transform=ax_top.transAxes,
        ha="right", va="bottom",
        fontsize=10, fontweight="bold",
    )

    # --- BOTTOM HISTOGRAM ---
    ax_bot.bar(
        x,
        pds_counts,
        width=0.9,
        color="#dddddd",
        edgecolor="k",
        linewidth=1.0,
        align="center",
    )

    ax_bot.set_ylim(ymax_padded, 0)

    # a) Enabled bottom spine for horizontal x-axis line
    ax_bot.spines["top"].set_visible(True)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.spines["bottom"].set_visible(True)

    ax_bot.set_xticks(x)
    ax_bot.tick_params(axis="x", bottom=True, labelbottom=True, direction="out", labelsize=8)
    ax_bot.set_xlabel("Lag (ms)", fontsize=10, labelpad=3)

    # Centered annotation: Patt. -> Comp.
    ax_bot.text(
        0.42, 0.05, "Patt.",
        transform=ax_bot.transAxes,
        ha="right", va="bottom",
        fontsize=10, color=pattern_color,
    )
    ax_bot.text(
        0.50, 0.05, r"$\rightarrow$",
        transform=ax_bot.transAxes,
        ha="center", va="bottom",
        fontsize=10, color="k",
    )
    ax_bot.text(
        0.58, 0.05, "Comp.",
        transform=ax_bot.transAxes,
        ha="left", va="bottom",
        fontsize=10, color=component_color,
    )

    # Tick sizes for Y-axis
    ax_top.tick_params(axis="y", labelsize=8)
    ax_bot.tick_params(axis="y", labelsize=8)

    # c) Y-label positioned slightly higher (y=0.52)
    fig.text(
        0.05, 0.52, "Number of functional connections",
        rotation=90,
        ha="center", va="center",
        fontsize=10, fontweight="regular",
    )

    print(
        f"counts used for mirrored-lag binomial test: "
        f"CDS->PDS={len(cds_to_pds_lags)}, PDS->CDS={len(pds_to_cds_lags)} "
        f"total={len(cds_to_pds_lags) + len(pds_to_cds_lags)}"
    )
    print(f"binom test of cds->pds vs pds->cds lags: p={p}")

    fig.subplots_adjust(
        left=0.22,
        right=0.98,
        bottom=0.12,
        top=0.95,
    )
    out_dir = f"./figures/ccg/{split}/"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}lags_cdstopds_vs_pdstocds_mirrored{date}.pdf", dpi=400)
    plt.close(fig)


def plot_perm_fwhm_violin(perm_diffs, true_diff, pval, out_stem, color="r"):
    fig, ax = plt.subplots(figsize=(1.35, 1.75))

    vp = ax.violinplot(perm_diffs, showextrema=False, widths=0.8)
    for body in vp["bodies"]:
        body.set_facecolor("0.6")
        body.set_edgecolor("0.6")
        body.set_alpha(0.7)

    ax.axhline(true_diff, color=color, linewidth=1.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(ticks=[], labels=[])

    ax.set_ylabel(r"$\Delta$FWHM (°)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)

    ax.text(0.5, 1.08, p_to_ast(pval), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=12, color="k")

    fig.tight_layout()
    fig.savefig(out_stem, bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)

def plot_diff_dirs(agg, plot_p_value=True, split=True, dirv=True):
    cell_types = ["pds", "cds"]
    cell_types_w_all = ["pds", "cds", "all"]
    maps = {"pds": "Patt.", "cds": "Comp.", "all": "All"}
    hists = []
    cats = []
    cp_diffs_by_cat = {}
    fwhm_by_cat = {}
    for ct1 in cell_types_w_all:
        for ct2 in cell_types:
            cats.append(ct1 + " to " + ct2)
            key = f"{ct1}_to_{ct2}_prefdir_diff"
            diff = agg[key]
            diff[diff > 180] -= 360
            diff[diff < -180] += 360
            cp_diff = np.abs(diff)
            cp_diffs_by_cat[ct1 + " to " + ct2] = cp_diff.copy()

            res = wilcoxon(cp_diff - 90.0, alternative="two-sided")
            p = float(res.pvalue)

            bins = np.arange(195, step=15)
            fig, ax = plt.subplots(figsize=(2.2, 2.2))
            counts, edges, patches = ax.hist(cp_diff, bins=bins, density=False, color="#dddddd", edgecolor="k", linewidth=1.0)
            hists.append(counts)
            ax.axvline(90, color="k", linestyle="--", linewidth=1.0)

            # fit half-Gaussian to histogram and overlay fit
            centers, (bias, amp, sigma), se_sigma = fit_half_gaussian_to_hist(counts, edges)
            fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
            se_fwhm = 2 * np.sqrt(2 * np.log(2)) * se_sigma 

            obs_fit = half_gaussian(0.5 * (edges[:-1] + edges[1:]), bias, amp, sigma)
            ss_res = np.sum((counts - obs_fit) ** 2)
            ss_tot = np.sum((counts - np.mean(counts)) ** 2)
            obs_r2 = 1.0 - (ss_res / ss_tot)
            print(f"obs_r2 {obs_r2}, fwhm {fwhm}, fwhm sigma {se_fwhm}, counts {np.sum(counts)}")

            xs = np.linspace(0, 180, 400)
            ax.plot(xs, half_gaussian(xs, bias, amp, sigma), color="k", linewidth=1.3)

            fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
            fwhm_by_cat[ct1 + " to " + ct2] = fwhm
            ax.text(
                1.04, 0.90,
                rf"FWHM = {fwhm:.1f}$^\circ$",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=8,
                color="k",
            )

            ax.set_xlabel(f"|Diff. in pref. dir (\N{DEGREE SIGN})|", fontsize=10, fontweight="regular")
            ax.set_ylabel("Num. of func. conn.", fontsize=10, fontweight="regular") 

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.set_ylim(ax.get_ylim()[0], np.max([ax.get_ylim()[1], 60]))

            y = 1.02
            fs = 12
            ax.text(0.45, 1.02, maps[ct1], transform=ax.transAxes, ha="right",  va="bottom", fontsize=10, color=pattern_color if ct1.lower()=="pds" else component_color)
            ax.text(0.5, 1.02, r"$\rightarrow$", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="k")
            ax.text(0.55, 1.02, maps[ct2], transform=ax.transAxes, ha="left",   va="bottom", fontsize=10, color=pattern_color if ct2.lower()=="pds" else component_color)
            plt.tight_layout()

            os.makedirs(f"./figures/ccg/{split}/", exist_ok=True)
            outpath = f"./figures/ccg/{split}/{key}.pdf"
            plt.savefig(outpath, bbox_inches="tight", pad_inches=0.02, dpi=400)
            plt.close(fig)

    cats = np.array(cats)
    hists = np.array(hists)

    # permutation test of FWHM difference: CDS->PDS vs CDS->CDS
    try:
        cp_diff = cp_diffs_by_cat["cds to pds"]
        cc_diff = cp_diffs_by_cat["cds to cds"]

        true_diff = fwhm_by_cat["cds to pds"] - fwhm_by_cat["cds to cds"]

        pooled = np.concatenate([cp_diff, cc_diff])
        n_cp = len(cp_diff)
        n_cc = len(cc_diff)

        n_perm = 5000
        rng = np.random.default_rng(0)
        bins = np.arange(195, step=15)

        perm_diffs = np.full(n_perm, np.nan)

        for i in range(n_perm):
            perm = rng.permutation(pooled)
            cp_perm = perm[:n_cp]
            cc_perm = perm[n_cp:]

            cp_counts, cp_edges = np.histogram(cp_perm, bins=bins)
            cc_counts, cc_edges = np.histogram(cc_perm, bins=bins)

            _, (_, _, cp_sigma), _ = fit_half_gaussian_to_hist(cp_counts, cp_edges)
            _, (_, _, cc_sigma), _ = fit_half_gaussian_to_hist(cc_counts, cc_edges)

            cp_fwhm = 2 * np.sqrt(2 * np.log(2)) * cp_sigma
            cc_fwhm = 2 * np.sqrt(2 * np.log(2)) * cc_sigma

            perm_diffs[i] = cp_fwhm - cc_fwhm

        p = (1 + np.sum(np.abs(perm_diffs) >= np.abs(true_diff))) / (n_perm + 1)
        print(f"permutation test, FWHM CDS->PDS vs CDS->CDS: p = {p}")

        os.makedirs(f"./figures/ccg/{split}/", exist_ok=True)
        outpath = f"./figures/ccg/{split}/cds_to_pds_vs_cds_to_cds_fwhm_perm.pdf"
        plot_perm_fwhm_violin(perm_diffs, true_diff, p, outpath, color='k')

    except Exception as e:
        print("not enough pairs for FWHM permutation test")
        print(e)

def _sig_pair_distance_counts_one_session(
    ccg_met,
    distance_vals,
    is_cds,
    is_pds,
    bins,
):
    sig = np.asarray(ccg_met["sig_indices_excit"], bool)

    d = np.asarray(distance_vals, float)
    is_cds = np.asarray(is_cds, bool)
    is_pds = np.asarray(is_pds, bool)

    n = len(is_cds)
    not_self = ~np.eye(n, dtype=bool)

    masks = {
        "pds_to_pds": is_pds[:, None] & is_pds[None, :]   & not_self,
        "pds_to_cds": is_pds[:, None] & is_cds[None, :],
        "cds_to_pds": is_cds[:, None] & is_pds[None, :],
        "cds_to_cds": is_cds[:, None] & is_cds[None, :] & not_self,
    }

    out = {}
    for k, m in masks.items():
        possible, _ = np.histogram(d[m], bins=bins)
        sig_count, _ = np.histogram(d[m & sig], bins=bins)
        out[k] = {
            "possible": possible.astype(float),
            "sig": sig_count.astype(float),
        }

    return out


def plot_sig_pair_probability_and_counts_by_distance(
    session_counts,
    split,
    bins,
    out_prefix,
    xlabel,
    out_root="figures/ccg",
):
    cats = ["pds_to_pds", "pds_to_cds", "cds_to_pds", "cds_to_cds"]

    labels = {
        "pds_to_pds": "Patt.→Patt.",
        "pds_to_cds": "Patt.→Comp.",
        "cds_to_pds": "Comp.→Patt.",
        "cds_to_cds": "Comp.→Comp.",
    }

    colors = {
        "pds_to_pds": pattern_color,
        "cds_to_cds": component_color,
        "pds_to_cds": pattern_color,
        "cds_to_pds": component_color,
    }

    linestyles = {
        "pds_to_pds": "-",
        "cds_to_cds": "-",
        "pds_to_cds": "-",   # PDS → CDS
        "cds_to_pds": "-",    # CDS → PDS
    }

    centers = 0.5 * (bins[:-1] + bins[1:])

    agg = {
        c: {
            "possible": np.zeros(len(bins) - 1, dtype=float),
            "sig": np.zeros(len(bins) - 1, dtype=float),
        }
        for c in cats
    }

    for sc in session_counts:
        for c in cats:
            agg[c]["possible"] += sc[c]["possible"]
            agg[c]["sig"] += sc[c]["sig"]

    os.makedirs(f"{out_root}/{split}", exist_ok=True)

    total_possible = np.sum([agg[c]["possible"] for c in cats], axis=0)
    total_sig = np.sum([agg[c]["sig"] for c in cats], axis=0)

    total_prob = np.divide(
        total_sig,
        total_possible,
        out=np.full_like(total_sig, np.nan, dtype=float),
        where=total_possible > 0,
    )

    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    ax.plot(centers, total_prob, "-o", color="k", markersize=3, lw=1.2)
    ax.set_xlabel(xlabel, fontsize=10, fontweight="regular")
    ax.set_ylabel("Prob. func. connection", fontsize=10, fontweight="regular")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out_root}/{split}/{out_prefix}_probability_all.pdf",
                bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.0, 2.4))

    for c in cats:
        if c =="pds_to_pds" or c == "cds_to_cds":
            continue
        prob = np.divide(
            agg[c]["sig"],
            agg[c]["possible"],
            out=np.full_like(agg[c]["sig"], np.nan, dtype=float),
            where=agg[c]["possible"] > 0,
        )

        ax.plot(
            centers,
            prob,
            marker="o",
            linestyle=linestyles[c],
            color=colors[c],
            label=labels[c],
            markersize=3,
            lw=1.3,
        )

    ax.set_xlabel(xlabel, fontsize=10, fontweight="regular")
    ax.set_ylabel("Prob. func. connection", fontsize=10, fontweight="regular")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out_root}/{split}/{out_prefix}_probability_by_combo.pdf",
                bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    ax.bar(
        centers,
        total_sig,
        width=np.diff(bins),
        color="#dddddd",
        edgecolor="k",
        linewidth=1.0,
        align="center",
    )
    ax.set_xlabel(xlabel, fontsize=10, fontweight="regular")
    ax.set_ylabel("Num. func. connections", fontsize=10, fontweight="regular")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out_root}/{split}/{out_prefix}_count_all.pdf",
                bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.0, 2.4))

    for c in cats:
        if c =="pds_to_pds" or c == "cds_to_cds":
            continue
        ax.plot(
            centers,
            agg[c]["sig"],
            marker="o",
            linestyle=linestyles[c],
            color=colors[c],
            label=labels[c],
            markersize=3,
            lw=1.3,
        )

    ax.set_xlabel(xlabel, fontsize=10, fontweight="regular")
    ax.set_ylabel("Num. func. connections", fontsize=10, fontweight="regular")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out_root}/{split}/{out_prefix}_count_by_combo.pdf",
                bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot ccg_gallery_excit.")
    parser.add_argument("--config", default="configs/default_config.yaml", type=str)
    parser.add_argument("--split", default="all", type=str)
    parser.add_argument('--start_time', default=None, type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = config["dataset_params"]["data_path"]
    ccg_config = config["ccg_params"]
    patcomp_split = config["analysis_params"]["patcomp_split"]
    ymax = config["neuron_params"]['neuron_y_thr']

    if args.start_time == None:
        ccg_start = ccg_config['start_time']
    else:
        ccg_start = args.start_time
    
    ccg_dict = load_ccgs(ccg_start if ccg_start != 100 else "")
    pat = load_pat()
    rfs = load_rfs()
    rf_pair_dist_counts = []
    out_root = "figures/ccg"
    os.makedirs(out_root, exist_ok=True)

    dates, _ = sort_dates_and_make_session_titles(config, args.split)

    # there are a few params that we experimented with changing so we can encode those into args.split to get it into our save path correctly 
    split_name = args.split + (str(ccg_start) if ccg_start != 100 else "")

    # aggregator for different variables across sessions
    agg = {
        "cds_to_pds_lags": [],
        "pds_to_cds_lags": [],
        "all_to_pds_prefdir_diff": [],
        "all_to_cds_prefdir_diff": [],
        "cds_to_pds_prefdir_diff": [],
        "cds_to_cds_prefdir_diff": [],
        "pds_to_pds_prefdir_diff": [],
        "pds_to_cds_prefdir_diff": [],
    }

    sig_pair_dist_counts = []
    total_n = 0
    total_sig_n = 0
    
    for date in dates:
        # skip sessions without plaid task
        if date not in pat["main"].keys():
            continue

        out_dir = os.path.join(out_root, str(split_name), str(date))
        os.makedirs(out_dir, exist_ok=True)

        data = load_session(data_path, date)
        neu_inc_bool = np.asarray(load_inc()["rf_plaid"][str(date)], dtype=bool)
        inc_idx = np.where(neu_inc_bool)[0]

        neuron_x = np.asarray(data["x"], float)[inc_idx]
        neuron_y = np.asarray(data["y"], float)[inc_idx]
        pat_dict = pat["main"][date]

        # get preferred grating direction, with interpolation 
        gr_tun = pat_dict['grating_tuning']
        n_dirs = gr_tun.shape[1]
        x = np.arange(n_dirs + 1) * (360.0 / float(n_dirs))
        cs = CubicSpline(x, np.concatenate((gr_tun, gr_tun[:, 0][:, None]), axis=1), axis=1, bc_type="periodic")
        angles_interp = np.linspace(0.0, 360.0, 500)
        pref_gr = angles_interp[np.argmax(cs(angles_interp), axis=1)]
        
        # get other metrics 
        pi = np.asarray(pat_dict["pi"], float)
        is_cds = pat_dict["is_cds"]
        is_pds = pat_dict["is_pds"]
        is_unc = pat_dict["is_unc"]
        pi_cls = pat_dict["pi_cls"]
        gr_tun = np.asarray(pat_dict["grating_tuning"], float)
        pl_tun = np.asarray(pat_dict["plaid_tuning"], float)
        gr_sem = np.asarray(pat_dict["grating_sem"], float)
        pl_sem = np.asarray(pat_dict["plaid_sem"], float)
        base = np.asarray(pat_dict["baseline_rates"], float)
        uq_dirs = np.asarray(30 * np.arange(12), float)

        ccg_met = ccg_dict[date]
        plt.close(plt.gcf())

        ccg_neuron_gallery_excit_targ(ccg_met,
            neuron_x,
            neuron_y,
            pi_vals=pi,
            is_cds=is_cds,
            is_pds=is_pds,
            patcomp_split=patcomp_split,
            gr_tun_pol=gr_tun,
            gr_sem_pol=gr_sem,
            pl_tun_pol=pl_tun,
            pl_sem_pol=pl_sem,
            base=base,
            uq_dirs=uq_dirs,
            outpath=os.path.join(out_dir, f"ccg_gallery_excit_targ1_{date}.pdf"),
            seed_neuron=None,
            n_show=4,
            ymin=0,
            ymax=ymax,
            gallery_seed=int(0),
        )
        
        # aggregating relevant variables 
        if date in rfs["main"]:
            rf_dict=rfs["main"][date]
            rf_x = np.asarray(rf_dict["rf_x"], float)
            rf_y = np.asarray(rf_dict["rf_y"], float)
            dx = rf_x[:, None] - rf_x[None, :]
            dy = rf_y[:, None] - rf_y[None, :]
            rf_dist = np.sqrt(dx * dx + dy * dy)

            rf_pair_dist_counts.append(_sig_pair_distance_counts_one_session(
                ccg_met=ccg_met,
                distance_vals=rf_dist,
                is_cds=is_cds,
                is_pds=is_pds,
                bins=np.arange(-1.5, 19.5, 3),
            ))
        else:
            print(f"[{date}] no RF metrics found; skipping RF-distance pair plot")

        n_pds_cds_pairs = int(np.sum(is_cds) * np.sum(is_pds))  # directed: CDS->PDS + PDS->CDS
        sig_pds_cds = int(np.sum(ccg_met["sig_indices_excit"] & ((is_cds[:, None] & is_pds[None, :]) | (is_pds[:, None] & is_cds[None, :])) ))
        print(f"[{date}] total PDS/CDS pairs={n_pds_cds_pairs}, total sig PDS/CDS CCGs={sig_pds_cds}")
        total_sig_n += sig_pds_cds
        total_n += n_pds_cds_pairs
        source_neuron, target_neuron = np.where(ccg_met['sig_indices_excit'])
        lags = ccg_met['peak_lag'][ccg_met['sig_indices_excit']]

        source_neuron = source_neuron.astype(int)
        target_neuron = target_neuron.astype(int)
        y_mask = (neuron_y[target_neuron] <= ymax) & (neuron_y[source_neuron] <= ymax)

        cds_to_pds = is_cds[source_neuron] & is_pds[target_neuron] & y_mask
        pds_to_cds = is_pds[source_neuron] & is_cds[target_neuron] & y_mask
        cds_to_cds = is_cds[source_neuron] & is_cds[target_neuron] & y_mask
        pds_to_pds = is_pds[source_neuron] & is_pds[target_neuron] & y_mask

        to_cds = is_cds[target_neuron] & y_mask & is_unc[source_neuron]
        to_pds = is_pds[target_neuron] & y_mask & is_unc[source_neuron]

        cds_to_pds_lags = lags[cds_to_pds]
        pds_to_cds_lags = lags[pds_to_cds]

        cds_to_pds_diff = pref_gr[source_neuron[cds_to_pds]] - pref_gr[target_neuron[cds_to_pds]]
        pds_to_cds_diff = pref_gr[source_neuron[pds_to_cds]] - pref_gr[target_neuron[pds_to_cds]]
        cds_to_cds_diff = pref_gr[source_neuron[cds_to_cds]] - pref_gr[target_neuron[cds_to_cds]]
        pds_to_pds_diff = pref_gr[source_neuron[pds_to_pds]] - pref_gr[target_neuron[pds_to_pds]]

        all_to_cds_diff = pref_gr[source_neuron[to_cds]] - pref_gr[target_neuron[to_cds]]
        all_to_pds_diff = pref_gr[source_neuron[to_pds]] - pref_gr[target_neuron[to_pds]]

        agg["cds_to_pds_lags"].append(cds_to_pds_lags)
        agg["pds_to_cds_lags"].append(pds_to_cds_lags)
        agg["cds_to_pds_prefdir_diff"].append(cds_to_pds_diff)
        agg["pds_to_cds_prefdir_diff"].append(pds_to_cds_diff)
        agg["cds_to_cds_prefdir_diff"].append(cds_to_cds_diff)
        agg["pds_to_pds_prefdir_diff"].append(pds_to_pds_diff)
        agg["all_to_pds_prefdir_diff"].append(all_to_pds_diff)
        agg["all_to_cds_prefdir_diff"].append(all_to_cds_diff)

        print(f"[{date}] wrote {os.path.join(out_dir, f'ccg_gallery_excit_{date}.pdf')}")

        plot_sig_ccg_probe_graph(
            ccg_met,
            neuron_x=neuron_x,
            neuron_y=neuron_y,
            pi_vals=pi,
            is_cds=is_cds,
            is_pds=is_pds,
            pi_cls=pi_cls,
            patcomp_split=patcomp_split,
            ymin=0,
            ymax=ymax,
            n_show=4,
            seed=0,
            outpath=os.path.join(out_dir, f"ccg_sig_probe_graph_{date}.pdf"),
        )

        # pairwise vertical distance (µm)
        dy = np.abs(neuron_y[:, None] - neuron_y[None, :])

        sig_pair_dist_counts.append(
            _sig_pair_distance_counts_one_session(
                ccg_met=ccg_met,
                distance_vals=dy,
                is_cds=is_cds,
                is_pds=is_pds,
                bins=np.arange(0, 1100, 100),
            )
        )
        
        # placeholder if we want to do something with fractures here
        pat_met = pi
        fractures_all = load_fractures(args.split)
        if date in fractures_all.keys():
            fractures = np.asarray(fractures_all[date]['grating'], float)
            if fractures.size == 0:
                fractures = None
        else:
            fractures = np.array([], float)

        # placeholder if we want individual session plots for this analysis
        try:
            plot_lags_mirrored(
                agg["cds_to_pds_lags"][-1],
                agg["pds_to_cds_lags"][-1],
                ccg_config,
                split=split_name,
                date=date,
            )
        except:
            pass

    # plot aggregated statistics across sessions
    agg["cds_to_pds_lags"] = np.concatenate(agg["cds_to_pds_lags"])
    agg["pds_to_cds_lags"] = np.concatenate(agg["pds_to_cds_lags"])

    agg["cds_to_pds_prefdir_diff"] = np.concatenate(agg["cds_to_pds_prefdir_diff"])
    agg["pds_to_cds_prefdir_diff"] = np.concatenate(agg["pds_to_cds_prefdir_diff"])
    agg["cds_to_cds_prefdir_diff"] = np.concatenate(agg["cds_to_cds_prefdir_diff"])
    agg["pds_to_pds_prefdir_diff"] = np.concatenate(agg["pds_to_pds_prefdir_diff"])
    agg["all_to_pds_prefdir_diff"] = np.concatenate(agg["all_to_pds_prefdir_diff"])
    agg["all_to_cds_prefdir_diff"] = np.concatenate(agg["all_to_cds_prefdir_diff"])

    plot_sig_pair_probability_and_counts_by_distance(
        rf_pair_dist_counts,
        split=split_name,
        bins=np.arange(-1.5, 19.5, 3),
        out_prefix="sig_pair_by_rf_center_distance",
        xlabel="Distance between RF centers (dva)",
    )

    plot_sig_pair_probability_and_counts_by_distance(
        sig_pair_dist_counts,
        split=split_name,
        bins=np.arange(0, 1100, 100),
        out_prefix="by_vertical_distance",
        xlabel="Distance between neurons (\u03BCm)",
    )

    plot_lags_mirrored(
        agg["cds_to_pds_lags"],
        agg["pds_to_cds_lags"],
        ccg_config,
        split=split_name,
    )

    plot_diff_dirs(agg, split=split_name)

    print(f"total_sig_n: {total_sig_n}")
    print(f"total_n: {total_n}")

if __name__ == "__main__":
    main()