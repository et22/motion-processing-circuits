import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

from utils import load_config, load_session, load_inc, load_pat, sort_dates_and_make_session_titles, set_plot_defaults, load_video

from tqdm import trange
component_color, pattern_color, unclass_color = set_plot_defaults()

def exp_func(x, baseline, tau, amp):
    return baseline - amp * np.exp(-x / tau)

def fit_exp_curve(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    baseline0 = y[-1]
    amp0 = baseline0 - np.min(y)
    tau0 = 200.0

    popt, _ = curve_fit(
        exp_func,
        x,
        y,
        p0=[baseline0, tau0, amp0],
        bounds=([0.0, 50, -180], [180.0, 1000, 180]),
        maxfev=20000,
    )
    return popt

def circ_abs_diff_deg(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)

def bin_y_diffs_100(y):
    y = np.asarray(y, float)
    return 100.0 * np.floor(y / 100.0)

def plot_pref_dir_diff_hist(adjdiff):
    color = "#AA0B07"

    bins = np.arange(0, 181, 15)

    fig, ax = plt.subplots(figsize=(2.4, 2.75))
    ax.hist(adjdiff, bins=bins, color=color, edgecolor="k", linewidth=1.0, alpha=1.0)
    ax.axvline(np.mean(adjdiff), color="k", linestyle="--", linewidth=1.3, zorder=3)
    ax.set_xlabel("|Diff. in pref. dir. (°)|", fontsize=10)
    ax.set_ylabel("Number of pairs", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-5, 185)
    ax.set_xticks([0, 60, 120, 180])
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.2)

    plt.tight_layout()
    return fig, ax

def plot_pref_dir_diff_vs_distance(ydiffs, dirdiffs, null,
                                   ylabel="Difference in preferred direction (°)",
                                   xmax=None):
    yvals = np.unique(ydiffs)
    obs = np.array([np.mean(dirdiffs[ydiffs == y]) for y in yvals])

    base = np.mean(null, axis=0)
    lo = np.percentile(null, 2.5, axis=0)
    hi = np.percentile(null, 97.5, axis=0)

    centers = yvals + 50.0  # [0,100) -> 50, [100,200) -> 150, ...

    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    ax.scatter(centers, obs, color="#AA0B07", marker=".", s=70, zorder=3)

    obs = obs[centers <= xmax]
    base = base[centers <= xmax]
    lo = lo[centers <= xmax]
    hi = hi[centers <= xmax]
    centers = centers[centers <= xmax]
    null = null[:, centers <= xmax]

    # fit exponential to observed curve and null mean curve
    obs_baseline, obs_tau, obs_amp = fit_exp_curve(centers, obs)

    obs_fit = exp_func(centers, obs_baseline, obs_tau, obs_amp)
    ss_res = np.sum((obs - obs_fit) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    obs_r2 = 1.0 - (ss_res / ss_tot)

    print(f"observed fit: amp={obs_amp:.2f}, tau={obs_tau:.1f}, R^2={obs_r2:.4f}")
    
    # fit exponential to each null curve, use amplitude as test statistic
    null_amps = np.full(null.shape[0], np.nan)
    for i in trange(null.shape[0]):
        _, _, null_amps[i] = fit_exp_curve(centers, null[i])

    global_pval = (1 + np.sum(null_amps >= obs_amp)) / (len(null_amps) + 1)
    print(f"amplitude permutation test p-value: {global_pval}")
    print(f"observed fit: amp={obs_amp:.2f}, tau={obs_tau:.1f}")

    # ax.plot(centers, obs, "-", color="#AA0B07", lw=1.3, zorder=3)
    xs = np.linspace(0, xmax, 300)
    ax.plot(xs, exp_func(xs, obs_baseline, obs_tau, obs_amp),
            color="#AA0B07", lw=1.3, zorder=3)
    
    ax.plot(centers, base, "k--", lw=1.3, zorder=2)
    ax.fill_between(centers, lo, hi, color="k", alpha=0.15, linewidth=0, zorder=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Distance between neurons (μm)", fontsize=10, fontweight="regular")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="regular")

    if global_pval >= 0.05:
        ast_text = ""
    if global_pval < 0.05:
        ast_text = "*"
    if global_pval < 0.01:
        ast_text = "**"
    if global_pval < 0.001:
        ast_text = "***"

    """
    if global_pval < 0.05:
        ax.text(
            0.96, 0.96, ast_text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=14, color="k"
        )
    """
    ax.text(
        0.92, 0.93, "$n_{pairs}$ = " + f"{len(dirdiffs):,}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8, color="k"
    )
    ax.text(
        0.92, 0.87, f"Amp. = {obs_amp:.0f}\n" + r"$\tau$" + f" = {obs_tau:.0f} μm",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8, color="k"
    )

    ax.set_xlim(0, xmax)

    ax.invert_yaxis()

    ymin, ymax = ax.get_ylim()
    yrng = ymax - ymin
    ax.set_ylim(ymin, ymax + 0.05 * yrng)

    # small violin inset for null amplitudes vs observed amplitude
    ax_in = fig.add_axes([0.75, 0.5, 0.12, 0.18])
    vp = ax_in.violinplot(null_amps, showextrema=False, widths=0.8)
    for body in vp["bodies"]:
        body.set_facecolor("0.6")
        body.set_edgecolor("0.6")
        body.set_alpha(0.7)

    ax_in.axhline(obs_amp, color="#AA0B07", linewidth=1.5)
    ax_in.spines["top"].set_visible(False)
    ax_in.spines["right"].set_visible(False)
    ax_in.spines["bottom"].set_visible(False)
    ax_in.set_xticks(ticks=[], labels=[])
    ax_in.set_ylabel("Amp.", fontsize=8)
    ax_in.tick_params(axis="y", labelsize=7)
    ax_in.text(0.5, 1.15, ast_text, transform=ax_in.transAxes,
               ha="center", va="top", fontsize=12, color="k")
    
    plt.tight_layout()
    return fig

def compute_pref_dir_from_tuning(gr_tun, n_interp=500):
    n_dirs = gr_tun.shape[1]
    x = np.arange(n_dirs + 1) * (360.0 / float(n_dirs))
    cs = CubicSpline(x, np.concatenate((gr_tun, gr_tun[:, 0][:, None]), axis=1), axis=1, bc_type="periodic")
    angles_interp = np.linspace(0.0, 360.0, 500)
    gr_tun = cs(angles_interp)
    pref_dir = angles_interp[np.argmax(gr_tun, axis=1)]

    return pref_dir

def pairwise_diffs(x):
    diffs = x[np.newaxis, :] - x[:, np.newaxis]  # [N,N]
    diffs = np.abs(diffs)
    return diffs[np.triu_indices(len(x), k=1)]

def pairwise_circ_diffs_deg(x):
    x = np.asarray(x, float)
    diffs = pairwise_diffs(x) % 360.0
    diffs = np.minimum(diffs, 360.0 - diffs)
    return diffs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml", type=str)
    parser.add_argument("--split", default="all", type=str)
    args = parser.parse_args()

    config = load_config(args.config)

    data_path = config["dataset_params"]["data_path"]

    n_perms = 5000
    seed = 0
    hist_max_dist = 100.0
    curve_xmax = 1000.0

    ordered_dates, _ = sort_dates_and_make_session_titles(config, args.split)

    pat = load_pat()
    inc = load_inc()

    histdiff_all = []
    ydiffs_all = []
    dirdiffs_all = []
    session_data = []

    for date in ordered_dates:
        if date not in pat["main"]:
            continue

        data = load_session(data_path, date)
        neu_inc = inc["rf_plaid"][str(date)]
        pat_main = pat["main"][date]

        gr_tun = pat_main["grating_tuning"]
        pref_gr = compute_pref_dir_from_tuning(gr_tun)

        neuron_y = np.asarray(data["y"][neu_inc], float)

        # all pairwise quantities
        y_diffs = pairwise_diffs(neuron_y)
        dir_diffs = pairwise_circ_diffs_deg(pref_gr)

        # histogram: all pairs within hist_max_dist
        hist_mask = y_diffs < hist_max_dist
        histdiff_all.append(dir_diffs[hist_mask])

        # distance curve: bins [0,100), [100,200)
        ydiffs_all.append(bin_y_diffs_100(y_diffs[y_diffs < curve_xmax]))
        dirdiffs_all.append(dir_diffs[y_diffs < curve_xmax])

        session_data.append({
            "neuron_y": neuron_y.copy(),
            "pref_gr": pref_gr.copy(),
        })

    histdiff_all = np.concatenate(histdiff_all)
    fig_hist, ax_hist = plot_pref_dir_diff_hist(histdiff_all)

    out_dir = f"figures/general/{args.split}/all_sessions/"
    os.makedirs(out_dir, exist_ok=True)

    fig_hist.savefig(
        f"{out_dir}/grating_tuning_pref_diff_hist_neurons_all_sessions.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    ydiffs_all = np.concatenate(ydiffs_all)
    dirdiffs_all = np.concatenate(dirdiffs_all)

    # null distribution for permutation test on distance curve
    yvals = np.unique(ydiffs_all)
    null = np.full((n_perms, len(yvals)), np.nan, float)
    rng = np.random.default_rng(seed)

    for i in range(n_perms):
        perm_y_all = []
        perm_d_all = []

        for sess in session_data:
            y_diffs = pairwise_diffs(sess["neuron_y"])
            perm_pref = rng.permutation(sess["pref_gr"])
            perm_diffs = pairwise_circ_diffs_deg(perm_pref)

            perm_y_all.append(bin_y_diffs_100(y_diffs[y_diffs < curve_xmax]))
            perm_d_all.append(perm_diffs[y_diffs < curve_xmax])

        perm_y_all = np.concatenate(perm_y_all)
        perm_d_all = np.concatenate(perm_d_all)

        null[i] = [np.mean(perm_d_all[perm_y_all == y]) for y in yvals]

    fig_dist = plot_pref_dir_diff_vs_distance(
        ydiffs_all,
        dirdiffs_all,
        null,
        ylabel="|Difference in pref. direction (°)|",
        xmax=curve_xmax,
    )

    fig_dist.savefig(
        f"{out_dir}/grating_tuning_pref_diff_vs_distance_neurons_all_sessions.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    ## video direction addition ## 
    video_metrics = load_video()
    dates = config['dataset_params']['dates_video']

    histdiff_all = []
    ydiffs_all = []
    dirdiffs_all = []
    session_data = []

    for date in dates:
        gr_tun = video_metrics[date]['video_dir_tuning']
        pref_gr = compute_pref_dir_from_tuning(gr_tun)
        neuron_y = video_metrics[date]['y']

        # all pairwise quantities
        y_diffs = pairwise_diffs(neuron_y)
        dir_diffs = pairwise_circ_diffs_deg(pref_gr)

        # histogram: all pairs within hist_max_dist
        hist_mask = y_diffs < hist_max_dist
        histdiff_all.append(dir_diffs[hist_mask])

        # distance curve: bins [0,100), [100,200)
        ydiffs_all.append(bin_y_diffs_100(y_diffs[y_diffs < curve_xmax]))
        dirdiffs_all.append(dir_diffs[y_diffs < curve_xmax])

        session_data.append({
            "neuron_y": neuron_y.copy(),
            "pref_gr": pref_gr.copy(),
        })

    histdiff_all = np.concatenate(histdiff_all)
    fig_hist, ax_hist = plot_pref_dir_diff_hist(histdiff_all)

    out_dir = f"figures/general/{args.split}/all_sessions/"
    os.makedirs(out_dir, exist_ok=True)

    fig_hist.savefig(
        f"{out_dir}/video_tuning_pref_diff_hist_neurons_all_sessions.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    ydiffs_all = np.concatenate(ydiffs_all)
    dirdiffs_all = np.concatenate(dirdiffs_all)

    # null distribution for permutation test on distance curve
    yvals = np.unique(ydiffs_all)
    null = np.full((n_perms, len(yvals)), np.nan, float)
    rng = np.random.default_rng(seed)

    for i in range(n_perms):
        perm_y_all = []
        perm_d_all = []

        for sess in session_data:
            y_diffs = pairwise_diffs(sess["neuron_y"])
            perm_pref = rng.permutation(sess["pref_gr"])
            perm_diffs = pairwise_circ_diffs_deg(perm_pref)

            perm_y_all.append(bin_y_diffs_100(y_diffs[y_diffs < curve_xmax]))
            perm_d_all.append(perm_diffs[y_diffs < curve_xmax])

        perm_y_all = np.concatenate(perm_y_all)
        perm_d_all = np.concatenate(perm_d_all)

        null[i] = [np.mean(perm_d_all[perm_y_all == y]) for y in yvals]

    fig_dist = plot_pref_dir_diff_vs_distance(
        ydiffs_all,
        dirdiffs_all,
        null,
        ylabel="|Difference in pref. direction (°)|",
        xmax=curve_xmax,
    )

    fig_dist.savefig(
        f"{out_dir}/video_tuning_pref_diff_vs_distance_neurons_all_sessions.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )


if __name__ == "__main__":
    main()