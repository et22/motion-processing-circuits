from matplotlib.colors import LinearSegmentedColormap

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.interpolate import griddata
from scipy.optimize import curve_fit

from utils import load_config, load_session, load_inc, load_pat, sort_dates_and_make_session_titles, load_fractures, set_plot_defaults

from tqdm import trange

pattern_color, component_color, unclass_color = set_plot_defaults()

def exp_func_same(x, baseline, tau, amp):
    return baseline + amp * np.exp(-x / tau)

def fit_exp_curve_same(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    baseline0 = y[-1]
    amp0 = np.max(y) - baseline0
    tau0 = 200.0

    popt, _ = curve_fit(
        exp_func_same,
        x,
        y,
        p0=[baseline0, tau0, amp0],
        bounds=([0.0, 50.0, -1.0], [1.0, 1000.0, 1.0]),
        maxfev=20000,
    )
    return popt

def binned_sum_cnt(x, bin_ids, n_bins):
    # sums and cnts have shape n_bins, each index corresponds to a distance bin
    # for sums with weights, if bin_id is found at position i in bin_ids, out[bin_id] += weight[i]
    # without weights, just adding 1
    # thus, bincount counts number of neuron_pairs in each bin, weighted by x (so this is a sum of all x within the bin)
    # weights is 1 so this is an unweighted sum of number of pairs in each bin (so this is a count of x within the bin)
    sums = np.bincount(bin_ids, weights=x, minlength=n_bins).astype(float)
    cnts = np.bincount(bin_ids, minlength=n_bins).astype(float)
    return sums, cnts

def pairwise_diffs_clipped(x):
    diffs = x[np.newaxis, :] - x[:, np.newaxis]  # [N,N]
    diffs = np.abs(diffs)
    diffs[diffs > 1] = 1 # for our purposes, we just care about whether pairs are same class (diff=0) or different class (diff \geq 1), so we can clip diffs greater than 1 to be 1
    return diffs[np.triu_indices(len(x), k=1)]

def pairwise_diffs(x):
    diffs = x[np.newaxis, :] - x[:, np.newaxis]  # [N,N]
    diffs = np.abs(diffs)
    return diffs[np.triu_indices(len(x), k=1)]

def pairwise_sums(x):
    sums = x[np.newaxis, :] + x[:, np.newaxis]  # [N, N]
    return sums[np.triu_indices(len(x), k=1)]

def pairwise_category_masks(x):
    pair_sums = pairwise_sums(x)
    is_pp = (pair_sums == 2)   # PDS-PDS
    is_cc = (pair_sums == 0)   # CDS-CDS
    is_mx = (pair_sums == 1)   # mixed
    return is_pp, is_cc, is_mx

def agg_metric_dist(neuron_x, neuron_y, metric, ymin = 0, ymax=2000, ylabel=None, max_dist=1000, bin_size=100.0, rng=None, n_perm=5000, pair_type=None):
    neuron_x, neuron_y, metric = _apply_y_mask(neuron_x, neuron_y, metric, ymin, ymax, ndims=1)

    if pair_type is None:
        diffs = pairwise_diffs_clipped(metric)
    else:
        is_pp, is_cc, is_mx = pairwise_category_masks(metric)

        pair_sums = pairwise_sums(metric)
        if pair_type == "pds_pds":
            diffs = is_pp.astype(float)
        elif pair_type == "cds_cds":
            diffs = is_cc.astype(float)
    
        if pair_type == "pds_pds":
            keep = is_pp | is_mx
        elif pair_type == "cds_cds":
            keep = is_cc | is_mx

    # same for distances, here we are just using vertical distance between neurons
    y_diffs = pairwise_diffs(neuron_y)
    dist = np.abs(y_diffs)

    # binning - bins from 0 to max_dist + bin_size
    edges = np.arange(0.0, max_dist + bin_size, bin_size)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = centers.size

    # want to exclude any distances outside the range we are plotting
    in_range = (dist < edges[-1])

    # digitize assigns each element of dist to its bin index (so this bins each pair), bin_ids has the same shape as dist and diffs and has the index of the bin that each pair belogns to
    bin_ids_full = np.digitize(dist, edges) - 1

    if pair_type is None:
        pair_keep = in_range
    else:
        pair_keep = in_range & keep

    obs_sums, obs_cnts = binned_sum_cnt(diffs[pair_keep], bin_ids_full[pair_keep], n_bins)
    obs = obs_sums / obs_cnts

    assert obs.shape[0] == n_bins
    assert obs_sums.shape[0] == n_bins
    assert obs_cnts.shape[0] == n_bins

    # flip if same class 
    if ylabel == "P(same class)":
        obs = 1.0 - obs
        obs_sums = obs_cnts - obs_sums # our metric is currently 1 for different class and 0 for same class, so obs_sums < obs_cnts; and obs_cnts - obs_sums gives us the number of 'same class pairs' in each bin

    # seed for permutation test 
    null = np.full((n_perm, n_bins), np.nan, float)
    null_sums = np.zeros((n_perm, n_bins), float)
    null_cnts = np.zeros((n_perm, n_bins), float)

    metric0 = np.asarray(metric) # metric is our class labels (i.e., is_pds), so we are shuffling the class labels for our permutation test
    n = metric0.shape[0]

    for i in range(n_perm):
        metric_perm = metric0[rng.permutation(n)]
        if pair_type is None:
            dif = pairwise_diffs_clipped(metric_perm)
            keep_perm = np.ones_like(dif, dtype=bool)
        else:
            is_pp, is_cc, is_mx = pairwise_category_masks(metric_perm)

            if pair_type == "pds_pds":
                dif = is_pp.astype(float)
            elif pair_type == "cds_cds":
                dif = is_cc.astype(float)
                
            if pair_type == "pds_pds":
                keep_perm = is_pp | is_mx
            elif pair_type == "cds_cds":
                keep_perm = is_cc | is_mx

        ss, cc = binned_sum_cnt(dif[in_range & keep_perm], bin_ids_full[in_range & keep_perm], n_bins)
        
        if ylabel == "P(same class)":
            ss = cc - ss

        null_sums[i] = ss # null sums contains sum of diff in each bin 
        null_cnts[i] = cc # null cnts contains cnts of diff in each bin

        null[i] = ss / cc

    return {"obs_sums": obs_sums, "obs_cnts": obs_cnts, "null_sums": null_sums, "null_cnts": null_cnts, "centers": centers, "edges": edges, "n_bins": n_bins, "metric": metric}

def plot_metric_dist_all(cache_list, ylabel=None, alpha=0.05, ymax=None, pair_type=None):
    # note here that we are pooling over pairs
    cache_list = [c for c in cache_list]

    centers = cache_list[0]["centers"] # all cache entries should have the same centers and edges since they are determined by the same binning procedure
    n_bins = cache_list[0]["n_bins"]
    n_perm = cache_list[0]["null_sums"].shape[0]

    # need to add obs sums and obs cnts together to the get the 'all session' distribution of sums and counts for each bin
    obs_sums = np.zeros(n_bins, float)
    obs_cnts = np.zeros(n_bins, float)
    total_pairs = 0
    for c in cache_list:
        obs_sums += c["obs_sums"]
        obs_cnts += c["obs_cnts"]
        total_pairs += c["obs_cnts"].sum()
    
    if pair_type is None:
        pass
    else:
        total_pairs = int((obs_cnts - (obs_cnts - obs_sums)/2).sum())

    if pair_type is None:
        obs = obs_sums / obs_cnts
    else:
        obs = obs_sums / (obs_cnts - (obs_cnts-obs_sums)/2)

    # same but for each permutation --- we've cached these from our single session calls
    null = np.full((n_perm, n_bins), np.nan, float)
    for i in range(n_perm):
        ss = np.zeros(n_bins, float)
        cc = np.zeros(n_bins, float)
        for c in cache_list:
            ss += c["null_sums"][i]
            cc += c["null_cnts"][i]
        if pair_type is None:
            null[i] = ss / cc
        else:
            null[i] = ss / (cc - (cc-ss) / 2)

    base = np.nanmean(null, axis=0)
    lo = np.nanpercentile(null, 2.5, axis=0)
    hi = np.nanpercentile(null, 97.5, axis=0)

    if pair_type is None:
        plot_col = "k"
    elif pair_type == "pds_pds":
        plot_col = pattern_color
    elif pair_type == "cds_cds":
        plot_col = component_color

    fig, ax = plt.subplots(1, 1, figsize=(2.7, 2.7))
    ax.scatter(centers, obs, color=plot_col, marker='.', s=70) #, lw=2.0)
    ax.plot(centers, base, "k--", lw=1.3)
    ax.fill_between(centers, lo, hi, color="k", alpha=0.15, linewidth=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Distance between neurons (\u03bcm)", fontsize=10, fontweight="regular")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="regular")
    ax.set_xlim(0, 1000)

    obs_baseline, obs_tau, obs_amp = fit_exp_curve_same(centers, obs)

    obs_fit = exp_func_same(centers, obs_baseline, obs_tau, obs_amp)
    ss_res = np.sum((obs - obs_fit) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    obs_r2 = 1.0 - (ss_res / ss_tot)

    print(f"observed fit: amp={obs_amp:.2f}, tau={obs_tau:.1f}, R^2={obs_r2:.4f}")

    null_amps = np.full(n_perm, np.nan)
    for i in trange(n_perm):
        _, _, null_amps[i] = fit_exp_curve_same(centers, null[i])

    global_pval = (1 + np.sum(null_amps >= obs_amp)) / (len(null_amps) + 1)
    print(f"amplitude permutation test for distance-dependence p-value: {global_pval}")
    print(f"observed fit: amp={obs_amp:.3f}, tau={obs_tau:.1f}")

    xs = np.linspace(0, 1000, 300)
    ax.plot(xs, exp_func_same(xs, obs_baseline, obs_tau, obs_amp), color=plot_col, lw=1.3, zorder=3)

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
        0.92, 0.93, "$n_{pairs}$ = " + f"{int((total_pairs)):,}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8, color="k"
    )
    ax.text(
        0.92, 0.87, f"Amp. = {obs_amp:.2f}\n" + r"$\tau$" + f" = {obs_tau:.0f} μm",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8, color="k"
    )

    ax_in = fig.add_axes([0.75, 0.45, 0.12, 0.18])
    vp = ax_in.violinplot(null_amps, showextrema=False, widths=0.8)
    for body in vp["bodies"]:
        body.set_facecolor("0.6")
        body.set_edgecolor("0.6")
        body.set_alpha(0.7)

    ax_in.axhline(obs_amp, color=plot_col, linewidth=1.5)
    ax_in.spines["top"].set_visible(False)
    ax_in.spines["right"].set_visible(False)
    ax_in.spines["bottom"].set_visible(False)
    ax_in.set_xticks(ticks=[], labels=[])
    ax_in.set_ylabel("Amp.", fontsize=8)
    ax_in.tick_params(axis="y", labelsize=7)
    ax_in.text(0.5, 1.15, ast_text, transform=ax_in.transAxes,
                ha="center", va="top", fontsize=12, color="k")

    plt.sca(ax)

def _apply_y_mask(neuron_x, neuron_y, vals, ymin=0, ymax=2000, ndims=1):
    y_mask = (neuron_y >= ymin) & (neuron_y <= ymax)
    if ndims == 1:
        vals = vals[y_mask]
    else:
        vals = vals[y_mask, :]
    neuron_x = neuron_x[y_mask]
    neuron_y = neuron_y[y_mask]
    return neuron_x, neuron_y, vals

def plot_metric_pattern(neuron_x, neuron_y, pi_vals, gr_tun, patcomp_split, ymin = 0, ymax=2000, fractures=None):
    neuron_x_full = np.asarray(neuron_x)
    neuron_y_full = np.asarray(neuron_y)
    _, _, gr_tun = _apply_y_mask(neuron_x_full, neuron_y_full, gr_tun, ymin, ymax, ndims=2)
    neuron_x, neuron_y, pi_vals = _apply_y_mask(neuron_x_full, neuron_y_full, pi_vals, ymin, ymax)

    neuron_x = neuron_x - np.min(neuron_x) - 52

    fig, axes = plt.subplots(1, 2, figsize=(3.0 * 5/8, 8.0 * 5/8))
    fig.subplots_adjust(left=0.18, right=0.76, bottom=0.06, top=0.96, wspace=0.25)  # reserve right margin for cbar

    uq_neuron_x = np.unique(neuron_x)
    uq_neuron_y = np.unique(neuron_y)
    neuron_xs = []
    neuron_ys = []
    pis = []
    for x in uq_neuron_x: 
        for y in uq_neuron_y: 
            mask = (neuron_x == x) & (neuron_y == y)
            if np.sum(mask) == 1:
                pis.append(np.nanmean(pi_vals[mask]))
                neuron_xs.append(x)
                neuron_ys.append(y)
            elif np.sum(mask) > 1:
                # max firing rate neuron selection w/in electrode
                tun_opt = gr_tun[mask, :]
                fr = np.argmax(np.nanmax(tun_opt, axis=1))
                pis.append(pi_vals[mask][fr])
                neuron_xs.append(x)
                neuron_ys.append(y)

    neuron_x = np.array(neuron_xs)
    neuron_y = np.array(neuron_ys)
    pi_vals = np.array(pis)
    tc = np.asarray(pis, dtype=float)

    # electrode position version
    ax = axes[0]
    plt.sca(axes[0])
    ax.set_aspect('equal')
    
    pi_cap = patcomp_split
    pi_cmap = LinearSegmentedColormap.from_list("pi_cmap", [component_color, "#dddddd", pattern_color], N=256)

    # normalizing colors between 0 and 1 for mapping 
    tc_clip = np.clip(tc, -pi_cap, pi_cap)
    norm_vals = (tc_clip + pi_cap) / (2 * pi_cap)
    colors = pi_cmap(norm_vals)
    
    ax.scatter(neuron_x, neuron_y, c=colors, alpha=1.0, marker='s', s=2.5)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylabel("Vertical position (\u03bcm)", fontsize=10, fontweight="regular")
    fig.supxlabel("Horiz. pos. (\u03bcm)", fontsize=10, fontweight="regular", y=-0.01)

    if fractures is not None: 
        for fracture in fractures:
            plt.axhline(fracture, color='k', linestyle='--', linewidth=1.0, zorder=10)

    ax.set_xticks([-50, 50])
    ax.set_xlim([-100, 100])
    ax.set_ylim(ymin-10, ymax+10)

    # interpolated version
    ax = axes[1]
    plt.sca(ax)

    if fractures is not None: 
        for fracture in fractures:
            plt.axhline(fracture, color='k', linestyle='--', linewidth=1.0, zorder=10)
            
    pi_norm = mpl.colors.Normalize(vmin=-pi_cap, vmax=pi_cap)
    pi_sm = mpl.cm.ScalarMappable(norm=pi_norm, cmap=pi_cmap)
    pi_sm.set_array([])

    cax = fig.add_axes([0.95, 0.14, 0.035, 0.3])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(pi_sm, cax=cax)
    cbar.set_ticks([-pi_cap, -pi_cap / 2.0, 0.0, pi_cap / 2.0, pi_cap])
    cbar.set_label("Pattern index", rotation=270, labelpad=-48, fontsize=10, fontweight="regular")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.yaxis.tick_left()

    x_u = np.sort(np.unique(neuron_x))
    y_u = np.sort(np.arange(ymin, ymax+20, 20))
    X, Y = np.meshgrid(x_u, y_u)

    Z = griddata(np.c_[neuron_x, neuron_y], pi_vals, (X, Y), method="nearest")

    im = ax.imshow(Z, origin="lower",extent=[x_u.min()-10, x_u.max()+10,y_u.min()-10, y_u.max()+10],cmap=pi_cmap,vmin=-pi_cap, vmax=pi_cap,interpolation="bicubic",aspect="auto")

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])
    ax.set_ylim(ymin-10, ymax+10)
    ax.set_xticks([-50, 50])
    ax.set_xlim([-70, 70])

def main():
    # parse args
    parser = argparse.ArgumentParser(description='Arguments for plotting pattern index over space')
    parser.add_argument('--config', default='configs/default_config.yaml', type=str)
    parser.add_argument("--split", default='all', type=str)
    args = parser.parse_args()

    # load config and grab relevant variables
    config_path = args.config
    config = load_config(config_path)

    data_path = config['dataset_params']['data_path']
    ymax = config['neuron_params']['neuron_y_thr']
    patcomp_split = config["analysis_params"]["patcomp_split"]

    # extract necessary parameters
    ordered_dates, date_to_session = sort_dates_and_make_session_titles(config, args.split)
    sess_cnt = 0
    all_pat_cache = []
    all_sd_pc_cache = []
    all_sd_pcu_cache = []
    all_sd_pi_cache = []

    all_sd_pp_cache = []
    all_sd_cc_cache = []


    rng =  np.random.default_rng(0) # global rng for perm test
    dates = []
    for date in ordered_dates:
        if date in load_pat()['main'].keys():
            dates.append(date)
            sess_cnt += 1

            # load data
            data = load_session(data_path, date)
            neu_inc = load_inc()["rf_plaid"][str(date)]

            # get pattern index
            pat_main = load_pat()['main'][date]
            
            pat_met = pat_main['pi']
            is_cds = pat_main["is_cds"]
            is_pds = pat_main["is_pds"]
            cls = pat_main["pi_cls"]

            # get preferred grating, with interpolation 
            gr_tun = pat_main['grating_tuning']
            
            neuron_x = data['x'][neu_inc]
            neuron_y = data['y'][neu_inc]

            path = f"figures/general/{args.split}/{date}/"
            os.makedirs(path, exist_ok=True)

            # plot pattern index in space without fractures
            plot_metric_pattern(neuron_x, neuron_y, pat_met, gr_tun,patcomp_split, ymax=ymax)
            plt.savefig(f"figures/general/{args.split}/{date}/pattern_index_space_{date}.pdf", bbox_inches="tight", pad_inches=0.02)

            # plot pattern index in space with fractures (if they exist for this session)
            fractures_all = load_fractures(args.split)
            if date in fractures_all.keys():
                fractures = np.asarray(fractures_all[date]['grating'], float)
                if fractures.size == 0:
                    fractures = None
            else:
                fractures = np.array([], float)

            plot_metric_pattern(neuron_x, neuron_y, pat_met, gr_tun,patcomp_split, ymax=ymax, fractures=fractures)
            plt.savefig(f"figures/general/{args.split}/{date}/pattern_index_space_fractures_{date}.pdf", bbox_inches="tight", pad_inches=0.02)
            plt.close(plt.gcf())

            # threshold pat_met between -patcomp_split and patcomp_split
            pat_met_clip = pat_met.copy()
            pat_met_clip[pat_met_clip < -patcomp_split] = -patcomp_split
            pat_met_clip[pat_met_clip > patcomp_split] = patcomp_split
            
            mask = (is_cds | is_pds)

            pi = agg_metric_dist(neuron_x[mask], neuron_y[mask], (is_pds[mask]).astype(float), ylabel="P(same class)", ymax=ymax, rng=rng, n_perm=5000)
            all_sd_pc_cache.append(pi)

            pi = agg_metric_dist(neuron_x[mask], neuron_y[mask], (is_pds[mask]).astype(float), ylabel="P(Patt.,Patt.)", ymax=ymax, rng=rng, n_perm=5000, pair_type="pds_pds")
            all_sd_pp_cache.append(pi)

            pi = agg_metric_dist(neuron_x[mask], neuron_y[mask], (is_pds[mask]).astype(float), ylabel="P(Comp.,Comp.)", ymax=ymax, rng=rng, n_perm=5000, pair_type="cds_cds")
            all_sd_cc_cache.append(pi)

            # including unc
            pi = agg_metric_dist(neuron_x, neuron_y, (cls).astype(float), ylabel="P(same class)", ymax=ymax, rng=rng, n_perm=5000)
            all_sd_pcu_cache.append(pi)

            pat_met[pat_met > 1.28] = 1.28
            pat_met[pat_met < -1.28] = -1.28
            pi = agg_metric_dist(neuron_x[mask], neuron_y[mask], pat_met.astype(float)[mask], ylabel="PI", ymax=ymax, rng=rng, n_perm=5000)
            all_sd_pi_cache.append(pi)

    dates = np.array(dates)
    print(dates)
    out_dir = f"figures/general/{args.split}/all_sessions/"
    os.makedirs(out_dir, exist_ok=True)
    
    plot_metric_dist_all(all_sd_pc_cache, ylabel="P(same class)", ymax=ymax)
    plt.gca().set_ylim([0.4, 0.85])
    plt.savefig(f"{out_dir}/same_diff_distance_all_sessions.pdf", bbox_inches="tight", pad_inches=0.02)


    plot_metric_dist_all(all_sd_pp_cache, ylabel="P(patt.|patt.)", ymax=ymax, pair_type="pds_pds")
    plt.gca().set_ylim([0.4, 0.85])
    plt.savefig(f"{out_dir}/same_diff_distance_pattpatt_all_sessions.pdf", bbox_inches="tight", pad_inches=0.02)

    plot_metric_dist_all(all_sd_cc_cache, ylabel="P(comp.|comp.)", ymax=ymax, pair_type="cds_cds")
    plt.gca().set_ylim([0.4, 0.85])
    plt.savefig(f"{out_dir}/same_diff_distance_compcomp_all_sessions.pdf", bbox_inches="tight", pad_inches=0.02)


    """
    for date in ["121025", "121125", "121625", "121725", "121825", "121925"]:
        plot_metric_dist_all([all_sd_pc_cache[np.where(dates == date)[0][0]]], ylabel="P(same class)", ymax=ymax)
        #plt.gca().set_ylim([0.3, 0.9])
        plt.savefig(f"{out_dir}/same_diff_distance_{date}_sessions.pdf", bbox_inches="tight", pad_inches=0.02)
    """

    plot_metric_dist_all(all_sd_pcu_cache, ylabel="P(same class)", ymax=ymax)
    plt.gca().set_ylim([0.4, 0.85])
    plt.savefig(f"{out_dir}/same_diff_distance_with_unc_all_sessions.pdf", bbox_inches="tight", pad_inches=0.02)

    plot_metric_dist_all(all_sd_pi_cache, ylabel="|Diff. in thr. pattern index|", ymax=ymax)
    plt.savefig(f"{out_dir}/same_diff_distance_patt_id.pdf", bbox_inches="tight", pad_inches=0.02)

    # plot_metric_dist_all(all_pat_cache, ylabel="|Diff. in pattern index|", ymax=ymax)
    # plt.savefig(f"{out_dir}/pattern_index_distance_all_sessions.pdf", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()