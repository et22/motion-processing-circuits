import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import load_config, load_session, load_inc, load_pat, load_fractures, sort_dates_and_make_session_titles, set_plot_defaults
from scipy.optimize import curve_fit

pattern_color, component_color, unclass_color = set_plot_defaults()

def nearest_fracture_dist(y_um, fractures_um):
    f = np.asarray(fractures_um, float)
    return np.min(np.abs(y_um[:, None] - f[None, :]), axis=1)

def binned_counts(mask01, bin_ids, n_bins):
    # mask01 is float 0/1 per sample
    return np.bincount(bin_ids, weights=mask01, minlength=n_bins).astype(float)

def cache_session_class(dist_um, is_cds, is_pds, is_unc, edges, n_perm=5000, seed=0, include_unc=True):
    rng = np.random.default_rng(seed) # for perm test 

    dist_um = np.asarray(dist_um, float)
    is_cds = np.asarray(is_cds, bool)
    is_pds = np.asarray(is_pds, bool)
    is_unc = np.asarray(is_unc, bool)

    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = centers.size

    # neurons very far from any fracture are excluded from analysis
    if include_unc:
        m = np.isfinite(dist_um) & (dist_um >= edges[0]) & (dist_um < edges[-1]) & (is_cds | is_pds | is_unc)
    else:
        m = np.isfinite(dist_um) & (dist_um >= edges[0]) & (dist_um < edges[-1]) & (is_cds | is_pds)

    dist0 = dist_um[m]
    cds0 = is_cds[m]
    pds0 = is_pds[m]
    unc0 = is_unc[m]

    # histograming into the distance bins 
    bin_ids = np.digitize(dist0, edges) - 1 # subtract one because the 0 index from digitize indicates 'to the left of 0' which is not applicable for us; bins are [0, 100), [100, 200), etc.

    obs_cds_cnts = binned_counts(cds0.astype(float), bin_ids, n_bins)
    obs_pds_cnts = binned_counts(pds0.astype(float), bin_ids, n_bins)
    obs_unc_cnts = binned_counts(unc0.astype(float), bin_ids, n_bins)

    # encode labels for permutation: 1=CDS, 2=PDS
    labels = np.zeros(dist0.size, dtype=int)
    labels[cds0] = 1
    labels[pds0] = 2
    labels[unc0] = 3
    null_cds_cnts = np.zeros((n_perm, n_bins), float)
    null_pds_cnts = np.zeros((n_perm, n_bins), float)
    null_unc_cnts = np.zeros((n_perm, n_bins), float)

    for i in range(n_perm):
        perm = rng.permutation(labels) # permute labels and bin count
        cds_p = (perm == 1).astype(float)
        pds_p = (perm == 2).astype(float)
        unc_p = (perm == 3).astype(float)
        null_cds_cnts[i] = binned_counts(cds_p, bin_ids, n_bins)
        null_pds_cnts[i] = binned_counts(pds_p, bin_ids, n_bins)
        null_unc_cnts[i] = binned_counts(unc_p, bin_ids, n_bins)

    return {
        "centers": centers,
        "edges": edges,
        "n_bins": n_bins,
        "obs_cds_cnts": obs_cds_cnts,
        "obs_pds_cnts": obs_pds_cnts,
        "obs_unc_cnts": obs_unc_cnts,
        "null_cds_cnts": null_cds_cnts,
        "null_pds_cnts": null_pds_cnts,
        "null_unc_cnts": null_unc_cnts,
    }

def p_to_ast(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""
    
def tanh_func(x, amp, c, temp):
    return -amp * np.tanh((x - c) / temp)

def fit_tanh_curve(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    amp0 = max(np.nanmax(np.abs(y)), 1e-3)
    c0 = np.nanmedian(x)
    temp0 = 75.0

    popt, _ = curve_fit(
        tanh_func,
        x,
        y,
        p0=[amp0, c0, temp0],
        bounds=([-10.0, np.min(x), 30.0], [10.0, np.max(x), 200.0]),
        maxfev=20000,
    )
    return popt  # amp, c, temp

def plot_amp_violin(null_amps, obs_amp, pval, ax, color="k"):
    vp = ax.violinplot(np.abs(null_amps), showextrema=False, widths=0.8)
    for body in vp["bodies"]:
        body.set_facecolor("0.6")
        body.set_edgecolor("0.6")
        body.set_alpha(0.7)

    ax.axhline(obs_amp, color=color, linewidth=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(ticks=[], labels=[])
    ax.set_ylabel("|Amp.|", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.text(0.5, 1.15, p_to_ast(pval), transform=ax.transAxes,
            ha="center", va="top", fontsize=12, color="k")

def plot_ratio(ratio, null_ratio, centers, total_neurons, ylabel="Component-to-pattern ratio", rat_ylim=4, out_path=None):
    base_ratio = np.nanmean(null_ratio, axis=0)
    lo_ratio = np.nanpercentile(null_ratio, 2.5, axis=0)
    hi_ratio = np.nanpercentile(null_ratio, 97.5, axis=0)

    # fit tanh to observed and null curves; use |amplitude| as test statistic
    obs_amp, obs_c, obs_temp = fit_tanh_curve(centers, ratio)

    obs_fit = tanh_func(centers, obs_amp, obs_c, obs_temp)
    ss_res = np.sum((ratio - obs_fit) ** 2)
    ss_tot = np.sum((ratio - np.mean(ratio)) ** 2)
    obs_r2 = 1.0 - (ss_res / ss_tot)

    print(f"r2 {obs_r2}")

    null_amps = np.full(null_ratio.shape[0], np.nan)
    for i in range(null_ratio.shape[0]):
        amp_i, _, _ = fit_tanh_curve(centers, null_ratio[i])
        null_amps[i] = amp_i

    global_pval = (1 + np.sum(np.abs(null_amps) >= np.abs(obs_amp))) / (len(null_amps) + 1)
    print(f"amplitude permutation test for distance-dependence p-value: {global_pval}")
    print(f"observed fit: amp={obs_amp:.3f}, c={obs_c:.1f}, temp={obs_temp:.1f}")

    fig, ax = plt.subplots(1, 1, figsize=(2.55, 2.9))

    ax.scatter(centers, ratio, color="k", marker=".", s=70)
    xs = np.linspace(np.min(centers), np.max(centers), 400)
    ax.plot(xs, tanh_func(xs, obs_amp, obs_c, obs_temp), color="k", lw=1.3, zorder=3)
    ax.plot(centers, base_ratio, color="k", ls="--", lw=1.3)
    ax.fill_between(centers, lo_ratio, hi_ratio, color="k", alpha=0.15, linewidth=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Distance from nearest fracture (\u03bcm)", fontsize=10, fontweight="regular")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="regular")

    ax.set_xlim(0, 500)
    if rat_ylim == 4:
        ax.set_yticks(
            [-np.log2(4), -np.log2(2), 0, np.log2(2), np.log2(4)],
            labels=["1 : 4", "1 : 2", "1 : 1", "2 : 1", "4 : 1"]
        )
        ax.set_ylim([-2, 2])
    else:
        ax.set_yticks(
            [-np.log2(8), -np.log2(4), -np.log2(2), 0, np.log2(2), np.log2(4), np.log2(8)],
            labels=["1 : 8", "1 : 4", "1 : 2", "1 : 1", "2 : 1", "4 : 1", "8 : 1"]
        )
        ax.set_ylim([-3, 3])

    ax.text(
        0.95, 0.98, "$n_{neurons}$ = " + f"{int(total_neurons):,}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8, color="k"
    )
    ax.text(
        0.95, 0.92,
        f"Amp. = {obs_amp:.2f}\n" + f"b = {obs_c:.0f} μm",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8, color="k"
    )

    ax_in = fig.add_axes([0.42, 0.7, 0.12, 0.18])
    plot_amp_violin(null_amps, obs_amp, global_pval, ax_in, color="k")

    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)

"""
def plot_ratio(ratio, null_ratio, centers, total_neurons,  ylabel = "Component-to-pattern ratio", rat_ylim = 4, out_path = None):
    base_ratio = np.nanmean(null_ratio, axis=0)
    lo_ratio= np.nanpercentile(null_ratio, 2.5, axis=0)
    hi_ratio = np.nanpercentile(null_ratio, 97.5, axis=0)

    # global p
    # global permutation test for overall curve difference from null
    obs_diff = np.sum(np.abs(ratio - base_ratio))
    null_diff = np.sum(
        np.abs(null_ratio - base_ratio[None, :]),
        axis=1
    )

    global_pval = (1 + np.sum(null_diff >= obs_diff)) / (len(null_diff) + 1)
    print(f"global permutation test for distance-dependence p-value: {global_pval}")

    fig, ax = plt.subplots(1, 1, figsize=(2.55, 2.9))

    ax.scatter(centers, ratio, color='k', marker=".", s=70)
    ax.plot(centers, ratio, color='k', lw=1.3)
    ax.plot(centers, base_ratio, color='k', ls="--", lw=1.3)
    ax.fill_between(centers, lo_ratio, hi_ratio, color="k", alpha=0.15, linewidth=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Distance from nearest fracture (\u03bcm)", fontsize=10, fontweight="regular")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="regular")

    ax.set_xlim(0, 500)
    if rat_ylim == 4:
        ax.set_yticks([-np.log2(4), -np.log2(2),  0, np.log2(2), np.log2(4)], labels=["1 : 4", "1 : 2", "1 : 1",  "2 : 1", "4 : 1"])
        ax.set_ylim([-2, 2])
    else:
        ax.set_yticks([-np.log2(8), -np.log2(4), -np.log2(2),  0, np.log2(2), np.log2(4), np.log2(8)], labels=["1 : 8", "1 : 4", "1 : 2", "1 : 1",  "2 : 1", "4 : 1", "8 : 1"])
        ax.set_ylim([-3, 3])        

    if global_pval < 0.05:
        ast_text = "*"
    if global_pval < 0.01:
        ast_text = "**"
    if global_pval < 0.001:
        ast_text = "***"

    if global_pval < 0.05:
        ax.text(
            0.96, 0.96, ast_text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=14, color="k"
        )
    
    ax.text(
        0.92, 0.9, "$n_{neurons}$ = " + f"{int(total_neurons):,}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8, color="k"
    )

    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
"""
def plot_all_sessions_pdf(cache_list, out_path=None, norm=True, include_unc=True):
    centers = cache_list[0]["centers"]

    n_bins = centers.size
    n_perm = cache_list[0]["null_cds_cnts"].shape[0]

    # pooled observed counts across sessions
    obs_cds_cnts = np.zeros(n_bins, float)
    obs_pds_cnts = np.zeros(n_bins, float)
    obs_unc_cnts = np.zeros(n_bins, float)

    for c in cache_list:
        obs_cds_cnts += c["obs_cds_cnts"]
        obs_pds_cnts += c["obs_pds_cnts"]
        obs_unc_cnts += c["obs_unc_cnts"]

    if include_unc == False:
        total_neurons = (obs_cds_cnts.sum() + obs_pds_cnts.sum())
    else:
        total_neurons = (obs_cds_cnts.sum() + obs_pds_cnts.sum() + obs_unc_cnts.sum())

    null_cds_cnts = np.full((n_perm, n_bins), np.nan, float)
    null_pds_cnts = np.full((n_perm, n_bins), np.nan, float)
    null_unc_cnts = np.full((n_perm, n_bins), np.nan, float)

    for i in range(n_perm):
        ss_cds = np.zeros(n_bins, float)
        ss_pds = np.zeros(n_bins, float)
        ss_unc = np.zeros(n_bins, float)

        for c in cache_list:
            ss_cds += c["null_cds_cnts"][i]
            ss_pds += c["null_pds_cnts"][i]
            ss_unc += c["null_unc_cnts"][i]
        
        null_cds_cnts[i] = ss_cds
        null_pds_cnts[i] = ss_pds
        null_unc_cnts[i] = ss_unc


    ratio = np.log2(obs_cds_cnts/obs_pds_cnts)
    null_ratio = np.log2(null_cds_cnts/null_pds_cnts)
    
    plot_ratio(ratio, null_ratio, centers, total_neurons,  ylabel = "Component-to-pattern ratio", out_path = out_path)

    if include_unc:
        ratio = np.log2(obs_cds_cnts/(obs_pds_cnts+obs_unc_cnts))
        null_ratio = np.log2(null_cds_cnts/(null_pds_cnts+obs_unc_cnts))
        
        plot_ratio(ratio, null_ratio, centers, total_neurons,  ylabel = "Component-to-unc+pattern ratio", rat_ylim = 6, out_path = out_path[:-4] + "_cu.pdf")
        
        ratio = np.log2(obs_pds_cnts/(obs_cds_cnts+obs_unc_cnts))
        null_ratio = np.log2(null_pds_cnts/(null_cds_cnts+obs_unc_cnts))
        
        plot_ratio(ratio, null_ratio, centers, total_neurons,  ylabel = "Pattern-to-unc+component ratio", rat_ylim = 6, out_path = out_path[:-4] + "_pu.pdf")
        
        ratio = np.log2(obs_unc_cnts/(obs_cds_cnts+obs_pds_cnts))
        null_ratio = np.log2(null_unc_cnts/(null_cds_cnts+obs_pds_cnts))
        
        plot_ratio(ratio, null_ratio, centers, total_neurons,  ylabel = "Unc-to-pattern+component ratio", rat_ylim = 6, out_path = out_path[:-4] + "_uu.pdf")
        
        


def main():
    parser = argparse.ArgumentParser(description="Distance-to-fracture distributions")
    parser.add_argument("--config", default="configs/default_config.yaml", type=str)
    parser.add_argument("--split", default="all", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = config["dataset_params"]["data_path"]
    fs = config['fractures']
    ymax = config["neuron_params"]["neuron_y_thr"]
    include_unc = False
    ordered_dates, date_to_session = sort_dates_and_make_session_titles(config, args.split)

    pat = load_pat()
    fractures_all = load_fractures(args.split)

    num_fracs = 0
    for key in fractures_all.keys():
        num_fracs += len(fractures_all[key]['grating'])

    print(f"total number of fractures: {num_fracs}")
    print(f"mean number of fracs per session: {num_fracs/len(fractures_all.keys())}")

    # distance bin edges for our plot/stats
    edges = np.arange(0.0, float(fs['max_frac_dist']) + float(fs['frac_bin_size']), float(fs['frac_bin_size']))

    all_cache = []
    sess_cnt = 0
    suffix=""
    dates = []
    for date in ordered_dates:
        # skip sessions without pat_idx or without fractures
        if date not in pat["main"] or date not in config['dataset_params']['dates_fracture']:
            continue

        # parse fracture list from saved structure
        fr_entry = fractures_all[date]
        fractures = fr_entry['grating']
        fractures = np.asarray(fractures, float)

        # skip sessions with no fractures
        if fractures.size == 0:
            continue
        
        dates.append(date)
        data = load_session(data_path, date)
        neu_inc = load_inc()["rf_plaid"][str(date)]

        neuron_y = data["y"][neu_inc].astype(float)
        pat_main = pat["main"][date]

        is_cds = pat_main["is_cds"]
        is_pds = pat_main["is_pds"]
        is_unc = pat_main["is_unc"]

        # restrict to depth range for fracture analysis 
        m_depth = (neuron_y >= 0) & (neuron_y <= float(ymax))
        neuron_y = neuron_y[m_depth]

        is_cds = is_cds[m_depth]
        is_pds = is_pds[m_depth]
        is_unc = is_unc[m_depth]

        # for all neurons, compute their distance to the nearest fracture
        dist = nearest_fracture_dist(neuron_y, fractures)

        # counts cds and pds in distance from fracture bins 
        cache = cache_session_class(dist_um=dist, is_cds=is_cds, is_pds=is_pds, is_unc=is_unc, edges=edges, n_perm=int(fs['frac_n_perm']), seed=int(fs['frac_seed']) + int(date), include_unc=True)
        all_cache.append(cache)

        sess_cnt += 1

    out_dir = f"figures/general/{args.split}/all_sessions/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"fracture_distance_pdf_all_sessions{suffix}_norm.pdf")
    plot_all_sessions_pdf(all_cache, out_path=out_path, include_unc=False)

    print(f"\nPlotted {sess_cnt} sessions.")

if __name__ == "__main__":
    main()
