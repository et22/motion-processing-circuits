import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from copy import deepcopy
from utils import load_config, sort_dates_and_make_session_titles, load_pat, load_session, load_inc, set_plot_defaults, save_waves

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu

from sklearn.metrics import balanced_accuracy_score

FS_COLOR = "#6a51a3"
RS_COLOR = "#238b45"

pattern_color, component_color, unclass_color = set_plot_defaults()

def p_to_ast(p):
    if p < 0.001:
        p_txt = "***"
    elif p < 0.01:
        p_txt = "**"
    elif p < 0.05: 
        p_txt = "*"
    else:
        p_txt = ""
    return p_txt

def plot_perm_accuracy_hist(perm_accs, acc, pval, out_stem, nbins=20, color='r'):
    perm_accs = np.asarray(perm_accs)

    fig, ax = plt.subplots(figsize=(1.5, 1.9))

    # histogram (gray bars)
    vp = plt.violinplot(perm_accs, showextrema=False)
    for body in vp['bodies']:
        body.set_facecolor("0.6")
        body.set_edgecolor("0.6")
        body.set_alpha(0.7)

    #ax.hist(perm_accs, bins=nbins, color="0.6", edgecolor="none")

    #ax.axhline(np.mean(perm_accs), color='k', linestyle='--', linewidth=1.5)

    # true accuracy (red vertical line)
    ax.axhline(acc, color=color, linewidth=1.5)

    # labels
    ax.set_ylabel("Accuracy")
    #ax.set_ylabel("Count")

    # clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(ticks=[], labels=[])

    #ax.legend(["Null", "Null Mean", "Measured"], frameon=False)

    ax.text(1.15, 0.68, p_to_ast(pval), color='k', fontsize=12)
    plt.ylim([0.45, 0.7])
    fig.tight_layout()
    fig.savefig(out_stem, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_waveforms_depth(metrics, times, outstem, setup_date):
    m = metrics["all_dates"] == str(setup_date)
    wf = metrics["normalized_waveforms"][m]
    ys = metrics["all_y"][m]

    order = np.argsort(ys)
    wf = wf[order]
    ys = ys[order]

    fig = plt.figure(figsize=(1.8, 8.0))
    ax = fig.add_subplot(111)

    xscale = 4
    yscale = 45
    xjitter = 30
    rng = np.random.default_rng(0)

    t_us = times

    palette = plt.get_cmap("Dark2").colors
    palette = np.array(palette, dtype=object)

    for i in range(len(ys)):
        x = xscale * t_us / 1000.0
        x = x[0, :len(wf[i])]
        x = x + rng.uniform(-xjitter / 2, xjitter / 2)
        y = yscale * wf[i] + ys[i]
        c = palette[rng.integers(len(palette))]
        ax.plot(x, y, color=c, lw=1.0, alpha=0.95)

    ax.set_xticks([])
    ax.set_yticks([])

    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_xlim(ax.get_xlim()[0]-(0.05*(ax.get_xlim()[1]-ax.get_xlim()[0])), ax.get_xlim()[1])

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xr = xmax - xmin
    yr = ymax - ymin

    x0 = xmin + 0 * xr
    y0 = ymin + 0.03 * yr

    ms_bar_len = 2.0 * xscale
    depth_bar_len_um = 200.0

    ax.annotate(
        "",
        xy=(x0 + ms_bar_len, y0),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="k", shrinkA=0, shrinkB=0),
    )

    ax.annotate(
        "",
        xy=(x0, y0 + depth_bar_len_um),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="k", shrinkA=0, shrinkB=0),
    )

    ax.text(x0 + 0.5 * ms_bar_len, y0 - 0.02 * yr, "2 ms", ha="center", va="top", fontsize=8)

    ax.text(
        x0 - 0.02 * xr,
        y0 + 0.5 * depth_bar_len_um,
        f"200 μm",
        ha="right",
        va="center",
        rotation=90,
        fontsize=8,
    )
    fig.savefig(outstem, dpi=800, bbox_inches="tight")

def norm_waveforms(all_waveforms, metric_dict):
    # all_waveforms is n_neurons x n_timepoints
    # subclustering analysis on PDS/CDS FS waveform templates

    # 1) normalization used for waveform plotting:
    # baseline to 0, then divide by trough magnitude
    all_waveforms = all_waveforms - all_waveforms[:, 0][:, None]
    all_waveforms = all_waveforms / -np.min(all_waveforms, axis=1, keepdims=True)

    # 2) realignment to trough for each neuron
    mins = np.argmin(all_waveforms, axis=1)
    window_size = 51  # [-15, +35]
    n_neurons, n_t = all_waveforms.shape
    wf_subset = np.full((n_neurons, window_size), np.nan)

    for i in range(n_neurons):
        start = mins[i] - 15
        end = mins[i] + 35 + 1
        if start >= 0 and end <= n_t:
            wf_subset[i, :] = all_waveforms[i, start:end]

    # only keep neurons with valid aligned windows
    valid = np.all(np.isfinite(wf_subset), axis=1)

    wf_subset = wf_subset[valid]
    metric_dict = deepcopy(metric_dict)
    for key in metric_dict.keys():
        metric_dict[key] = metric_dict[key][valid]

    metric_dict["normalized_waveforms"] = wf_subset

    metric_dict["peak_amp"] = np.max(metric_dict["normalized_waveforms"], axis=1)
    return metric_dict

def waveform_classifier(metrics, use_hand_feats=False, use_pc_feats=False, kernel_func=None):
    # goal is to report CV accuracy vs a within-session shuffled null 
    # focus on cds vs pds binary classification
    inc_mask = metrics["is_cds"] | metrics["is_pds"]

    y = metrics["is_pds"][inc_mask]
    if use_hand_feats:
        X = np.concatenate([metrics["peak_amp"][inc_mask, None], metrics["all_ttp_us_raw"][inc_mask, None]], axis=1)
    else:
        X = metrics["normalized_waveforms"][inc_mask, :]
        X_ear = X[:, 1:15]
        X_lat = X[:, 16:]
        X = np.concatenate((X_ear, X_lat), axis=1)
    
    if use_pc_feats:
        reducer = PCA(n_components=2, random_state=0)
        X = reducer.fit_transform(X)

    print("Pat/(Pat+Comp): " + str(np.mean(y))) # this is about 0.5 (0.51), so we don't need to subsample 

    dates = metrics["all_dates"][inc_mask]

    skf = StratifiedKFold(shuffle=True, random_state=42)
    
    accs = []
    Cs = []
    for j, (train_index, test_index) in enumerate(skf.split(X, y)):
        # hyperparameter selection insde CV loop (nested CV)
        C_values = 10.0 ** np.arange(-1, 3)
        C_accs = []
        for C in C_values:
            clf_inner = Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(C=C, kernel=kernel_func, random_state=42)),
            ])
            skf_inner = StratifiedKFold(shuffle=True, random_state=42)
            X_train_outer = X[train_index, :]
            y_train_outer = y[train_index]
            inner_accs = []
            for i, (inner_train_index, inner_test_index) in enumerate(skf_inner.split(X_train_outer, y_train_outer)):
                clf_inner.fit(X_train_outer[inner_train_index, :], y_train_outer[inner_train_index])
                y_test_inner = y_train_outer[inner_test_index]
                y_pred_inner = clf_inner.predict(X_train_outer[inner_test_index, :])
                acc = balanced_accuracy_score(y_test_inner, y_pred_inner) #np.mean(y_test_inner == y_pred_inner)
                inner_accs.append(acc)
            inner_acc = np.mean(inner_accs) # mean CV accuracy 
            C_accs.append(inner_acc)

        C = C_values[np.argmax(C_accs)]
        Cs.append(C)
    
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(C=C, kernel=kernel_func, random_state=42)),
        ])        
        clf.fit(X[train_index, :], y[train_index])
        y_test = y[test_index]
        y_pred = clf.predict(X[test_index, :])
        acc = balanced_accuracy_score(y_test, y_pred) #np.mean(y_test == y_pred)
        accs.append(acc)

    acc = np.mean(accs) # mean CV accuracy 
    print("true cv accuracy:", acc)

    # repeat n_perm times, n_perm = 100
    n_perm = 1000 # can reduce if want to run faster but using this for final manuscript run
    rng = np.random.default_rng(0)
    perm_accs = []

    for perm_id in range(n_perm):
        y_perm = deepcopy(y)
        y_perm = rng.permutation(y_perm)

        # within-session --- decided against bc confused people; randomly permute y within each date
        #for d in np.unique(dates):
        #    m = dates == d
        #    y_perm[m] = rng.permutation(y_perm[m])

        fold_accs = []
        for i, (train_index, test_index) in enumerate(skf.split(X, y_perm)):
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(C=Cs[i], kernel=kernel_func, random_state=42)),
            ])   
            clf.fit(X[train_index, :], y_perm[train_index])
            y_test = y_perm[test_index]
            y_pred = clf.predict(X[test_index, :])
            fold_accs.append(balanced_accuracy_score(y_test, y_pred))

        perm_accs.append(np.mean(fold_accs))

    perm_accs = np.asarray(perm_accs)
    pval = (1 + np.sum(perm_accs >= acc)) / (n_perm + 1)
    print("null cv accuracy:", np.mean(perm_accs))
    print("p-value:", pval)
    return perm_accs, acc, pval



def define_features_and_fsrs(ds, outstem, t):
    fig, axs = plt.subplots(1, 2, figsize=(4.4 / 1.3, 2.8 / 1.3), sharey=True)
    ax0, ax1 = axs

    classified = ds["is_cds"] | ds["is_pds"]
    fs = classified & ds["is_fs"]
    rs = classified & ds["is_rs"]
    ex_idx = min(905, len(ds["normalized_waveforms"]) - 1)

    wf = ds["normalized_waveforms"][ex_idx]
    trough_i = np.argmin(wf)
    peak_i = np.argmax(wf)
    peak_amp = wf[peak_i]

    ax0.plot(t, wf, color="k", lw=1.5)
    ax0.axhline(0, color="k", lw=0.8, ls="--", zorder=0)
    ax0.axvline(t[peak_i], color="k", lw=0.8, ls=":", zorder=0)

    y_arrow = -0.98
    xpad = 8
    ax0.annotate(
        "",
        xy=(t[peak_i] + xpad, y_arrow),
        xytext=(t[trough_i] - xpad, y_arrow),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color="k"),
    )
    ax0.text(
        t[peak_i] + 50,
        y_arrow + 0.10,
        "Trough-to-peak\nduration",
        ha="left",
        va="bottom",
        fontsize=8,
    )

    ypad = 0.04
    ax0.annotate(
        "",
        xy=(t[peak_i], peak_amp + ypad),
        xytext=(t[peak_i], -ypad),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color="k"),
    )
    ax0.text(
        t[peak_i] + 50,
        0.5 * peak_amp + 0.38,
        "Normalized\npeak amplitude",
        ha="left",
        va="center",
        fontsize=8,
    )

    ax0.plot(t[trough_i], wf[trough_i], "o", ms=3, color="k")
    ax0.plot(t[peak_i], wf[peak_i], "o", ms=3, color="k")
    ax0.set_xlabel(f"Time from trough (μs)")

    rng = np.random.default_rng(0)
    for mask, color in [(fs, FS_COLOR), (rs, RS_COLOR)]:
        idx = np.where(mask)[0]
        if len(idx) > 0:
            draw = rng.choice(idx, size=min(200, len(idx)), replace=False)
            for i in draw:
                ax1.plot(t, ds["normalized_waveforms"][i], color=color, lw=0.5, alpha=0.06)

    def plot_mean(ax, mask, color, label):
        mu = np.nanmedian(ds["normalized_waveforms"][mask], axis=0)
        ax.plot(t, mu, color=color, lw=1.5, label=label)

    plot_mean(ax1, fs, FS_COLOR, f"FS (n={np.sum(fs)})")
    plot_mean(ax1, rs, RS_COLOR, f"RS (n={np.sum(rs)})")

    ax1.set_xlabel(f"Time from trough (μs)")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")

    axs[0].set_ylim(-1.25, 1.2)

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])

    fig.tight_layout()
    plt.savefig(outstem, dpi=800, bbox_inches="tight")

def subset_norm_metrics(metric_dict, fsrs_split, fs=True):
    metric_dict = deepcopy(metric_dict)
    if fs:
        idx = metric_dict["all_ttp_us_raw"] <= fsrs_split
    else:
        idx = metric_dict["all_ttp_us_raw"] > fsrs_split
    for key in metric_dict.keys():
        metric_dict[key] = metric_dict[key][idx]
    return metric_dict

def comppat_feature(ds, outstem, use_fs=True, use_all=False):
    if use_all:
        fs_classified = (ds["is_cds"] | ds["is_pds"])
    else:
        if use_fs:
            fs_classified = ds["is_fs"] & (ds["is_cds"] | ds["is_pds"])
        else:
            fs_classified = ds["is_rs"] & (ds["is_cds"] | ds["is_pds"])
    cds_vals = ds["peak_amp"][fs_classified & ds["is_cds"]]
    pds_vals = ds["peak_amp"][fs_classified & ds["is_pds"]]

    _, p = mannwhitneyu(cds_vals, pds_vals, alternative="two-sided")
    print("===metric comp vs patt, diff===")
    print(np.nanmedian(cds_vals))
    print(np.nanmedian(pds_vals))
    print(np.nanmedian(pds_vals) - np.nanmedian(cds_vals))
    print(p)
    fig, ax = plt.subplots(figsize=(2.2 / 1.3, 3.0 / 1.3))

    data = [cds_vals, pds_vals]
    pos = [0, 1]

    vp = ax.violinplot(
        data,
        positions=pos,
        widths=0.8,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )

    for body, c in zip(vp["bodies"], [component_color, pattern_color]):
        body.set_facecolor(c)
        body.set_edgecolor("none")
        body.set_alpha(0.22)

    bp = ax.boxplot(
        data,
        positions=pos,
        widths=0.28,
        patch_artist=True,
        showcaps=True,
        showfliers=False,
        medianprops=dict(color="k", linewidth=1.2),
        whiskerprops=dict(color="k", linewidth=1.0),
        capprops=dict(color="k", linewidth=1.0),
        boxprops=dict(linewidth=1.0, color="k"),
    )

    for patch, c in zip(bp["boxes"], [component_color, pattern_color]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(0)
    for x0, vals, c in zip(pos, data, [component_color, pattern_color]):
        jitter = rng.normal(0, 0.06, size=len(vals))
        ax.scatter(np.full(len(vals), x0) + jitter, vals, s=8, color=c, alpha=0.32, linewidth=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.83, 1, p_to_ast(p), transform=ax.transAxes, ha="left", va="top", fontsize=12)

    if use_all:
        ax.set_xticks(pos, ["Comp.", "Patt."])
    elif use_fs:
        ax.set_xticks(pos, ["FS, Comp.", "FS, Patt."])
    else:
        ax.set_xticks(pos, ["RS, Comp.", "RS, Patt."])

    ax.set_ylabel("Normalized peak amplitude")
    ax.set_ylim([0, 1])

    plt.savefig(outstem, dpi=800, bbox_inches="tight")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml", type=str)
    parser.add_argument("--split", default="all", type=str)
    args = parser.parse_args()

    use_hand_feats = False
    use_pc_feats = False
    kernel_func = 'rbf'

    if args.split == "main":
        setup_date = "121825"
    else:
        setup_date = "041825"

    config = load_config(args.config)
    fsrs_split = config["analysis_params"]["fsrs_split"]
    
    outdir = f"figures/general/{args.split}/waveforms"
    os.makedirs(outdir, exist_ok=True)

    data_path = config["dataset_params"]["data_path"]
    ordered_dates, _ = sort_dates_and_make_session_titles(config, args.split)
    pat_outputs = load_pat()["main"]

    is_cds_cache = []
    is_pds_cache = []
    pi_cls_cache = []
    
    ttp_cache = []
    raw_wf_cache = []
    date_cache = []
    y_cache = []
    x_cache = []
    pi_cache = []
    neu_id_cache = []

    for date in ordered_dates:
        if date not in pat_outputs.keys():
            continue

        data = load_session(data_path, date)
        neu_inc = load_inc()["rf_plaid"][str(date)]

        pat_dict = pat_outputs[date]

        pi = np.asarray(pat_dict["pi"])
        is_cds = np.asarray(pat_dict["is_cds"], dtype=bool)
        is_pds = np.asarray(pat_dict["is_pds"], dtype=bool)
        pi_cls = np.asarray(pat_dict["pi_cls"], dtype=int)

        waveforms = np.asarray(data["waveforms"][neu_inc, :], dtype=float)
        times_us = 1000 * np.asarray(data["waveform_times"][neu_inc, :], dtype=float)
        ys = np.asarray(data["y"][neu_inc], dtype=float)
        xs = np.asarray(data["x"][neu_inc], dtype=float) - np.min(data["x"][neu_inc]) - 52

        ttp_us = []
        for i in range(len(ys)):
            tr = np.argmin(waveforms[i])
            pk = np.argmax(waveforms[i])
            ttp_us.append(np.round(times_us[i, pk] - times_us[i, tr]).astype(int))
        ttp_us = np.asarray(ttp_us)

        raw_wf_cache.append(waveforms)
        is_cds_cache.append(is_cds)
        is_pds_cache.append(is_pds)
        pi_cls_cache.append(pi_cls)
        ttp_cache.append(ttp_us)
        date_cache.append(np.array([str(date)] * len(pi)))
        y_cache.append(ys)
        x_cache.append(xs)
        pi_cache.append(pi)
        neu_id_cache.append(np.arange(len(pi)))

    all_waveforms_raw = np.concatenate(raw_wf_cache, axis=0)
    all_is_cds = np.concatenate(is_cds_cache, axis=0)
    all_is_pds = np.concatenate(is_pds_cache, axis=0)
    all_pi_cls = np.concatenate(pi_cls_cache, axis=0)
    all_ttp_us_raw = np.concatenate(ttp_cache, axis=0)
    all_dates = np.concatenate(date_cache, axis=0)
    all_y = np.concatenate(y_cache, axis=0)
    all_x = np.concatenate(x_cache, axis=0)
    all_pi = np.concatenate(pi_cache, axis=0)
    all_neu_ids = np.concatenate(neu_id_cache, axis=0)

    metric_dict = {
        "all_waveforms_raw": all_waveforms_raw,
        "all_ttp_us_raw": all_ttp_us_raw,
        "all_dates": all_dates,
        "all_y": all_y,
        "all_x": all_x,
        "is_cds": all_is_cds,
        "is_pds": all_is_pds,
        "pi_cls": all_pi_cls,
        "pi": all_pi,
        "is_fs": all_ttp_us_raw <= fsrs_split,
        "is_rs": all_ttp_us_raw > fsrs_split,
        "all_neu_ids": all_neu_ids,
    }

    # norm_metrics contains normalized metric dictionary, above
    norm_metrics = norm_waveforms(all_waveforms_raw, metric_dict)

    define_features_and_fsrs(norm_metrics, outdir + "/waveform_example.pdf",  times_us[0, :51] - times_us[0, 15])
    
    print("\nall")
    comppat_feature(norm_metrics, outdir + "/all_peak_amp.pdf", use_fs=False, use_all=True)
    print("\nfs")
    comppat_feature(norm_metrics, outdir + "/fs_peak_amp.pdf", use_fs=True)
    print("\nrs")
    comppat_feature(norm_metrics, outdir + "/rs_peak_amp.pdf", use_fs=False)

    plot_waveforms_depth(norm_metrics, times_us, outdir + "/waveform_depth.pdf", setup_date)
    all_perm_accs, all_acc, pval = waveform_classifier(norm_metrics, use_hand_feats, use_pc_feats, kernel_func)
    plot_perm_accuracy_hist(all_perm_accs, all_acc, pval, outdir + "/overall_classifier.pdf")

    fs_dict = subset_norm_metrics(norm_metrics, fsrs_split, fs=True)
    rs_dict = subset_norm_metrics(norm_metrics, fsrs_split, fs=False)

    fs_perm_accs, fs_acc, pval = waveform_classifier(fs_dict, use_hand_feats, use_pc_feats, kernel_func)
    plot_perm_accuracy_hist(fs_perm_accs, fs_acc, pval, outdir + "/fs_classifier.pdf", color=FS_COLOR)

    rs_perm_accs, rs_acc, pval = waveform_classifier(rs_dict, use_hand_feats, use_pc_feats, kernel_func)
    plot_perm_accuracy_hist(rs_perm_accs, rs_acc, pval, outdir + "/rs_classifier.pdf", color=RS_COLOR)  
    
    save_waves(norm_metrics)

if __name__ == "__main__":
    main()  