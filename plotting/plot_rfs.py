import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from scipy.interpolate import RectBivariateSpline
from scipy.stats import linregress

from utils import (
    load_config,
    load_session,
    load_inc,
    load_rfs,                 
    sort_dates_and_make_session_titles,
    set_plot_defaults,
)

pattern_color, component_color, unclass_color = set_plot_defaults()


RF_CMAP = "RdBu_r"
RF_VMIN = 0.0
RF_VMAX = 1.0


def _add_rf_meridians(ax, linewidth=1.4):
    ax.axvline(
        0,
        linestyle="--",
        color="white",
        linewidth=linewidth,
        alpha=0.95,
        zorder=10,
    )
    ax.axhline(
        0,
        linestyle="--",
        color="white",
        linewidth=linewidth,
        alpha=0.95,
        zorder=10,
    )


def _style_rf_colorbar(cbar, fontsize=8):
    cbar.ax.tick_params(labelsize=fontsize - 1)
    cbar.set_label(
        "Normalized Firing Rate",
        weight="regular",
        rotation=-90,
        labelpad=16,
        fontsize=fontsize,
    )

def gaussian_2d(pos, amp, xo, yo, sigma, offset):
    x, y = pos
    return offset + amp * np.exp(-((x - xo) ** 2 + (y - yo) ** 2) / (2.0 * sigma ** 2))


def eval_gaussian_map(x_coords, y_coords, fit_params):
    amp, xo, yo, sigma, offset = fit_params
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = gaussian_2d((X, Y), amp, xo, yo, sigma, offset)
    return Z


def interpolate_rf_map(map2d, x_coords, y_coords, upsample=6):
    x_coords = np.asarray(x_coords, float)
    y_coords = np.asarray(y_coords, float)
    map2d = np.asarray(map2d, float)

    valid = np.isfinite(map2d)
    if np.sum(valid) < 4:
        return x_coords, y_coords, map2d

    # fill missing values conservatively so spline does not fail
    z = map2d.copy()
    if not np.all(valid):
        fill_val = np.nanmedian(z[valid])
        z[~valid] = fill_val

    spline = RectBivariateSpline(y_coords, x_coords, z, kx=2, ky=2)
    x_fine = np.linspace(x_coords.min(), x_coords.max(), upsample * len(x_coords))
    y_fine = np.linspace(y_coords.min(), y_coords.max(), upsample * len(y_coords))
    z_fine = spline(y_fine, x_fine)
    return x_fine, y_fine, z_fine


def _format_rf_axis(ax, x_coords, y_coords, show_xlabel=False, show_ylabel=False):
    # --- X axis ---
    if show_xlabel:
        ax.set_xticks(x_coords)
        ax.set_xticklabels(
            [f"{x:g}°" if ii % 2 == 0 else "" for ii, x in enumerate(x_coords)],
            fontsize=6,
        )
        ax.set_xlabel("Horizontal position (°)", fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel("")

    # --- Y axis ---
    if show_ylabel:
        ax.set_yticks(y_coords)
        ax.set_yticklabels(
            [f"{y:g}°" if ii % 2 == 0 else "" for ii, y in enumerate(y_coords)],
            fontsize=6,
        )
        ax.set_ylabel("Vertical position (°)", fontsize=8)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel("")

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=2.5,
        width=0.8,
        bottom=show_xlabel,
        left=show_ylabel,
        labelbottom=show_xlabel,
        labelleft=show_ylabel,
    )

    ax.set_aspect("equal")


def _plot_rf_panel_grid(
    maps,
    x_coords,
    y_coords,
    neuron_ids,
    out_path,
    mode="raw",
    fit_params=None,
    centers=None,
    fwhm=None,
    title=None,
):
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    axes = axes.ravel()

    for k, ax in enumerate(axes):
        idx = neuron_ids[k]

        if mode == "raw":
            im = ax.imshow(
                (maps[idx] - np.min(maps[idx])) / (np.max(maps[idx]) - np.min(maps[idx]) + 1e-6),
                origin="lower",
                extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
                interpolation="nearest",
                aspect="equal",
                cmap=RF_CMAP,
                vmin=RF_VMIN,
                vmax=RF_VMAX,
            )

        elif mode == "interp":
            x_f, y_f, z_f = interpolate_rf_map(maps[idx], x_coords, y_coords, upsample=8)
            im = ax.imshow(
                (z_f - np.min(z_f)) / (np.max(z_f) - np.min(z_f) + 1e-6),
                origin="lower",
                extent=[x_f.min(), x_f.max(), y_f.min(), y_f.max()],
                interpolation="bicubic",
                aspect="equal",
                cmap=RF_CMAP,
                vmin=RF_VMIN,
                vmax=RF_VMAX,
            )

        elif mode == "fit":
            z_fit = eval_gaussian_map(x_coords, y_coords, fit_params[idx])
            x_f, y_f, z_f = interpolate_rf_map(z_fit, x_coords, y_coords, upsample=8)
            im = ax.imshow(
                (z_f - np.min(z_f)) / (np.max(z_f) - np.min(z_f) + 1e-6),
                origin="lower",
                extent=[x_f.min(), x_f.max(), y_f.min(), y_f.max()],
                interpolation="nearest",
                aspect="equal",
                cmap=RF_CMAP,
                vmin=RF_VMIN,
                vmax=RF_VMAX,
            )

        elif mode == "raw+fwhm":
            im = ax.imshow(
                (maps[idx] - np.min(maps[idx])) / (np.max(maps[idx]) - np.min(maps[idx]) + 1e-6),
                origin="lower",
                extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
                interpolation="nearest",
                aspect="equal",
                cmap=RF_CMAP,
                vmin=RF_VMIN,
                vmax=RF_VMAX,
            )

            cx, cy = centers[idx]
            radius = 0.5 * fwhm[idx]
            circ = Circle((cx, cy), radius=radius, fill=False, linewidth=1.2, color="w")
            ax.add_patch(circ)
            ax.plot(cx, cy, marker="+", markersize=7, color="w", mew=1.2)        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        _add_rf_meridians(ax)
        
        # if bottom row or left column, show axis labels; otherwise hide them for clarity

        _format_rf_axis(
            ax,
            x_coords,
            y_coords,
            show_xlabel=(k // 5 == 4),  # bottom row only
            show_ylabel=(k % 5 == 0),   # left column only
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.3, fraction=0.025, pad=0.02)
    _style_rf_colorbar(cbar, fontsize=8)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)


def plot_rf_size_vs_eccentricity_session(rf_dict, date, out_path):
    rf_size = np.asarray(rf_dict["rf_size"], float)
    ecc = np.asarray(rf_dict["eccentricity"], float)
    fit_success = np.asarray(rf_dict["rf_fit_success"], bool)

    mask = fit_success
    rf_size = rf_size[mask]
    ecc = ecc[mask]

    fig, ax = plt.subplots(1, 1, figsize=(2.8, 2.8))
    ax.scatter(ecc, rf_size, color="k", s=18, alpha=0.9, linewidths=0.0)

    lo = 0.0
    hi = max(5.0, np.nanmax([np.nanmax(ecc) if ecc.size else 0, np.nanmax(rf_size) if rf_size.size else 0]) * 1.1)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Eccentricity (°)", fontsize=9)
    ax.set_ylabel("RF size (°)", fontsize=9)
    ax.tick_params(axis="both", which="both", direction="out", length=4, width=1.0, labelsize=8)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)

    return ecc, rf_size


def p_to_ast(p):
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return "p < 0.01"
    elif p < 0.05:
        return "p < 0.05"
    else:
        return "n.s."
    
def plot_rf_size_vs_eccentricity_all(ecc, rf_size, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(2.8, 2.8))
    ax.scatter(ecc, rf_size, color="k", s=6, alpha=0.9, linewidths=0.0)

    # -----------------------------
    # identity line
    # -----------------------------
    lo = 0.0
    hi = max(
        5.0,
        np.nanmax([
            np.nanmax(ecc) if ecc.size else 0,
            np.nanmax(rf_size) if rf_size.size else 0
        ]) * 1.1
    )
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)

    # -----------------------------
    # linear fit (y = m x + b) + p-value
    # -----------------------------
    mask = np.isfinite(ecc) & np.isfinite(rf_size)
    if np.sum(mask) >= 2:
        x = ecc[mask]
        y = rf_size[mask]

        res = linregress(x, y)
        m = res.slope
        b = res.intercept
        r = res.rvalue
        r2 = r ** 2
        p = res.pvalue  # two-sided test for slope != 0

        # plot fit
        x_fit = np.linspace(lo, hi, 200)
        y_fit = m * x_fit + b
        ax.plot(x_fit, y_fit, color="red", linewidth=1.5)

        # compact p formatting
        p_str = p_to_ast(p)

        # annotation (top-left)
        txt = f"m = {m:.2f}\n$R^2$ = {r2:.2f}\nb = {b:.2f}\n{p_str}"
        ax.text(
            0.03, 0.97, txt,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=7
        )

    # -----------------------------
    # formatting
    # -----------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Eccentricity (°)", fontsize=9)
    ax.set_ylabel("RF size (°)", fontsize=9)
    ax.tick_params(axis="both", which="both", direction="out", length=4, width=1.0, labelsize=8)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)

def _smooth_over_depth(values, depths, bin_size, step, ymin, ymax):
    window_starts = np.arange(ymin, ymax - bin_size + 1, step)
    centers = window_starts + 0.5 * bin_size
    means = np.full(window_starts.shape, np.nan, dtype=float)
    for i, start in enumerate(window_starts):
        m = (depths >= start) & (depths < start + bin_size)
        if np.any(m):
            means[i] = np.nanmean(values[m])
    return means, centers

def plot_rf_trajectory_panel(
    ax,
    date,
    rf_dict,
    y_depth,
    date_to_session,
    dates_by_monkey,
    smooth_depth=200,
    step_size=20.0,
    y_min=0.0,
    y_max=2000.0,
    cmap_name="magma",
):
    monkey_markers = {"A": "^", "H": "o", "T": "x"}

    monkey_limits = {
        "A": {"xlim": (-3, 15), "ylim": (-12, 6)},
        "H": {"xlim": (-20, 0), "ylim": (-5, 15)},
        "T": {"xlim": (-3, 15), "ylim": (-15, 15)},
    }

    monkey_ticks = {
        "A": {"xticks": [-3, 3, 6, 9, 12, 15], "yticks": [-12, -9, -6, -3, 3, 6]},
        "H": {"xticks": [-20, -15, -10, -5], "yticks": [-5, 5, 10, 15]},
        "T": {"xticks": [-3, 3, 6, 9, 12, 15], "yticks": [-15, -10, -5, 5, 10, 15]},
    }

    mk = None
    for m, dlist in dates_by_monkey.items():
        if date in dlist:
            mk = m
            break

    fit_success = np.asarray(rf_dict["rf_fit_success"], bool)
    x_rf = np.asarray(rf_dict["rf_int_x"], float)[fit_success]
    y_rf = np.asarray(rf_dict["rf_int_y"], float)[fit_success]
    y_depth = np.asarray(y_depth, float)[fit_success]

    x_s, depth_s = _smooth_over_depth(x_rf, y_depth, smooth_depth, step_size, y_min, y_max)
    y_s, _ = _smooth_over_depth(y_rf, y_depth, smooth_depth, step_size, y_min, y_max)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_linewidth(1.3)
    ax.spines["left"].set_linewidth(1.3)

    ax.axhline(0, linewidth=0.9, alpha=0.35, zorder=1)
    ax.axvline(0, linewidth=0.9, alpha=0.35, zorder=1)

    ax.set_xlim(*monkey_limits[mk]["xlim"])
    ax.set_ylim(*monkey_limits[mk]["ylim"])
    ax.set_xticks(monkey_ticks[mk]["xticks"])
    ax.set_yticks(monkey_ticks[mk]["yticks"])

    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="both", which="both", direction="inout", length=4, width=0.9, labelsize=7)

    ax.set_title(date_to_session.get(date, str(date)), fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([f"{t:g}°" for t in ax.get_xticks()])
    ax.set_yticklabels([f"{t:g}°" for t in ax.get_yticks()])

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.text(x1 + 0.1 * (x1 - x0), 0, "HM", ha="right", va="center", fontsize=8, clip_on=False)
    ax.text(0, y0 - 0.07 * (y1 - y0), "VM", ha="center", va="bottom", fontsize=8, clip_on=False)

    cmap = plt.get_cmap(cmap_name)
    depth_norm = Normalize(vmin=0.0, vmax=2000.0)

    ax.scatter(
        x_s,
        y_s,
        c=depth_s,
        cmap=cmap,
        norm=depth_norm,
        s=18,
        marker=monkey_markers.get(mk, "o"),
        linewidths=0.8,
        alpha=0.95,
        zorder=4,
    )

    ax.set_facecolor("white")
    return ax

def make_dates_by_monkey(config):
    dataset_params = config["dataset_params"]
    monkeys = ["A", "H", "T"]
    out = {}
    for mk in monkeys:
        key = f"dates_{mk}"
        out[mk] = list(dataset_params.get(key, []))
    return out

def main():
    parser = argparse.ArgumentParser(description="Arguments for plotting receptive field figures")
    parser.add_argument("--config", default="configs/default_config.yaml", type=str)
    parser.add_argument("--split", default="all", type=str)
    args = parser.parse_args()

    seed = 0
    rng = np.random.default_rng(seed)

    config_path = args.config
    config = load_config(config_path)
    data_path = config["dataset_params"]["data_path"]

    ordered_dates, date_to_session = sort_dates_and_make_session_titles(config, args.split)
    rf_metrics = load_rfs()

    out_root = f"figures/rfs/{args.split}"
    os.makedirs(out_root, exist_ok=True)

    overall_success = 0
    overall_total = 0

    dates_by_monkey = make_dates_by_monkey(config)

    ecss = []
    rf_sizes = []
    for date in ordered_dates:
        if date not in rf_metrics["main"]:
            continue

        rf_dict = rf_metrics["main"][date]
        fit_success = np.asarray(rf_dict["rf_fit_success"], bool)
        n_total = len(fit_success)
        n_success = int(np.sum(fit_success))
        overall_success += n_success
        overall_total += n_total

        print(f"{date}: RF fit success = {n_success}/{n_total} ({100.0 * n_success / max(n_total, 1):.1f}%)")

        path = f"{out_root}/{date}"
        os.makedirs(path, exist_ok=True)

        rf_map = np.asarray(rf_dict["rf_map"], float)
        rf_x_coords = np.asarray(rf_dict["rf_x_coords"], float)
        rf_y_coords = np.asarray(rf_dict["rf_y_coords"], float)
        rf_fit = np.asarray(rf_dict["rf_fit"], float)
        rf_size = np.asarray(rf_dict["rf_size"], float)
        rf_int_x = np.asarray(rf_dict["rf_int_x"], float)
        rf_int_y = np.asarray(rf_dict["rf_int_y"], float)

        # load depth
        data = load_session(data_path, date)
        neu_inc = load_inc()["rf_plaid"][str(date)]
        y_depth = np.asarray(data["y"])[neu_inc]

        # choose 25 random neurons
        success_ids = np.where(fit_success)[0]
        n_show = min(25, len(success_ids))
        chosen = rng.choice(success_ids, size=n_show, replace=False)
        chosen = chosen[np.argsort(y_depth[chosen])] # sort for nicer plotting order
        if n_show < 25:
            # pad if needed
            chosen = np.concatenate([chosen, rng.choice(success_ids, size=25 - n_show, replace=True)])

        centers = np.c_[rf_int_x, rf_int_y]

        # 1) raw RF maps
        _plot_rf_panel_grid(
            rf_map,
            rf_x_coords,
            rf_y_coords,
            chosen,
            out_path=f"{path}/rf_raw_maps_{date}.png",
            mode="raw",
            title=f"{date}: raw receptive fields",
        )

        # 2) interpolated RF maps
        _plot_rf_panel_grid(
            rf_map,
            rf_x_coords,
            rf_y_coords,
            chosen,
            out_path=f"{path}/rf_interpolated_maps_{date}.png",
            mode="interp",
            title=f"{date}: interpolated receptive fields",
        )

        # 3) gaussian fit RF maps
        _plot_rf_panel_grid(
            rf_map,
            rf_x_coords,
            rf_y_coords,
            chosen,
            out_path=f"{path}/rf_gaussian_fit_maps_{date}.png",
            mode="fit",
            fit_params=rf_fit,
            title=f"{date}: gaussian fit receptive fields",
        )

        # 4) raw RF maps with FWHM overlay
        _plot_rf_panel_grid(
            rf_map,
            rf_x_coords,
            rf_y_coords,
            chosen,
            out_path=f"{path}/rf_raw_maps_fwhm_overlay_{date}.png",
            mode="raw+fwhm",
            centers=centers,
            fwhm=rf_size,
            title=f"{date}: raw receptive fields + FWHM",
        )

        # 5) size vs eccentricity
        ecc, rfs = plot_rf_size_vs_eccentricity_session(
            rf_dict,
            date,
            out_path=f"{path}/rf_size_vs_eccentricity_{date}.png"
        )
        ecss.append(ecc)
        rf_sizes.append(rfs)

        # 6) trajectory of RF centers over depth


        fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.0))

        plot_rf_trajectory_panel(
            ax,
            date,
            rf_dict,
            y_depth=y_depth,
            date_to_session=date_to_session,
            dates_by_monkey=dates_by_monkey,
            cmap_name="magma",
        )

        depth_norm = Normalize(vmin=0.0, vmax=2000.0)
        sm = mpl.cm.ScalarMappable(norm=depth_norm, cmap=plt.get_cmap("magma"))
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.1, shrink=0.7)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(
            "Vertical position (µm)",
            weight="regular",
            rotation=-90,
            labelpad=14,
            fontsize=8,
        )

        fig.savefig(f"{path}/rf_center_trajectory_{date}.png", bbox_inches="tight", pad_inches=0.02, dpi=400)
        plt.close(fig)

    if overall_total > 0:
        print(
            f"Overall RF fit success = {overall_success}/{overall_total} "
            f"({100.0 * overall_success / overall_total:.1f}%)"
        )

    eccs = np.concatenate(ecss) if len(ecss) > 0 else np.array([])
    rf_szs = np.concatenate(rf_sizes) if len(rf_sizes) > 0 else np.array([])
    plot_rf_size_vs_eccentricity_all(eccs, rf_szs, out_path=f"{out_root}/rf_size_vs_eccentricity_all.png")


if __name__ == "__main__":
    main()