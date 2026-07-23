import matplotlib.colors as mcolors

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.interpolate import CubicSpline

from utils import load_config, load_video, set_plot_defaults

pattern_color, component_color, unclass_color = set_plot_defaults()

def get_custom_reds_cmap():
    base_cmap = mpl.colormaps.get_cmap("Reds")
    custom_colors = [
        (0.0, base_cmap(0.0)),
        (0.25, base_cmap(0.0)),
        (0.5, base_cmap(0.2)),
        (0.7, base_cmap(0.5)),
        (0.85, base_cmap(0.8)),
        (1.0, base_cmap(0.97)),
    ]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("customReds", custom_colors)
    return custom_cmap


def _apply_y_mask(neuron_y, vals, ymin=0, ymax=2000):
    y_mask = (neuron_y >= ymin) & (neuron_y <= ymax)
    vals = vals[y_mask, :]
    neuron_y = neuron_y[y_mask]
    return neuron_y, vals


def plot_tuning_heatmap(neuron_y, tun, gr_tun, ymin=0, ymax=2000, binsize=40):
    _, gr_tun = _apply_y_mask(neuron_y, gr_tun, ymin, ymax)
    neuron_y, tun = _apply_y_mask(neuron_y, tun, ymin, ymax)

    fig, ax = plt.subplots(1, 1, figsize=(2.85 * 5/8, 8.0 * 5/8))
    fig.subplots_adjust(left=0.18, right=0.76, bottom=0.06, top=0.96, wspace=0.25)  # reserve right margin for cbar

    y_vals = np.arange(ymin, ymax+binsize, binsize)
    tuning = np.full((y_vals.shape[0]-1, tun.shape[1]), np.nan, dtype=float)

    for i in range(len(y_vals)-1):
        y = y_vals[i]
        mask = (neuron_y >= y) & (neuron_y < y_vals[i+1])
        if np.sum(mask) == 0:
            pass
        if np.sum(mask) == 1:
            tuning[i, :] = tun[mask, :]
        elif np.sum(mask) > 1:
            # pick neuron with max fr if multiple per bin
            tun_opt = tun[mask, :]
            fr = np.argmax(np.nanmax(gr_tun[mask, :], axis=1))

            tuning[i, :] = tun_opt[fr, :]

    heatmap = np.roll(tuning, int(tuning.shape[1] // 4), axis=1)
    max_fr = np.max(heatmap, axis=1)
    heatmap = heatmap - np.nanmin(heatmap, 1, keepdims=True)
    heatmap = heatmap / np.nanmax(heatmap, 1, keepdims=True)
    heatmap[np.isnan(heatmap)] = 0
    cmap = get_custom_reds_cmap()

    y0 = float(y_vals[0])
    y1 = float(y_vals[-1])
    dy = binsize
    im = ax.imshow(
        heatmap,
        cmap=cmap,
        origin='lower',
        extent=[0, heatmap.shape[1], 0, 2000],
        interpolation='nearest'
    )
    ax.set_ylim([ymin, ymax])
    ax.set_aspect('auto')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Direction (°)', fontweight='regular', fontsize=10)
    ax.set_ylabel("Vertical position (μm)", fontweight='regular', fontsize=10)

    ax.set_xticks(np.linspace(0, heatmap.shape[1], 5), np.linspace(-90, 270, 5).astype(int))

    cax = fig.add_axes([0.82, 0.14, 0.035, 0.3])  # [left, bottom, width, height] in figure coords
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Normalized Firing Rate", fontsize=10, labelpad=-34, fontweight="regular", rotation=270)
    cb.ax.yaxis.set_label_position("left")
    ax.text(0.98, 1.02, "  ", transform=ax.transAxes, ha="right", va="top", fontsize=10, fontweight="regular", clip_on=False)

    return max_fr, y_vals


def main():
    # parse args
    parser = argparse.ArgumentParser(description='Arguments for plotting video optic-flow tuning over space')
    parser.add_argument('--config', default='configs/default_config.yaml', type=str)
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    dates = config['dataset_params']['dates_video']
    ymax = config['neuron_params']['neuron_y_thr']

    video_metrics = load_video()

    sess_cnt = 0
    for date in dates:
        vid = video_metrics[date]
        tun = np.asarray(vid['video_dir_tuning'], dtype=float)  
        neuron_y = vid['y']

        n_dirs = tun.shape[1]
        x = np.arange(n_dirs + 1) * (360.0 / float(n_dirs))
        cs = CubicSpline(x, np.concatenate((tun, tun[:, 0][:, None]), axis=1), axis=1, bc_type="periodic")
        angles_interp = np.linspace(0.0, 360.0, 500)
        tun = cs(angles_interp)

        outdir = f"figures/video/{date}"
        os.makedirs(outdir, exist_ok=True)

        plot_tuning_heatmap(neuron_y, tun, tun, ymax=ymax)

        plt.savefig(f"{outdir}/video_dir_tuning_heatmap_{date}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(plt.gcf())
        sess_cnt += 1

    print(f"Done! Plotted video optic-flow tuning over space for {sess_cnt} sessions.")

if __name__ == "__main__":
    main()