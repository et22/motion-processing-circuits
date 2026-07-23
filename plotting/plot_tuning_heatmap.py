import matplotlib.colors as mcolors

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.interpolate import CubicSpline
from utils import load_config, load_session, load_inc, load_pat, sort_dates_and_make_session_titles, set_plot_defaults, save_fractures

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
        
def plot_tuning_heatmap(neuron_y, tun, gr_tun, ymin = 0, ymax=2000, binsize=40, draw_fractures=True, thr_frac_deg=150.0, thr_frac_sim=45, overlay_frac=None):
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

    fracture_y = _detect_fractures_from_heatmap(heatmap, y_vals, ymin=ymin, ymax=ymax, thr_deg=thr_frac_deg, thr_sim=thr_frac_sim)

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
    ax.set_ylabel("Vertical position (\u03bcm)", fontweight='regular', fontsize=10)

    ax.set_xticks(np.linspace(0, heatmap.shape[1], 5), np.linspace(-90, 270, 5).astype(int))

    cax = fig.add_axes([0.82, 0.14, 0.035, 0.3])  # [left, bottom, width, height] in figure coords
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Normalized Firing Rate", fontsize=10, labelpad=-34, fontweight="regular", rotation=270)
    cb.ax.yaxis.set_label_position("left")
    ax.text(0.98, 1.02, "  ", transform=ax.transAxes, ha="right", va="top", fontsize=10, fontweight="regular", clip_on=False)

    if overlay_frac is not None:
        for y in overlay_frac:
            ax.hlines(y, xmin=0, xmax=heatmap.shape[1], colors="k", linestyles="--", linewidth=1.0, alpha=0.9, zorder=10)

    if draw_fractures and (len(fracture_y) > 0):
        for y in fracture_y:
            ax.hlines(y, xmin=0, xmax=heatmap.shape[1], colors="k", linestyles="--", linewidth=1.0, alpha=0.9, zorder=10)

    return fracture_y, max_fr, y_vals

def _circ_abs_diff_deg(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)

def _fill_invalid_rows_nearest(H, valid_row):
    H = np.asarray(H, float).copy()
    valid_row = np.asarray(valid_row, bool)

    if valid_row.all():
        return H

    idx = np.arange(H.shape[0])
    valid_idx = idx[valid_row]
    invalid_idx = idx[~valid_row]

    nearest_pos = np.abs(invalid_idx[:, None] - valid_idx[None, :]).argmin(axis=1)
    nearest_valid_idx = valid_idx[nearest_pos]

    H[invalid_idx] = H[nearest_valid_idx]
    return H

def _circ_median_deg(x):
    x = np.asarray(x, float) % 360.0
    d = _circ_abs_diff_deg(x[:, None], x[None, :])
    return x[np.argmin(d.sum(axis=1))]

def _detect_fractures_from_heatmap(heatmap,y_vals,ymin=0,ymax=None,thr_deg=160.0,thr_sim=45,hmap_bin=40.0):
    H = np.asarray(heatmap, float)
    n_y, n_ang = H.shape
    angles = np.linspace(0.0, 360.0, n_ang, endpoint=False)
    valid_row = np.any(H > 0, axis=1)

    H = _fill_invalid_rows_nearest(H, valid_row)
    valid_row = np.ones_like(valid_row, dtype=bool)
    pref_idx = np.argmax(H, axis=1)
    pref_deg = angles[pref_idx]

    d = np.full(n_y - 1, np.nan, dtype=float)
    pair_valid = valid_row[:-1] & valid_row[1:]
    d[pair_valid] = _circ_abs_diff_deg(pref_deg[:-1][pair_valid], pref_deg[1:][pair_valid])

    d2 = np.full(n_y - 2, np.nan, dtype=float)
    pair2_valid = valid_row[:-2] & valid_row[2:]
    d2[pair2_valid] = _circ_abs_diff_deg(pref_deg[:-2][pair2_valid], pref_deg[2:][pair2_valid])

    jump = np.zeros(n_y - 1, dtype=bool)
    jump[pair_valid] = ((d[pair_valid]) >= float(thr_deg)) # what counts as a fracture or 'jump' 

    # keep big jumps that are stable on both sides --- with some noise tolerance
    # our criteria here is specifically --- if on the side of the fracture, bin -1 and bin -2 are very similar or bin -1 and -3 (corresponds to 40 and 80 microns), we label it as 'stable'
    # and the fracture needs to be 'stable' on both sides to be labeled a fracture --- otherwise it could just be noise 
    small_thr = float(thr_sim)
    keep = np.zeros_like(jump, dtype=bool)
    for i in np.where(jump)[0]:
        ok_prev = (
            (i - 1 >= 0 and (d[i - 1]) <= small_thr) or
            (i - 2 >= 0 and (d[i - 2]) <= small_thr) or
            (i - 2 >= 0 and (d2[i - 2]) <= small_thr)
        )
        ok_next = (
            (i + 1 < d.size  and (d[i + 1]) <= small_thr) or
            (i + 2 < d.size  and (d[i + 2]) <= small_thr) or
            (i + 1 < d2.size and (d2[i + 1]) <= small_thr)
        )

        # 3-bin circular-median jump across the boundary
        prev_med = _circ_median_deg(pref_deg[max(0, i-2):i+1])
        next_med = _circ_median_deg(pref_deg[i+1:min(n_y, i+4)])
        ok_med_jump = _circ_abs_diff_deg(prev_med, next_med) >= float(thr_deg)  # tune

        if ok_prev and ok_next and ok_med_jump:
            keep[i] = True

    jump = keep

    fracture_y = []

    # this loop is mapping from the fracture index to the actual y-value 
    for i in np.where(jump)[0]:
        yb = float(y_vals[i+1])
        # cannot reasonably estimate fractures near edges of the probe 
        if ymin + 60 <= yb <= ymax - 60:
            fracture_y.append(yb)

    # we return a list of y-values where each is a fracture location 
    return fracture_y

def main():
    # parse args
    parser = argparse.ArgumentParser(description='Arguments for plotting tuning')
    parser.add_argument('--config', default='configs/default_config.yaml', type=str)
    parser.add_argument("--split", default="all", type=str)

    args = parser.parse_args()

    # load config and grab relevant variables
    config_path = args.config
    config = load_config(config_path)
    fs = config['fractures']
    data_path = config['dataset_params']['data_path']
    ymax = config['neuron_params']['neuron_y_thr']

    # extract necessary parameters
    ordered_dates, date_to_session = sort_dates_and_make_session_titles(config, args.split)
    sess_cnt = 0
    fractures_out = {}
    for date in ordered_dates:
        if date in load_pat()['main'].keys():
            outdir = f"figures/general/{args.split}/{date}"
            os.makedirs(outdir, exist_ok=True)

            sess_cnt += 1

            # load data
            data = load_session(data_path, date)
            neu_inc = load_inc()["rf_plaid"][str(date)]

            # get pattern index
            pat = load_pat()
            suffix = ""

            # get preferred grating, with interpolation
            gr_tun = pat['main'][date]['grating_tuning']
            n_dirs = gr_tun.shape[1]
            x = np.arange(n_dirs + 1) * (360.0 / float(n_dirs))
            cs = CubicSpline(x, np.concatenate((gr_tun, gr_tun[:, 0][:, None]), axis=1), axis=1, bc_type="periodic")
            angles_interp = np.linspace(0.0, 360.0, 500)
            gr_tun = cs(angles_interp)

            pl_tun = pat['main'][date]['plaid_tuning']
            n_dirs = pl_tun.shape[1]
            x = np.arange(n_dirs + 1) * (360.0 / float(n_dirs))
            cs = CubicSpline(x, np.concatenate((pl_tun, pl_tun[:, 0][:, None]), axis=1), axis=1, bc_type="periodic")
            angles_interp = np.linspace(0.0, 360.0, 500)
            pl_tun = cs(angles_interp)

            neuron_y = data['y'][neu_inc]
            fr_gr, max_fr, y_vals = plot_tuning_heatmap(neuron_y, gr_tun, gr_tun, ymax=ymax, draw_fractures=True, thr_frac_deg=fs['thr_frac_deg'], thr_frac_sim=fs['thr_frac_sim'])

            plt.savefig(f"figures/general/{args.split}/{date}/grating_tuning_heatmap_{date}_{suffix}.pdf", bbox_inches="tight", pad_inches=0.02)
            
            #plt.scatter(max_fr, y_vals)
            #plt.savefig(f"figures/general/{args.split}/{date}/grating_tuning_fr_{date}_{suffix}.pdf", bbox_inches="tight", pad_inches=0.02)


            fr_pl, max_fr, y_vals = plot_tuning_heatmap(neuron_y, pl_tun, gr_tun, ymax=ymax, draw_fractures=False, overlay_frac=fr_gr)
            plt.savefig(f"figures/general/{args.split}/{date}/plaid_tuning_heatmap_{date}_{suffix}.pdf", bbox_inches="tight", pad_inches=0.02)

            if (args.split == "all" and date in config['dataset_params']['dates_fracture']) or (args.split == "main" and date in config['dataset_params']['dates_fracture_main']) or (args.split == "supplement" and date in config['dataset_params']['dates_fracture_supplement']):
                fractures_out[date] = {
                    "grating": fr_gr,
                }
    save_fractures(fractures_out, args.split)

if __name__ == "__main__":
    main()