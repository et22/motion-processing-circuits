import os
import argparse
import numpy as np
import matplotlib as mpl

from utils import load_config, load_session, load_pat, set_plot_defaults, load_inc, sort_dates_and_make_session_titles

from scipy.interpolate import CubicSpline
from scipy.stats import mannwhitneyu

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

PATTERN_COLOR, COMPONENT_COLOR, UNCLASS_COLOR = set_plot_defaults()

def get_class_color_and_name(pat_main, neuron_idx):
    if pat_main["is_pds"][neuron_idx]:
        return PATTERN_COLOR, "pattern"
    elif pat_main["is_cds"][neuron_idx]:
        return COMPONENT_COLOR, "component"
    else:
        return UNCLASS_COLOR, "unclassified"
    
def main():
    # parse args
    parser = argparse.ArgumentParser(description='Arguments for plotting tuning')
    parser.add_argument('--config', default='configs/default_config.yaml', type=str)
    parser.add_argument("--split", default="all", type=str)

    args = parser.parse_args()

    # load config and grab relevant variables
    config_path = args.config
    config = load_config(config_path)
    data_path = config['dataset_params']['data_path']

    # extract necessary parameters
    ordered_dates, date_to_session = sort_dates_and_make_session_titles(config, args.split)
    sess_cnt = 0

    config = load_config(args.config)
    data_path = config["dataset_params"]["data_path"]

    comp_bandwidths = []
    patt_bandwidths = []
    
    for date in ordered_dates:
        if date in load_pat()['main'].keys():
            sess_cnt += 1

            pat = load_pat()
            pat_main = pat["main"][date]

            data = load_session(data_path, date)
            inc = load_inc()["rf_plaid"][date]

            # included-neuron depths, aligned to pat_main neuron order

            n_neurons = pat_main["pi"].shape[0]

            # get preferred grating, with interpolation
            gr_tun = pat['main'][date]['grating_tuning']
            n_dirs = gr_tun.shape[1]
            x = np.arange(n_dirs + 1) * (360.0 / float(n_dirs))
            cs = CubicSpline(x, np.concatenate((gr_tun, gr_tun[:, 0][:, None]), axis=1), axis=1, bc_type="periodic")
            angles_interp = np.linspace(0.0, 360.0, 500, endpoint=False)
            gr_tun = cs(angles_interp)

            for i in range(n_neurons):
                cls_color, cls_name = get_class_color_and_name(pat_main, i)

                grating_tuning = gr_tun[i]
                grating_tuning = grating_tuning - np.min(grating_tuning)
                grating_tuning = grating_tuning / np.max(grating_tuning)
                max_pt = np.argmax(grating_tuning)

                up_cnt = max_pt
                down_cnt = max_pt
                for _ in range(len(grating_tuning)):
                    if grating_tuning[up_cnt] > 0.5:
                        up_cnt += 1
                        up_cnt = up_cnt % len(grating_tuning)
                    
                    if grating_tuning[down_cnt] > 0.5:
                        down_cnt -= 1
                        down_cnt = down_cnt % len(grating_tuning)
                    
                    if grating_tuning[down_cnt] < 0.5 and grating_tuning[up_cnt] < 0.5:
                        break
                
                if up_cnt < down_cnt:
                    bandwidth = angles_interp[up_cnt] + 360 - angles_interp[down_cnt]
                else:
                    bandwidth = angles_interp[up_cnt] - angles_interp[down_cnt]
                
                if pat_main["is_cds"][i]: 
                    comp_bandwidths.append(bandwidth)
                elif pat_main["is_pds"][i]:
                    patt_bandwidths.append(bandwidth)
    
    res = mannwhitneyu(comp_bandwidths, patt_bandwidths)
    print(f"mann whitney u p-value: {res.pvalue}")
    print(f"med comp: {np.nanmedian(comp_bandwidths)}")
    print(f"med patt: {np.nanmedian(patt_bandwidths)}")

if __name__ == "__main__":
    main()