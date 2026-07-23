import numpy as np
import argparse

from tqdm import tqdm
from utils import load_config, load_session, load_inc, save_rfs, load_rf_task_map

from scipy.optimize import curve_fit

def gaussian_2d(pos, amp, xo, yo, sigma, offset):
    x, y = pos
    return offset + amp * np.exp(-((x - xo) ** 2 + (y - yo) ** 2) / (2.0 * sigma ** 2))

def fit_gaussian_2d(data2d, x_coords, y_coords):
    X, Y = np.meshgrid(x_coords, y_coords)  # both [Ny, Nx]
    x = X.ravel() # flattening stuff!
    y = Y.ravel()
    z = data2d.ravel()

    # mask any nans and any infs in z if they are there
    if not np.all(np.isfinite(z)):
        m = np.isfinite(z)
        x, y, z = x[m], y[m], z[m]

    z0 = z

    # inital guess --- peak near argmax, sigma from coordinate ranges
    imax = int(np.nanargmax(z0))
    xo0, yo0 = x[imax], y[imax]
    amp0 = float(np.nanmax(z0))
    offset0 = float(np.nanpercentile(z0, 5))
    sigma0 = 2  # gentle default sigma guess in DVA units, based on estimate from RF maps

    p0 = (amp0, xo0, yo0, sigma0, offset0)

    # bounds --- 
    # -we use 2 instead of one in case we didn't sample the exact RF center with our grid
    # -assuming our rf center is in mapped region 
    # -very broad range of potentials sigmas
    # -broad range of potential offsets (strictly positive, see assumptions above)
    bounds = (
        (0.0, np.nanmin(x_coords), np.nanmin(y_coords), 1e-3, 0),
        (1.5*np.nanmax(z0), np.nanmax(x_coords), np.nanmax(y_coords), 50.0, np.nanmax(z0)), 
    )

    popt, _ = curve_fit(
        gaussian_2d,
        (x, y),
        z0,
        p0=p0,
        bounds=bounds,
        maxfev=200000,
    )
    return popt  # (amp, xo, yo, sigma, offset)

def get_pref_locs(coords, rates):
    uq_locs, loc_ids = np.unique(coords, axis=0, return_inverse=True) # get coordinates that stimuli were presented at 
    pref_locs = np.zeros(rates.shape[1], dtype=int)
    for i in range(rates.shape[1]):
        means = [np.mean(rates[loc_ids == j, i]) for j in range(uq_locs.shape[0])] # iterate over each location and compute mean response
        pref_locs[i] = int(np.argmax(means)) # argmax over locations

    return pref_locs, loc_ids, uq_locs

def compute_rf_metrics(rates, coords, directions, rf_fit_crit):
    coords = np.asarray(coords)
    rates = np.asarray(rates)
    directions = np.asarray(directions)

    n_trials, n_neurons = rates.shape

    pref_locs, loc_ids, uq_locs = get_pref_locs(coords, rates)

    pref_xy = uq_locs[pref_locs]
    rf_x = pref_xy[:, 0].astype(float)
    rf_y = pref_xy[:, 1].astype(float)

    eccentricity = np.full(n_neurons, np.nan, dtype=float)

    # grid coordinates
    rf_x_coords = np.unique(coords[:, 0]).astype(float)
    rf_y_coords = np.unique(coords[:, 1]).astype(float)

    Nx = rf_x_coords.size
    Ny = rf_y_coords.size

    x_to_ix = {x: i for i, x in enumerate(rf_x_coords)}
    y_to_iy = {y: i for i, y in enumerate(rf_y_coords)}

    # outputs
    rf_map = np.full((n_neurons, Ny, Nx), np.nan, dtype=float)

    rf_size = np.full(n_neurons, np.nan, dtype=float)
    rf_int_x = np.full(n_neurons, np.nan, dtype=float)
    rf_int_y = np.full(n_neurons, np.nan, dtype=float)
    rf_fit_success = np.zeros(n_neurons, dtype=bool)
    rf_fit = np.full((n_neurons, 5), np.nan, dtype=float)  # amp, xo, yo, sigma, offset
    rf_fit_explained_var = np.full(n_neurons, np.nan, dtype=float)

    loc_masks = []
    for loc in uq_locs:
        m = (coords[:, 0] == loc[0]) & (coords[:, 1] == loc[1])
        loc_masks.append(m)

    for i in range(n_neurons):
        for loc_idx, loc in enumerate(uq_locs):
            x, y = loc
            ix = x_to_ix[x]
            iy = y_to_iy[y]

            m_loc = loc_masks[loc_idx]
            if np.any(m_loc):
                rf_map[i, iy, ix] = np.nanmean(rates[m_loc, i])

        # fit 2D Gaussian to spatial RF map
        map2d = rf_map[i]
        if not np.any(np.isfinite(map2d)):
            continue

        try:
            amp, xo, yo, sigma, offset = fit_gaussian_2d(map2d, rf_x_coords, rf_y_coords)

            rf_fit[i] = np.array([amp, xo, yo, sigma, offset], dtype=float)
            rf_int_x[i] = xo
            rf_int_y[i] = yo
            rf_size[i] = 2.355 * sigma  # FWHM
            eccentricity[i] = np.sqrt(xo * xo + yo * yo)

            X, Y = np.meshgrid(rf_x_coords, rf_y_coords)
            z = map2d.ravel()
            x_flat = X.ravel()
            y_flat = Y.ravel()

            valid = np.isfinite(z)
            z = z[valid]
            x_flat = x_flat[valid]
            y_flat = y_flat[valid]

            z0 = z
            zhat = gaussian_2d((x_flat, y_flat), amp, xo, yo, sigma, offset)

            ss_res = np.nansum((z0 - zhat) ** 2)
            ss_tot = np.nansum((z0 - np.nanmean(z0)) ** 2)

            if ss_tot > 0:
                rf_fit_explained_var[i] = 1.0 - (ss_res / ss_tot)
            else:
                rf_fit_explained_var[i] = np.nan
            
            if rf_fit_explained_var[i] > rf_fit_crit:
                rf_fit_success[i] = True
            else:
                rf_fit_success[i] = False

        except Exception:
            pass

    return (
        rf_x,
        rf_y,
        eccentricity,
        rf_size,
        rf_int_x,
        rf_int_y,
        rf_x_coords,
        rf_y_coords,
        rf_fit_success,
        rf_map,
        rf_fit,
        rf_fit_explained_var,
    )

def main():
    # parse args
    parser = argparse.ArgumentParser(description='Arguments for rf metric calculation')
    parser.add_argument('--config', default='configs/default_config.yaml', type=str)
    args = parser.parse_args()

    # load config and grab relevant variables
    config_path = args.config
    config = load_config(config_path)
    data_path = config['dataset_params']['data_path']

    # extract necessary parameters
    dates = config['dataset_params']['dates']

    start_time_fr = config['analysis_params']['start_time_fr']
    prestim_time = config['analysis_params']['prestim_time']
    poststim_time = config['analysis_params']['poststim_time']
    probe_dur = config['stimulus_params']['probe']
    rf_fit_crit = config['analysis_params']['rf_fit_crit']

    # specify timing
    tun_offset = start_time_fr   
    stim_start_idx = prestim_time + tun_offset
    stim_end_idx = prestim_time + probe_dur

    rf_metrics = dict()
    rf_metrics["main"] = dict()
    rf_metrics["all"] = dict()

    num_sessions = 0

    print(f"Computing rf metrics...")
    rf_task_map = load_rf_task_map()
    for date in tqdm(dates):
        print(date)
        # load data
        data = load_session(data_path, date)
        task_name, task_id = rf_task_map[date]['task_name'], rf_task_map[date]['task_id']

        if task_name == "rf_task" and task_id != -1:
            inc = load_inc()["rf_plaid"][date] # only using included neurons

            num_sessions += 1
            rf_metrics["all"][date] = []

            # for fixed time interval after stimulus onset 
            for tid in range(len(data[task_name])):
                rates = np.nanmean(data[task_name][tid]['spikes'][:, stim_start_idx:stim_end_idx, inc], axis=1)

                coords = data[task_name][tid]['coords']
                directions = data[task_name][tid]['direction']
                
                # main pattern index and tuning curve calculation
                rf_x, rf_y, eccentricity, rf_size, rf_int_x, rf_int_y, rf_x_coords, rf_y_coords, rf_fit_success, rf_map, rf_fit, rf_fit_explained_var = compute_rf_metrics(rates, coords, directions, rf_fit_crit)

                rf_dict = {"rf_x": rf_x, "rf_y": rf_y, "eccentricity": eccentricity, "rf_size": rf_size, "rf_int_x": rf_int_x, "rf_int_y": rf_int_y, "rf_x_coords": rf_x_coords, "rf_y_coords": rf_y_coords, "rf_fit_success": rf_fit_success, "rf_map": rf_map, "rf_fit": rf_fit, "rf_fit_explained_var": rf_fit_explained_var}
                if tid == task_id:
                    rf_metrics["main"][date] = rf_dict

                rf_metrics["all"][date].append(rf_dict)
    
    save_rfs(rf_metrics)

    print(f"Done! Computed rf metrics for {num_sessions} sessions.")

if __name__ == "__main__":
    main()
