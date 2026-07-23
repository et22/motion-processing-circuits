import numpy as np
import argparse

from tqdm import tqdm
from utils import load_config, load_session, load_inc, save_pat, load_date_task_map

def get_pref_locs(coords, rates):
    uq_locs, loc_ids = np.unique(coords, axis=0, return_inverse=True) # get coordinates that stimuli were presented at 
    pref_locs = np.zeros(rates.shape[1], dtype=int)
    for i in range(rates.shape[1]):
        means = [np.mean(rates[loc_ids == j, i]) for j in range(uq_locs.shape[0])] # iterate over each location and compute mean response
        pref_locs[i] = int(np.argmax(means)) # argmax over locations

    return pref_locs, loc_ids, uq_locs

def compute_pattern_metrics(rates, coords, directions, is_plaid, is_grating, patcomp_thresh):
    n_neurons = rates.shape[1]

    pref_locs, loc_ids, _ = get_pref_locs(coords, rates)

    uq_dirs = np.sort(np.unique(directions))

    grating_tuning = np.full((n_neurons, uq_dirs.size), np.nan, dtype=float)
    grating_sem = np.full((n_neurons, uq_dirs.size), np.nan, dtype=float)

    plaid_tuning = np.full((n_neurons, uq_dirs.size), np.nan, dtype=float)
    plaid_sem = np.full((n_neurons, uq_dirs.size), np.nan, dtype=float)

    pi = np.full(n_neurons, np.nan, dtype=float)
    rc = np.full(n_neurons, np.nan, dtype=float)
    rp = np.full(n_neurons, np.nan, dtype=float)
    zc = np.full(n_neurons, np.nan, dtype=float)
    zp = np.full(n_neurons, np.nan, dtype=float)
    is_cds = np.full(n_neurons, False, dtype=bool)
    is_pds = np.full(n_neurons, False, dtype=bool)
    is_unc = np.full(n_neurons, False, dtype=bool)
    pi_cls = np.zeros(n_neurons, dtype=int)


    for i in range(n_neurons):
        mask_loc = (loc_ids == pref_locs[i])
        g_mask = is_grating & mask_loc
        p_mask = is_plaid & mask_loc
        
        d_g = {}
        s_g = {}
        for d in uq_dirs:
            d_g[d] = np.nanmean(rates[g_mask & (directions == d), i])
            s_g[d] = np.nanstd(rates[g_mask & (directions == d), i], ddof=1)/np.sqrt(np.sum(~np.isnan(rates[g_mask & (directions == d), i])))
        
        d_p = {}
        s_p = {}
        for d in uq_dirs:
            d_p[d] = np.nanmean(rates[p_mask & (directions == d), i])
            s_p[d] = np.nanstd(rates[p_mask & (directions == d), i], ddof=1)/np.sqrt(np.sum(~np.isnan(rates[p_mask & (directions == d), i])))

        measured, comp_pred, patt_pred = np.full(uq_dirs.size, np.nan, dtype=float), np.full(uq_dirs.size, np.nan, dtype=float), np.full(uq_dirs.size, np.nan, dtype=float)
        measured_sem, patt_pred_sem = np.full(uq_dirs.size, np.nan, dtype=float), np.full(uq_dirs.size, np.nan, dtype=float)

        for j, d in enumerate(uq_dirs):
            d = int(d)
            th1, th2 = int((d - 60) % 360), int((d + 60) % 360)
            measured[j] = d_p[d]
            measured_sem[j] = s_p[d]
            comp_pred[j] = d_g[th1] + d_g[th2]
            
            patt_pred[j] = d_g[d]
            patt_pred_sem[j] = s_g[d]

        # following smith et al. 2005 for definition of partial corr., verified this matches pingouin implementation too!
        r_p = np.clip(np.corrcoef(measured, patt_pred)[0, 1], a_min=-0.9999, a_max=0.9999)
        r_c = np.clip(np.corrcoef(measured, comp_pred)[0, 1], a_min=-0.9999, a_max=0.9999)
        r_pc = np.clip(np.corrcoef(comp_pred, patt_pred)[0, 1], a_min=-0.9999, a_max=0.9999)

        R_p = (r_p - r_c * r_pc) / np.sqrt((1-r_c ** 2) * (1-r_pc ** 2))
        R_c = (r_c - r_p * r_pc) / np.sqrt((1-r_p ** 2) * (1-r_pc ** 2))

        # fisher z transform, divided by standard deviation 
        df = uq_dirs.size - 3 
        assert df == 9

        Z_p = 0.5 * np.log((1 + R_p) / (1 - R_p)) * np.sqrt(df)
        Z_c = 0.5 * np.log((1 + R_c) / (1 - R_c)) * np.sqrt(df)
        
        # pattern index
        pi[i] = Z_p - Z_c # pattern is positive, component is negative

        # store other correlations
        rp[i] = R_p
        rc[i] = R_c
        zp[i] = Z_p
        zc[i] = Z_c

        # also store tuning curves
        grating_tuning[i, :] = patt_pred
        plaid_tuning[i, :] = measured

        # also store tuning curve SEM
        grating_sem[i, :] = patt_pred_sem
        plaid_sem[i, :] = measured_sem

        is_cds[i] = (pi[i] <= -patcomp_thresh) & (zc[i] >= patcomp_thresh)
        is_pds[i] = (pi[i] >= patcomp_thresh) & (zp[i] >= patcomp_thresh)
        is_unc[i] = np.logical_not(is_cds[i] | is_pds[i])

        if is_cds[i]:
            pi_cls[i] = -1
        elif is_pds[i]: 
            pi_cls[i] = 1 
        else: 
            pi_cls[i] = 0

    return pi, grating_tuning, plaid_tuning, grating_sem, plaid_sem, rp, rc, zp, zc, is_cds, is_pds, is_unc, pi_cls

def main():
    # parse args
    parser = argparse.ArgumentParser(description='Arguments for plotting tuning')
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

    patcomp_thresh = config['analysis_params']['patcomp_split']

    # specify timing
    tun_offset = start_time_fr   
    stim_start_idx = prestim_time + tun_offset

    pattern_metrics = dict()
    pattern_metrics["main"] = dict()
    pattern_metrics["all"] = dict()

    num_sessions = 0

    print(f"Computing pattern metrics...")
    date_task_map = load_date_task_map()
    for date in tqdm(dates):
        plaid_dur = int(config["stimulus_params"]["plaid"])

        print(date)
        # load data
        data = load_session(data_path, date)
        task_name, task_id = date_task_map[date]['task_name'], date_task_map[date]['task_id']
        stim_end_idx = prestim_time + int(data[task_name][task_id]['probe_duration']) #plaid_dur

        if task_name == "plaid_task":
            inc = load_inc()["rf_plaid"][date] # only using included neurons

            num_sessions += 1
            pattern_metrics["all"][date] = []

            # for fixed time interval after stimulus onset 
            for tid in range(len(data[task_name])):
                rates = np.nanmean(data[task_name][tid]['spikes'][:, stim_start_idx:stim_end_idx, inc], axis=1)

                coords = data[task_name][tid]['coords']
                directions = data[task_name][tid]['direction']
                is_plaid = data[task_name][tid]['plaid'] == 1
                is_grating = data[task_name][tid]['plaid'] == 0
                
                # baseline rate fetching
                task_index = data[task_name][tid]['task_index']
                for i in range(len(data["baseline_task"])):
                    if data["baseline_task"][i]['task_index'] == task_index:
                        baseline_id = i 
                        break
                spikes_baseline = data["baseline_task"][baseline_id]["spikes"]
                baseline_rates = np.nanmean(spikes_baseline[:, -prestim_time:, inc], axis=(0,1)) # use 100 ms before stimulus to define baseline, use baseline of the same task

                # main pattern index and tuning curve calculation
                pi, grating_tuning, plaid_tuning, grating_sem, plaid_sem, rp, rc, zp, zc, is_cds, is_pds, is_unc, pi_cls = compute_pattern_metrics(rates, coords, directions, is_plaid, is_grating, patcomp_thresh)

                # * 1000 converts appropriate variables to hz
                pat_dict = {"pi": pi, "grating_tuning": grating_tuning * 1000, "plaid_tuning": plaid_tuning * 1000, "grating_sem": grating_sem * 1000, "plaid_sem": plaid_sem * 1000, "rp": rp, "rc": rc, "zp": zp, "zc": zc, "probe_size": data[task_name][tid]["probe_size"], "probe_class": data[task_name][tid]["probe_class"], "task_mod": data[task_name][tid]["task_mod"], "baseline_rates": baseline_rates * 1000, "is_cds": is_cds, "is_pds": is_pds, "is_unc": is_unc, "pi_cls": pi_cls}
                
                if tid == task_id:
                    pattern_metrics["main"][date] = pat_dict

                pattern_metrics["all"][date].append(pat_dict)
    
    save_pat(pattern_metrics)

    print(f"Done! Computed pattern metrics for {num_sessions} sessions.")

if __name__ == "__main__":
    main()
