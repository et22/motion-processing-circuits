import argparse
import datetime
from logging import config
import os
import numpy as np

from utils import load_config, load_session, save_inc, get_pref_task_name_id, save_date_task_map, save_rf_task_map
from scipy.stats import mannwhitneyu

from analysis.video_metrics import compute_video_metrics

def get_neuron_inclusion_index_video(date, data, config, rate_thr, neuron_y_thr, exclude_as, rel_crit):
    start_time_fr = config['analysis_params']['start_time_fr']
    prestim_time = config['analysis_params']['prestim_time']
    video_dur = config['stimulus_params']['video']

    stim_start_idx = prestim_time + start_time_fr
    stim_end_idx = prestim_time + video_dur

    td = data["video_task"][0]
    spikes = td["spikes"]
    video_ids = td["video_id"]
    is_train_probe = td["is_train_probe"]

    vid_dict = compute_video_metrics(
        spikes, video_ids, is_train_probe,
        stim_start_idx, stim_end_idx, crop=None,
    )

    reliab = vid_dict["reliability"]
    max_test = vid_dict["max_test_rate"]
    n_neurons = len(max_test)
    include = np.zeros(n_neurons, dtype=bool)

    for neu_idx in range(len(max_test)):

        rate_gate = max_test[neu_idx] >= rate_thr
        reliab_gate = reliab[neu_idx] >= rel_crit 

        # not AS
        if exclude_as:
            wave_gate = False
        else:
            wave_gate = True

        ttp = 1000 * (data['waveform_times'][neu_idx, np.argmax(data['waveforms'][neu_idx, :])] - data['waveform_times'][neu_idx, np.argmin(data['waveforms'][neu_idx, :])])
        if ttp >= 0 and (np.abs(np.min(data['waveforms'][neu_idx, :])) > np.abs(np.max(data['waveforms'][neu_idx, :]))): 
            wave_gate = True

        y_inc = data["y"][neu_idx] <= neuron_y_thr

        include[neu_idx] = reliab_gate & rate_gate & wave_gate & y_inc
    
    return include

def get_pref_locs(coords, rates):
    uq_locs, loc_ids = np.unique(coords, axis=0, return_inverse=True) # get coordinates that stimuli were presented at 
    pref_locs = np.zeros(rates.shape[1], dtype=int)
    for i in range(rates.shape[1]):
        means = [np.mean(rates[loc_ids == j, i]) for j in range(uq_locs.shape[0])] # iterate over each location and compute mean response
        pref_locs[i] = int(np.argmax(means)) # argmax over locations

    return pref_locs, loc_ids, uq_locs

def get_neuron_inclusion_index(date, data, config, task_name="plaid_task", task_id=0, alpha=0.05, rate_thr=5.0, dsi_thr=0.25, neuron_y_thr=2000, exclude_as=True):
    if task_name == "plaid_task":
        stim_dur = int(data[task_name][task_id]['probe_duration']) 
        print("assumed duration: ",   int(config["stimulus_params"]["plaid"]))
        print("actual duration: ", stim_dur)
    elif task_name == "rf_task":
        stim_dur = int(data[task_name][task_id]['probe_duration']) 
        #print("assumed duration: ",   int(config["stimulus_params"]["probe"]))
        #print("actual duration: ", stim_dur)

    tun_offset = int(config["analysis_params"]["start_time_fr"])
    prestim_time = int(config["analysis_params"]["prestim_time"])
    stim_start_idx = prestim_time    
    stim_end_idx = prestim_time + stim_dur

    task_index =  data[task_name][task_id]['task_index']
    spikes_all = data[task_name][task_id]["spikes"]

    for i in range(len(data["baseline_task"])):
        if data["baseline_task"][i]['task_index'] == task_index:
            baseline_id = i 
            break

    spikes_baseline = data["baseline_task"][baseline_id]["spikes"]
    directions = data[task_name][task_id]["direction"]
    coords = data[task_name][task_id]["coords"]

    if task_name == "plaid_task":
        is_plaid = data["plaid_task"][task_id]["plaid"]
        n_plaids = 1
    else: 
        is_plaid = np.zeros(directions.shape, dtype=int)
        n_plaids = 1

    n_trials, T, n_neurons = spikes_all.shape
    all_dirs = np.sort(np.unique(directions))

    s0 = stim_start_idx + tun_offset
    s1 = stim_end_idx

    rates = np.mean(spikes_all[:, s0:s1, :], axis=1)  # (n_trials, n_neurons)
    pref_locs, loc_ids, uq_locs = get_pref_locs(coords, rates)

    alpha_corr = alpha # don't bonferonni correct for number of directions (more liberal screen for inclusion), if we wanted to we'd use / (n_plaids*len(all_dirs))

    cond_masks = []
    is_dir = []
    for p in range(n_plaids):
        for d in all_dirs:
            cond_masks.append((is_plaid == p) & (directions == d))
            is_dir.append(p == 0)
    is_dir = np.array(is_dir, dtype=bool)
    include = np.zeros(n_neurons, dtype=bool)

    for neu_idx in range(n_neurons):
        spikes = spikes_all[:, :, neu_idx].astype(float)

        baseline_rates = np.nanmean(spikes_baseline[:, -prestim_time:, neu_idx], axis=1) * 1000.0 # use 100 ms before stimulus to define baseline, use baseline of the same task
        stim_rates = np.nanmean(spikes[:, s0:s1], axis=1) * 1000.0

        rate_gate = False # baseline_mean > rate_thr
        sig_gate = False
        
        mask_pref_loc = (loc_ids == pref_locs[neu_idx])
        sr = np.zeros(len(all_dirs))
        for idx, m in enumerate(cond_masks):
            stim_c = stim_rates[m & mask_pref_loc]
            if is_dir[idx]:
                sr[idx] = np.nanmean(stim_c)
        
        assert len(all_dirs) == 12

        prefcond = np.nanargmax(sr).astype(int)

        m = cond_masks[prefcond]
        stim_c = stim_rates[m & mask_pref_loc]
        if stim_c.size >= 2 and baseline_rates.size >= 2:
            _, p = mannwhitneyu(stim_c, baseline_rates, alternative='greater', nan_policy='omit')
            if p < alpha_corr:
                sig_gate = True
            if np.nanmean(stim_c) > rate_thr: 
                rate_gate = True

        anticond = (prefcond + len(all_dirs)//2) % len(all_dirs) # + 6 % 12
        preffr = sr[prefcond]
        antifr = sr[anticond]
        den = preffr + antifr
        dsi = np.nan if den <= 0 else (preffr - antifr) / den

        dsi_gate = dsi >= dsi_thr

        # not AS
        if exclude_as:
            wave_gate = False
        else:
            wave_gate = True

        ttp = 1000 * (data['waveform_times'][neu_idx, np.argmax(data['waveforms'][neu_idx, :])] - data['waveform_times'][neu_idx, np.argmin(data['waveforms'][neu_idx, :])])
        if ttp >= 0 and (np.abs(np.min(data['waveforms'][neu_idx, :])) > np.abs(np.max(data['waveforms'][neu_idx, :]))): 
            wave_gate = True

        y_inc = data["y"][neu_idx] <= neuron_y_thr

        include[neu_idx] = sig_gate & rate_gate & wave_gate & dsi_gate & y_inc
    
    return include

def main():
    # parse args
    parser = argparse.ArgumentParser(description='Arguments for selecting which neurons to include/exclude.')
    parser.add_argument('--config', default='configs/default_config.yaml', type=str)
    args = parser.parse_args()

    # load config and grab relevant variables
    config_path = args.config
    config = load_config(config_path)
    data_path = config['dataset_params']['data_path']

    # extract necessary parameters
    dates = config['dataset_params']['dates']

    counts = []

    inc = dict()
    inc["rf_plaid"] = dict()

    date_task_map = {}
    rf_task_map = {}
    for date in dates:
        data = load_session(data_path, date)
        task_name, task_id = get_pref_task_name_id(data)
        assert task_name == "plaid_task"
        date_task_map[date] = {}
         
        date_task_map[date]['task_name'] = task_name
        date_task_map[date]['task_id'] = task_id
        date_task_map[date]['n_trials'] = data["plaid_task"][task_id]['spikes'].shape[0]
        date_task_map[date]['task_mod'] = data['plaid_task'][task_id]['task_mod']
        date_task_map[date]['probe_size'] = data['plaid_task'][task_id]['probe_size']
        date_task_map[date]['probe_class'] = data['plaid_task'][task_id]['probe_class']
        uq_locs, loc_ids = np.unique(data['plaid_task'][task_id]['coords'], axis=0, return_inverse=True) # get coordinates that stimuli were presented at 
        date_task_map[date]['num_locs'] = len(uq_locs)

        neu_inc = get_neuron_inclusion_index(date, data, config, task_name=task_name, task_id=task_id, alpha=config['neuron_params']['alpha_thr'], rate_thr=config['neuron_params']['rate_thr'],dsi_thr=config['neuron_params']['dsi_thr'], neuron_y_thr=config['neuron_params']['neuron_y_thr'], exclude_as=config['neuron_params']['exclude_as'])

        print(f"session: {date}, {np.sum(neu_inc)}/{len(neu_inc)}={np.around(np.sum(neu_inc)/len(neu_inc), 2)}")
        counts.append(np.sum(neu_inc))
        inc["rf_plaid"][date] = neu_inc

        # make task map for the rf_task as well, since we'll need it for the rf metrics analysis. 
        task_name, task_id = get_pref_task_name_id(data, task_name = 'rf_task')
        assert task_name == "rf_task"
        rf_task_map[date] = {}
        rf_task_map[date]['task_name'] = task_name
        rf_task_map[date]['task_id'] = task_id
        if len(data["rf_task"]) > 0:
            rf_task_map[date]['n_trials'] = data["rf_task"][task_id]['spikes'].shape[0]
            rf_task_map[date]['probe_size'] = data['rf_task'][task_id]['probe_size']
            uq_locs, loc_ids = np.unique(data['rf_task'][task_id]['coords'], axis=0, return_inverse=True) # get coordinates that stimuli were presented at 
            rf_task_map[date]['num_locs'] = len(uq_locs)
        else:
            rf_task_map[date]['task_id'] = -1

    print(f"plaid task only counts...")
    print(f"total number of neurons: {np.sum(counts)}")
    print(f"min number of neurons per session: {np.min(counts)}")
    print(f"max number of neurons per session: {np.max(counts)}")
    print(f"mean number of neurons per session: {np.mean(counts)}")

    inc["video"] = dict()
    counts = []
    # specific video inclusion criteria section
    dates = config['dataset_params']['dates_video'] # only use dates_video subset for this
    for date in dates:
        data = load_session(data_path, date)

        neu_inc = get_neuron_inclusion_index_video(date, data, config, rate_thr=config['neuron_params']['rate_thr'], neuron_y_thr=config['neuron_params']['neuron_y_thr'], exclude_as=config['neuron_params']['exclude_as'], rel_crit=config['neuron_params']['video_rel_thr'])
        inc["video"][date] = neu_inc
        counts.append(np.sum(neu_inc))

    print(f"video task only counts...")
    print(f"total number of neurons: {np.sum(counts)}")
    print(f"min number of neurons per session: {np.min(counts)}")
    print(f"max number of neurons per session: {np.max(counts)}")
    print(f"mean number of neurons per session: {np.mean(counts)}")

    print(f"all task counts...")
    counts = []
    all_dates = list(set(config['dataset_params']['dates']) | set(config['dataset_params']['dates_video']))
    for date in all_dates: 
        vid_inc = None
        pl_inc = None
        if date in inc["video"].keys(): 
            vid_inc = inc["video"][date]
        if date in inc["rf_plaid"].keys():
            pl_inc = inc["rf_plaid"][date]
        
        if pl_inc is not None and vid_inc is not None:
            counts.append(np.sum(pl_inc | vid_inc))
        elif pl_inc is not None: 
            counts.append(np.sum(pl_inc))
        elif vid_inc is not None: 
            counts.append(np.sum(vid_inc))

    print(f"total number of neurons: {np.sum(counts)}")
    print(f"min number of neurons per session: {np.min(counts)}")
    print(f"max number of neurons per session: {np.max(counts)}")
    print(f"mean number of neurons per session: {np.mean(counts)}")

    os.makedirs("figures", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    save_date_task_map(date_task_map)
    save_inc(inc)
    save_rf_task_map(rf_task_map)

if __name__ == "__main__":
    main()
