import argparse
import numpy as np
from tqdm import trange, tqdm
from scipy.stats import poisson, norm
from scipy.ndimage import convolve1d

from utils import load_config, load_session, save_ccgs, load_inc, load_date_task_map

def get_significant_ccgs_excit(ccg_true, ccg_slow, ccg_rate, ccg_config):
    # ccg true has shape n_timepoints x n_neurons x n_neurons
    # ccg slow has shape n_timepoints x n_neurons x n_neurons
    # 
    # see also English et al. 2017, Stark and Abeles 2009

    # unpack config
    ac_range = ccg_config["ac_range"]
    sig_range = ccg_config["sig_range"]

    p_fast_crit = ccg_config["p_fast_crit"]
    p_caus_crit = ccg_config["p_caus_crit"]

    min_coinc = ccg_config["min_coinc"]

    # get indices
    zero_index = int(ccg_true.shape[0] // 2)

    # compute p_fast
    ccg_true_sig_range = ccg_true[zero_index+sig_range[0]:zero_index+sig_range[1]+1, :, :]
    ccg_slow_sig_range = ccg_slow[zero_index+sig_range[0]:zero_index+sig_range[1]+1, :, :]

    p_fast = 1 - poisson.cdf(k=ccg_true_sig_range-1, mu=ccg_slow_sig_range) - 0.5 * poisson.pmf(k=ccg_true_sig_range, mu=ccg_slow_sig_range)
    
    # compute p_causal
    max_anticausal = np.max(ccg_true[zero_index+ac_range[0]:zero_index+ac_range[1]+1, :, :], axis=0)
    max_anti_causal = np.repeat(max_anticausal[None, :, :], ccg_true_sig_range.shape[0], axis=0)

    p_causal = 1 - poisson.cdf(k=ccg_true_sig_range-1, mu=max_anti_causal) - 0.5 * poisson.pmf(k=ccg_true_sig_range, mu=max_anti_causal)

    # 1. criteria from english et al.
    sig_indices = np.any(np.logical_and(p_fast < p_fast_crit, p_causal < p_caus_crit), axis=0)

    # 2. overall peak must be within sig_range
    sig_indices = sig_indices & np.logical_and(np.argmax(ccg_rate, axis=0) >= zero_index+sig_range[0], np.argmax(ccg_rate, axis=0) <= zero_index+sig_range[1])

    # 3. peak > min_coinc criteria
    sig_indices = sig_indices & (np.max(ccg_true, axis=0) > min_coinc)

    # 4. stdev criteria - compute stdev :20 and -20: of ccg_rate, peak must be > 2 stdev above mean of these flanks
    # note could include to target the few phasic pairs that sneak in, but doesn't affect the results much
    """ 
    flank_exclude = ccg_config.get("flank_exclude", 20)
    print(ccg_rate.shape)
    stdev_mult = ccg_config.get("stdev_mult", 2)

    flank_ccg_rate = np.concatenate([ccg_rate[-flank_exclude:, :, :],ccg_rate[:flank_exclude:, :, :],],axis=0,)

    flank_mean = np.mean(flank_ccg_rate, axis=0)
    flank_std = np.std(flank_ccg_rate, axis=0)

    peak_rate = np.max(ccg_rate[zero_index+sig_range[0]:zero_index+sig_range[1]+1, :, :], axis=0,)

    sig_indices = sig_indices & (peak_rate > (flank_mean + stdev_mult * flank_std))
    """

    return sig_indices


def get_significant_ccgs_inhib(ccg_true, ccg_slow, ccg_rate, ccg_config):
    # ccg true has shape n_timepoints x n_neurons x n_neurons
    # ccg slow has shape n_timepoints x n_neurons x n_neurons
    # 
    # see also English et al. 2017, Stark and Abeles 2009

    # unpack config
    ac_range = ccg_config["ac_range"]
    sig_range = ccg_config["sig_range"]

    p_fast_crit = ccg_config["p_fast_crit"]
    p_caus_crit = ccg_config["p_caus_crit"]

    # get indices
    zero_index = int(ccg_true.shape[0] // 2)

    # compute p_fast
    ccg_true_sig_range = ccg_true[zero_index+sig_range[0]:zero_index+sig_range[1]+1, :, :]
    ccg_slow_sig_range = ccg_slow[zero_index+sig_range[0]:zero_index+sig_range[1]+1, :, :]

    # prob of obtaining observed or lower 
    p_fast = poisson.cdf(k=ccg_true_sig_range, mu=ccg_slow_sig_range) - 0.5 * poisson.pmf(k=ccg_true_sig_range, mu=ccg_slow_sig_range)

    # compute p_causal
    min_anticausal = np.min(ccg_true[zero_index+ac_range[0]:zero_index+ac_range[1]+1, :, :], axis=0)
    min_anti_causal = np.repeat(min_anticausal[None, :, :], ccg_true_sig_range.shape[0], axis=0)

    # prob of obtaining observed or lower 
    p_causal = poisson.cdf(k=ccg_true_sig_range, mu=min_anti_causal) - 0.5 * poisson.pmf(k=ccg_true_sig_range, mu=min_anti_causal)

    # 1. criteria from english et al. 
    sig_indices = np.any(np.logical_and(p_fast < p_fast_crit, p_causal < p_caus_crit), axis=0)

    # 2. additional heuristic criteria to exclude noise - overall trough must be within sig_range
    # apply heuristic
    sig_indices = sig_indices & np.logical_and(np.argmin(ccg_rate, axis=0) >= zero_index+sig_range[0], np.argmin(ccg_rate, axis=0) <= zero_index+sig_range[1])

    return sig_indices

def convolve_ccgs(ccg_true, ccg_config, duration):
    # ccg_true has shape 2*maxlag+1 x n_neurons x n_neurons
    # convolves ccgs with a partially hollow gaussian kernel to generate 'slow' prediction
    # see English et al. 2017, Stark and Abeles 2009

    # gaussian with a standard deviation of 10 ms and hollow fraction of 60% 
    hollow_frac = ccg_config['hollow_frac']
    kernel_stdev = ccg_config['kernel_stdev']
    kernel_clip = ccg_config['kernel_clip']

    # construct gaussian with hollow window kernel, hollowed only at 0
    indices = np.arange(-kernel_clip, kernel_clip+1)
    kernel = norm.pdf(indices, loc=0, scale=kernel_stdev)
    kernel[indices == 0] = kernel[indices == 0] * (1 - hollow_frac)
    kernel = kernel / np.sum(kernel)

    # opportunity-correct the ccg (duration = T, maxlag = ccg_config['maxlag'])
    maxlag = ccg_config['maxlag']
    lags = np.arange(-maxlag, maxlag + 1)
    opp = (duration - np.abs(lags)).astype(float) # T - |lag|
    ccg_rate = ccg_true / opp[:, None, None]

    # convolve
    ccg_slow = convolve1d(ccg_rate, kernel, axis=0, mode='reflect') # reflect is okay because edges are unused!

    # bring back to counts space
    ccg_slow = ccg_slow * opp[:, None, None]

    return ccg_slow, ccg_rate

def compute_ccgs_all(spikes, ccg_config, duration):
    # unpack ccg_config
    maxlag = ccg_config['maxlag']

    ccg = dict()
    ccg_true = compute_ccgs(spikes, maxlag=maxlag)
    ccg_slow, ccg_rate = convolve_ccgs(ccg_true, ccg_config, duration)

    ccg["ccg_true"] = ccg_true
    ccg["ccg_slow"] = ccg_slow
    ccg["ccg_rate"] = ccg_rate

    ccg["n_pre"] = spikes.sum(axis=(0, 1))  # (n_neurons,)

    return ccg

def compute_ccgs(spikes, maxlag=30):
    n_units = spikes.shape[2]
    
    output = compute_ccgs_sum(spikes, maxlag=maxlag)

    # zero out diagonal, i.e., autocorrelograms
    for i in range(n_units):
        output[:, i, i] = 0

    return output

def compute_ccgs_sum(spikes, maxlag=30):
    """
    ccg computes all cross-correlograms between a group of neurons

    :param spikes:   3d array of spike raster with dimensions n_trials by n_times by n_units
    :param maxlag:   maximum lag of cross-correlation in bins, default is 30

    :return output: numpy array of size (2*maxlag+1, n, n) where element [t,i,j] corresponds to the number of events
                    of cell j firing at timelag t relative to cell i
    """
    n_trials, n_times, n_units = spikes.shape
    output = np.zeros((2 * maxlag + 1, n_units, n_units), dtype=np.float32)

    for i in trange(n_trials):
        times, ids = np.where(spikes[i, :, :])
        duration = n_times
        output = ccg(times, ids, duration, output=output, maxlag=maxlag)

    return output

# helper function to efficiently compute CCGs for population using 'sum' method
def ccg(
    times: np.ndarray,
    ids: np.ndarray,
    duration: float,
    output: np.ndarray,
    maxlag: int = 30,
):
    """
    ccg computes all cross-correlograms between a group of neurons (lightning fast!)

    :param times:    1d array of all spike times from all units in millisecond integer times
    :param ids:      1d array of the cell id corresponding to each spike time in times
    :param duration: duration of recording in milliseconds
    :param output:   ccg output array that coincidence spikes are added to
    :param maxlag:   maximum lag of cross-correlation in bins, default is 30

    :return output: numpy array of size (2*maxlag+1, n, n) where element [t,i,j] corresponds to the number of events
                    of cell j firing at timelag t relative to cell i

    referenced https://github.com/petersenpeter/CellExplorer/blob/master/calc_CellMetrics/mex/CCGHeart.c
    """
    # sort times and groups
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    ids = ids[sort_idx]

    # construct d, a map from time bin to the index in times array of first element >= bin
    d = np.zeros(duration + 1, dtype=np.int32)

    idx = 0
    for bin in range(duration + 1):
        while idx < times.size and bin > times[idx]:
            idx += 1
        d[bin] = idx

    # iterate through spike times and compute CCG
    for idx1, (time1, id1) in enumerate(zip(times, ids)):
        # if not the last spike
        if idx1 < times.size - 1:
            start = idx1 + 1
            end = d[min(time1 + maxlag + 1, d.size - 1)]
            if start != end:
                bins = times[start:end] - time1
                id2s = ids[start:end]
                output[bins + maxlag, id1, id2s] += 1
                output[maxlag - bins, id2s, id1] += 1

    return output

# get spike transmission probability 
def compute_synaptic_efficacy(ccgs, ccg_config):
    maxlag = ccg_config["maxlag"]
    sig_range = ccg_config["sig_range"]

    ccg_excess = ccgs["ccg_true"] - ccgs["ccg_slow"]  # baseline-subtracted counts

    # sum excess counts over the causal peak window
    peak_counts = ccg_excess[
        maxlag + sig_range[0]:maxlag + sig_range[1] + 1, :, :
    ].sum(axis=0)  # shape: (n_units, n_units)

    n_pre = ccgs["n_pre"]
    efficacy = np.where(n_pre[:, None] > 0, peak_counts / n_pre[:, None], np.nan)

    return efficacy

# get spike transmission probability 
def compute_synaptic_efficacy_post(ccgs, ccg_config):
    maxlag = ccg_config["maxlag"]
    sig_range = ccg_config["sig_range"]

    ccg_excess = ccgs["ccg_true"] - ccgs["ccg_slow"]  # baseline-subtracted counts

    # sum excess counts over the causal peak window
    peak_counts = ccg_excess[
        maxlag + sig_range[0]:maxlag + sig_range[1] + 1, :, :
    ].sum(axis=0)  # shape: (n_units, n_units)

    n_pre = ccgs["n_pre"]
    efficacy = np.where(n_pre[None, :] > 0, peak_counts / n_pre[None, :], np.nan)

    return efficacy    

def compute_synaptic_efficacy_sqrt(ccgs, ccg_config):
    maxlag = ccg_config["maxlag"]
    sig_range = ccg_config["sig_range"]

    ccg_excess = ccgs["ccg_true"] - ccgs["ccg_slow"] 

    # sum excess counts over causal window
    peak_counts = ccg_excess[
        maxlag + sig_range[0]:maxlag + sig_range[1] + 1, :, :
    ].sum(axis=0)  # (n_units, n_units)

    n_pre = ccgs["n_pre"]            
    n_post = ccgs["n_pre"]             

    denom = np.sqrt(n_pre[:, None] * n_post[None, :])

    efficacy_sqrt = np.where(denom > 0, peak_counts / denom, np.nan)

    return efficacy_sqrt

# gets peaks and lags of ccgs
def get_peak_trough_lag(ccg, ccg_config):
    maxlag = ccg_config["maxlag"]

    ccg_corr = (ccg["ccg_true"] - ccg["ccg_slow"])/np.sqrt(ccg["ccg_slow"])
    
    peak = np.max(ccg_corr, axis=0)
    trough = np.min(ccg_corr, axis=0)
    peak_lag = np.abs(np.argmax(ccg_corr, axis=0) - maxlag)
    trough_lag = np.abs(np.argmin(ccg_corr, axis=0) - maxlag)

    return peak, trough, peak_lag, trough_lag

def add_ccg_properties(ccgs, ccg_config):
    sig_indices_excit = get_significant_ccgs_excit(ccgs['ccg_true'], ccgs['ccg_slow'],  ccgs['ccg_rate'], ccg_config)
    sig_indices_inhib = get_significant_ccgs_inhib(ccgs['ccg_true'], ccgs['ccg_slow'], ccgs['ccg_rate'], ccg_config)
    peak, trough, peak_lag, trough_lag = get_peak_trough_lag(ccgs, ccg_config)

    ccgs['sig_indices_excit'] = sig_indices_excit
    ccgs['sig_indices_inhib'] = sig_indices_inhib
    ccgs['peak_lag'] = peak_lag
    ccgs['trough_lag'] = trough_lag
    ccgs['peak'] = peak
    ccgs['trough'] = trough
    ccgs['efficacy'] = compute_synaptic_efficacy(ccgs, ccg_config)
    ccgs['efficacy_sqrt'] = compute_synaptic_efficacy_sqrt(ccgs, ccg_config)
    ccgs['efficacy_post'] = compute_synaptic_efficacy_post(ccgs, ccg_config)
    
    return ccgs

def main():
    # parse args
    parser = argparse.ArgumentParser(description='Arguments for plotting tuning')
    parser.add_argument('--config', default='configs/default_config.yaml', type=str)
    parser.add_argument('--start_time', default=None, type=int)
    args = parser.parse_args()

    # load config and grab relevant variables
    config_path = args.config
    config = load_config(config_path)
    data_path = config['dataset_params']['data_path']

    ccg_config = config['ccg_params']
    trial_type = ccg_config['trial_type']

    # extract necessary parameters
    dates = config['dataset_params']['dates']

    prestim_time = config['analysis_params']['prestim_time']
    if args.start_time == None:
        ccg_start = ccg_config['start_time']
    else:
        ccg_start = args.start_time

    # specify timing
    stim_start_idx = prestim_time + ccg_start

    ccg_metrics = dict()

    num_sessions = 0

    print(f"Computing ccgs metrics for plaid tasks...")
    date_task_map = load_date_task_map()

    for date in tqdm(dates):
        # load data
        data = load_session(data_path, date)
        task_name, task_id = date_task_map[date]['task_name'], date_task_map[date]['task_id']

        stim_end_idx = prestim_time + int(data[task_name][task_id]['probe_duration']) #ccg_start + ccg_dur

        if task_name == "plaid_task":
            inc = load_inc()["rf_plaid"][date] # only using included neurons, much more efficient

            num_sessions += 1

            if trial_type == "all":
                trial_inc = (data[task_name][task_id]['plaid'] == 1) | (data[task_name][task_id]['plaid'] == 0)
            elif trial_type == "plaid_only":
                trial_inc = data[task_name][task_id]['plaid'] == 1
            elif trial_type == "grating_only":
                trial_inc = data[task_name][task_id]['plaid'] == 0
            else:
                raise ValueError(f"Invalid trial type: {trial_type}")

            spikes = data[task_name][task_id]['spikes'][:, stim_start_idx:stim_end_idx, inc]
            spikes = spikes[trial_inc, :, :]
            ccg_metrics[date] = compute_ccgs_all(spikes, ccg_config, duration=spikes.shape[1])
            ccg_metrics[date] = add_ccg_properties(ccg_metrics[date], ccg_config)

    save_ccgs(ccg_metrics, ccg_start if ccg_start != 100 else "")
    print(f"Done! Computed ccgs for {num_sessions} sessions.")

if __name__ == "__main__":
    main()