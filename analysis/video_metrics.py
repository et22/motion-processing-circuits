import numpy as np
import argparse
import os
import pickle

import cv2

from tqdm import tqdm
from utils import load_config, load_session, save_video, load_inc
from copy import deepcopy

SESSION_CROPS = {
    # A crop 1
    "012925": {'top': 115, 'left': 260, 'height': 150, 'width': 150},
    "013025": {'top': 115, 'left': 260, 'height': 150, 'width': 150},
    "013125": {'top': 115, 'left': 260, 'height': 150, 'width': 150},
    "020425": {'top': 115, 'left': 260, 'height': 150, 'width': 150},
    "021125": {'top': 115, 'left': 260, 'height': 150, 'width': 150},
    "121825": {'top': 115, 'left': 260, 'height': 150, 'width': 150},
    # H crop
    "042425": {'top': 0,   'left': 125, 'height': 150, 'width': 150},
    "042525": {'top': 0,   'left': 125, 'height': 150, 'width': 150},
    "121925": {'top': 0,   'left': 125, 'height': 150, 'width': 150},
    # A crop 2
    "070126": {'top': 90,  'left': 360, 'height': 150, 'width': 150},
}

OF_NAME = 'farneback'
VIDEO_DIR = "./data/videos"
CACHE_DIR = f"./data/videos/{OF_NAME}_cache"

def crop_frame(frame, crop):
    t, l, h, w = crop['top'], crop['left'], crop['height'], crop['width']
    return frame[t:t + h, l:l + w]


def compute_optic_flow_target(video_path, crop=None):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    if crop is not None:
        prev_gray = crop_frame(prev_gray, crop)

    flows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if crop is not None:
            gray = crop_frame(gray, crop)

        if OF_NAME == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(np.mean(flow, axis=(0, 1)))
        elif OF_NAME == 'lk':
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                        minDistance=7, blockSize=7)
            if pts is not None:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, pts, None,
                    winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                good = status.ravel() == 1
                if good.sum() > 0:
                    disp = (nxt[good] - pts[good]).reshape(-1, 2)   # per-point (dx, dy)
                    flows.append(np.mean(disp, axis=0))
                else:
                    flows.append(np.array([0.0, 0.0]))
            else:
                flows.append(np.array([0.0, 0.0]))

        prev_gray = gray
    cap.release()

    mean_flow = np.mean(flows, axis=0)
    return mean_flow, np.sqrt(np.sum(np.square(mean_flow)))


def compute_video_flow_tuning(spikes, video_ids, is_train_probe, stim_start_idx,
                              stim_end_idx, crop):
    rates = get_video_rates(spikes, stim_start_idx, stim_end_idx)  # (n_pres, n_neurons)
    all_video_ids = np.asarray(video_ids)

    os.makedirs(CACHE_DIR, exist_ok=True)
    crop_key = (crop['top'], crop['left'], crop['height'], crop['width'])
    cache_path = os.path.join(CACHE_DIR, f"optic_flow_cache_{crop_key}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            of_cache = pickle.load(f)
    else:
        of_cache = {}

    all_of, all_mag = [], []
    for stimulus in tqdm(all_video_ids, leave=False):
        if stimulus not in of_cache:
            vp = os.path.join(VIDEO_DIR, f"{stimulus}.mp4")
            if not os.path.exists(vp):
                return None
            of, mag = compute_optic_flow_target(vp, crop=crop)
            of_cache[stimulus] = (of, mag)
        else:
            of, mag = of_cache[stimulus]
        all_of.append(of)
        all_mag.append(mag)

    with open(cache_path, "wb") as f:
        pickle.dump(of_cache, f)

    all_of = np.array(all_of)     # (n_pres, 2)  [flow_x, flow_y]
    all_mag = np.array(all_mag)  

    all_mag[all_mag <= 0] = 1
    target = all_of / all_mag[:, None]

    angles_deg = (np.degrees(np.arctan2(target[:, 0], -target[:, 1])) + 360 + 90) % 360

    # bin into 12 direction bins
    bin_edges = np.linspace(-15, 345, 13) # to get centering appropriate 
    angles_deg_for_bin = deepcopy(angles_deg)
    angles_deg_for_bin[angles_deg_for_bin >= 345] = angles_deg_for_bin[angles_deg_for_bin >= 345] - 360
    video_bins = np.digitize(angles_deg_for_bin, bin_edges) - 1
    video_bins[video_bins == 12] = 11

    # map each video id -> its direction bin, then assign each presentation
    video_to_bin = dict(zip(all_video_ids, video_bins))
    trial_bins = np.array([video_to_bin[v] for v in all_video_ids])
    test_rates = rates  

    n_neurons = test_rates.shape[1]
    video_dir_tuning = np.full((n_neurons, 12), np.nan)
    for b in range(12):
        mask = trial_bins == b
        if np.any(mask):
            video_dir_tuning[:, b] = np.nanmean(test_rates[mask], axis=0)

    dir_bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    uq, uq_idx = np.unique(all_video_ids, return_index=True)
    return {
        "video_dir_tuning": video_dir_tuning,       # (n_neurons, 12), Hz
        "dir_bin_centers": dir_bin_centers,         # (12,)
        "of_direction": angles_deg[uq_idx],         # (n_unique_video,)
        "of_magnitude": all_mag[uq_idx],            # (n_unique_video,)
        "of_stims": uq,                             # (n_unique_video,)
    }


def get_video_rates(spikes, stim_start_idx, stim_end_idx):
    return np.nanmean(spikes[:, stim_start_idx:stim_end_idx, :], axis=1) * 1000.0


def build_test_response_matrix(rates, video_ids, is_train_probe):
    rates = np.asarray(rates, dtype=float)
    video_ids = np.asarray(video_ids)
    is_train_probe = np.asarray(is_train_probe, dtype=bool)
    n_neurons = rates.shape[1]

    test_stims = np.array(sorted(set(video_ids[~is_train_probe].tolist())))
    n_test = test_stims.size

    max_reps = 0
    for stim in test_stims:
        max_reps = max(max_reps, int(np.sum(video_ids == stim)))

    test_activity = np.full((n_test, n_neurons), np.nan, dtype=float)
    test_activity_reps = np.full((n_test, n_neurons, max_reps), np.nan, dtype=float)

    for si, stim in enumerate(test_stims):
        mask = (video_ids == stim)
        stim_rates = rates[mask, :]
        test_activity[si, :] = np.nanmean(stim_rates, axis=0)
        n_reps = stim_rates.shape[0]
        test_activity_reps[si, :, :n_reps] = stim_rates.T

    return test_stims, test_activity, test_activity_reps, n_test


def compute_response_reliability(test_activity_reps, n_bootstraps=50, seed=5):
    np.random.seed(seed)
    n_test, n_neurons, _ = test_activity_reps.shape
    reliability = np.full(n_neurons, np.nan, dtype=float)

    for neu_idx in range(n_neurons):
        row_subset = np.sum(np.isnan(test_activity_reps[:, neu_idx, :]), axis=0) != n_test
        curr_act = test_activity_reps[:, neu_idx, row_subset]  # (n_test, n_valid_reps)

        rs = np.full(n_bootstraps, np.nan, dtype=float)
        for bootstrap_idx in range(n_bootstraps):
            length = curr_act.shape[1]
            if length < 2:
                continue
            indices = np.arange(length)
            np.random.shuffle(indices)
            midpoint = length // 2
            act1 = np.nanmean(curr_act[:, indices[:midpoint]], axis=1)
            act2 = np.nanmean(curr_act[:, indices[midpoint:]], axis=1)
            nan_idx = np.isnan(act1) | np.isnan(act2)
            act1 = act1[~nan_idx]
            act2 = act2[~nan_idx]
            if act1.size >= 2 and np.std(act1) > 0 and np.std(act2) > 0:
                rs[bootstrap_idx] = np.corrcoef(act1, act2)[0, 1]

        reliability[neu_idx] = np.nanmean(rs)

    return reliability


def compute_video_metrics(spikes, video_ids, is_train_probe,
                          stim_start_idx, stim_end_idx, crop=None,
                          n_bootstraps=50, seed=5):
    rates = get_video_rates(spikes, stim_start_idx, stim_end_idx)  # (n_pres, n_neurons), Hz
    video_ids = np.asarray(video_ids)
    is_train_probe = np.asarray(is_train_probe, dtype=bool)

    test_stims, test_activity, test_activity_reps, n_test = build_test_response_matrix(
        rates, video_ids, is_train_probe
    )

    reliability = compute_response_reliability(test_activity_reps, n_bootstraps=n_bootstraps, seed=seed)
    video_tuning = test_activity.T

    mean_train_rate = np.nanmean(rates[is_train_probe, :], axis=0)
    mean_test_rate = np.nanmean(test_activity, axis=0)
    max_test_rate = np.nanmax(test_activity, axis=0)

    out = {
        "reliability": reliability,
        "video_tuning": video_tuning,
        "test_stims": test_stims,
        "mean_train_rate": mean_train_rate,
        "mean_test_rate": mean_test_rate,
        "max_test_rate": max_test_rate,
        "n_test": n_test,
    }

    if crop is not None:
        flow = compute_video_flow_tuning(
            spikes, video_ids, is_train_probe, stim_start_idx, stim_end_idx, crop
        )
        if flow is not None:
            out.update(flow)

    return out


def main():
    # parse args
    parser = argparse.ArgumentParser(description='Arguments for video metric calculation')
    parser.add_argument('--config', default='configs/default_config.yaml', type=str)
    args = parser.parse_args()

    # load config and grab relevant variables
    config_path = args.config
    config = load_config(config_path)
    data_path = config['dataset_params']['data_path']

    dates = config['dataset_params']['dates_video']

    start_time_fr = config['analysis_params']['start_time_fr']
    prestim_time = config['analysis_params']['prestim_time']
    video_dur = config['stimulus_params']['video']

    tun_offset = start_time_fr
    stim_start_idx = prestim_time + tun_offset
    stim_end_idx = prestim_time + video_dur

    video_metrics = dict()

    num_sessions = 0

    # sessions to sum reliable-neuron counts over
    print(f"Computing video metrics...")
    for date in tqdm(dates):
        print(date)
        data = load_session(data_path, date)
        neu_inc = load_inc()["video"][str(date)]

        num_sessions += 1

        td = data["video_task"][0]
        spikes = td["spikes"][:, :, neu_inc]
        x = data["x"][neu_inc]
        y = data["y"][neu_inc]
        video_ids = td["video_id"]
        is_train_probe = td["is_train_probe"]

        crop = SESSION_CROPS.get(date, None)
        if crop is None:
            print(f"  {date}: no crop defined, skipping optic flow for this session")

        vid_dict = compute_video_metrics(spikes, video_ids, is_train_probe, stim_start_idx, stim_end_idx, crop=crop)

        vid_dict['x'] = x
        vid_dict['y'] = y

        video_metrics[date] = (vid_dict)

    save_video(video_metrics)

if __name__ == "__main__":
    main()