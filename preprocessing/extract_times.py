import argparse
import numpy as np
import yaml
from datetime import datetime

from pathlib import Path

from preprocessing.readSGLX import readMeta, SampRate, makeMemMapRaw, ExtractDigital, ChannelCountsNI, GainCorrectNI
from scipy.signal import find_peaks

def main():
    parser = argparse.ArgumentParser(description="This script extracts sync times.")
    parser.add_argument("--date", type=str, help="Recording date")

    args = parser.parse_args()

    date = args.date
    print(f"===========extracting sync times for {date}...==============")

    # check whether photodiode signal is present
    config = yaml.safe_load(open("preprocessing/default_config.yaml"))
    cutoff_date = datetime.strptime(config["photodiode_cutoff"], "%m%d%y")
    input_date = datetime.strptime(date, "%m%d%y")
    is_after_cutoff = input_date > cutoff_date

    # dictionary for saving neural data
    neural_data = dict()

    # load spike times and cluster ids from kilosort
    results_dir = Path(f"./data/rawdata/neural_data/{date}_kilosort4/kilosort4/")

    st = np.load(results_dir / 'spike_times.npy')
    clu = np.load(results_dir / 'spike_clusters.npy')

    templates = np.load(results_dir / "templates.npy")
    ops = np.load(results_dir / "ops.npy", allow_pickle=True).item()
    probe = ops['probe']

    chan_best = (templates**2).sum(axis=1).argmax(axis=-1)
    x_coords, y_coords = probe['xc'][chan_best], probe['yc'][chan_best]
    waveforms = templates[np.arange(templates.shape[0]), :, chan_best]
    neural_data['waveforms'] = waveforms
        
    ap_bin_path = Path(f'./data/rawdata/sync_data/{date}_g0_t0.imec0.ap.bin')
    metaNeural = readMeta(ap_bin_path)
    sRateNeural = SampRate(metaNeural)
    st = st / sRateNeural
    max_t = np.max(st)

    neural_data["times"] = st
    neural_data["srate"] = sRateNeural
    neural_data["ids"] = clu
    neural_data["x_coords"] = x_coords
    neural_data["y_coords"] = y_coords

    print("loading digital data...")

    # load digital data
    tStart = 0 
    tEnd = np.floor(max_t)
    dw = 0    
    dLineList = [0]

    nidq_bin_path = Path(f'./data/rawdata/sync_data/{date}_g0_t0.nidq.bin')
    meta = readMeta(nidq_bin_path)
    print("digital data loaded...")

    sRate = SampRate(meta)
    firstSamp = int(sRate*tStart)

    rawData = makeMemMapRaw(nidq_bin_path, meta)
    nFileSamp_nidq = rawData.shape[1]
    lastSamp = min(int(sRate * tEnd), nFileSamp_nidq - 1)

    requested_last = int(sRate * tEnd)
    if lastSamp < requested_last:
        print(f"warning: nidq file shorter than expected ({nFileSamp_nidq} samples); "
            f"truncating requested range from {requested_last} to {lastSamp}")
        
    MN, MA, XA, DW = ChannelCountsNI(meta)
    nAnalog = MN + MA + XA
    print(f"sample rate: {sRate:.4f} Hz")
    print(f"channel counts -> MN:{MN} MA:{MA} XA:{XA} DW:{DW}")

    chanList = list(range(nAnalog))
    selectData = rawData[chanList, firstSamp:lastSamp + 1]
    analogDat = 1e3 * GainCorrectNI(selectData, chanList, meta)[-1, :]  # get last analog channel
    analogThresh = 0.5 * np.max(analogDat)
    pdiode_peaks = find_peaks(analogDat, distance=sRate/200, height=analogThresh)
    peak_ids = pdiode_peaks[0]
    peak_times = peak_ids / sRate
    
    print(f'number of photodiode events: {len(peak_ids)}')
    print(np.max(analogDat))

    min_isolated_gap = 0.02  # seconds

    gap_before = np.diff(peak_times, prepend=-np.inf)  
    gap_after = np.diff(peak_times, append=np.inf)     

    pdiode_on_ids = peak_ids[gap_before > min_isolated_gap]
    pdiode_off_ids = peak_ids[gap_after > min_isolated_gap]

    print("extracting digital data...")
    digArray = ExtractDigital(rawData, firstSamp, lastSamp, dw, dLineList, meta)
    print("digital data extracted...")
    tDat = np.arange(firstSamp, lastSamp+1, dtype='uint64')
    tDat = tDat/sRate 
    
    dig_line = digArray[0, :]
    changes = np.flatnonzero(np.diff(dig_line.astype(int)))
    onset_ids = changes[dig_line[changes + 1] == True] + 1
    offset_ids = changes[dig_line[changes + 1] == False] + 1

    print(f'number of onset events: {len(onset_ids)}')
    print(f'number of offset events: {len(offset_ids)}')

    onset_times = tDat[np.array(onset_ids)]
    offset_times = tDat[np.array(offset_ids)]

    neural_data["sync_on"] = onset_times # seconds
    neural_data["sync_off"] = offset_times # seconds
    neural_data["pdiode_on"] = pdiode_on_ids /sRate # seconds
    neural_data["pdiode_off"] = pdiode_off_ids / sRate # seconds
    neural_data["has_pdiode"] = is_after_cutoff

    print("saving file...")
    output_file = f'./data/procdata/{date}.npz'
    np.savez_compressed(output_file, **neural_data)
    print("file saved...")

if __name__ == "__main__":
    main()