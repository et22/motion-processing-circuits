# Motion Processing in Area MT

## Overview
This repository contains the code for the following manuscript: 

Trepka et al., (2026) A modular neural circuit for computing the motion of objects. bioRxiv. https://www.biorxiv.org/content/10.64898/2026.06.07.730718v1

<p align="center">
  <img src="https://github.com/et22/motion-processing-circuits/blob/main/assets/figure1.png?raw=true" alt="Figure 1" width="750">
</p>

## Code
### Quick start
To replicate all results in the paper, first download the dataset and place it in a directory called ```data/``` structured as: 

```text
data/
├── output_*.pkl
├── ...
└── videos/
    ├── *.mp4
    └── ...
```

Next, install the required dependencies: 

```bash
pip install -r requirements.txt
```

Finally, run the bash script: 
```bash
bash run_all.sh
```

### Code organization
The code is divided into ```configs/```, ```analysis/```, and ```plotting/``` directories. The scripts in ```analysis/```save intermediate results for later use in ```plotting/``` and relevant parameters and paths are defined in ```configs/```. ```run_all.sh``` demonstrates the order in which scripts should be run to reproduce results. The most important scripts are highlighted below: 

```analysis/neuron_inclusion.py``` selects which subset of neurons to include in downstream analyses (based on parameters defined in ```configs/```) using firing rate and selectivity criteria. This is an important step for filtering the Kilosort4 output. 

```analysis/pattern_metrics.py``` computes tuning curves, pattern index, and component and pattern neuron labels.  

```plotting/plot_tuning_heatmap.py``` plots tuning curves over space along the probe.

The ```preprocessing/``` directory is provided for reference. The code in ```preprocessing/``` was used to generate the dataset described below. 

## Dataset

### Overview
The Motion Processing Neuropixels Dataset is available on [Kaggle](https://www.kaggle.com/datasets/ethantrepka1/motion-processing-neuropixels/). The dataset contains neural activity recordings encompassing area MT while animals viewed visual stimuli, including drifting gratings, plaids, and natural videos. The dataset contains spike-sorted neural data aligned to stimulus onset and corresponding stimulus information. Data from 25 recording sessions in two animals is included. Recording sessions contain one or more tasks (probe, plaid, video) and one or more variants of the same task, e.g., the plaid task with stimuli of different sizes. 

The stimulus information and neural data for each recording session is provided in the ```data/``` directory. The natural video stimuli can be found [here](https://www.kaggle.com/datasets/ethantrepka1/stsbench).

### Dataset Organization
The stimulus-aligned neural data and stimulus info for each recording session is contained in the file ```./output_{date}.pkl.gz```. The data for one session can be loaded as follows: 

```python
import gzip, pickle
with gzip.open("../data/output_{date}.pkl_gz", "rb") as f:
    data = pickle.load(f)
```

``` data``` is a dictionary with neuron-specific and task-specific keys. The neuron-specific fields are ```['y', 'x', 'waveforms', 'waveform_times']``` and the task-specific fields are ```['rf_task', 'plaid_task', 'video_task', 'baseline_task']```. Each of these fields is described in detail below. 

#### Neuron-specific fields
The neuron-specific fields describe the following properties for each neuron: 
- `x` and `y`
    - shape: (num_neurons)
    - dtype: float
    - description: x- and y-position of the best channel for the neuron in µm. y-position is relative to the tip of the recording electrode with zero corresponding to channel closest to the tip. 
- `waveforms`
    - shape: (num_neurons, 61)
    - dtype: float
    - description: spike waveform template for each neuron from its best channel. 
- `waveform_times`
    - shape: (num_neurons, 61)
    - dtype: float
    - description: times in ms corresponding to the waveform samples for each neuron. 

#### Task-specific fields
The task-specific fields each contain a list of dictionaries, one for each task variant that was run in a given recording session. 

The receptive field mapping task (```rf_task```),  pattern motion task (```plaid_task```), and natural video task (```video_task```) contain the following information: 
- `spikes`
  - shape: (num_stimuli, num_timepoints, num_neurons)
  - dtype: bool
  - description: spike train aligned to probe onset. Note that there is a 100 ms buffer before probe onset and 100 ms buffer after probe offset included in spike train (see times), such that index 100 is 0 ms from probe onset and index -1 is 100 ms after probe offset.
- `coords`
  - shape: (num_stimuli, 2)
  - dtype: float
  - description: coordinates (x,y) that stimulus was centered at in degrees of visual angle.
- `times`
  - shape: (num_timepoints,)
  - dtype: float
  - description: times relative to stimulus onset, corresponding to the num_timepoints axis in spikes. 

The receptive field mapping task (```rf_task```) and pattern motion task (```plaid_task```) also contain: 
- `direction`
  - shape: (num_stimuli,)
  - dtype: int
  - description: stimulus drift direction in degrees. 
- `probe_size`
  - shape: 1
  - dtype: int
  - descript: size of the stimulus

In addition, the pattern motion task (```plaid_task```) contains: 
- `plaid`
  - shape: (num_stimuli,)
  - dtype: int
  - description: id corresponding to each plaid stimulus condition. 0 = drifting grating, 1 = plaid, and 2 = triplaid. 
- `probe_class`
  - shape: 1
  - dtype: str
  - descript: either 'equal_contrast' or 'unequal_contrast', see manuscript for details

The natural video task (```video_task```) also contains: 
- `video_id`
  - shape: (num_stimuli,)
  - dtype: str
  - description: unique id for each video presented, corresponding to the video filename. 
- `probe_order`
  - shape: (num_stimuli,)
  - dtype: int
  - description: order in which stimuli were presented within a trial (e.g., whether a stimulus was first or third in the trial)
- `is_train_probe`
  - shape: (num_stimuli,)
  - dtype: bool
  - description: whether the stimulus was part of a train trial (unique) or test trial (repeated stimuli). 1 = train trial, 0 = test trial.
- `fix_coord_deg`
  - shape: (num_trials, 2)
  - dtype: int
  - description: coordinates of the fixation point in degrees of visual angle. Note that this is the same across all trials, so the trial dimension can be neglected.
- `px_per_deg`
  - shape: 1
  - dtype: float
  - description: pixels of the video (640 x 360) per degree of visual angle. 

## Contact
Feel free to email trepka@stanford.edu with any questions or add an issue to the GitHub repo.

## Acknowledgements
Code for analyses in this repo was largely written by hand. Some functions incorporated AI generated code. All AI generated code included here was proofread to ensure correctness. 
