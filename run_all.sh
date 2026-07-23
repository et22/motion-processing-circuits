#!/bin/bash
python -W ignore -m analysis.neuron_inclusion
python -W ignore -m analysis.pattern_metrics
python -W ignore -m analysis.rf_metrics
python -W ignore -m analysis.ccg_metrics
python -W ignore -m analysis.video_metrics

python -W ignore -m plotting.plot_tuning_heatmap
python -W ignore -m plotting.plot_pattern_space
python -W ignore -m plotting.plot_diffdir
python -W ignore -m plotting.plot_class
python -W ignore -m plotting.plot_ccgs
python -W ignore -m plotting.plot_video_tuning_heatmap
python -W ignore -m plotting.plot_video_tuning_corr
python -W ignore -m plotting.plot_dist_from_fractures

python -W ignore -m plotting.plot_class --split main
python -W ignore -m plotting.plot_class --split supplement

python -W ignore -m analysis.ccg_metrics --start_time 0
python -W ignore -m plotting.plot_ccgs --start_time 0

python -W ignore -m plotting.plot_tuning_bandwidth
python -W ignore -m plotting.plot_waveforms
