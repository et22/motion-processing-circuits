import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from plotting.plot_utils import _plot_single_polar_tuning
from utils import load_config, load_session, load_pat, set_plot_defaults, load_inc

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

def annotate_scale(ax, scale):
    if np.isfinite(scale):
        ax.text(
            0.80,
            -0.98,
            f"{int(round(scale))} sp/s",
            fontsize=6,
            ha="center",
            va="top",
        )

def annotate_depth(ax, depth_um):
    ax.text(
        -1.20,
        1.12,
        f"depth = {int(round(depth_um))} µm",
        fontsize=6,
        ha="left",
        va="top",
    )

def annotate_metrics(ax, rp, rc, pi):
    ax.text(
        -1.20,
        1.12,
        f"Rp = {rp:.2f}\nRc = {rc:.2f}\nPI = {pi:.2f}",
        fontsize=6,
        ha="left",
        va="top",
    )

def make_single_plot(
    outpath,
    tuning,
    tuning_sem,
    baseline,
    scale,
    color="k",
    ls="-",
    add_error=True,
    add_baseline=False,
    depth_um=None,
    rp=None,
    rc=None,
    pi=None,
):
    fig, ax = plt.subplots(figsize=(1.2, 1.2))
    _plot_single_polar_tuning(
        ax=ax,
        tuning=tuning,
        tuning_sem=tuning_sem,
        baseline_rate_hz=baseline,
        scale=scale,
        color=color,
        ls=ls,
        add_error=add_error,
        add_baseline=add_baseline,
        size_mult=0.4,
    )
    annotate_scale(ax, scale)

    if depth_um is not None:
        annotate_depth(ax, depth_um)

    if (rp is not None) and (rc is not None) and (pi is not None):
        annotate_metrics(ax, rp, rc, pi)

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml", type=str)
    args = parser.parse_args()

    date = "121625"

    config = load_config(args.config)
    data_path = config["dataset_params"]["data_path"]

    pat = load_pat()
    pat_main = pat["main"][date]

    data = load_session(data_path, date)
    inc = load_inc()["rf_plaid"][date]

    # included-neuron depths, aligned to pat_main neuron order
    y_all = np.asarray(data["y"])
    y = y_all[inc]

    outdir = os.path.join("figures", "pattern_tuning_all_neurons", date)
    os.makedirs(outdir, exist_ok=True)

    n_neurons = pat_main["pi"].shape[0]
    print(n_neurons)

    for i in range(n_neurons):
        neuron_dir = os.path.join(outdir, f"neuron_{i:04d}")
        os.makedirs(neuron_dir, exist_ok=True)

        cls_color, cls_name = get_class_color_and_name(pat_main, i)

        grating_tuning = pat_main["grating_tuning"][i]
        plaid_tuning = pat_main["plaid_tuning"][i]
        grating_sem = pat_main["grating_sem"][i]
        plaid_sem = pat_main["plaid_sem"][i]
        baseline = pat_main["baseline_rates"][i]

        rp = pat_main["rp"][i]
        rc = pat_main["rc"][i]
        pi = pat_main["pi"][i]
        depth_um = y[i]

        pattern_pred = grating_tuning
        component_pred = np.roll(grating_tuning, -2) + np.roll(grating_tuning, 2)
        component_pred = component_pred - np.min(component_pred)

        # 1
        make_single_plot(
            os.path.join(neuron_dir, "01_grating_black.pdf"),
            grating_tuning, grating_sem, baseline, np.nanmax(grating_tuning),
            color="k", ls="-", add_error=True, add_baseline=True,
        )

        # 2
        make_single_plot(
            os.path.join(neuron_dir, "02_plaid_black.pdf"),
            plaid_tuning, plaid_sem, baseline, np.nanmax(plaid_tuning),
            color="k", ls="-", add_error=True, add_baseline=True,
        )

        # 3
        make_single_plot(
            os.path.join(neuron_dir, "03_grating_classcolor.pdf"),
            grating_tuning, grating_sem, baseline, np.nanmax(grating_tuning),
            color=cls_color, ls="-", add_error=True, add_baseline=True,
        )

        # 4
        make_single_plot(
            os.path.join(neuron_dir, "04_plaid_classcolor.pdf"),
            plaid_tuning, plaid_sem, baseline, np.nanmax(plaid_tuning),
            color=cls_color, ls="-", add_error=True, add_baseline=True,
        )

        # 5
        make_single_plot(
            os.path.join(neuron_dir, "05_pattern_prediction.pdf"),
            pattern_pred, np.zeros_like(pattern_pred), 0.0, np.nanmax(pattern_pred),
            color=PATTERN_COLOR, ls="--", add_error=False, add_baseline=False,
        )

        # 6
        make_single_plot(
            os.path.join(neuron_dir, "06_component_prediction.pdf"),
            component_pred, np.zeros_like(component_pred), 0.0, np.nanmax(component_pred),
            color=COMPONENT_COLOR, ls="--", add_error=False, add_baseline=False,
        )

        # 7
        make_single_plot(
            os.path.join(neuron_dir, "07_plaid_classcolor_with_metrics.pdf"),
            plaid_tuning, plaid_sem, baseline, np.nanmax(plaid_tuning),
            color=cls_color, ls="-", add_error=True, add_baseline=True,
            rp=rp, rc=rc, pi=pi,
        )

        # Versions with depth annotated
        make_single_plot(
            os.path.join(neuron_dir, "01_grating_black_depth.pdf"),
            grating_tuning, grating_sem, baseline, np.nanmax(grating_tuning),
            color="k", ls="-", add_error=True, add_baseline=True,
            depth_um=depth_um,
        )

        make_single_plot(
            os.path.join(neuron_dir, "02_plaid_black_depth.pdf"),
            plaid_tuning, plaid_sem, baseline, np.nanmax(plaid_tuning),
            color="k", ls="-", add_error=True, add_baseline=True,
            depth_um=depth_um,
        )

        make_single_plot(
            os.path.join(neuron_dir, "03_grating_classcolor_depth.pdf"),
            grating_tuning, grating_sem, baseline, np.nanmax(grating_tuning),
            color=cls_color, ls="-", add_error=True, add_baseline=True,
            depth_um=depth_um,
        )

        make_single_plot(
            os.path.join(neuron_dir, "04_plaid_classcolor_depth.pdf"),
            plaid_tuning, plaid_sem, baseline, np.nanmax(plaid_tuning),
            color=cls_color, ls="-", add_error=True, add_baseline=True,
            depth_um=depth_um,
        )

        make_single_plot(
            os.path.join(neuron_dir, "05_pattern_prediction_depth.pdf"),
            pattern_pred, np.zeros_like(pattern_pred), 0.0, np.nanmax(pattern_pred),
            color=PATTERN_COLOR, ls="--", add_error=False, add_baseline=False,
            depth_um=depth_um,
        )

        make_single_plot(
            os.path.join(neuron_dir, "06_component_prediction_depth.pdf"),
            component_pred, np.zeros_like(component_pred), 0.0, np.nanmax(component_pred),
            color=COMPONENT_COLOR, ls="--", add_error=False, add_baseline=False,
            depth_um=depth_um,
        )

        make_single_plot(
            os.path.join(neuron_dir, "07_plaid_classcolor_with_metrics_depth.pdf"),
            plaid_tuning, plaid_sem, baseline, np.nanmax(plaid_tuning),
            color=cls_color, ls="-", add_error=True, add_baseline=True,
            depth_um=depth_um, rp=rp, rc=rc, pi=pi,
        )

        print(f"saved neuron {i:04d} ({cls_name} {pi:.2f})")

    print(f"Done. Saved plots to: {outdir}")


if __name__ == "__main__":
    main()