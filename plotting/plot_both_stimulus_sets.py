import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, wilcoxon

from plot_utils import _plot_single_polar_tuning

from utils import (
    load_config,
    load_session,
    load_pat,
    set_plot_defaults,
)

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

PATTERN_COLOR, COMPONENT_COLOR, UNCLASS_COLOR = set_plot_defaults()


def _plot_condition_overlay(
    ax,
    eq_tuning,
    eq_sem,
    un_tuning,
    un_sem,
    eq_baseline,
    un_baseline,
    color="k",
    add_legend=False,
):
    scale = np.nanmax(np.concatenate([eq_tuning, un_tuning]))

    _plot_single_polar_tuning(
        ax, eq_tuning, eq_sem, eq_baseline, scale, color=color, ls="-"
    )
    _plot_single_polar_tuning(
        ax, un_tuning, un_sem, un_baseline, scale, color=color, ls=":"
    )

    if add_legend:
        handles = [
            Line2D([0], [0], color=color, lw=1.5, ls="-", label="equal"),
            Line2D([0], [0], color=color, lw=1.5, ls=":", label="unequal"),
        ]
        ax.legend(
            handles=handles,
            loc="lower right",
            bbox_to_anchor=(1.3, 0.0),
            frameon=False,
            fontsize=8,
            handlelength=2.0,
            borderaxespad=0.2,
        )


def _draw_delta_inset(ax_sc, delta):
    delta = np.asarray(delta, float)
    delta = delta[np.isfinite(delta)]
    delta = -delta

    ax_in = ax_sc.inset_axes([0.72, 0.82, 0.28, 0.22])

    bins = np.linspace(
        np.nanmin(delta) - 0.1,
        np.nanmax(delta) + 0.1,
        16,
    )

    ax_in.hist(delta, bins=bins, color="0.75", edgecolor="none")
    ax_in.axvline(0, color="k", ls="--", lw=1.0)

    mu = np.nanmean(delta)
    ax_in.axvline(mu, color="tab:red", lw=1.5)

    _, p_delta = wilcoxon(delta)

    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.spines["top"].set_visible(False)
    ax_in.spines["right"].set_visible(False)

    star = "*" if p_delta < 0.05 else ""
    ax_in.text(
        0.57,
        1.06,
        star,
        ha="right",
        va="top",
        transform=ax_in.transAxes,
        fontsize=8,
    )
    ax_in.text(
        1.1,
        1.3,
        r"$\Delta PI = PI_{eq} - PI_{uneq}$",
        ha="right",
        va="top",
        transform=ax_in.transAxes,
        fontsize=8,
    )

    return str(p_delta), mu


def main():
    parser = argparse.ArgumentParser(description="Plot example polar tuning curves and PI comparison across two plaid stimulus sets.")
    parser.add_argument("--config", default="configs/default_config.yaml", type=str)
    args = parser.parse_args()

    #  params
    date = "121725"
    ex_component=9
    ex_unclass=38
    ex_pattern=33
    eq_task_id = 0
    un_task_id = 1 

    config = load_config(args.config)

    data_path = config["dataset_params"]["data_path"]

    pat = load_pat()
    data = load_session(data_path, date)

    assert data["plaid_task"][eq_task_id]["probe_class"] == "equal_contrast"
    assert data["plaid_task"][un_task_id]["probe_class"] == "unequal_contrast"

    eq_pat = pat["all"][date][eq_task_id]
    un_pat = pat["all"][date][un_task_id]

    pi_eq = eq_pat["pi"]
    pi_un = un_pat["pi"]

    finite = np.isfinite(pi_eq) & np.isfinite(pi_un)
    pi_eq = pi_eq[finite]
    pi_un = pi_un[finite]

    r_s, p_r = pearsonr(pi_eq, pi_un)
    delta = pi_un - pi_eq

    fig = plt.figure(figsize=(10.0, 3.4))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        width_ratios=[0.65, 0.65, 0.65, 2.0],
        wspace=0.18,
        hspace=0.18,
    )

    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1])
    ax13 = fig.add_subplot(gs[0, 2])
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])
    ax23 = fig.add_subplot(gs[1, 2])
    ax_sc = fig.add_subplot(gs[:, 3])

    _plot_condition_overlay(
        ax11,
        eq_pat["grating_tuning"][ex_component],
        eq_pat["grating_sem"][ex_component],
        un_pat["grating_tuning"][ex_component],
        un_pat["grating_sem"][ex_component],
        eq_pat["baseline_rates"][ex_component],
        un_pat["baseline_rates"][ex_component],
        color=COMPONENT_COLOR,
        add_legend=True,
    )
    _plot_condition_overlay(
        ax12,
        eq_pat["grating_tuning"][ex_unclass],
        eq_pat["grating_sem"][ex_unclass],
        un_pat["grating_tuning"][ex_unclass],
        un_pat["grating_sem"][ex_unclass],
        eq_pat["baseline_rates"][ex_unclass],
        un_pat["baseline_rates"][ex_unclass],
        color=UNCLASS_COLOR,
        add_legend=False,
    )
    _plot_condition_overlay(
        ax13,
        eq_pat["grating_tuning"][ex_pattern],
        eq_pat["grating_sem"][ex_pattern],
        un_pat["grating_tuning"][ex_pattern],
        un_pat["grating_sem"][ex_pattern],
        eq_pat["baseline_rates"][ex_pattern],
        un_pat["baseline_rates"][ex_pattern],
        color=PATTERN_COLOR,
        add_legend=False,
    )

    _plot_condition_overlay(
        ax21,
        eq_pat["plaid_tuning"][ex_component],
        eq_pat["plaid_sem"][ex_component],
        un_pat["plaid_tuning"][ex_component],
        un_pat["plaid_sem"][ex_component],
        eq_pat["baseline_rates"][ex_component],
        un_pat["baseline_rates"][ex_component],
        color=COMPONENT_COLOR,
        add_legend=False,
    )
    _plot_condition_overlay(
        ax22,
        eq_pat["plaid_tuning"][ex_unclass],
        eq_pat["plaid_sem"][ex_unclass],
        un_pat["plaid_tuning"][ex_unclass],
        un_pat["plaid_sem"][ex_unclass],
        eq_pat["baseline_rates"][ex_unclass],
        un_pat["baseline_rates"][ex_unclass],
        color=UNCLASS_COLOR,
        add_legend=False,
    )
    _plot_condition_overlay(
        ax23,
        eq_pat["plaid_tuning"][ex_pattern],
        eq_pat["plaid_sem"][ex_pattern],
        un_pat["plaid_tuning"][ex_pattern],
        un_pat["plaid_sem"][ex_pattern],
        eq_pat["baseline_rates"][ex_pattern],
        un_pat["baseline_rates"][ex_pattern],
        color=PATTERN_COLOR,
        add_legend=False,
    )

    x = pi_eq
    y = pi_un
    lim = np.nanmax(np.abs(np.concatenate([x, y])))
    lim = max(2.0, np.ceil((lim + 0.5)))

    ax_sc.scatter(x, y, s=18, c="k", alpha=0.8, linewidths=0)
    ax_sc.plot([-lim, lim], [-lim, lim], "k--", lw=1.1, alpha=0.7)
    ax_sc.set_xlim([-lim, lim])
    ax_sc.set_ylim([-lim, lim])
    ax_sc.set_aspect("equal")
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)
    ax_sc.set_xlabel("Pattern index (equal contrast)")
    ax_sc.set_ylabel("Pattern index (unequal contrast)")
    ax_sc.text(
        0.05,
        0.96,
        rf"$R_p = {r_s:.2f}$",
        transform=ax_sc.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    pdelta, mudelta =_draw_delta_inset(ax_sc, delta)

    outdir = os.path.join("figures", "general")
    os.makedirs(outdir, exist_ok=True)
    out_pdf = os.path.join(outdir, f"plaid_examples_pi_compare_{date}.pdf")
    out_png = os.path.join(outdir, f"plaid_examples_pi_compare_{date}.png")

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")

    print(f"date: {date}")
    print(f"equal task id: {eq_task_id}")
    print(f"unequal task id: {un_task_id}")
    print(f"n neurons: {pi_eq.size}")
    print("wilcoxon shift test of pi eq vs uneq: p=" + str(pdelta))
    print(f"mean delta of eq - uneq: {mudelta}")
    print(f"pearson r of pi eq vs uneq: {r_s:.2f}")
    print(f"pearson r test of pi eq vs uneq: p=" + str(p_r))

    print(f"saved: {out_pdf}")

if __name__ == "__main__":
    main()