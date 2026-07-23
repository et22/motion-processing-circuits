import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import load_config, load_pat, sort_dates_and_make_session_titles, set_plot_defaults

pattern_color, component_color, unclass_color = set_plot_defaults()
comp_to_lab = {"PDS": "Pattern", "CDS": "Component", "Unc": "Unclassified"}

def fisher_z_inverse(out):
    coef = np.exp(2 * out)
    return (coef - 1) / (coef + 1)

def classify_pi(pi_cls):
    cls = np.full(pi_cls.shape, "Unc", dtype=object)
    cls[pi_cls == 1] = "PDS"
    cls[pi_cls == -1] = "CDS"
    return cls

def plot_zc_vs_zp(results, save_path, patcomp_split):
    fig, ax = plt.subplots(figsize=(2.7, 2.7))
    classes = {"PDS": pattern_color, "CDS": component_color, "Unc": unclass_color}

    zc = np.asarray(results["Zc"])
    zp = np.asarray(results["Zp"])
    cls = np.asarray(results["classification"])

    for c, color in classes.items():
        mask = cls == c
        ax.scatter(
            zc[mask],
            zp[mask],
            c=color,
            s=10,
            alpha=1.0,
            marker='.',
            label=comp_to_lab[c],
        )

    x = np.linspace(0, 7, 100)
    y = x + patcomp_split
    ax.plot(x, y, "k--", lw=1)

    y = np.linspace(0, 7, 100)
    x = y + patcomp_split
    ax.plot(x, y, "k--", lw=1)

    ax.plot([patcomp_split, patcomp_split], [-3, 0], "k--", lw=1)
    ax.plot([-3, 0], [patcomp_split, patcomp_split], "k--", lw=1)

    ax.set_xticks([-5, -2.5, 0, 2.5, 5, 7.5])
    ax.set_yticks([-5, -2.5, 0, 2.5, 5, 7.5])
    ax.set_xlim([-5, 8.0])
    ax.set_ylim([-5, 8.0])
    ax.set_xlabel("Z-component correlation (Zc)")
    ax.set_ylabel("Z-pattern correlation (Zp)")
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(-0.06, 0),handletextpad=0.1)
    ax.set_position([0.2, 0.18, 0.72, 0.72])
    plt.savefig(os.path.join("./figures/class/", "zc_zp_" + save_path), dpi=800)
    plt.close(fig)


def plot_rc_vs_rp(results, save_path, patcomp_split):
    fig, ax = plt.subplots(figsize=(2.7, 2.7))
    classes = {"PDS": pattern_color, "CDS": component_color, "Unc": unclass_color}

    rc = np.asarray(results["Rc"])
    rp = np.asarray(results["Rp"])
    cls = np.asarray(results["classification"])

    for c, color in classes.items():
        mask = cls == c
        ax.scatter(
            rc[mask],
            rp[mask],
            c=color,
            s=10,
            alpha=1.0,
            marker='.',
            label=comp_to_lab[c],
        )

    df = np.sqrt(1 / (12 - 3))
    x = np.linspace(0, 10, 100)
    y = x + patcomp_split
    ax.plot(fisher_z_inverse(x * df), fisher_z_inverse(y * df), "k--", lw=1)

    y = np.linspace(0, 10, 100)
    x = y + patcomp_split
    ax.plot(fisher_z_inverse(x * df), fisher_z_inverse(y * df), "k--", lw=1)

    ax.plot(
        fisher_z_inverse(np.array([patcomp_split, patcomp_split]) * df),
        fisher_z_inverse(np.array([-3, 0]) * df),
        "k--",
        lw=1,
    )
    ax.plot(
        fisher_z_inverse(np.array([-3, 0]) * df),
        fisher_z_inverse(np.array([patcomp_split, patcomp_split]) * df),
        "k--",
        lw=1,
    )

    ax.set_xlabel("Component correlation (Rc)")
    ax.set_ylabel("Pattern correlation (Rp)")
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(-0.06, 0),handletextpad=0.1)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xticks([-1, -0.5, 0, 0.5, 1])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_position([0.2, 0.18, 0.72, 0.72])
    plt.savefig(os.path.join("./figures/class/", "rc_rp_" + save_path), dpi=800)
    plt.close(fig)


def plot_pi_hist(results, save_path, patcomp_split):
    pi = np.asarray(results["PI"])
    cls = np.asarray(results["classification"])

    n_pds = int(np.sum(cls == "PDS"))
    n_cds = int(np.sum(cls == "CDS"))
    n_un = int(np.sum(cls == "Unc"))
    print(f"[{save_path}] PDS: {n_pds} | CDS: {n_cds} | Unclassified: {n_un}")

    bw = 0.32
    pi_min = np.nanmin(pi)
    pi_max = np.nanmax(pi) 

    n_up = int(np.ceil(max(0.0, pi_max) / bw))
    n_dn = int(np.ceil(max(0.0, -pi_min) / bw))

    neg_edges = -np.arange(n_dn * bw, 0, -bw)
    pos_edges = np.arange(0, (n_up + 1) * bw, bw)
    bins = np.concatenate((neg_edges, pos_edges))

    fig, ax = plt.subplots(figsize=(2.7, 2.7))
    counts, edges, patches = ax.hist(
        pi, bins=bins, edgecolor="none", linewidth=0, alpha=0.9
    )

    for patch, e_left, e_right in zip(patches, edges[:-1], edges[1:]):
        if e_left <= -patcomp_split:
            color = component_color
        elif e_left >= patcomp_split:
            color = pattern_color
        else:
            color = unclass_color

        patch.set_facecolor(color)
        patch.set_edgecolor(color) 
        #patch.set_linewidth(0)

    ax.axvline(patcomp_split, linestyle="--", color="k", linewidth=1)
    ax.axvline(-patcomp_split, linestyle="--", color="k", linewidth=1)

    ax.set_xlabel("Pattern Index (Zp − Zc)")
    ax.set_ylabel("Count")
    ax.set_xlim([edges[0], edges[-1]])

    ax.text(
        0.7, 1.06, f"n = {n_pds}",
        color=pattern_color,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8
    )
    ax.text(
        0.4, 1.06, f"n = {n_un}",
        color=unclass_color,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8
    )
    ax.text(
        0.03, 1.06, f"n = {n_cds}",
        color=component_color,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    plt.tight_layout()
    plt.xlim([-8, 8])
    ax.set_position([0.2, 0.18, 0.72, 0.72])
    plt.savefig(os.path.join("./figures/class/", "pi_" + save_path), dpi=800)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Aggregate classification space across sessions and plot.")
    parser.add_argument("--config", default="configs/default_config.yaml", type=str)
    parser.add_argument("--split", default="all", type=str)
    args = parser.parse_args()
    os.makedirs("./figures/class/", exist_ok=True)
    config = load_config(args.config)
    patcomp_split = config["analysis_params"]["patcomp_split"]
    ordered_dates, date_to_session = sort_dates_and_make_session_titles(config, args.split)

    pat = load_pat()
    
    results = {
        "PI": [],
        "Rc": [],
        "Rp": [],
        "Zc": [],
        "Zp": [],
        "classification": [],
    }

    for date in ordered_dates:
        date = str(date)
        if date not in pat["main"]:
            print(date)
            print("here")
            continue

        pi = np.asarray(pat["main"][date]["pi"])
        pi_cls = np.asarray(pat["main"][date]["pi_cls"])
        rc = np.asarray(pat["main"][date]["rc"])
        rp = np.asarray(pat["main"][date]["rp"])
        zc = np.asarray(pat["main"][date]["zc"])
        zp = np.asarray(pat["main"][date]["zp"])

        cls = classify_pi(pi_cls)

        results["PI"].extend(pi.tolist())
        results["Rc"].extend(rc.tolist())
        results["Rp"].extend(rp.tolist())
        results["Zc"].extend(zc.tolist())
        results["Zp"].extend(zp.tolist())
        results["classification"].extend(cls.tolist())

    results["PI"] = np.asarray(results["PI"])
    results["Rc"] = np.asarray(results["Rc"])
    results["Rp"] = np.asarray(results["Rp"])
    results["Zc"] = np.asarray(results["Zc"])
    results["Zp"] = np.asarray(results["Zp"])
    results["classification"] = np.asarray(results["classification"])

    print(f"n neurons = {len(results['PI'])}")
    print(f"n PDS = {np.sum(results['classification'] == 'PDS')}")
    print(f"n CDS = {np.sum(results['classification'] == 'CDS')}")
    print(f"n Unc = {np.sum(results['classification'] == 'Unc')}")

    save_path = f"{args.split}.pdf"

    plot_rc_vs_rp(results, save_path, patcomp_split)
    plot_zc_vs_zp(results, save_path, patcomp_split)
    plot_pi_hist(results, save_path, patcomp_split)

if __name__ == "__main__":
    main()
