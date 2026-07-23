import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def _plot_single_polar_tuning(ax, tuning, tuning_sem, baseline_rate_hz, scale, color="k", ls="-", add_error=True, add_baseline=False, size_mult=1.0):
    tuning = np.asarray(tuning, float)
    tuning_sem = np.asarray(tuning_sem, float)
    n_dirs = tuning.size

    dirs = np.arange(n_dirs) * (360.0 / float(n_dirs))
    angles = np.deg2rad(dirs)
    r = np.asarray(tuning, float) / scale 
    sem_r = np.asarray(tuning_sem, float) / scale
    br = float(baseline_rate_hz) / float(scale)
    xs = r * np.cos(angles)
    ys = r * np.sin(angles)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([-1.3, 1.3], [0.0, 0.0], color="k", lw=0.5, zorder=0)
    ax.plot([0.0, 0.0], [-1.3, 1.3], color="k", lw=0.5, zorder=0)

    ax.scatter(xs, ys, color=color, marker=".", s=12*size_mult, zorder=3)

    if add_error:
        cap_len = 0.03
        for a, ri, sei in zip(angles, r, sem_r):
            ux, uy = np.cos(a), np.sin(a)
            tx, ty = -uy, ux
            x0, y0 = (ri - sei) * ux, (ri - sei) * uy
            x1, y1 = (ri + sei) * ux, (ri + sei) * uy

            ax.plot([x0, x1], [y0, y1], color=color, lw=0.5, zorder=2)
            ax.plot(
                [x0 - cap_len * tx, x0 + cap_len * tx],
                [y0 - cap_len * ty, y0 + cap_len * ty],
                color=color,
                lw=0.5,
                zorder=2,
            )
            ax.plot(
                [x1 - cap_len * tx, x1 + cap_len * tx],
                [y1 - cap_len * ty, y1 + cap_len * ty],
                color=color,
                lw=0.5,
                zorder=2,
            )

    x = np.arange(n_dirs + 1) * (360.0 / float(n_dirs))
    cs = CubicSpline(
        x,
        np.concatenate((tuning, [tuning[0]]), axis=0),
        bc_type="periodic",
    )
    ang_i = np.linspace(0.0, 360.0, 500)
    tun_i = cs(ang_i)
    r_i = np.asarray(tun_i, float) / scale
    xs_i = r_i * np.cos(np.deg2rad(ang_i))
    ys_i = r_i * np.sin(np.deg2rad(ang_i))
    ax.plot(np.r_[xs_i, xs_i[0]], np.r_[ys_i, ys_i[0]], color=color, lw=0.5, ls=ls, zorder=1)
    if add_baseline:
        ax.add_patch(plt.Circle((0, 0), br, fill=False, ec=color, lw=0.5, ls='--', zorder=1))
        
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)