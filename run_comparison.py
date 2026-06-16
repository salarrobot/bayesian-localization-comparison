"""
Run and compare four localization algorithms on the SAME simulated robot.

Pipeline
--------
1. Simulate the autonomous robot once, recording (odometry, measurements,
   ground-truth pose) for every time step.
2. Replay that identical data through all four filters
   (Histogram, EKF, UKF, Particle Filter), recording each filter's
   estimated trajectory, error and CPU time.
3. Save comparison plots:  trajectories, error-vs-time, RMSE bar chart,
   and an overview.
4. Render a side-by-side animated GIF of all filters tracking the robot.

Usage
-----
    python run_comparison.py                 # full run + GIF
    python run_comparison.py --no-anim       # plots/metrics only (fast)
    python run_comparison.py --steps 500 --seed 3 --fps 20

All outputs are written to the ./outputs directory.
"""

import argparse
import csv
import os
import sys
from time import perf_counter

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless / batch rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import (Environment, RobotSimulator, wrap_angle, DT,
                         MIN_X, MAX_X, MIN_Y, MAX_Y)
from algorithms import (HistogramFilter, EKFLocalization,
                        UKFLocalization, ParticleFilter)

ORDER = ["Histogram", "EKF", "UKF", "Particle Filter"]


# =====================================================================
# ----------------------------- helpers -------------------------------
# =====================================================================
def make_filters(env, init_pose, init_cov, seed):
    return [
        HistogramFilter(env, init_pose, init_cov),
        EKFLocalization(env, init_pose, init_cov),
        UKFLocalization(env, init_pose, init_cov),
        ParticleFilter(env, init_pose, init_cov, seed=seed + 100),
    ]


def simulate(env, steps, seed):
    """Run the ground-truth robot once and record everything."""
    sim = RobotSimulator(env, seed=seed)
    records = []
    for _ in range(steps):
        u, meas, truth = sim.step(DT)
        records.append((u, meas, truth))
    return records


def setup_axis(ax, title=None):
    ax.set_xlim(MIN_X, MAX_X)
    ax.set_ylim(MIN_Y, MAX_Y)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)


def cov_ellipse(ax, mean, cov, color, nstd=2.0):
    if cov is None:
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-6)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2.0 * nstd * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, w, h, angle=angle, fill=False,
                         edgecolor=color, lw=1.6, alpha=0.9))


# =====================================================================
# ------------------ run all filters over the records -----------------
# =====================================================================
def run_filters(env, records, init_pose, init_cov, seed, stride):
    filters = make_filters(env, init_pose, init_cov, seed)
    names = [f.name for f in filters]

    est_traj = {n: [] for n in names}
    timings = {n: 0.0 for n in names}
    truth_traj = []
    frames = []                              # light-weight viz snapshots

    for k, (u, meas, truth) in enumerate(records):
        truth_traj.append(truth)
        for f in filters:
            t0 = perf_counter()
            f.predict(u, DT)
            f.update(meas)
            timings[f.name] += perf_counter() - t0
            est_traj[f.name].append(f.estimate())

        if k % stride == 0:
            snap = {"k": k, "truth": truth, "filters": {}}
            for f in filters:
                pts = f.particles_xy()
                if pts is not None and len(pts) > 250:
                    sel = np.linspace(0, len(pts) - 1, 250).astype(int)
                    pts = pts[sel]
                snap["filters"][f.name] = {
                    "est": f.estimate(),
                    "cov": f.cov_xy(),
                    "particles": pts,
                    "grid": f.belief_grid(),
                }
            frames.append(snap)

    return names, est_traj, timings, truth_traj, frames


# =====================================================================
# ------------------------------ metrics ------------------------------
# =====================================================================
def compute_metrics(names, est_traj, truth_traj, timings, steps):
    truth = np.array(truth_traj)
    metrics = {}
    for n in names:
        est = np.array(est_traj[n])
        pos_err = np.hypot(est[:, 0] - truth[:, 0], est[:, 1] - truth[:, 1])
        head_err = np.abs(wrap_angle(est[:, 2] - truth[:, 2]))
        metrics[n] = {
            "pos_rmse": float(np.sqrt(np.mean(pos_err ** 2))),
            "pos_mean": float(np.mean(pos_err)),
            "pos_max": float(np.max(pos_err)),
            "head_rmse_deg": float(np.degrees(np.sqrt(np.mean(head_err ** 2)))),
            "time_ms": float(1000.0 * timings[n] / steps),
            "pos_err": pos_err,
            "head_err_deg": np.degrees(head_err),
        }
    return metrics


def print_and_save_metrics(metrics, outdir):
    hdr = f"{'Algorithm':<16}{'Pos RMSE [m]':>14}{'Pos mean [m]':>14}" \
          f"{'Head RMSE [deg]':>18}{'Time/step [ms]':>17}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for n in ORDER:
        m = metrics[n]
        print(f"{n:<16}{m['pos_rmse']:>14.3f}{m['pos_mean']:>14.3f}"
              f"{m['head_rmse_deg']:>18.2f}{m['time_ms']:>17.3f}")
    print()

    with open(os.path.join(outdir, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["algorithm", "pos_rmse_m", "pos_mean_m", "pos_max_m",
                    "head_rmse_deg", "time_per_step_ms"])
        for n in ORDER:
            m = metrics[n]
            w.writerow([n, f"{m['pos_rmse']:.4f}", f"{m['pos_mean']:.4f}",
                        f"{m['pos_max']:.4f}", f"{m['head_rmse_deg']:.4f}",
                        f"{m['time_ms']:.4f}"])


# =====================================================================
# ------------------------------- plots -------------------------------
# =====================================================================
def plot_trajectories(env, est_traj, truth_traj, metrics, filters, outdir):
    truth = np.array(truth_traj)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, f in zip(axes.ravel(), filters):
        setup_axis(ax, f"{f.name}   (RMSE {metrics[f.name]['pos_rmse']:.2f} m)")
        env.draw_map(ax, color="#bbbbbb", lw=1.0)
        env.draw_beacons(ax)
        est = np.array(est_traj[f.name])
        ax.plot(truth[:, 0], truth[:, 1], "k-", lw=1.5, label="ground truth")
        ax.plot(est[:, 0], est[:, 1], color=f.color, lw=1.3,
                alpha=0.9, label="estimate")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Estimated trajectory vs. ground truth", fontsize=14)
    fig.tight_layout()
    path = os.path.join(outdir, "plot_trajectories.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_errors(metrics, filters, outdir):
    t = np.arange(len(metrics[filters[0].name]["pos_err"])) * DT
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for f in filters:
        a1.plot(t, metrics[f.name]["pos_err"], color=f.color,
                lw=1.2, label=f.name)
        a2.plot(t, metrics[f.name]["head_err_deg"], color=f.color,
                lw=1.2, label=f.name)
    a1.set_ylabel("position error [m]")
    a1.set_title("Localization error over time")
    a1.grid(alpha=0.3); a1.legend(ncol=4, fontsize=9)
    a2.set_ylabel("heading error [deg]")
    a2.set_xlabel("time [s]")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "plot_errors.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_rmse_bars(metrics, filters, outdir):
    names = [f.name for f in filters]
    colors = [f.color for f in filters]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    keys = [("pos_rmse", "Position RMSE [m]"),
            ("head_rmse_deg", "Heading RMSE [deg]"),
            ("time_ms", "CPU time per step [ms]")]
    for ax, (key, title) in zip(axes, keys):
        vals = [metrics[n][key] for n in names]
        bars = ax.bar(names, vals, color=colors, alpha=0.9)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Accuracy and cost comparison", fontsize=14)
    fig.tight_layout()
    path = os.path.join(outdir, "plot_rmse_bars.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_overview(env, est_traj, truth_traj, filters, outdir):
    truth = np.array(truth_traj)
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_axis(ax, "All estimates overlaid on ground truth")
    env.draw_map(ax, color="#999999", lw=1.2)
    env.draw_beacons(ax)
    ax.plot(truth[:, 0], truth[:, 1], "k-", lw=2.5, label="ground truth")
    for f in filters:
        est = np.array(est_traj[f.name])
        ax.plot(est[:, 0], est[:, 1], color=f.color, lw=1.2,
                alpha=0.85, label=f.name)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    path = os.path.join(outdir, "plot_overview.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# =====================================================================
# ----------------------------- animation -----------------------------
# =====================================================================
def render_animation(env, frames, est_traj, truth_traj, metrics, outdir, fps):
    truth_all = np.array(truth_traj)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()
    colors = {n: c for n, c in
              [("Histogram", "#ff7f0e"), ("EKF", "#d62728"),
               ("UKF", "#9467bd"), ("Particle Filter", "#2ca02c")]}

    def draw(i):
        snap = frames[i]
        k = snap["k"]
        fig.suptitle(f"SLAM localization comparison   —   t = {k * DT:5.1f} s",
                     fontsize=14)
        tr = truth_all[:k + 1]
        for ax, name in zip(axes, ORDER):
            ax.cla()
            fd = snap["filters"][name]
            col = colors[name]

            # filter-specific belief layer (drawn first, in the background)
            grid = fd["grid"]
            if grid is not None:
                ax.pcolormesh(env.x_edges, env.y_edges, grid.T,
                              cmap="Blues", shading="auto",
                              vmin=0.0, vmax=max(1e-9, grid.max()), alpha=0.85)
            pts = fd["particles"]
            if pts is not None:
                ax.scatter(pts[:, 0], pts[:, 1], s=4, color=col, alpha=0.35)
            cov_ellipse(ax, fd["est"][:2], fd["cov"], col)

            env.draw_map(ax, color="#888888", lw=1.0)
            env.draw_beacons(ax)

            # trails
            ax.plot(tr[:, 0], tr[:, 1], "k-", lw=1.2, alpha=0.7)
            est = np.array(est_traj[name][:k + 1])
            ax.plot(est[:, 0], est[:, 1], color=col, lw=1.2, alpha=0.9)

            # current poses
            txx, tyy, tth = snap["truth"]
            ax.add_patch(Circle((txx, tyy), 0.45, facecolor="#1f77b4",
                                edgecolor="k", zorder=5))
            ex, ey, eth = fd["est"]
            ax.arrow(ex, ey, 1.2 * np.cos(eth), 1.2 * np.sin(eth),
                     head_width=0.5, head_length=0.5, fc=col, ec=col, zorder=6)

            err = np.hypot(ex - txx, ey - tyy)
            setup_axis(ax, f"{name}    error = {err:4.2f} m")
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), blit=False)
    gif_path = os.path.join(outdir, "comparison.gif")
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=85)
    plt.close(fig)
    return gif_path


# =====================================================================
# -------------------------------- main -------------------------------
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4,
                    help="keep every Nth step as an animation frame")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-anim", action="store_true",
                    help="skip the (slow) GIF rendering")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    env = Environment()

    # all filters share the same (deliberately offset) initial guess
    true_start = np.array([3.5, 4.0, np.pi / 2])
    init_pose = true_start + np.array([1.2, -1.0, 0.25])
    init_cov = np.diag([1.5 ** 2, 1.5 ** 2, 0.35 ** 2])

    print(f"Simulating {args.steps} steps (seed={args.seed}) ...")
    records = simulate(env, args.steps, args.seed)

    print("Running 4 filters on identical sensor data ...")
    names, est_traj, timings, truth_traj, frames = run_filters(
        env, records, init_pose, init_cov, args.seed, args.stride)

    metrics = compute_metrics(names, est_traj, truth_traj, timings, args.steps)
    print_and_save_metrics(metrics, args.outdir)

    # the filter objects are only used here to carry name/color into plots
    plot_filters = make_filters(env, init_pose, init_cov, args.seed)
    print("Saving plots ...")
    p1 = plot_trajectories(env, est_traj, truth_traj, metrics, plot_filters, args.outdir)
    p2 = plot_errors(metrics, plot_filters, args.outdir)
    p3 = plot_rmse_bars(metrics, plot_filters, args.outdir)
    p4 = plot_overview(env, est_traj, truth_traj, plot_filters, args.outdir)
    for p in (p1, p2, p3, p4):
        print(f"  -> {p}")

    if not args.no_anim:
        print(f"Rendering animation ({len(frames)} frames) ...")
        gif = render_animation(env, frames, est_traj, truth_traj,
                               metrics, args.outdir, args.fps)
        print(f"  -> {gif}")

    print("\nDone.")


if __name__ == "__main__":
    main()
