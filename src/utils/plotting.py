"""Publication-quality plots for the radar-imu-fusion project. All functions
save a PNG to `save_path` and also return the created Figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AXIS_LABELS = ["x (East)", "y (North)", "z (Up)"]
COLORS = {
    "truth": "#111111",
    "imu_only": "#d62728",
    "imu_gnss": "#1f77b4",
    "radar_doppler": "#2ca02c",
    "radar_full": "#17becf",
    "full_fusion": "#9467bd",
    "gnss_dropout": "#ff7f0e",
}


def _save(fig, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_comparison(true_pos: np.ndarray, estimates: dict[str, np.ndarray], save_path):
    """2D top-down trajectory comparison: ground truth vs one or more estimates.

    IMU-only dead reckoning drifts kilometers off course, which would swamp
    the axis scale and hide the (much smaller) GNSS/radar errors -- so this
    is drawn as two panels: full extent, and a panel zoomed to the ground
    truth's own footprint.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for ax in axes:
        ax.plot(true_pos[:, 0], true_pos[:, 1], color=COLORS["truth"], lw=2.5, label="Ground truth", zorder=5)
        for name, pos in estimates.items():
            ax.plot(pos[:, 0], pos[:, 1], color=COLORS.get(name, None), lw=1.4, alpha=0.9, label=name)
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)

    axes[0].set_title("Full extent")
    axes[0].legend(loc="best", fontsize=9)

    margin = 50.0
    x0, x1 = true_pos[:, 0].min() - margin, true_pos[:, 0].max() + margin
    y0, y1 = true_pos[:, 1].min() - margin, true_pos[:, 1].max() + margin
    axes[1].set_aspect("auto")
    axes[1].set_xlim(x0, x1)
    axes[1].set_ylim(y0, y1)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("Zoomed to ground-truth footprint")

    fig.suptitle("Top-down trajectory comparison")
    _save(fig, save_path)


def plot_error_with_bounds(time: np.ndarray, error: np.ndarray, sigma: np.ndarray,
                            labels: list[str], title: str, ylabel: str, save_path):
    """3 stacked subplots of `error[:, i]` with +/- 3*sigma[:, i] bounds."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(time, error[:, i], color="#1f77b4", lw=0.9, label="error")
        ax.fill_between(time, -3 * sigma[:, i], 3 * sigma[:, i], color="#1f77b4", alpha=0.2, label=r"$\pm 3\sigma$")
        ax.set_ylabel(f"{labels[i]}\n[{ylabel}]")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(title)
    _save(fig, save_path)


def plot_bias_estimation(time: np.ndarray, ba_est: np.ndarray, bg_est: np.ndarray,
                          ba_true: np.ndarray, bg_true: np.ndarray, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    axis_names = ["x", "y", "z"]
    for i in range(3):
        axes[0, i].plot(time, ba_true[:, i], color="#111111", lw=1.5, label="true")
        axes[0, i].plot(time, ba_est[:, i], color="#d62728", lw=1.0, ls="--", label="estimated")
        axes[0, i].set_title(f"accel bias {axis_names[i]}")
        axes[0, i].grid(alpha=0.3)
        axes[1, i].plot(time, bg_true[:, i], color="#111111", lw=1.5, label="true")
        axes[1, i].plot(time, bg_est[:, i], color="#d62728", lw=1.0, ls="--", label="estimated")
        axes[1, i].set_title(f"gyro bias {axis_names[i]}")
        axes[1, i].grid(alpha=0.3)
        axes[1, i].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("m/s^2")
    axes[1, 0].set_ylabel("rad/s")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("IMU bias estimation")
    _save(fig, save_path)


def plot_nees(time: np.ndarray, nees: np.ndarray, dof: int, save_path):
    from scipy.stats import chi2
    lower, upper = chi2.ppf(0.025, dof), chi2.ppf(0.975, dof)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(time, nees, color="#1f77b4", lw=0.8, label="NEES")
    ax.axhline(dof, color="#111111", ls="-", lw=1.0, label=f"E[NEES] = {dof}")
    ax.axhline(lower, color="#d62728", ls="--", lw=1.0, label="95% chi2 bounds")
    ax.axhline(upper, color="#d62728", ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("NEES")
    ax.set_title("Normalized Estimation Error Squared (filter consistency)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, save_path)


def plot_doppler_residuals(times: np.ndarray, residuals: np.ndarray, is_inlier: np.ndarray,
                            sigma_d: float, save_path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    # Outliers are sparse but high-magnitude and would visually bury the dense
    # near-zero inlier band if drawn on top, so draw them first, small and faint.
    ax.scatter(times[~is_inlier], residuals[~is_inlier], s=5, color="#d62728", alpha=0.25,
               linewidths=0, label="outlier (RANSAC-rejected)", rasterized=True)
    ax.scatter(times[is_inlier], residuals[is_inlier], s=3, color="#2ca02c", alpha=0.6,
               linewidths=0, label="inlier (used)", rasterized=True)
    ax.axhline(3 * sigma_d, color="#111111", ls="--", lw=1.0, label=r"$\pm 3\sigma$")
    ax.axhline(-3 * sigma_d, color="#111111", ls="--", lw=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Doppler residual [m/s]")
    ax.set_title("Radar Doppler innovations (moving-object / clutter outliers in red)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, save_path)


def plot_gnss_dropout(time: np.ndarray, true_pos: np.ndarray, est_pos: np.ndarray,
                       dropout_window: tuple[float, float], save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(true_pos[:, 0], true_pos[:, 1], color=COLORS["truth"], lw=2.0, label="Ground truth")
    ax.plot(est_pos[:, 0], est_pos[:, 1], color=COLORS["gnss_dropout"], lw=1.4, label="Radar+Camera+GNSS estimate")
    mask = (time >= dropout_window[0]) & (time <= dropout_window[1])
    ax.plot(est_pos[mask, 0], est_pos[mask, 1], color="#d62728", lw=2.2, label="During GNSS dropout")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_title("Trajectory during GNSS dropout")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    err = np.linalg.norm(est_pos - true_pos, axis=1)
    ax.plot(time, err, color=COLORS["gnss_dropout"], lw=1.2)
    ax.axvspan(dropout_window[0], dropout_window[1], color="#d62728", alpha=0.15, label="GNSS dropout window")
    ax.set_xlabel("time [s]"); ax.set_ylabel("position error [m]")
    ax.set_title("Graceful degradation: position error through the dropout")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    _save(fig, save_path)


def plot_polar_vs_cartesian_noise(range_m: float, sigma_r: float, sigma_az: float, sigma_el: float, save_path):
    """Illustrate why radar noise is modeled in polar coordinates: a diagonal
    polar covariance maps to a rotated, range-dependent ("banana-shaped")
    ellipse in Cartesian, elongated across-range (azimuth uncertainty scales
    with range) while staying tight along-range (range uncertainty is
    constant). At range=50m with sigma_az=1.5deg, the cross-range 1-sigma
    extent is ~1.3m versus ~0.05m along range -- a >25x anisotropy that a
    single scalar Cartesian sigma could never represent.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    n_std = 3
    az0 = 0.0
    x0, y0 = range_m * np.cos(az0), range_m * np.sin(az0)

    # --- Cartesian view: the true (nonlinear) image of the polar 3-sigma box.
    # Mapping a rectangle through x=r*cos(az), y=r*sin(az) bends its two
    # constant-azimuth edges into arcs, producing a curved "banana" -- the
    # linearized (Jacobian-propagated) ellipse is a flat-sliver approximation
    # of this that only agrees locally.
    r_lo, r_hi = range_m - n_std * sigma_r, range_m + n_std * sigma_r
    az_lo, az_hi = az0 - n_std * sigma_az, az0 + n_std * sigma_az
    top = np.stack([r_hi * np.cos(np.linspace(az_lo, az_hi, 100)), r_hi * np.sin(np.linspace(az_lo, az_hi, 100))], axis=1)
    right = np.stack([np.linspace(r_hi, r_lo, 20) * np.cos(az_hi), np.linspace(r_hi, r_lo, 20) * np.sin(az_hi)], axis=1)
    bottom = np.stack([r_lo * np.cos(np.linspace(az_hi, az_lo, 100)), r_lo * np.sin(np.linspace(az_hi, az_lo, 100))], axis=1)
    left = np.stack([np.linspace(r_lo, r_hi, 20) * np.cos(az_lo), np.linspace(r_lo, r_hi, 20) * np.sin(az_lo)], axis=1)
    banana = np.concatenate([top, right, bottom, left])

    ax = axes[0]
    ax.fill(banana[:, 0], banana[:, 1], facecolor="#1f77b4", alpha=0.4, edgecolor="#1f77b4", lw=1.5,
            label=f"{n_std}$\\sigma$ region (Cartesian, exact)")
    ax.scatter([0], [0], color="#111111", marker="^", s=80, label="radar")
    ax.scatter([x0], [y0], color="#111111", s=30, zorder=5, label="target")
    ax.plot([0, x0], [0, y0], color="#999999", lw=0.8, ls=":")
    span = max(banana[:, 1].max() - banana[:, 1].min(), banana[:, 0].max() - banana[:, 0].min()) * 0.7 + 2
    ax.set_xlim(x0 - span, x0 + span)
    ax.set_ylim(y0 - span, y0 + span)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Cartesian: banana-shaped, range-dependent region\n(range={range_m:.0f} m)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # --- Polar view: the same uncertainty is a simple axis-aligned rectangle ---
    ax = axes[1]
    az_deg_sigma = np.degrees(sigma_az)
    rect_r = [range_m - 3 * sigma_r, range_m + 3 * sigma_r, range_m + 3 * sigma_r, range_m - 3 * sigma_r, range_m - 3 * sigma_r]
    rect_az = [-3 * az_deg_sigma, -3 * az_deg_sigma, 3 * az_deg_sigma, 3 * az_deg_sigma, -3 * az_deg_sigma]
    ax.plot(rect_az, rect_r, color="#2ca02c", lw=2.0)
    ax.fill(rect_az, rect_r, color="#2ca02c", alpha=0.3, label=r"$3\sigma$ region (polar, diagonal $R$)")
    ax.scatter([0], [range_m], color="#111111", s=30, zorder=5, label="target")
    ax.set_xlabel("azimuth [deg]"); ax.set_ylabel("range [m]")
    ax.set_title("Polar: axis-aligned, diagonal covariance\n(natural radar noise model)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle("Why radar measurement models use polar coordinates")
    _save(fig, save_path)
