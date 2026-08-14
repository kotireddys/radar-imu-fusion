"""GNSS position simulator, used for comparison/fallback and the dropout demo.

    p_meas = p_vehicle + R_body_to_world @ t_body_to_gnss + noise,  noise ~ N(0, diag(sigma_h^2, sigma_h^2, sigma_v^2))

Not part of the explicit project file tree in the spec but required by the
GNSS position update / dropout scenarios, so it lives alongside the other
sensor simulators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.trajectory import Trajectory
from src.utils.rotations import quat_to_rot


@dataclass
class GNSSMeasurements:
    time: np.ndarray       # (N,)
    position: np.ndarray   # (N,3) measured antenna position, world frame
    rate_hz: float


def simulate_gnss(traj: Trajectory, params: dict, seed: int = 0,
                   dropout_window: tuple[float, float] | None = None) -> GNSSMeasurements:
    """Simulate GNSS fixes at `params['rate_hz']`.

    If `dropout_window = (t0, t1)` is given, fixes within that window are
    dropped entirely (simulates e.g. a tunnel / urban canyon).
    """
    rng = np.random.default_rng(seed)
    rate = params["rate_hz"]
    dt = 1.0 / rate
    frame_times = np.arange(0.0, traj.time[-1], dt)

    t_bg = np.array(params["t_body_to_gnss"], dtype=float)
    sigma_h = params["sigma_horizontal_m"]
    sigma_v = params["sigma_vertical_m"]

    if dropout_window is not None:
        t0, t1 = dropout_window
        frame_times = frame_times[(frame_times < t0) | (frame_times > t1)]

    n = len(frame_times)
    positions = np.zeros((n, 3))
    for k, t in enumerate(frame_times):
        idx = traj.index_at(t)
        R_wb = quat_to_rot(traj.quaternion[idx])
        true_pos = traj.position[idx] + R_wb @ t_bg
        noise = np.array([
            rng.normal(0, sigma_h), rng.normal(0, sigma_h), rng.normal(0, sigma_v),
        ])
        positions[k] = true_pos + noise

    return GNSSMeasurements(time=frame_times, position=positions, rate_hz=rate)
