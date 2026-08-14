"""Ground-truth 3D vehicle trajectory generator.

The trajectory is built from smoothed speed and path-curvature profiles
(Gaussian-smoothed piecewise-constant targets). Gaussian smoothing is
infinitely differentiable, which guarantees a continuous (in fact smooth)
acceleration and yaw-rate history -- exactly the "no discontinuous jerks"
requirement for a believable vehicle trajectory. Elevation is a slow
sinusoid superimposed on the horizontal path. Orientation is chosen so the
body x-axis (forward) points along the instantaneous velocity vector
(coordinated motion, no sideslip) with roll held at zero, since a ground
vehicle yaws and pitches over hills but does not bank like an aircraft.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class Trajectory:
    """Ground-truth trajectory sampled at `rate_hz`."""

    time: np.ndarray              # (N,)
    position: np.ndarray          # (N,3) world ENU [m]
    velocity: np.ndarray          # (N,3) world ENU [m/s]
    acceleration: np.ndarray      # (N,3) body-frame specific force incl. gravity [m/s^2]
    quaternion: np.ndarray        # (N,4) [w,x,y,z], rotates body -> world
    angular_velocity: np.ndarray  # (N,3) body frame [rad/s]
    rate_hz: float

    def index_at(self, t: float) -> int:
        return int(np.clip(round(t * self.rate_hz), 0, len(self.time) - 1))


def _piecewise_constant(t: np.ndarray, segments: list[tuple[float, float, float]]) -> np.ndarray:
    """Evaluate a piecewise-constant target defined by [(t0, t1, value), ...]."""
    out = np.zeros_like(t)
    for t0, t1, val in segments:
        out[(t >= t0) & (t < t1)] = val
    if segments:
        out[t >= segments[-1][1] - 1e-9] = segments[-1][2]
    return out


def _batch_quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Vectorized quaternion [w,x,y,z] (N,4) -> rotation matrices (N,3,3)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = q.shape[0]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _batch_log_so3(R: np.ndarray) -> np.ndarray:
    """Vectorized SO(3) log map: rotation matrices (N,3,3) -> axis-angle (N,3)."""
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    wx = R[:, 2, 1] - R[:, 1, 2]
    wy = R[:, 0, 2] - R[:, 2, 0]
    wz = R[:, 1, 0] - R[:, 0, 1]
    sin_theta = np.sin(theta)
    safe_sin = np.where(sin_theta == 0.0, 1.0, sin_theta)
    factor = theta / (2.0 * safe_sin)
    return factor[:, None] * np.stack([wx, wy, wz], axis=1)


def _batch_euler_to_quat(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Vectorized ZYX Euler angles -> quaternion [w,x,y,z]."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.stack([w, x, y, z], axis=1)


def generate_trajectory(params: dict, seed: int = 42) -> Trajectory:
    """Generate a smooth 3D ground-truth vehicle trajectory from `params`
    (the `trajectory:` block of the sim config).
    """
    duration = params["duration_s"]
    rate = params["rate_hz"]
    dt = 1.0 / rate
    n = int(round(duration * rate)) + 1
    t = np.arange(n) * dt

    cruise = params["cruise_speed_mps"]

    # --- speed profile: accelerate, cruise, slow for tight turn, re-accelerate ---
    speed_segments = [
        (0.0, 5.0, 0.0),
        (5.0, 70.0, cruise),
        (70.0, 95.0, 0.55 * cruise),
        (95.0, duration + 1.0, cruise),
    ]
    speed_raw = _piecewise_constant(t, speed_segments)
    speed = gaussian_filter1d(speed_raw, sigma=2.5 * rate, mode="nearest")

    # --- path curvature profile: straight -> left turn -> straight -> right turn -> straight -> gentle left ---
    radii = params["turn_radii_m"]
    curvature_segments = [
        (0.0, 35.0, 0.0),
        (35.0, 55.0, 1.0 / radii[1]),
        (55.0, 80.0, 0.0),
        (80.0, 95.0, -1.0 / radii[2]),
        (95.0, 105.0, 0.0),
        (105.0, 118.0, 1.0 / radii[3]),
        (118.0, duration + 1.0, 0.0),
    ]
    curvature_raw = _piecewise_constant(t, curvature_segments)
    curvature = gaussian_filter1d(curvature_raw, sigma=1.5 * rate, mode="nearest")

    yaw_rate = speed * curvature
    yaw = np.concatenate(([0.0], np.cumsum(0.5 * (yaw_rate[1:] + yaw_rate[:-1]) * dt)))

    vx = speed * np.cos(yaw)
    vy = speed * np.sin(yaw)

    # --- elevation: slow sinusoid, amplitude/2 peak deviation each way ---
    z_amplitude = params["elevation_amplitude_m"] / 2.0
    z_period = duration * 0.85
    z = z_amplitude * np.sin(2 * np.pi * t / z_period)
    vz = z_amplitude * (2 * np.pi / z_period) * np.cos(2 * np.pi * t / z_period)

    velocity = np.stack([vx, vy, vz], axis=1)
    position = np.zeros((n, 3))
    position[1:] = np.cumsum(0.5 * (velocity[1:] + velocity[:-1]) * dt, axis=0)

    acceleration_world = np.gradient(velocity, dt, axis=0, edge_order=2)

    horizontal_speed = np.hypot(vx, vy)
    pitch = np.arctan2(vz, np.maximum(horizontal_speed, 1e-6))
    roll = np.zeros(n)

    quaternion = _batch_euler_to_quat(roll, pitch, yaw)
    R = _batch_quat_to_rot(quaternion)

    gravity = np.array([0.0, 0.0, -params["gravity_mps2"]])
    specific_force_world = acceleration_world - gravity
    acceleration_body = np.einsum("nji,nj->ni", R, specific_force_world)

    angular_velocity = np.zeros((n, 3))
    R_rel = np.einsum("nji,njk->nik", R[:-2], R[2:])  # R(t-dt)^T @ R(t+dt)
    angular_velocity[1:-1] = _batch_log_so3(R_rel) / (2 * dt)
    angular_velocity[0] = angular_velocity[1]
    angular_velocity[-1] = angular_velocity[-2]

    return Trajectory(
        time=t,
        position=position,
        velocity=velocity,
        acceleration=acceleration_body,
        quaternion=quaternion,
        angular_velocity=angular_velocity,
        rate_hz=rate,
    )
