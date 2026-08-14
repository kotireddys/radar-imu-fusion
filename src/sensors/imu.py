"""IMU simulator: accelerometer + gyroscope with bias random walk and white noise.

    accel_meas = R_body_to_world.T @ (accel_world - gravity) + b_a + noise_a
    gyro_meas  = omega_body + b_g + noise_g

Since `Trajectory.acceleration` already stores the true body-frame specific
force ``R^T @ (a_world - g)`` (see trajectory.py), the noiseless accelerometer
signal is simply that array resampled at the IMU rate; the gyro signal is
likewise `Trajectory.angular_velocity` resampled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.trajectory import Trajectory


@dataclass
class IMUMeasurements:
    time: np.ndarray               # (N,)
    accel: np.ndarray              # (N,3) measured specific force, body frame [m/s^2]
    gyro: np.ndarray               # (N,3) measured angular velocity, body frame [rad/s]
    true_accel_bias: np.ndarray    # (N,3) ground-truth bias, for evaluation only
    true_gyro_bias: np.ndarray     # (N,3)
    rate_hz: float


def simulate_imu(traj: Trajectory, params: dict, seed: int = 0) -> IMUMeasurements:
    """Sample the trajectory at `params['rate_hz']` and add IMU bias + noise."""
    rng = np.random.default_rng(seed)
    rate = params["rate_hz"]
    step = max(1, int(round(traj.rate_hz / rate)))
    idx = np.arange(0, len(traj.time), step)
    n = len(idx)
    dt = 1.0 / rate

    sigma_a = params["accel_noise_density"]
    sigma_g = params["gyro_noise_density"]
    sigma_ba = params["accel_bias_random_walk"]
    sigma_bg = params["gyro_bias_random_walk"]

    accel_bias = np.zeros((n, 3))
    gyro_bias = np.zeros((n, 3))
    accel_bias[0] = np.array(params["init_accel_bias"])
    gyro_bias[0] = np.array(params["init_gyro_bias"])
    ba_steps = sigma_ba * np.sqrt(dt) * rng.standard_normal((n - 1, 3))
    bg_steps = sigma_bg * np.sqrt(dt) * rng.standard_normal((n - 1, 3))
    accel_bias[1:] = accel_bias[0] + np.cumsum(ba_steps, axis=0)
    gyro_bias[1:] = gyro_bias[0] + np.cumsum(bg_steps, axis=0)

    accel_noise = sigma_a * rng.standard_normal((n, 3))
    gyro_noise = sigma_g * rng.standard_normal((n, 3))

    accel_true = traj.acceleration[idx]
    gyro_true = traj.angular_velocity[idx]

    accel_meas = accel_true + accel_bias + accel_noise
    gyro_meas = gyro_true + gyro_bias + gyro_noise

    return IMUMeasurements(
        time=traj.time[idx],
        accel=accel_meas,
        gyro=gyro_meas,
        true_accel_bias=accel_bias,
        true_gyro_bias=gyro_bias,
        rate_hz=rate,
    )
