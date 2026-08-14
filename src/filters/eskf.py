"""Error-State Kalman Filter (ESKF) for IMU-driven vehicle state estimation.

Nominal state (16 elements)::

    x = [p(3), v(3), q(4), b_a(3), b_g(3)]

Error state (15 elements, tangent space)::

    dx = [dp(3), dv(3), dtheta(3), db_a(3), db_g(3)]

`dtheta` is a rotation-vector (so(3)) perturbation, not a quaternion
difference: the true rotation is recovered as ``R_true ~= R_nom @ Exp(dtheta)``.
See Sola (2017), "Quaternion kinematics for the error-state Kalman filter".
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.filters.lie_group import exp_quat, inject_rotation_error
from src.utils.rotations import quat_mul, quat_normalize, quat_to_rot, skew

N_ERR = 15


@dataclass
class NominalState:
    p: np.ndarray   # (3,) world position
    v: np.ndarray   # (3,) world velocity
    q: np.ndarray   # (4,) [w,x,y,z], body -> world
    ba: np.ndarray  # (3,) accel bias
    bg: np.ndarray  # (3,) gyro bias

    def copy(self) -> "NominalState":
        return replace(self, p=self.p.copy(), v=self.v.copy(), q=self.q.copy(),
                        ba=self.ba.copy(), bg=self.bg.copy())


def compute_F_jacobian(R: np.ndarray, a_unbiased: np.ndarray, omega_unbiased: np.ndarray, dt: float) -> np.ndarray:
    """Discrete error-state transition matrix F = I + Fx*dt (15x15)."""
    Fx = np.zeros((N_ERR, N_ERR))
    Fx[0:3, 3:6] = np.eye(3)
    Fx[3:6, 6:9] = -R @ skew(a_unbiased)
    Fx[3:6, 9:12] = -R
    Fx[6:9, 6:9] = -skew(omega_unbiased)
    Fx[6:9, 12:15] = -np.eye(3)
    return np.eye(N_ERR) + Fx * dt


def compute_Q_discrete(sigma_a: float, sigma_g: float, sigma_ba: float, sigma_bg: float, dt: float) -> np.ndarray:
    """Discrete-time process noise covariance (15x15), block-diagonal.

    Accel/gyro white noise enters the velocity/attitude error scaled by dt
    (it is integrated once); bias random walks accumulate variance sigma^2*dt.
    Rotating the accel noise by R does not change its covariance since R is
    orthonormal (isotropic noise), so no R dependence appears here.
    """
    Q = np.zeros((N_ERR, N_ERR))
    Q[3:6, 3:6] = (sigma_a ** 2) * (dt ** 2) * np.eye(3)
    Q[6:9, 6:9] = (sigma_g ** 2) * (dt ** 2) * np.eye(3)
    Q[9:12, 9:12] = (sigma_ba ** 2) * dt * np.eye(3)
    Q[12:15, 12:15] = (sigma_bg ** 2) * dt * np.eye(3)
    return Q


class ESKF:
    """Error-State Kalman Filter with IMU propagation and pluggable updates."""

    def __init__(self, state0: NominalState, P0: np.ndarray, gravity_mps2: float,
                 sigma_a: float, sigma_g: float, sigma_ba: float, sigma_bg: float):
        self.state = state0.copy()
        self.P = P0.copy()
        self.gravity = np.array([0.0, 0.0, -gravity_mps2])
        self.sigma_a = sigma_a
        self.sigma_g = sigma_g
        self.sigma_ba = sigma_ba
        self.sigma_bg = sigma_bg

    # ------------------------------------------------------------------
    # Prediction (IMU propagation)
    # ------------------------------------------------------------------
    def predict(self, accel_meas: np.ndarray, gyro_meas: np.ndarray, dt: float) -> None:
        R = quat_to_rot(self.state.q)
        a_unbiased = accel_meas - self.state.ba
        omega_unbiased = gyro_meas - self.state.bg

        a_world = R @ a_unbiased + self.gravity

        p_new = self.state.p + self.state.v * dt + 0.5 * a_world * dt ** 2
        v_new = self.state.v + a_world * dt
        dq = exp_quat(omega_unbiased * dt)
        q_new = quat_normalize(quat_mul(self.state.q, dq))

        self.state.p, self.state.v, self.state.q = p_new, v_new, q_new
        # ba, bg unchanged (constant in the process model; random walk lives in Q)

        F = compute_F_jacobian(R, a_unbiased, omega_unbiased, dt)
        Q = compute_Q_discrete(self.sigma_a, self.sigma_g, self.sigma_ba, self.sigma_bg, dt)
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------
    # Update (generic EKF update on the error state)
    # ------------------------------------------------------------------
    def update(self, H: np.ndarray, innovation: np.ndarray, R_meas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply a measurement update given Jacobian H (mx15), innovation
        (z - h(x_nom)), and measurement covariance R_meas (mxm). Returns
        (dx, S) -- the injected error state and innovation covariance
        (the latter useful for NEES / innovation-consistency plots).
        """
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ innovation

        I_KH = np.eye(N_ERR) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T  # Joseph form
        self._inject_and_reset(dx)
        return dx, S

    def _inject_and_reset(self, dx: np.ndarray) -> None:
        self.state.p = self.state.p + dx[0:3]
        self.state.v = self.state.v + dx[3:6]
        self.state.q = inject_rotation_error(self.state.q, dx[6:9])
        self.state.ba = self.state.ba + dx[9:12]
        self.state.bg = self.state.bg + dx[12:15]

        G = np.eye(N_ERR)
        G[6:9, 6:9] = np.eye(3) - skew(0.5 * dx[6:9])
        self.P = G @ self.P @ G.T

    # ------------------------------------------------------------------
    def covariance_diag(self) -> np.ndarray:
        return np.diag(self.P).copy()


def build_initial_covariance(params: dict) -> np.ndarray:
    """Diagonal initial error-state covariance from the `eskf:` config block."""
    P0 = np.zeros((N_ERR, N_ERR))
    P0[0:3, 0:3] = params["init_std_pos_m"] ** 2 * np.eye(3)
    P0[3:6, 3:6] = params["init_std_vel_mps"] ** 2 * np.eye(3)
    P0[6:9, 6:9] = params["init_std_theta_rad"] ** 2 * np.eye(3)
    P0[9:12, 9:12] = params["init_std_bias_accel"] ** 2 * np.eye(3)
    P0[12:15, 12:15] = params["init_std_bias_gyro"] ** 2 * np.eye(3)
    return P0
