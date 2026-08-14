"""SO(3) exponential / logarithm maps and small-angle quaternion utilities.

These implement the Lie-group machinery used by the ESKF to move between the
tangent-space rotation error ``delta_theta in R^3`` and the manifold
(quaternion / rotation matrix), following Sola, "Quaternion kinematics for
the error-state Kalman filter" (2017).
"""
from __future__ import annotations

import numpy as np

from src.utils.rotations import quat_mul, quat_normalize, quat_to_rot, skew

_EPS = 1e-8


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """so(3) -> SO(3). Rodrigues' formula: rotation matrix for axis-angle phi."""
    theta = np.linalg.norm(phi)
    K = skew(phi)
    if theta < _EPS:
        return np.eye(3) + K + 0.5 * K @ K
    axis = phi / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """SO(3) -> so(3). Inverse Rodrigues: axis-angle vector for rotation matrix R."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < _EPS:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    w_hat = (R - R.T) / (2 * np.sin(theta))
    w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])
    return theta * w


def exp_quat(phi: np.ndarray) -> np.ndarray:
    """so(3) -> unit quaternion [w,x,y,z] via the quaternion exponential map.

    For a rotation vector phi = theta * axis:
        q = [cos(theta/2), axis * sin(theta/2)]
    """
    theta = np.linalg.norm(phi)
    if theta < _EPS:
        # First-order Taylor expansion, numerically stable near identity.
        return quat_normalize(np.array([1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]]))
    axis = phi / theta
    half = theta / 2.0
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def log_quat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w,x,y,z] -> rotation vector phi in so(3)."""
    q = quat_normalize(q)
    w = np.clip(q[0], -1.0, 1.0)
    xyz = q[1:4]
    n = np.linalg.norm(xyz)
    if n < _EPS:
        return 2.0 * xyz  # sign(w) ~ 1 near identity
    angle = 2.0 * np.arctan2(n, w)
    return angle * (xyz / n)


def inject_rotation_error(q: np.ndarray, delta_theta: np.ndarray) -> np.ndarray:
    """Inject a tangent-space rotation error into a nominal quaternion.

    q_new = q ⊗ exp(delta_theta)   (right/local perturbation, ESKF convention)
    """
    dq = exp_quat(delta_theta)
    return quat_normalize(quat_mul(q, dq))


def rotation_error(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Small-angle rotation error vector between two quaternions (world-frame axis-angle
    such that R_true ~= exp(err) @ R_est).
    """
    R_true = quat_to_rot(q_true)
    R_est = quat_to_rot(q_est)
    return log_so3(R_true @ R_est.T)
