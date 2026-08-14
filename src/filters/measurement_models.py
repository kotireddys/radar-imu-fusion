"""Measurement models for the ESKF: radar Doppler ego-velocity (with RANSAC
outlier rejection), radar landmark position (polar), camera bearing
(pinhole), and GNSS position.

Each `*_jacobian`/`*_update` helper returns ``(H, innovation, R_meas)`` where
``H`` is the mx15 Jacobian of the measurement w.r.t. the *error state*
``[dp, dv, dtheta, dba, dbg]``, ``innovation = z_meas - h(x_nominal)``, and
``R_meas`` is the measurement noise covariance -- ready to hand to
`ESKF.update`.

Landmark data association is assumed solved: detections carry the
ground-truth landmark id from the simulator, mirroring a system with a
working association front-end (nearest-neighbor / JCBB) -- the ESKF math is
the point of this project, not association.
"""
from __future__ import annotations

import numpy as np

from src.utils.rotations import quat_to_rot, skew


def _unit_vector(azimuth: float, elevation: float) -> np.ndarray:
    ce = np.cos(elevation)
    return np.array([ce * np.cos(azimuth), ce * np.sin(azimuth), np.sin(elevation)])


# ---------------------------------------------------------------------------
# a) Radar Doppler ego-velocity update (with RANSAC outlier rejection)
# ---------------------------------------------------------------------------

def ransac_doppler_inliers(unit_vectors: np.ndarray, doppler: np.ndarray, R_br: np.ndarray,
                            R_wb: np.ndarray, iterations: int, inlier_threshold: float,
                            rng: np.random.Generator) -> np.ndarray:
    """RANSAC over the linear ego-velocity model  d_i = A_i @ v_world,
    A_i = -e_i^T @ R_br @ R_wb^T. Returns indices of the largest consensus set.

    Moving-object detections do not fit this ego-motion-only model and are
    rejected as outliers -- that is the whole point of running RANSAC here.
    """
    n = len(doppler)
    if n < 3:
        return np.array([], dtype=int)

    A = -(unit_vectors @ R_br) @ R_wb.T  # (n,3)
    best_inliers = np.array([], dtype=int)

    for _ in range(iterations):
        sample = rng.choice(n, size=3, replace=False)
        A_s, d_s = A[sample], doppler[sample]
        if abs(np.linalg.det(A_s)) < 1e-9:
            continue
        try:
            v_cand = np.linalg.solve(A_s, d_s)
        except np.linalg.LinAlgError:
            continue
        residual = doppler - A @ v_cand
        inliers = np.nonzero(np.abs(residual) < inlier_threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    return best_inliers


def doppler_ego_velocity_model(state, detections: list, R_br: np.ndarray, sigma_doppler: float,
                                ransac_iterations: int, ransac_threshold: float, ransac_min_inliers: int,
                                rng: np.random.Generator):
    """Build the stacked Doppler update from a radar frame's detections.

    Returns (H, innovation, R_meas, inlier_mask) or None if too few inliers
    survive RANSAC to form a reliable update.
    """
    if len(detections) < 3:
        return None

    az = np.array([d.azimuth for d in detections])
    el = np.array([d.elevation for d in detections])
    doppler = np.array([d.doppler for d in detections])
    e = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)

    R_wb = quat_to_rot(state.q)  # body -> world
    inliers = ransac_doppler_inliers(e, doppler, R_br, R_wb, ransac_iterations, ransac_threshold, rng)
    if len(inliers) < ransac_min_inliers:
        return None

    e_in = e[inliers]
    d_in = doppler[inliers]
    Rt = R_wb.T
    v = state.v

    e_Rbr = e_in @ R_br                       # (n,3)
    h_pred = -e_Rbr @ (Rt @ v)                 # (n,)
    innovation = d_in - h_pred

    H = np.zeros((len(inliers), 15))
    H[:, 3:6] = -e_Rbr @ Rt                    # d h / d(dv)
    H[:, 6:9] = -e_Rbr @ skew(Rt @ v)           # d h / d(dtheta)

    R_meas = (sigma_doppler ** 2) * np.eye(len(inliers))
    return H, innovation, R_meas, inliers


# ---------------------------------------------------------------------------
# b) Radar landmark position update (polar: range, azimuth, elevation)
# ---------------------------------------------------------------------------

def radar_landmark_model(state, detection, landmark_pos: np.ndarray, R_br: np.ndarray, t_br: np.ndarray,
                          sigma_r: float, sigma_az: float, sigma_el: float):
    """Polar measurement model h(x) = [range, azimuth, elevation] of a known
    landmark, observed through the radar extrinsics.

    Polar is the natural representation here: radar noise (sigma_r, sigma_az,
    sigma_el) is diagonal in polar coordinates, but the equivalent Cartesian
    noise is a rotated, range-dependent ("banana-shaped") ellipsoid -- far
    more precise along the line of sight than across it. Working directly in
    polar keeps R_meas exactly diagonal instead of an awkward, range-varying
    dense matrix.
    """
    p, R = state.p, quat_to_rot(state.q)
    rel_body = R.T @ (landmark_pos - p)
    p_radar = R_br @ rel_body + t_br

    x, y, z = p_radar
    r = np.linalg.norm(p_radar)
    rxy = max(np.hypot(x, y), 1e-9)

    z_pred = np.array([r, np.arctan2(y, x), np.arcsin(np.clip(z / r, -1.0, 1.0))])
    z_meas = np.array([detection.r, detection.azimuth, detection.elevation])
    innovation = z_meas - z_pred
    innovation[1] = np.arctan2(np.sin(innovation[1]), np.cos(innovation[1]))  # wrap azimuth

    # dz/d(p_radar): standard Cartesian -> polar Jacobian (3x3)
    d_range = np.array([x, y, z]) / r
    d_az = np.array([-y, x, 0.0]) / (rxy ** 2)
    d_el = np.array([-x * z, -y * z, rxy ** 2]) / (r ** 2 * rxy)
    Jz = np.stack([d_range, d_az, d_el])

    d_pradar_dp = -R_br @ R.T
    d_pradar_dtheta = R_br @ skew(rel_body)

    H = np.zeros((3, 15))
    H[:, 0:3] = Jz @ d_pradar_dp
    H[:, 6:9] = Jz @ d_pradar_dtheta

    R_meas = np.diag([sigma_r ** 2, sigma_az ** 2, sigma_el ** 2])
    return H, innovation, R_meas


# ---------------------------------------------------------------------------
# c) Camera bearing (pinhole) update
# ---------------------------------------------------------------------------

def camera_bearing_model(state, detection, landmark_pos: np.ndarray, R_bc: np.ndarray, t_bc: np.ndarray,
                          fx: float, fy: float, cx: float, cy: float, sigma_uv: float):
    p, R = state.p, quat_to_rot(state.q)
    rel_body = R.T @ (landmark_pos - p)
    p_cam = R_bc @ rel_body + t_bc
    x, y, z = p_cam
    z = max(z, 1e-3)

    z_pred = np.array([fx * x / z + cx, fy * y / z + cy])
    z_meas = np.array([detection.u, detection.v])
    innovation = z_meas - z_pred

    Jproj = np.array([
        [fx / z, 0.0, -fx * x / z ** 2],
        [0.0, fy / z, -fy * y / z ** 2],
    ])
    d_pcam_dp = -R_bc @ R.T
    d_pcam_dtheta = R_bc @ skew(rel_body)

    H = np.zeros((2, 15))
    H[:, 0:3] = Jproj @ d_pcam_dp
    H[:, 6:9] = Jproj @ d_pcam_dtheta

    R_meas = (sigma_uv ** 2) * np.eye(2)
    return H, innovation, R_meas


# ---------------------------------------------------------------------------
# d) GNSS position update
# ---------------------------------------------------------------------------

def gnss_position_model(state, z_meas: np.ndarray, t_bg: np.ndarray, sigma_h: float, sigma_v: float):
    p, R = state.p, quat_to_rot(state.q)
    h = p + R @ t_bg
    innovation = z_meas - h

    H = np.zeros((3, 15))
    H[:, 0:3] = np.eye(3)
    H[:, 6:9] = -R @ skew(t_bg)

    R_meas = np.diag([sigma_h ** 2, sigma_h ** 2, sigma_v ** 2])
    return H, innovation, R_meas
