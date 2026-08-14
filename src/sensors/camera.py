"""Forward-facing pinhole camera simulator (bearing-only landmark observations).

    p_cam = R_body_to_cam @ R_world_to_body @ (p_landmark - p_vehicle) + t_body_to_cam
    u = fx * p_cam_x / p_cam_z + cx
    v = fy * p_cam_y / p_cam_z + cy

Points are kept only if in front of the camera (p_cam_z > 0) and inside the
image bounds; pixel noise sigma_uv is added on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.trajectory import Trajectory
from src.utils.rotations import quat_to_rot


@dataclass
class CameraDetection:
    u: float
    v: float
    landmark_id: int


@dataclass
class CameraFrame:
    time: float
    detections: list[CameraDetection] = field(default_factory=list)


def simulate_camera(traj: Trajectory, landmarks: np.ndarray, params: dict, seed: int = 0) -> list[CameraFrame]:
    """Simulate camera frames at `params['rate_hz']` observing `landmarks`."""
    rng = np.random.default_rng(seed)
    rate = params["rate_hz"]
    dt = 1.0 / rate
    frame_times = np.arange(0.0, traj.time[-1], dt)

    R_bc = np.array(params["R_body_to_cam"], dtype=float)
    t_bc = np.array(params["t_body_to_cam"], dtype=float)
    fx, fy = params["fx"], params["fy"]
    cx, cy = params["cx"], params["cy"]
    width, height = params["image_width"], params["image_height"]
    sigma_uv = params["sigma_pixel"]

    frames: list[CameraFrame] = []
    for t in frame_times:
        idx = traj.index_at(t)
        ego_pos = traj.position[idx]
        R_bw = quat_to_rot(traj.quaternion[idx]).T

        rel_body = (landmarks - ego_pos) @ R_bw.T
        p_cam = rel_body @ R_bc.T + t_bc

        in_front = p_cam[:, 2] > 0.1
        z = np.where(in_front, p_cam[:, 2], 1.0)
        u = fx * p_cam[:, 0] / z + cx
        v = fy * p_cam[:, 1] / z + cy

        in_bounds = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)

        frame = CameraFrame(time=float(t))
        for i in np.nonzero(in_bounds)[0]:
            um = u[i] + rng.normal(0, sigma_uv)
            vm = v[i] + rng.normal(0, sigma_uv)
            frame.detections.append(CameraDetection(um, vm, int(i)))
        frames.append(frame)

    return frames
