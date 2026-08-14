"""Forward-facing FMCW radar simulator.

Ground-truth per detection (radar frame, body-aligned unless configured
otherwise)::

    p_radar   = R_body_to_radar @ (R_world_to_body @ (p_landmark - p_vehicle)) + t_body_to_radar
    range     = ||p_radar||
    azimuth   = atan2(p_radar_y, p_radar_x)
    elevation = asin(p_radar_z / range)

Doppler (radial velocity) is frame-invariant under pure rotation, so it is
computed directly in the world frame, ignoring the (small) lever-arm offset
of the radar from the vehicle origin, exactly as the simplified EKF
measurement model does::

    e_radial = (p_landmark - p_vehicle) / ||p_landmark - p_vehicle||
    doppler  = dot(e_radial, v_target_world - v_vehicle_world)

For static landmarks (v_target = 0) this reduces to ``doppler = -dot(e_radial, v_vehicle)``.
Moving objects violate this ego-motion-only model -- that is the point: the
ESKF's Doppler RANSAC must reject them as outliers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.trajectory import Trajectory
from src.utils.rotations import quat_to_rot


@dataclass
class RadarDetection:
    r: float            # range [m]
    azimuth: float       # [rad]
    elevation: float      # [rad]
    doppler: float        # radial velocity [m/s]
    landmark_id: int      # >=0 static landmark id, -1 clutter, >=1000 moving object
    is_dynamic: bool      # True for moving-object detections (ground-truth label)
    is_clutter: bool       # True for injected clutter (ground-truth label)


@dataclass
class RadarFrame:
    time: float
    detections: list[RadarDetection] = field(default_factory=list)


@dataclass
class MovingObject:
    start_pos: np.ndarray
    velocity: np.ndarray
    t0: float

    def position(self, t: float) -> np.ndarray:
        return self.start_pos + self.velocity * (t - self.t0)


def _generate_landmarks(traj: Trajectory, n: int, rng: np.random.Generator) -> np.ndarray:
    """Scatter landmarks near the ego corridor (buildings, signs, guardrails)."""
    sample_idx = rng.integers(0, len(traj.time), size=n)
    anchors = traj.position[sample_idx]
    lateral = rng.uniform(-150.0, 150.0, size=n)
    along = rng.uniform(-20.0, 20.0, size=n)
    heights = rng.uniform(0.0, 8.0, size=n)
    landmarks = anchors.copy()
    landmarks[:, 0] += lateral
    landmarks[:, 1] += along
    landmarks[:, 2] = heights
    return landmarks


def _generate_moving_objects(traj: Trajectory, n: int, rng: np.random.Generator) -> list[MovingObject]:
    objects = []
    for _ in range(n):
        t0 = rng.uniform(10.0, traj.time[-1] - 10.0)
        idx0 = traj.index_at(t0)
        ego_pos = traj.position[idx0]
        lateral_offset = rng.choice([-1.0, 1.0]) * rng.uniform(15.0, 40.0)
        along_offset = rng.uniform(-60.0, 60.0)
        start_pos = ego_pos + np.array([lateral_offset, along_offset, 0.0])
        speed = rng.uniform(8.0, 22.0)
        heading = rng.uniform(0, 2 * np.pi)
        velocity = speed * np.array([np.cos(heading), np.sin(heading), 0.0])
        objects.append(MovingObject(start_pos=start_pos, velocity=velocity, t0=t0))
    return objects


def simulate_radar(traj: Trajectory, params: dict, seed: int = 0) -> tuple[list[RadarFrame], np.ndarray, list[MovingObject]]:
    """Simulate radar frames at `params['rate_hz']`.

    Returns (frames, static_landmarks, moving_objects) -- the landmarks and
    moving objects are also returned so `run_fusion.py` / the camera
    simulator can share the same world.
    """
    rng = np.random.default_rng(seed)
    rate = params["rate_hz"]
    dt = 1.0 / rate
    frame_times = np.arange(0.0, traj.time[-1], dt)

    landmarks = _generate_landmarks(traj, params["n_static_landmarks"], rng)
    moving_objects = _generate_moving_objects(traj, params["n_moving_objects"], rng)

    R_br = np.array(params["R_body_to_radar"], dtype=float)
    t_br = np.array(params["t_body_to_radar"], dtype=float)
    fov_az = np.radians(params["fov_azimuth_deg"])
    fov_el = np.radians(params["fov_elevation_deg"])
    r_min, r_max = params["range_min_m"], params["range_max_m"]
    sigma_r = params["sigma_range_m"]
    sigma_az = np.radians(params["sigma_azimuth_deg"])
    sigma_el = np.radians(params["sigma_elevation_deg"])
    sigma_d = params["sigma_doppler_mps"]
    p_clutter = params["clutter_probability"]
    p_miss = params["missed_detection_probability"]

    frames: list[RadarFrame] = []

    for t in frame_times:
        idx = traj.index_at(t)
        ego_pos = traj.position[idx]
        ego_vel = traj.velocity[idx]
        R_wb = quat_to_rot(traj.quaternion[idx])
        R_bw = R_wb.T

        # --- candidate targets: static landmarks + moving objects ---
        targets_world = list(landmarks)
        targets_vel = [np.zeros(3)] * len(landmarks)
        targets_id = list(range(len(landmarks)))
        targets_dynamic = [False] * len(landmarks)
        for mi, obj in enumerate(moving_objects):
            targets_world.append(obj.position(t))
            targets_vel.append(obj.velocity)
            targets_id.append(1000 + mi)
            targets_dynamic.append(True)

        targets_world = np.array(targets_world)
        targets_vel = np.array(targets_vel)

        rel_world = targets_world - ego_pos
        rel_body = rel_world @ R_bw.T
        p_radar = rel_body @ R_br.T + t_br

        rng_meas_true = np.linalg.norm(p_radar, axis=1)
        rng_meas_true = np.maximum(rng_meas_true, 1e-6)
        az_true = np.arctan2(p_radar[:, 1], p_radar[:, 0])
        el_true = np.arcsin(np.clip(p_radar[:, 2] / rng_meas_true, -1.0, 1.0))

        visible = (
            (rng_meas_true >= r_min) & (rng_meas_true <= r_max)
            & (np.abs(az_true) <= fov_az) & (np.abs(el_true) <= fov_el)
        )

        world_range = np.maximum(np.linalg.norm(rel_world, axis=1), 1e-6)
        e_radial_world = rel_world / world_range[:, None]
        doppler_true = np.einsum("ij,ij->i", e_radial_world, targets_vel - ego_vel)

        frame = RadarFrame(time=float(t))
        vis_indices = np.nonzero(visible)[0]
        for i in vis_indices:
            if rng.uniform() < p_miss:
                continue
            is_clutter = rng.uniform() < p_clutter
            if is_clutter:
                r = rng.uniform(r_min, r_max)
                az = rng.uniform(-fov_az, fov_az)
                el = rng.uniform(-fov_el, fov_el)
                d = rng.uniform(-20.0, 20.0)
                det = RadarDetection(r, az, el, d, -1, False, True)
            else:
                r = rng_meas_true[i] + rng.normal(0, sigma_r)
                az = az_true[i] + rng.normal(0, sigma_az)
                el = el_true[i] + rng.normal(0, sigma_el)
                d = doppler_true[i] + rng.normal(0, sigma_d)
                det = RadarDetection(r, az, el, d, int(targets_id[i]), bool(targets_dynamic[i]), False)
            frame.detections.append(det)
        frames.append(frame)

    return frames, landmarks, moving_objects
