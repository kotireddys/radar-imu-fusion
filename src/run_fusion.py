"""Main entry point: simulate all sensors once, run the ESKF over one or more
scenarios (different sensor combinations), and generate all result plots.

Usage
-----
    python -m src.run_fusion --scenario all
    python -m src.run_fusion --scenario radar_doppler
    python -m src.run_fusion --scenario gnss_dropout
"""
from __future__ import annotations

import argparse
import time as pytime
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from src.trajectory import generate_trajectory, Trajectory
from src.sensors.imu import simulate_imu, IMUMeasurements
from src.sensors.radar import simulate_radar
from src.sensors.camera import simulate_camera
from src.sensors.gnss import simulate_gnss, GNSSMeasurements
from src.filters.eskf import ESKF, NominalState, build_initial_covariance
from src.filters.lie_group import inject_rotation_error, log_so3
from src.filters.measurement_models import (
    doppler_ego_velocity_model,
    radar_landmark_model,
    camera_bearing_model,
    gnss_position_model,
)
from src.utils.rotations import quat_to_rot
from src.utils import plotting

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

SCENARIOS = {
    "imu_only": dict(sensors=set(), gnss_dropout=False,
                      desc="IMU dead-reckoning baseline (no corrections)"),
    "imu_gnss": dict(sensors={"gnss"}, gnss_dropout=False,
                      desc="IMU + GNSS (standard fusion, for comparison)"),
    "radar_doppler": dict(sensors={"radar_doppler"}, gnss_dropout=False,
                           desc="IMU + radar Doppler ego-velocity only"),
    "radar_full": dict(sensors={"radar_doppler", "radar_landmark"}, gnss_dropout=False,
                        desc="IMU + radar Doppler + radar landmark positions (radar-only nav)"),
    "full_fusion": dict(sensors={"radar_doppler", "radar_landmark", "camera"}, gnss_dropout=False,
                         desc="IMU + radar + camera"),
    "gnss_dropout": dict(sensors={"radar_doppler", "radar_landmark", "camera", "gnss"}, gnss_dropout=True,
                          desc="IMU + radar + camera + GNSS, with GNSS dropout t=40-80s"),
}


@dataclass
class FilterHistory:
    time: np.ndarray = None
    p_est: np.ndarray = None
    v_est: np.ndarray = None
    q_est: np.ndarray = None
    ba_est: np.ndarray = None
    bg_est: np.ndarray = None
    p_true: np.ndarray = None
    v_true: np.ndarray = None
    q_true: np.ndarray = None
    ba_true: np.ndarray = None
    bg_true: np.ndarray = None
    sigma_p: np.ndarray = None
    sigma_v: np.ndarray = None
    sigma_theta: np.ndarray = None
    sigma_ba: np.ndarray = None
    sigma_bg: np.ndarray = None
    nees: np.ndarray = None
    doppler_time: np.ndarray = None
    doppler_residual: np.ndarray = None
    doppler_is_inlier: np.ndarray = None
    n_gnss_updates: int = 0
    n_radar_doppler_updates: int = 0
    n_radar_landmark_updates: int = 0
    n_camera_updates: int = 0


def _doppler_all_residuals(state: NominalState, detections: list, R_br: np.ndarray, inlier_idx: np.ndarray):
    """Residuals (z - h(x)) for *every* detection in a frame (not just RANSAC
    inliers), used purely for the diagnostic residual scatter plot.
    """
    az = np.array([d.azimuth for d in detections])
    el = np.array([d.elevation for d in detections])
    doppler = np.array([d.doppler for d in detections])
    e = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    R_wb = quat_to_rot(state.q)
    h_pred = -(e @ R_br) @ (R_wb.T @ state.v)
    residual = doppler - h_pred
    is_inlier = np.zeros(len(detections), dtype=bool)
    is_inlier[inlier_idx] = True
    return residual, is_inlier


def run_scenario(name: str, traj: Trajectory, imu: IMUMeasurements, radar_frames: list,
                  landmarks: np.ndarray, camera_frames: list, gnss: GNSSMeasurements,
                  cfg: dict, seed: int = 0) -> FilterHistory:
    """Drive the ESKF over the full trajectory for one sensor combination."""
    scenario = SCENARIOS[name]
    sensors = scenario["sensors"]

    init_rng = np.random.default_rng(cfg["seed"])
    p0 = traj.position[0] + init_rng.normal(0, cfg["eskf"]["init_std_pos_m"], 3)
    v0 = traj.velocity[0] + init_rng.normal(0, cfg["eskf"]["init_std_vel_mps"], 3)
    theta0 = init_rng.normal(0, cfg["eskf"]["init_std_theta_rad"], 3)
    q0 = inject_rotation_error(traj.quaternion[0], theta0)
    state0 = NominalState(p=p0, v=v0, q=q0, ba=np.zeros(3), bg=np.zeros(3))
    P0 = build_initial_covariance(cfg["eskf"])

    eskf = ESKF(state0, P0, cfg["trajectory"]["gravity_mps2"],
                cfg["imu"]["accel_noise_density"], cfg["imu"]["gyro_noise_density"],
                cfg["imu"]["accel_bias_random_walk"], cfg["imu"]["gyro_bias_random_walk"])

    rng = np.random.default_rng(seed + 1000)
    dt = 1.0 / imu.rate_hz
    R_br = np.array(cfg["radar"]["R_body_to_radar"], dtype=float)
    t_br = np.array(cfg["radar"]["t_body_to_radar"], dtype=float)
    R_bc = np.array(cfg["camera"]["R_body_to_cam"], dtype=float)
    t_bc = np.array(cfg["camera"]["t_body_to_cam"], dtype=float)
    t_bg = np.array(cfg["gnss"]["t_body_to_gnss"], dtype=float)
    cam_p = cfg["camera"]
    max_cam = cfg["eskf"]["max_camera_features_per_frame"]
    max_radar_lm = cfg["eskf"]["max_radar_landmark_updates_per_frame"]

    n = len(imu.time)
    hist = FilterHistory(
        time=imu.time.copy(),
        p_est=np.zeros((n, 3)), v_est=np.zeros((n, 3)), q_est=np.zeros((n, 4)),
        ba_est=np.zeros((n, 3)), bg_est=np.zeros((n, 3)),
        p_true=np.zeros((n, 3)), v_true=np.zeros((n, 3)), q_true=np.zeros((n, 4)),
        ba_true=imu.true_accel_bias.copy(), bg_true=imu.true_gyro_bias.copy(),
        sigma_p=np.zeros((n, 3)), sigma_v=np.zeros((n, 3)), sigma_theta=np.zeros((n, 3)),
        sigma_ba=np.zeros((n, 3)), sigma_bg=np.zeros((n, 3)),
        nees=np.zeros(n),
    )
    doppler_times, doppler_residuals, doppler_inliers = [], [], []

    radar_ptr = camera_ptr = gnss_ptr = 0

    for k in range(n):
        t = imu.time[k]
        eskf.predict(imu.accel[k], imu.gyro[k], dt)

        if "radar_doppler" in sensors or "radar_landmark" in sensors:
            while radar_ptr < len(radar_frames) and radar_frames[radar_ptr].time <= t:
                frame = radar_frames[radar_ptr]
                inlier_idx = np.array([], dtype=int)

                if "radar_doppler" in sensors and len(frame.detections) >= 3:
                    result = doppler_ego_velocity_model(
                        eskf.state, frame.detections, R_br, cfg["radar"]["sigma_doppler_mps"],
                        cfg["eskf"]["ransac_iterations"], cfg["eskf"]["ransac_inlier_threshold_mps"],
                        cfg["eskf"]["ransac_min_inliers"], rng)
                    inlier_idx = result[3] if result is not None else np.array([], dtype=int)

                    # Innovation (z - h(x_nominal)) recorded *before* the update is applied.
                    residual, is_inlier = _doppler_all_residuals(eskf.state, frame.detections, R_br, inlier_idx)
                    doppler_times.append(np.full(len(frame.detections), t))
                    doppler_residuals.append(residual)
                    doppler_inliers.append(is_inlier)

                    if result is not None:
                        H, innovation, R_meas, _ = result
                        eskf.update(H, innovation, R_meas)
                        hist.n_radar_doppler_updates += 1

                if "radar_landmark" in sensors:
                    static_dets = [d for d in frame.detections if not d.is_clutter and not d.is_dynamic]
                    if len(static_dets) > max_radar_lm:
                        idx_sel = rng.choice(len(static_dets), size=max_radar_lm, replace=False)
                        static_dets = [static_dets[i] for i in idx_sel]
                    for det in static_dets:
                        H, innovation, R_meas = radar_landmark_model(
                            eskf.state, det, landmarks[det.landmark_id], R_br, t_br,
                            cfg["radar"]["sigma_range_m"], np.radians(cfg["radar"]["sigma_azimuth_deg"]),
                            np.radians(cfg["radar"]["sigma_elevation_deg"]))
                        eskf.update(H, innovation, R_meas)
                        hist.n_radar_landmark_updates += 1

                radar_ptr += 1

        if "camera" in sensors:
            while camera_ptr < len(camera_frames) and camera_frames[camera_ptr].time <= t:
                frame = camera_frames[camera_ptr]
                dets = frame.detections
                if len(dets) > max_cam:
                    idx_sel = rng.choice(len(dets), size=max_cam, replace=False)
                    dets = [dets[i] for i in idx_sel]
                for det in dets:
                    H, innovation, R_meas = camera_bearing_model(
                        eskf.state, det, landmarks[det.landmark_id], R_bc, t_bc,
                        cam_p["fx"], cam_p["fy"], cam_p["cx"], cam_p["cy"], cam_p["sigma_pixel"])
                    eskf.update(H, innovation, R_meas)
                    hist.n_camera_updates += 1
                camera_ptr += 1

        if "gnss" in sensors:
            while gnss_ptr < len(gnss.time) and gnss.time[gnss_ptr] <= t:
                H, innovation, R_meas = gnss_position_model(
                    eskf.state, gnss.position[gnss_ptr], t_bg,
                    cfg["gnss"]["sigma_horizontal_m"], cfg["gnss"]["sigma_vertical_m"])
                eskf.update(H, innovation, R_meas)
                hist.n_gnss_updates += 1
                gnss_ptr += 1

        idx_gt = traj.index_at(t)
        hist.p_est[k], hist.v_est[k], hist.q_est[k] = eskf.state.p, eskf.state.v, eskf.state.q
        hist.ba_est[k], hist.bg_est[k] = eskf.state.ba, eskf.state.bg
        hist.p_true[k], hist.v_true[k], hist.q_true[k] = traj.position[idx_gt], traj.velocity[idx_gt], traj.quaternion[idx_gt]

        P_diag = eskf.covariance_diag()
        hist.sigma_p[k] = np.sqrt(P_diag[0:3])
        hist.sigma_v[k] = np.sqrt(P_diag[3:6])
        hist.sigma_theta[k] = np.sqrt(P_diag[6:9])
        hist.sigma_ba[k] = np.sqrt(P_diag[9:12])
        hist.sigma_bg[k] = np.sqrt(P_diag[12:15])

        R_true = quat_to_rot(traj.quaternion[idx_gt])
        R_est = quat_to_rot(eskf.state.q)
        err = np.concatenate([
            hist.p_est[k] - hist.p_true[k],
            hist.v_est[k] - hist.v_true[k],
            log_so3(R_est.T @ R_true),
            eskf.state.ba - imu.true_accel_bias[k],
            eskf.state.bg - imu.true_gyro_bias[k],
        ])
        try:
            hist.nees[k] = err @ np.linalg.solve(eskf.P, err)
        except np.linalg.LinAlgError:
            hist.nees[k] = np.nan

    if doppler_times:
        hist.doppler_time = np.concatenate(doppler_times)
        hist.doppler_residual = np.concatenate(doppler_residuals)
        hist.doppler_is_inlier = np.concatenate(doppler_inliers)
    else:
        hist.doppler_time = np.array([])
        hist.doppler_residual = np.array([])
        hist.doppler_is_inlier = np.array([], dtype=bool)

    return hist


def _pos_error_summary(hist: FilterHistory) -> str:
    err = np.linalg.norm(hist.p_est - hist.p_true, axis=1)
    return f"final pos err={err[-1]:6.2f} m | mean={err.mean():6.2f} m | max={err.max():7.2f} m"


def generate_all_sensor_data(cfg: dict):
    traj = generate_trajectory(cfg["trajectory"], seed=cfg["seed"])
    imu = simulate_imu(traj, cfg["imu"], seed=cfg["seed"])
    radar_frames, landmarks, moving_objects = simulate_radar(traj, cfg["radar"], seed=cfg["seed"])
    camera_frames = simulate_camera(traj, landmarks, cfg["camera"], seed=cfg["seed"])
    gnss_full = simulate_gnss(traj, cfg["gnss"], seed=cfg["seed"])
    gnss_dropout = simulate_gnss(traj, cfg["gnss"], seed=cfg["seed"],
                                  dropout_window=(cfg["gnss"]["dropout_start_s"], cfg["gnss"]["dropout_end_s"]))
    return traj, imu, radar_frames, landmarks, moving_objects, camera_frames, gnss_full, gnss_dropout


def main():
    parser = argparse.ArgumentParser(description="Radar-IMU-Camera ESKF sensor fusion demo")
    parser.add_argument("--scenario", default="all", choices=list(SCENARIOS.keys()) + ["all"])
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "sim_params.yaml"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Generating trajectory and sensor data ...")
    t0 = pytime.time()
    traj, imu, radar_frames, landmarks, moving_objects, camera_frames, gnss_full, gnss_dropout = generate_all_sensor_data(cfg)
    print(f"  done in {pytime.time() - t0:.1f}s | traj={len(traj.time)} imu={len(imu.time)} "
          f"radar_frames={len(radar_frames)} camera_frames={len(camera_frames)} landmarks={len(landmarks)}")

    scenario_names = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    histories: dict[str, FilterHistory] = {}
    for name in scenario_names:
        gnss = gnss_dropout if SCENARIOS[name]["gnss_dropout"] else gnss_full
        print(f"Running scenario '{name}': {SCENARIOS[name]['desc']}")
        t0 = pytime.time()
        hist = run_scenario(name, traj, imu, radar_frames, landmarks, camera_frames, gnss, cfg, seed=cfg["seed"])
        histories[name] = hist
        print(f"  {_pos_error_summary(hist)} | updates: gnss={hist.n_gnss_updates} "
              f"doppler={hist.n_radar_doppler_updates} landmark={hist.n_radar_landmark_updates} "
              f"camera={hist.n_camera_updates} | {pytime.time() - t0:.1f}s")

    if args.no_plots:
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    print("Generating plots ...")

    # 1) trajectory comparison across the classic three (if available)
    compare = {k: histories[k].p_est for k in ["imu_only", "imu_gnss", "radar_doppler"] if k in histories}
    if compare:
        plotting.plot_trajectory_comparison(traj.position, compare, RESULTS_DIR / "01_trajectory_comparison.png")

    # 2-6, 7: per-scenario diagnostic plots for every scenario that ran
    for name, hist in histories.items():
        pos_err = hist.p_est - hist.p_true
        vel_err = hist.v_est - hist.v_true
        theta_err = np.array([log_so3(quat_to_rot(hist.q_est[k]).T @ quat_to_rot(hist.q_true[k])) for k in range(len(hist.time))])

        plotting.plot_error_with_bounds(hist.time, pos_err, hist.sigma_p, AXIS := ["x", "y", "z"],
                                         f"Position error ({name})", "m", RESULTS_DIR / f"02_position_error_{name}.png")
        plotting.plot_error_with_bounds(hist.time, vel_err, hist.sigma_v, AXIS,
                                         f"Velocity error ({name})", "m/s", RESULTS_DIR / f"03_velocity_error_{name}.png")
        plotting.plot_error_with_bounds(hist.time, theta_err, hist.sigma_theta, ["roll", "pitch", "yaw"],
                                         f"Orientation error ({name})", "rad", RESULTS_DIR / f"04_orientation_error_{name}.png")
        plotting.plot_bias_estimation(hist.time, hist.ba_est, hist.bg_est, hist.ba_true, hist.bg_true,
                                       RESULTS_DIR / f"05_bias_estimation_{name}.png")
        plotting.plot_nees(hist.time, hist.nees, dof=15, save_path=RESULTS_DIR / f"06_nees_{name}.png")

        if len(hist.doppler_time):
            plotting.plot_doppler_residuals(hist.doppler_time, hist.doppler_residual, hist.doppler_is_inlier,
                                             cfg["radar"]["sigma_doppler_mps"], RESULTS_DIR / f"07_doppler_residuals_{name}.png")

    # 8) GNSS dropout scenario
    if "gnss_dropout" in histories:
        hist = histories["gnss_dropout"]
        plotting.plot_gnss_dropout(hist.time, hist.p_true, hist.p_est,
                                    (cfg["gnss"]["dropout_start_s"], cfg["gnss"]["dropout_end_s"]),
                                    RESULTS_DIR / "08_gnss_dropout.png")

    # 9) polar vs Cartesian radar noise illustration (standalone)
    plotting.plot_polar_vs_cartesian_noise(50.0, cfg["radar"]["sigma_range_m"],
                                            np.radians(cfg["radar"]["sigma_azimuth_deg"]),
                                            np.radians(cfg["radar"]["sigma_elevation_deg"]),
                                            RESULTS_DIR / "09_polar_vs_cartesian_noise.png")

    print(f"Plots written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
