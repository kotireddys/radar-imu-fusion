import numpy as np

from src.filters.eskf import ESKF, NominalState, compute_F_jacobian, compute_Q_discrete, build_initial_covariance
from src.filters.measurement_models import gnss_position_model
from src.utils.rotations import quat_identity

G = 9.81


def _make_filter(P_scale=1.0):
    state0 = NominalState(p=np.zeros(3), v=np.zeros(3), q=quat_identity(), ba=np.zeros(3), bg=np.zeros(3))
    P0 = np.eye(15) * P_scale
    return ESKF(state0, P0, gravity_mps2=G, sigma_a=0.1, sigma_g=0.01, sigma_ba=0.001, sigma_bg=0.0001)


def test_predict_stationary_imu_keeps_state_at_rest():
    """An IMU at rest measures accel=[0,0,g] (specific force cancels gravity)
    and gyro=0; the nominal state should not move.
    """
    eskf = _make_filter()
    accel_meas = np.array([0.0, 0.0, G])
    gyro_meas = np.zeros(3)
    for _ in range(100):
        eskf.predict(accel_meas, gyro_meas, dt=0.005)
    assert np.allclose(eskf.state.p, np.zeros(3), atol=1e-8)
    assert np.allclose(eskf.state.v, np.zeros(3), atol=1e-8)
    assert np.allclose(eskf.state.q, quat_identity(), atol=1e-8)


def test_predict_free_fall_accelerates_downward():
    """Zero specific force (accel_meas=0, e.g. free fall) should integrate to
    v_z = -g*t under gravity.
    """
    eskf = _make_filter()
    dt = 0.005
    for _ in range(200):
        eskf.predict(np.zeros(3), np.zeros(3), dt)
    t = 200 * dt
    assert np.isclose(eskf.state.v[2], -G * t, atol=1e-6)


def test_predict_covariance_grows_monotonically_without_updates():
    eskf = _make_filter(P_scale=1e-6)
    prev_trace = np.trace(eskf.P)
    for _ in range(50):
        eskf.predict(np.array([0.0, 0.0, G]), np.zeros(3), dt=0.005)
        new_trace = np.trace(eskf.P)
        assert new_trace >= prev_trace
        prev_trace = new_trace


def test_F_jacobian_is_identity_plus_small_perturbation_at_low_dt():
    R = np.eye(3)
    F = compute_F_jacobian(R, a_unbiased=np.array([1.0, 0.0, 0.0]), omega_unbiased=np.zeros(3), dt=1e-6)
    assert np.allclose(F, np.eye(15), atol=1e-4)


def test_F_jacobian_position_velocity_coupling():
    R = np.eye(3)
    dt = 0.01
    F = compute_F_jacobian(R, a_unbiased=np.zeros(3), omega_unbiased=np.zeros(3), dt=dt)
    assert np.allclose(F[0:3, 3:6], np.eye(3) * dt)


def test_Q_discrete_is_positive_semidefinite_and_diagonal():
    Q = compute_Q_discrete(sigma_a=0.1, sigma_g=0.01, sigma_ba=0.001, sigma_bg=0.0001, dt=0.005)
    assert np.all(np.diag(Q) >= 0)
    assert np.allclose(Q, np.diag(np.diag(Q)))


def test_gnss_update_reduces_uncertainty():
    eskf = _make_filter(P_scale=1.0)
    trace_before = np.trace(eskf.P)
    H, innovation, R_meas = gnss_position_model(eskf.state, z_meas=np.zeros(3), t_bg=np.zeros(3),
                                                  sigma_h=1.5, sigma_v=3.0)
    eskf.update(H, innovation, R_meas)
    trace_after = np.trace(eskf.P)
    assert trace_after < trace_before


def test_gnss_update_pulls_position_toward_measurement():
    eskf = _make_filter(P_scale=1.0)
    eskf.state.p = np.array([10.0, -5.0, 2.0])
    z_meas = np.zeros(3)
    H, innovation, R_meas = gnss_position_model(eskf.state, z_meas, t_bg=np.zeros(3), sigma_h=1.5, sigma_v=3.0)
    eskf.update(H, innovation, R_meas)
    # position should move toward 0 but with these near-equal (P vs R) magnitudes,
    # not overshoot past it
    assert np.all(np.abs(eskf.state.p) < np.abs(np.array([10.0, -5.0, 2.0])))


def test_update_resets_error_state_covariance_stays_psd():
    eskf = _make_filter(P_scale=1.0)
    H, innovation, R_meas = gnss_position_model(eskf.state, z_meas=np.array([1.0, 2.0, 3.0]),
                                                  t_bg=np.zeros(3), sigma_h=1.5, sigma_v=3.0)
    eskf.update(H, innovation, R_meas)
    eigvals = np.linalg.eigvalsh(eskf.P)
    assert np.all(eigvals > -1e-9)


def test_build_initial_covariance_matches_config_stds():
    params = dict(init_std_pos_m=0.5, init_std_vel_mps=0.1, init_std_theta_rad=0.05,
                  init_std_bias_accel=0.05, init_std_bias_gyro=0.005)
    P0 = build_initial_covariance(params)
    assert np.isclose(P0[0, 0], 0.5 ** 2)
    assert np.isclose(P0[3, 3], 0.1 ** 2)
    assert np.isclose(P0[6, 6], 0.05 ** 2)
    assert np.isclose(P0[9, 9], 0.05 ** 2)
    assert np.isclose(P0[12, 12], 0.005 ** 2)
