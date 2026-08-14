import numpy as np

from src.filters.lie_group import (
    exp_so3, log_so3, exp_quat, log_quat, inject_rotation_error, rotation_error,
)
from src.utils.rotations import quat_to_rot, quat_identity, quat_normalize


def test_exp_log_so3_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        phi = rng.normal(scale=0.5, size=3)
        R = exp_so3(phi)
        phi2 = log_so3(R)
        assert np.allclose(phi, phi2, atol=1e-8)


def test_exp_so3_zero_is_identity():
    assert np.allclose(exp_so3(np.zeros(3)), np.eye(3))


def test_log_so3_identity_is_zero():
    assert np.allclose(log_so3(np.eye(3)), np.zeros(3))


def test_exp_so3_is_orthonormal():
    rng = np.random.default_rng(1)
    phi = rng.normal(scale=1.0, size=3)
    R = exp_so3(phi)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_exp_quat_matches_exp_so3():
    rng = np.random.default_rng(2)
    for _ in range(20):
        phi = rng.normal(scale=0.5, size=3)
        R_from_so3 = exp_so3(phi)
        R_from_quat = quat_to_rot(exp_quat(phi))
        assert np.allclose(R_from_so3, R_from_quat, atol=1e-8)


def test_exp_log_quat_roundtrip():
    rng = np.random.default_rng(3)
    for _ in range(20):
        phi = rng.normal(scale=0.5, size=3)
        q = exp_quat(phi)
        phi2 = log_quat(q)
        assert np.allclose(phi, phi2, atol=1e-8)


def test_exp_quat_zero_is_identity():
    q = exp_quat(np.zeros(3))
    assert np.allclose(q, quat_identity())


def test_inject_rotation_error_zero_is_noop():
    rng = np.random.default_rng(4)
    q = quat_normalize(rng.normal(size=4))
    q2 = inject_rotation_error(q, np.zeros(3))
    assert np.allclose(q, q2, atol=1e-10)


def test_inject_rotation_error_matches_right_perturbation():
    rng = np.random.default_rng(5)
    q = quat_normalize(rng.normal(size=4))
    dtheta = rng.normal(scale=0.05, size=3)
    q_new = inject_rotation_error(q, dtheta)
    R_new = quat_to_rot(q_new)
    R_expected = quat_to_rot(q) @ exp_so3(dtheta)
    assert np.allclose(R_new, R_expected, atol=1e-8)


def test_rotation_error_is_zero_for_identical_quats():
    rng = np.random.default_rng(6)
    q = quat_normalize(rng.normal(size=4))
    err = rotation_error(q, q)
    assert np.allclose(err, np.zeros(3), atol=1e-8)


def test_rotation_error_recovers_known_perturbation():
    rng = np.random.default_rng(7)
    q = quat_normalize(rng.normal(size=4))
    dtheta = rng.normal(scale=0.02, size=3)
    q_true = inject_rotation_error(q, dtheta)
    # rotation_error(q_true, q_est) returns the *world-frame* (left) axis-angle
    # s.t. R_true ~= exp(err) @ R_est, whereas dtheta was injected as a
    # *body-frame* (right) perturbation R_true = R_est @ Exp(dtheta). To first
    # order these are related by err ~= R_est @ dtheta (a frame rotation), not
    # numerically equal unless R_est = I.
    err = rotation_error(q_true, q)
    expected = quat_to_rot(q) @ dtheta
    assert np.linalg.norm(err) < 0.1
    assert np.allclose(err, expected, atol=1e-3)
