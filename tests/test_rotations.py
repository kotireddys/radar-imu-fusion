import numpy as np

from src.utils.rotations import (
    skew, quat_normalize, quat_identity, quat_mul, quat_conj,
    quat_to_rot, rot_to_quat, euler_to_quat, quat_to_euler,
)


def test_skew_matches_cross_product():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(skew(a) @ b, np.cross(a, b))


def test_skew_is_antisymmetric():
    a = np.array([1.0, 2.0, 3.0])
    S = skew(a)
    assert np.allclose(S, -S.T)


def test_quat_identity_is_no_rotation():
    q = quat_identity()
    R = quat_to_rot(q)
    assert np.allclose(R, np.eye(3))


def test_quat_mul_with_identity():
    rng = np.random.default_rng(1)
    q = quat_normalize(rng.normal(size=4))
    e = quat_identity()
    assert np.allclose(quat_mul(q, e), q)
    assert np.allclose(quat_mul(e, q), q)


def test_quat_conj_is_inverse_for_unit_quat():
    rng = np.random.default_rng(2)
    q = quat_normalize(rng.normal(size=4))
    result = quat_mul(q, quat_conj(q))
    assert np.allclose(result, quat_identity(), atol=1e-10)


def test_quat_rot_roundtrip():
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = quat_normalize(rng.normal(size=4))
        if q[0] < 0:
            q = -q  # canonical hemisphere for comparison
        R = quat_to_rot(q)
        q2 = rot_to_quat(R)
        if q2[0] < 0:
            q2 = -q2
        assert np.allclose(q, q2, atol=1e-8)


def test_rotation_matrix_is_orthonormal():
    rng = np.random.default_rng(4)
    q = quat_normalize(rng.normal(size=4))
    R = quat_to_rot(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_euler_quat_roundtrip_small_angles():
    rng = np.random.default_rng(5)
    for _ in range(20):
        roll, pitch, yaw = rng.uniform(-1.0, 1.0, size=3)  # avoid gimbal lock at +-pi/2
        q = euler_to_quat(roll, pitch, yaw)
        r2, p2, y2 = quat_to_euler(q)
        assert np.allclose([roll, pitch, yaw], [r2, p2, y2], atol=1e-8)


def test_composed_rotation_matches_quat_product():
    rng = np.random.default_rng(6)
    q1 = quat_normalize(rng.normal(size=4))
    q2 = quat_normalize(rng.normal(size=4))
    R1, R2 = quat_to_rot(q1), quat_to_rot(q2)
    R12 = quat_to_rot(quat_mul(q1, q2))
    assert np.allclose(R1 @ R2, R12, atol=1e-8)
