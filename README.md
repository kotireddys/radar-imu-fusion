# radar-imu-fusion

Radar-centric sensor fusion for autonomous vehicle state estimation: a from-scratch
**Error-State Kalman Filter (ESKF)** fusing **radar Doppler ego-velocity + radar
landmark detections + IMU + camera bearings**, with GNSS available as a fallback /
comparison sensor. Pure Python/NumPy, no ROS dependency.

This is a portfolio project built to demonstrate the specific skill set that matters
for radar-first autonomy stacks: polar measurement models, Doppler-based ego-motion
with RANSAC outlier rejection, error-state (manifold) filtering on SO(3), and honest
filter-consistency evaluation (NEES).

## Why radar-centric?

Cameras lose bearing precision with range and fail in fog/rain/darkness. LiDAR is
expensive and also degrades in weather. Radar is comparatively cheap, sees through
weather, and — critically — measures **Doppler (radial velocity) directly**, which
means a single radar frame with enough static detections can solve for the vehicle's
full 3D ego-velocity without any position fix at all. That's a fundamentally
different (and complementary) source of information than what a camera or GNSS
provides, and it's the centerpiece of this project.

## Architecture

```
                     ┌──────────────────────┐
                     │  Ground-truth 3D      │
                     │  trajectory (1 kHz)   │
                     └──────────┬───────────┘
                                │
      ┌────────────┬───────────┼───────────┬─────────────┐
      ▼             ▼           ▼           ▼             ▼
  ┌───────┐    ┌────────-─┐ ┌─────────┐ ┌─────────┐   ┌───────-──┐
  │  IMU  │    │  Radar   │ │ Camera  │ │  GNSS   │   │ (dropout │
  │200 Hz │    │  13 Hz   │ │  30 Hz  │ │  5 Hz   │   │ scenario)│
  └───┬───┘    └────┬─────┘ └────┬────┘ └────┬────┘   └────────-─┘
      │             │            │           │
      │ predict     │ update     │ update    │ update
      │ (propagate) │ (Doppler   │ (bearing, │ (position,
      │             │  RANSAC +  │  known    │  lever arm)
      │             │  landmark, │  landmark)│
      │             │  polar)    │           │
      ▼             ▼            ▼           ▼
   ┌─────────────────────────────────────────────--──┐
   │                Error-State KF                   │
   │  nominal: p, v, q, b_a, b_g   (16)              │
   │  error:   δp, δv, δθ, δb_a, δb_g  (15, tangent)│
   └───────────────────-┬─────────────────────────────┘
                        ▼
              State estimate + covariance
                        ▼
          results/*.png  (error, NEES, residuals, ...)
```

## Mathematical formulation

**Nominal state** (16 elements): `x = [p(3), v(3), q(4), b_a(3), b_g(3)]`, world-frame
position/velocity, body→world unit quaternion `[w,x,y,z]`, and accel/gyro biases.

**Error state** (15 elements, tangent space): `δx = [δp(3), δv(3), δθ(3), δb_a(3), δb_g(3)]`.
`δθ` is a rotation-vector (so(3)) perturbation, not a quaternion difference:
`R_true ≈ R_nominal · Exp(δθ)` (Solà 2017).

**Prediction** (IMU, 200 Hz):

```
a_world = R·(a_meas − b_a) + g
p ← p + v·dt + ½·a_world·dt²
v ← v + a_world·dt
q ← q ⊗ Exp((ω_meas − b_g)·dt)

F = I + Fx·dt,   Fx = [ [0,  I,  0,      0,  0],
                        [0,  0,  −R[a×], −R, 0],
                        [0,  0,  −[ω×],  0, −I],
                        [0,  0,  0,      0,  0],
                        [0,  0,  0,      0,  0]]
P ← F·P·Fᵀ + Q_d
```

**Update** (generic, any sensor): given Jacobian `H` (m×15) and innovation
`y = z − h(x̂)`,

```
S = H·P·Hᵀ + R
K = P·Hᵀ·S⁻¹
δx = K·y
P ← (I−KH)·P·(I−KH)ᵀ + K·R·Kᵀ      (Joseph form)
```

**Injection + reset**: `p ← p+δp`, `v ← v+δv`, `q ← q ⊗ Exp(δθ)`, `b ← b+δb`,
then `δx ← 0` and `P ← G·P·Gᵀ` with `G = I − blkdiag(0, 0, [δθ/2×], 0, 0)`.

### Measurement models (`src/filters/measurement_models.py`)

| Sensor | z | h(x) | Notes |
|---|---|---|---|
| Radar Doppler | `d_i` per detection | `−êᵢᵀ·R_br·Rᵀ·v` | RANSAC first rejects moving-object/clutter detections; only inliers enter the stacked update. `H` includes the `δθ` cross-term since `R` appears in `h`. |
| Radar landmark | `[range, az, el]` | polar transform of a known landmark | Kept in **polar**, not Cartesian — see below. |
| Camera bearing | `[u, v]` pixels | pinhole projection | Standard `[fx/z, 0, −fx·x/z²; ...]` Jacobian. |
| GNSS | `p_meas` | `p + R·t_bg` | Lever-arm-corrected absolute position. |

## Why polar measurement models for radar?

Radar directly measures range, azimuth, and Doppler — noise is naturally **diagonal
in polar coordinates** (`σ_r ≈ 0.05 m`, `σ_az ≈ 1.5°`). Converting a detection to
Cartesian `(x, y, z)` and then treating the noise as circular/diagonal there is
wrong: azimuth uncertainty scales *with range* (arc length = `r·σ_az`), producing a
**range-dependent, rotated ("banana-shaped") ellipse** in Cartesian — tight along the
line of sight, loose across it. At 50 m with `σ_az = 1.5°`, that's ~5 cm along range
vs. ~1.3 m across range: a 25× anisotropy a single Cartesian covariance can't
represent cleanly. Working directly in polar keeps `R_meas` exactly diagonal and the
Jacobian is a standard, well-conditioned Cartesian→polar transform. See
`results/09_polar_vs_cartesian_noise.png`.

## An honest finding: Doppler alone can't observe heading

Running the `radar_doppler`-only scenario surfaces something worth stating plainly:
**a single Doppler radar frame constrains only the body-frame velocity `R^T·v`, never
`R` and `v` separately.** For *any* rotation error `δθ`, choosing
`δv = −R·[R^Tv ×]·δθ` leaves every Doppler residual unchanged — an exact gauge
freedom, not just "weak" observability. Concretely: a body-mounted Doppler sensor is
a very good speedometer, but a speedometer alone cannot tell you which way you're
facing.

The consequence shows up directly in the results: `radar_doppler` yaw error grows to
~1.7 rad over 120 s (see `results/04_orientation_error_radar_doppler.png`), and the
NEES plot for that scenario (`results/06_nees_radar_doppler.png`) is badly
inconsistent — the covariance collapses along the unobservable direction, a
well-documented EKF pathology for gauge-symmetric measurements (see Huang/Mourikis/
Roumeliotis on observability-constrained EKF for the visual-inertial analogue). This
is *expected*, physically correct, and exactly why real Doppler-radar systems always
pair it with an absolute heading/position reference. The moment landmark position
fixes are added (`radar_full`), the gauge freedom is broken, yaw locks on, and both
accuracy and NEES become well-behaved — see the comparison in the results table
below.

## Project structure

```
radar-imu-fusion/
├── config/sim_params.yaml       # every tunable parameter, with units/meaning
├── src/
│   ├── trajectory.py            # ground-truth 3D trajectory generator
│   ├── sensors/{imu,radar,camera,gnss}.py
│   ├── filters/{eskf,measurement_models,lie_group}.py
│   ├── utils/{rotations,plotting}.py
│   └── run_fusion.py            # main entry point
├── tests/                       # unit tests (pytest)
└── results/                     # generated plots (png)
```

## Usage

```bash
pip install -r requirements.txt

# Run everything (all 6 scenarios) and generate all plots into results/
python -m src.run_fusion --scenario all

# Run just one scenario
python -m src.run_fusion --scenario radar_doppler
python -m src.run_fusion --scenario gnss_dropout

# Skip plotting (just print the accuracy summary)
python -m src.run_fusion --scenario all --no-plots

# Unit tests (a globally-installed ROS `launch_testing` pytest11 plugin can break
# autoloading in some environments; disable plugin autoload if you see an
# unrelated `ModuleNotFoundError: lark` from pytest startup)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -v
```

Scenarios (`--scenario`): `imu_only`, `imu_gnss`, `radar_doppler`, `radar_full`
(Doppler + landmark, i.e. radar-only navigation), `full_fusion` (+ camera),
`gnss_dropout` (full fusion with a 40–80 s GNSS blackout), or `all`.

## Results

120 s simulated highway-like trajectory (accel/brake phases, three turns, mild
elevation change), 500 static landmarks, 4 moving objects, 5% radar clutter, 10%
missed-detection rate. All scenarios share the same IMU noise/bias realization for a
fair comparison.

| Scenario | Final pos. error | Mean pos. error | Max pos. error |
|---|---:|---:|---:|
| `imu_only` (dead reckoning) | 4488.4 m | 1429.6 m | 4488.4 m |
| `imu_gnss` | 0.66 m | 0.79 m | 1.95 m |
| `radar_doppler` only | 858.8 m | 235.1 m | 858.8 m |
| `radar_full` (Doppler + landmarks) | 0.08 m | 0.11 m | 0.58 m |
| `full_fusion` (+ camera) | 0.06 m | 0.04 m | 0.10 m |
| `gnss_dropout` (40–80 s blackout) | 0.06 m | 0.04 m | 0.10 m |

Takeaways, in order:

1. **IMU-only dead reckoning is useless past a few seconds** — 4.5 km of drift over
   120 s, as expected for MEMS-grade bias/noise with nothing to correct it
   (`results/01_trajectory_comparison.png`, `results/02_position_error_imu_only.png`).
2. **Doppler-only aiding cuts drift ~5×** (4488 m → 859 m) purely by constraining
   velocity magnitude, even though (per above) it cannot fix heading — a genuinely
   useful partial correction, and a clean illustration of the difference between
   "reduces drift" and "bounds error."
3. **Adding known-landmark radar position fixes (`radar_full`) collapses error to
   centimeters** — this is "radar-only navigation": no camera, no GNSS, just Doppler
   + polar landmark ranging.
4. **Full fusion (+ camera) is the tightest**, and critically, **survives a 40 s GNSS
   blackout with no visible degradation** (`results/08_gnss_dropout.png`) — radar and
   camera fully cover for the missing GNSS.
5. **Bias estimation converges** from a cold start (filter initialized at zero bias)
   toward the true random-walk bias trajectory within seconds
   (`results/05_bias_estimation_full_fusion.png`).
6. **RANSAC cleanly separates static-world Doppler inliers from moving-object/clutter
   outliers** — see the tight ±3σ inlier band vs. the widely-scattered rejected
   points, including two clearly visible moving-vehicle Doppler tracks, in
   `results/07_doppler_residuals_full_fusion.png`.
7. **NEES** (`results/06_nees_*.png`) is bounded and reasonable for every fully
   observable scenario, and dramatically, instructively inconsistent for
   `radar_doppler`-only — see the discussion above.

Regenerate all plots (they are not tracked as "final"/frozen — rerun any time):

```bash
python -m src.run_fusion --scenario all
```

## References

- K. Retan, *SE(3) radar odometry with Doppler and polar measurement models* (radar
  ego-motion via Doppler + RANSAC, the direct inspiration for the Doppler update here).
- J. Solà, *Quaternion kinematics for the error-state Kalman filter*, 2017 (the ESKF
  formulation, error-state Jacobians, and injection/reset used throughout).
- C. Forster, L. Carlone, F. Dellaert, D. Scaramuzza, *On-Manifold Preintegration for
  Real-Time Visual-Inertial Odometry*, IEEE T-RO 2017 (IMU noise/bias modeling and
  manifold conventions).
- G. Huang, A. Mourikis, S. Roumeliotis, *Observability-based rules for designing
  consistent EKF SLAM estimators* (background for the Doppler-only unobservability
  / NEES-inconsistency discussion above).
