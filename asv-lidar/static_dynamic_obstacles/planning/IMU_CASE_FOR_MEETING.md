# IMU: the case for installing it, and the fallback

Date: 2026-11-11. Preparation for the supervision meeting before the two calibration sessions
and the field trials. Open item O-2 of `FIELD_CALIBRATION_PLAN.md` ("Is the IMU installed, raw
at 100 Hz or more, on the bridge clock?"), with P-1 and P-5 of the same plan.

## 1. The case

**1. The deployed policy was trained on the assumption of an IMU.**
- Three of the policy's 70 inputs are surge, sway and yaw rate (the ego branch,
  `src/observation.py`). Own velocity also enters the course error of the path branch and the
  CPA estimates of the target and context branches.
- The frozen simulator puts 1-sigma noise of 0.05 m/s on surge and sway and 1 deg/s on yaw rate
  (`src/constants.py`, `EGO_SPEED_NOISE`, `EGO_YAW_RATE_NOISE_DPS`). The source records the
  premise: "IMU CONFIRMED (05 section 4.7) -- one will be added, logging raw gyro and
  accelerometer at 100 Hz+, time-synced to the LiDAR"; the 1 deg/s is a gyro noise floor, and
  "0.05 assumes the IMU helps".
- Without an IMU the bridge differentiates the vessel's pose, as in July: 2 Hz at 0.1 m and
  0.1 deg resolution. Surge from it is good to about 0.08 m/s (`src/constants.py`) and moves in
  steps of about 0.2 m/s (0.1 m in 0.5 s), over a third of the 0.56 m/s cruise speed.
- The bench replay of the July logs at that resolution put the ego inputs 0.11-0.16 away from
  the simulator's on average, in the policy's normalised units (`field_deployment/README_P3_BRIDGE.md`,
  open items); a 0.05 m/s error is 0.045 in those units (speed scale 1.116 m/s).
- Without an IMU, the trials therefore test the policy on an ego input outside what it was
  trained on, and a field result cannot be separated from that change of sensing.

**2. A higher pose rate alone does not fix the live estimate.** At 10 Hz (P-1) a pose difference
spans 0.1 s instead of 0.5 s, so at the same resolution the velocity estimate is about five
times noisier; the estimator has to change with the rate. A gyro measures yaw rate directly at
100 Hz or more.

**3. Calibration: the fast terms.** The fit compares simulated with logged heading and position
(`bluefin/simulate.py`), so a pose-only identification works, and offline a centred five-point
derivative of the July poses gives yaw rate to about 0.5 deg/s and surge to about 0.05 m/s,
enough to initialise a fitting window. The IMU's contribution is the fast dynamics that 2 Hz
poses cannot resolve (Nyquist limit 1 Hz):
- servo rate, lag and transport delay separated to +/-0.1 s (T-1, T-2), seen first in the yaw
  rate after a rudder step;
- yaw inertia and damping from the zig-zags (S1-E);
- yaw-rate decay and sway after a stop from a turn (S1-C2, T-13);
- surge deceleration in coast-downs and crash stops: drag (S1-C) and reverse efficiency to
  +/-0.1 (T-9).

The gyro and accelerometer add yaw rate and acceleration as measured outputs of the fit. The
steady turn rate is currently a prior, not a measurement (calibration plan section 2).

**4. Timing.** The July model's 0.73 s rudder delay is one lump of communication latency, servo
lag and estimator lag. With the IMU, LiDAR and commands on one clock (P-5), a clock offset stays
out of the fitted actuator lag; a sharp yaw impulse, which the gyro sees clearly, gives the
synchronising mark.

**5. The safety layer's physics.**
- The layer's stop holds full astern until the measured surge is below 0.05 m/s
  (`ESTOP_STOP_SPEED`, "speed-estimate noise floor"); surge from pose differences does not
  resolve that.
- The main residual collision mode in simulation is drift into the boundary after a committed
  stop: 16 of 70 own-vessel collisions on test set v4 with E1. Its prediction needs yaw rate and
  sway.
- In the trials the layer runs in shadow mode (D-4), so this bears on the quality of the shadow
  record (C-20), not on the vessel's safety.

**6. Attribution of the sim-to-field gap (C-18).** With measured ego motion, the field ego input
can be compared with the simulator's directly, so a gap can be placed in the dynamics or in
perception and localisation. Without it the two are mixed.

**7. Expected, not shown.** The pose comes from scan matching on a 2D LiDAR (RPLidar C1) with no
motion prior. A gyro steadies the heading in turns and aligns scan and pose in time, which may
reduce the phantom moving tracks at the basin wall. The registered map frame (P-7) is the main
fix; the IMU is not claimed to remove them.

## 2. Fallback if the IMU cannot be installed

1. **Calibration only: a standalone IMU logger.** A small self-logging IMU, or a phone with a
   sensor-logging application, sealed and fixed near the hull reference point, logging gyro and
   accelerometer at 100 Hz or more. Its clock is aligned afterwards on the yaw impulses at the
   start and end of each session (already in P-5). It supplies the fit's yaw rate and
   accelerations without changes to the vessel's computer; it is not used live.
2. **Live ego estimate without the IMU.** Telemetry at 10 Hz or more with finer pose resolution
   (P-1), and surge, sway and yaw rate from a filter over the last 0.5 s of poses (for example a
   constant-velocity Kalman filter), sampled at the 2 Hz decisions. This is a new bridge module;
   `src/` stays frozen. Its noise is measured at rest (S1-A) and against the logger's gyro in the
   turning circles (S1-D).
3. **Checked in simulation before the trials.** The deployed policy on the validation set with
   the measured ego noise of that estimator (the environment exposes `ego_speed_noise` and
   `ego_yaw_rate_noise_dps`). If success holds, the trials proceed as planned; if it drops, the
   expected effect is known in advance and reported as a limitation of the field input.
4. **What a pose-only calibration loses.** Turning-circle rate and radius and the zig-zag
   overshoot angles are unaffected. Weaker: the actuator delay to +/-0.1 s (T-2 may be recorded
   as not met) and yaw inertia. Reverse efficiency rests on the pose and the side camera already
   planned for S1-C2.
5. **In the paper.** The measured ego-input quality is reported against the simulator's
   assumption, as part of the sim-to-field gap.

## 3. Paragraph for the meeting request

> The deployed policy was trained assuming gyro-measured yaw rate and IMU-aided speed. Without
> an IMU these come from differentiating 2 Hz poses, which our bench replay shows differ markedly
> from the simulator's. The calibration's turning, zig-zag and stop manoeuvres also need yaw rate
> and acceleration at 100 Hz to separate the fast actuator and yaw terms. Backup: a standalone
> IMU logger for the calibration, and a higher-rate pose-based estimate checked in simulation
> first.
