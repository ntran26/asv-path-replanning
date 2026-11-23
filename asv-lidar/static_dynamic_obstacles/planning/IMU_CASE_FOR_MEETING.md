# IMU: why the vessel needs one, and the fallback

Prepared 2026-11-11 for the supervision meeting before the calibration sessions and field trials. Simplified 2026-11-23. This is open item O-2 of `FIELD_CALIBRATION_PLAN.md`.

## In short

The policy steers using the vessel's own speed and turning rate. It was trained in a simulator that assumed these come from an inertial measurement unit (IMU): a small gyroscope and accelerometer that measure turning and acceleration directly.

The vessel currently has no IMU. Its speed and turning rate can only be worked out from changes in its position between updates. That gives coarse, jumpy values, unlike anything the policy saw in training.

An IMU makes the trials a test of the policy rather than of a different sensor. It is also the only way to measure the fast parts of the vessel's behaviour in the calibration.

## Why it matters

1. **The policy expects it.** Speed and turning rate are among the policy's inputs, and they also feed its estimates of how close a target ship will pass. Replaying the July logs on the bench showed that position-derived values sit well away from what the simulator provides. Without an IMU, a field result could not be separated from this change in sensing.

2. **Faster position updates are not enough.** Updating position more often shortens the time between readings. That makes a speed worked out from them noisier, not cleaner. A gyroscope measures turning directly.

3. **The calibration needs the fast details.** Several behaviours play out within a fraction of a second:
   - the delay between a rudder command and the vessel starting to turn;
   - how quickly a turn builds up and dies away;
   - how the vessel slows and drifts during a stop.

   Position updates twice a second cannot capture these; an IMU measures them directly.

4. **Clean timing.** The present vessel model lumps communication delay, steering delay and estimation delay into one number. If the IMU, the LiDAR and the commands are logged on one clock, these can be told apart. A sharp turn of the hull gives a clear mark to line the records up.

5. **The safety layer.**
   - The emergency stop keeps reversing until the vessel is almost stationary, which needs an accurate low-speed reading.
   - The most common remaining collision in simulation is the stopped vessel drifting into the wall. Predicting that needs turning rate and sideways speed.
   - In the trials the safety layer only records what it would have done ("shadow mode"). So this affects the quality of that record, not the vessel's safety.

6. **Explaining the gap between simulation and the water.** With measured motion, a difference between field and simulated behaviour can be placed either in the vessel's dynamics or in its perception and positioning. Without it, the two are mixed together.

**A possible further benefit, not claimed:** a gyroscope steadies the heading estimate in turns. That may reduce false moving objects seen at the basin walls. The main fix for those is the surveyed map frame (P-7).

## Sampling rate: a requirement to confirm, not a certainty

The plan asks for raw gyroscope and accelerometer readings at 100 times a second (100 Hz) or more. That figure is a requirement, not a known property of the installed unit.

- **Where it comes from.** The vessel-model plan (`planning_records/05_VESSEL_MODEL_AND_SIM2REAL.md`, section 4.7) names suitable sensors that can be logged in raw form: a BNO055 in raw mode, an ICM-20948, or a Pixhawk flight-controller log.
- **Why that rate.**
  - 100 readings a second resolve timing to a hundredth of a second. That is well inside the tenth-of-a-second precision the calibration aims for, for delays and turn response.
  - The policy itself decides twice a second and needs far less.
  - The rate is set by the calibration, not by the policy.
- **Why it cannot be assumed.** The logged rate depends on three things: the sensor, its settings (raw output, not the sensor's own fused orientation, whose filtering hides the delays being measured), and how the vessel's computer reads and timestamps it. A sensor's rated rate is not the logged rate.
- **How it is confirmed.** In calibration session 1, the logged timestamps give the rate actually achieved, and any gaps. O-2 is closed only on that measurement.
- **If the rate is lower.** The rudder delay and the turn response are measured less precisely. The steady turning circle and the zig-zag overshoots are unaffected.

## Fallback if an IMU cannot be installed

1. **For the calibration only: a standalone IMU logger.** A small self-logging IMU, or a phone running a sensor-logging app, is sealed and fixed near the hull's reference point. Its clock is lined up afterwards on sharp turns at the start and end of each session. It is not used to steer.
2. **For live steering: a better position-based estimate.** Faster, finer position telemetry is filtered over the last half-second to give speed and turning rate. This is a new module in the bridge; the simulation code stays unchanged. Its noise is measured at rest and against the logger's gyroscope during the turning manoeuvres.
3. **Checked in simulation before the trials.** The policy is run on the validation set with the measured noise of that estimate.
   - If performance holds, the trials go ahead as planned.
   - If it drops, the effect is known in advance and reported as a limitation.
4. **What a position-only calibration loses.** It loses precision in the rudder delay and in how quickly turns build up. The turning circle and zig-zag results are unaffected.
5. **In the paper.** The quality of the field speed and turning-rate inputs is reported against the simulator's assumption, as part of the sim-to-field comparison.

## Paragraph for the meeting request

> The navigation policy was trained assuming the vessel measures its own turning rate and speed with an IMU. Without one, these have to be estimated from position changes, which the bench replay of the July logs shows differ markedly from what the policy was trained on. The calibration manoeuvres also need direct, high-rate measurements of turning and acceleration to capture the vessel's fast responses. The request is to fit an IMU that logs raw readings on the same clock as the LiDAR. If that is not possible, the backup is a standalone IMU logger for the calibration and an improved position-based estimate, checked in simulation before the trials.
