# 05_VESSEL_MODEL_AND_SIM2REAL — Part 2: basin test plan

Two sessions, which is the hard constraint. Session 1 identifies, session 2
validates. **Session 2 must not become a second identification session** — with
no independent check, the model's accuracy claim rests on the same data that
produced it, which is the Reviewer 1.4 concession from Paper 2 repeating itself.
The contingency ordering in §6 exists so that a bad session 1 does not silently
consume session 2.

Everything here is driven by what the July logs cannot reach. Part 1 identified
the vessel model from 11 minutes of field data and reached 6/6 holdout runs on
the paired heading test, but three things remain assumptions:

| gap | current state | fixed by |
|---|---|---|
| sustained-turn behaviour | **prior, not measured** — turn rate, radius and speed loss at held helm are imposed bands | S1-D turning circles |
| thrust vs RPM | `T12` anchored at one operating point; square law assumed | S1-C RPM sweep |
| actuator decomposition | `rud_rate`, `rud_tau`, `rud_delay` trade off freely; 0.73 s is an effective lump | S1-B servo tests |

---

## 1. Pre-session work (no basin time)

Do all of this before booking. Basin time is the scarce resource; every item
below is something that, if discovered on the day, costs a block.

**P-1 Logging rate.** Raise the telemetry and log rate as far as the vessel
scheduler allows, at minimum for session 1. The servo tests (S1-B) cannot
separate rate limit from delay at 2 Hz — Nyquist is 1 Hz and the servo transient
is expected to be faster than that. Target ≥ 10 Hz for pose and command; if the
vessel cannot, see P-3. **Decide and test this on the bench, not in the basin.**

**P-2 Do not hard-code 2 Hz.** Every script should derive dt from timestamps and
assert against the `#CONFIG` `dt_cmd` value. Session 1 deliberately runs at a
different rate; a hard-coded constant corrupts exactly that data. (The July logs
are a consistent 2 Hz — median 0.5000 s, sd 1.0 ms, zero dropped pose frames
across 1,074 intervals — but that is a configuration, not a constant.)

**P-3 Independent rudder measurement.** The transmitted command is known; the
*actual deflection* is not, and that is the whole point of S1-B. If there is no
rudder position feedback, mount a phone at 60 fps viewing a protractor scale
taped to the rudder post, with a synchronising event visible to both streams (a
bright LED flash triggered from the bridge, or the rudder slammed to a hard
stop). This costs nothing and is decisive. **Without it, S1-B produces another
effective lumped delay and part 1's limitation stands.**

**P-4 Rate-limiter switch.** The bridge's 50 %/s command limiter must be
disableable by a config flag, not a code edit. S1-B needs it off; every other
block needs it on (it is part of the deployment plant — see part 1 §9). Log the
flag state in `#CONFIG`.

**P-5 Clock discipline.** IMU (100 Hz+), LiDAR and command must land on one
clock, or carry a measurable offset. A constant offset appears in the fit as
actuator lag, which is the parameter S1-B exists to measure — so this is not a
tidiness issue. Log a stationary period for gyro bias and a sharp yaw impulse
visible in both streams at the start and end of every session.

**P-6 Pre-register the analysis.** Per the Paper 2 lesson: write the empty
tables, the metric definitions and the acceptance thresholds (§5) *before* the
session, and commit the fit script that will be run. Session 1's mid-session
check (S1-F) executes that committed script unchanged.

**P-7 Landmark survey plan.** Mark and measure the fixed features visible in the
basin video — recessed doorways, signage panels, the continuous aluminium deck
rail, benches, the raised platform. These are the scan-to-map registration
targets. Do not survey anything on the far end where furniture moves between
sessions.

---

## 2. Session 1 — identification (~4 h)

Ordered by irreversibility: the blocks that cannot be recovered from logs come
first, and the priority-1 block (D) sits before the discretionary ones.

### S1-A Instrumentation and sensing baseline (25 min)

1. Vessel stationary, held at a known pose, 5 min continuous log.
2. Sharp manual yaw impulse, then stationary again.
3. Repeat at three positions: mid-basin, near the light wall, **near the matte
   black wall**.

Yields: gyro bias; localisation drift rate while stationary; LiDAR return count
per revolution and range noise; **return rate vs bearing against the black wall**
— the open item that the July logs could not settle because heading coverage was
too narrow to separate vessel-fixed occlusion from wall-fixed reflectivity.

Also record: the aft self-occlusion sector, and whether the suspension lines
produce 1–2 beam returns that survive the clustering threshold.

### S1-B Actuator identification (35 min) — **limiter OFF**

Vessel restrained or in open water with propulsion off, rudder visible to the
P-3 camera.

- Step inputs: 0 → ±25 %, ±50 %, ±100 %, and ±100 % → ∓100 % reversals. 5 repeats
  each.
- One slow ramp at 10 %/s as a linearity check.

Yields: servo rate limit, first-order lag and transport delay **separately**.
This is the only block where the limiter is off.

### S1-C Surge identification (45 min) — limiter ON

- Acceleration from rest, straight, at RPM command {6, 9, 12, 15, 18, 21, 24}.
  2 runs each, run until speed is visibly settled.
- Coast-down from settled speed at RPM {9, 15, 21}: cut propulsion, log until
  near stationary. 3 runs each.

Coast-down is the cleanest drag measurement available — no thrust term to trade
off against. Together with the accel runs this separates thrust from drag, which
the July data (18 of 20 runs at a single RPM) cannot do at all.

### S1-D Turning circles (70 min) — **priority 1**

- Helm {25, 50, 75, 100} %, **both signs**, at RPM 12. Minimum 2 full
  revolutions each, entered from settled straight-line speed.
- Repeat helm {50, 100} % at RPM 18.

Yields: steady turn rate, turning radius, speed loss in the turn, and the
advance/transfer/tactical diameter set. This replaces the imposed prior bands in
part 1 §7 with measurements, and is the single most valuable block in either
session. Both signs matter — the fitted model has a drift-moment term whose
asymmetry is currently unconstrained.

Basin size will limit this. If a full circle does not fit, log the largest arc
achievable and record the constraint; a settled arc of 180° still yields steady
r and u.

### S1-E Zig-zag (40 min) — limiter ON

- 10°/10° and 20°/20° zig-zags, both initial directions, RPM 12, 3 repeats each.

Classic manoeuvring identification, directly comparable to the literature, and
the manoeuvre part 1's own notes flagged as the priority for fitting heading
directly. Overshoot angles are a standard reportable quantity.

### S1-F Mid-session check (25 min) — **do not skip**

Run the P-6 committed script on the data just collected, in the basin, before
packing up. Check the §5 acceptance thresholds. Anything that fails gets
re-shot now. This block is why session 2 stays free for validation.

---

## 3. Session 2 — validation and transfer (~4 h)

### S2-A Repeatability subset (45 min)

Repeat, without refitting anything: turning circles at helm {50, 100} %,
one RPM sweep point, one 20°/20° zig-zag.

Yields **day-to-day parameter variation**, which is what part 1's domain
randomisation ranges currently lack — the bootstrap there resamples runs within
a single session and is explicitly a lower bound on true uncertainty. This block
turns `scale > 1` from a judgement call into a measurement.

### S2-B Open-loop replay validation (45 min)

Replay fixed command sequences — recorded from session 1 and from the trained
policy — with no feedback, and compare against the model's free-run prediction.
Same metric as part 1 §6 V1: heading and position RMSE vs horizon, against the
naive freeze-heading and constant-rate baselines.

This is the honest sim-to-real number and it uses data the model never saw.

### S2-C Closed-loop policy runs (90 min)

The frozen evaluation scenarios. Policy in the loop, limiter ON, full logging.

Run each scenario with the policy trained on (i) the part 1 model and (ii) the
session-1-updated model, if training time allows both. That comparison is the
evidence that system identification improved transfer — which is the claim the
paper wants to make and which no amount of simulation can support on its own.

### S2-D Contingency buffer (40 min)

Unallocated by design. If session 1 lost a block, it is recovered here at the
cost of S2-C scenarios, not at the cost of S2-A or S2-B.

---

## 4. Deliverables

Per the Paper 2 Reviewer 1.4 lesson, **the archived identification dataset is
itself a deliverable**, not just the model fitted from it.

1. Raw logs, unmodified, with `#CONFIG` recording rate, limiter state and clock
   offsets for every run.
2. A manoeuvre index: run ID → manoeuvre type → parameters → validity flag.
3. The landmark survey.
4. Standard manoeuvring quantities in reportable form: turning circle advance,
   transfer, tactical diameter, steady radius and speed loss; zig-zag first and
   second overshoot angles; servo rate, lag and delay; thrust curve.
5. Updated `params_final.json` with day-to-day intervals, and a re-run of the
   part 1 acceptance suite.

---

## 5. Pre-committed acceptance thresholds

Fill before the session; evaluate at S1-F.

| ID | quantity | source block | threshold | result |
|---|---|---|---|---|
| T-1 | servo rate limit | S1-B | resolved to ±20 %, separated from delay | |
| T-2 | transport delay | S1-B | resolved to ±0.1 s | |
| T-3 | thrust curve exponent | S1-C | fitted, not assumed; ≥5 RPM points usable | |
| T-4 | steady turn rate at full helm | S1-D | measured at ≥3 helm settings, both signs | |
| T-5 | turning radius | S1-D | measured; replaces the 1.5–4 L prior band | |
| T-6 | zig-zag overshoot | S1-E | both manoeuvres, ≥2 valid repeats | |
| T-7 | black-wall return rate | S1-A | return rate vs bearing resolved at ≥3 headings | |
| T-8 | localisation drift | S1-A | stationary drift rate quantified | |

---

## 6. Contingency ordering

If session 1 runs short, drop in this order — last dropped is most valuable:

1. S1-C RPM sweep at the extreme settings (keep 9/12/18)
2. S1-E 10°/10° zig-zag (keep 20°/20°)
3. S1-B ramp linearity check (keep the steps)
4. S1-D RPM 18 circles (keep RPM 12, all helm settings, both signs)

**Never drop:** S1-A, S1-D at RPM 12, S1-B steps, S1-F.

If a block is lost entirely, recover it in S2-D and accept fewer S2-C scenarios.
Do not recover it by refitting on session 2 data that was meant for validation —
if that happens, say so in the paper rather than reporting the holdout as
independent.

---

## 7. Open items

- **O-1** Can the vessel telemetry scheduler exceed 2 Hz, and by how much?
  Determines whether S1-B is decisive or produces another lumped delay.
- **O-2** Is 2 Hz the right *control* rate? With ~0.73 s effective dead time and
  yaw dynamics of order 1 s, 2 Hz sits near the edge of controllability and
  plausibly shapes the bang-bang command pattern seen in the July logs. Worth
  measuring the achievable rate even if the policy keeps running at 2 Hz.
- **O-3** Does a full turning circle fit in the basin at RPM 12? If not, S1-D
  becomes settled-arc measurement and the tactical diameter quantities are
  unavailable.
- **O-4** Rudder position feedback: does it exist, or is P-3's camera method
  required?
- **O-5** Is the matte black wall still present and still matte? The basin video
  shows only the light-walled side.
