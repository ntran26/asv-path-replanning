"""Offline reverse-response identification from DEVELOPMENT measurements only.

Prediction-error fitting follows Ljung (2002), Section 1:
https://liu.diva-portal.org/smash/get/diva2%3A316694/FULLTEXT01.pdf
DOI: 10.1007/BF01211648. This is a batch grey-box fit, not an implementation
of recursive identification and not a safety guarantee. The additional body
surge acceleration is identified ONLY at the recorded -24 RPM. Sparse short
pulses cannot independently identify onset delay and thrust strength.

The fixed onboard forward model and reconstructed issued-rudder history are
used. Ground-truth state is kept in separate scoring fields and never enters
the fitting objective. No environment, policy, or episode is constructed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from classical import common as cc
import constants as cfg
import safety_v2 as v2

EXCLUDED = {"DV3-CRP-VS-04", "DV3-CRS-CV-04"}
SIGMA = np.array([cfg.EGO_SPEED_NOISE, cfg.EGO_SPEED_NOISE,
                  np.deg2rad(cfg.EGO_YAW_RATE_NOISE_DPS)])
SOURCES = ("v4_selected_contacts_steps.csv", "wall_oracle_state_steps.csv")


@dataclass
class Pulse:
    source: str
    mode: str
    case: str
    step: int
    measured: np.ndarray
    truth: np.ndarray
    commands: np.ndarray
    servo: float
    pending: np.ndarray

    @property
    def decisions(self):
        return len(self.commands)


def active_duration(age: float, delay: float, dt: float):
    """Time spent after brake onset within this integration interval."""
    return max(0.0, min(dt, age + dt - delay))


def read_pulses(paths):
    pulses, dropped = [], []
    for path in paths:
        frame = pd.read_csv(path)
        for (mode, case), group in frame.groupby(["mode", "case"], sort=False):
            rows = list(group.sort_values("step").itertuples(index=False))
            if [r.step for r in rows] != list(range(len(rows))):
                raise ValueError(f"{path.name}/{case}: missing command history")
            histories, actuator = [], cc.Actuators()
            for row in rows:
                pending = actuator.buffer
                if pending is None:
                    pending = [-cc.MAX_RUDDER_RAD * row.rudder] * cc.DELAY_STEPS
                histories.append((actuator.servo, np.array(pending, copy=True)))
                actuator.issue(SimpleNamespace(command_rate_limit=False), row.rudder)
            i = 0
            while i < len(rows):
                if rows[i].rpm >= 0:
                    i += 1
                    continue
                end = i + 1
                while end < len(rows) and rows[end].rpm < 0:
                    end += 1
                if end == len(rows):
                    dropped.append(dict(source=path.name, case=case, step=i,
                                        reason="no recorded successor measurement"))
                    break
                window = rows[i:end + 1]
                stale = any(str(getattr(r, "pose_stale", False)).lower()
                            in ("true", "1") for r in window)
                if stale:
                    dropped.append(dict(source=path.name, case=case, step=i,
                                        reason="stale measurement"))
                else:
                    commands = np.array([[r.rudder, r.rpm] for r in rows[i:end]])
                    if not np.all(commands[:, 1] == -24):
                        raise ValueError("Only observed -24 RPM response is identified")
                    measured = np.array([[r.measured_u_mps, r.measured_v_mps,
                                          r.measured_r_radps] for r in window])
                    if mode in ("oracle", "oracle_state"):
                        truth = np.array([[r.snapshot_u_mps, r.snapshot_v_mps,
                                           r.snapshot_r_radps] for r in window])
                    else:
                        truth = np.array([[r.surge_before, r.sway_before,
                                           r.yaw_before_radps] for r in window])
                    if not np.isfinite(measured).all():
                        raise ValueError(f"Non-finite measurement: {case}/{i}")
                    pulses.append(Pulse(path.name, mode, case, i, measured,
                                        truth, commands, *histories[i]))
                i = end
    return pulses, dropped


def predict(pulses, acceleration, delay=0.0, initial=None, legacy=False):
    """Batched complete pulse paths, including each 0.5 s endpoint.

    The local body equations are pose independent; no measured or true global
    pose is needed. Reverse force is a constant additional surge acceleration
    after onset, applied with the same nonnegative-surge clipping as the
    onboard model. Brake age is carried across all decisions of a pulse.
    Legacy timing reproduces the old first-substep timestamp convention.
    """
    n = len(pulses)
    if not n:
        return []
    initial = (np.array([p.measured[0] for p in pulses]) if initial is None
               else np.array(initial, dtype=float, copy=True))
    state = np.zeros((7, n))
    state[:3] = initial.T
    state[0] = np.maximum(0, state[0])
    state[4] = [p.servo for p in pulses]
    pending = [np.array([p.pending[j] for p in pulses])
               for j in range(cc.DELAY_STEPS)]
    params = {k: np.full(n, v) for k, v in cc.IDENTIFIED.items()}
    paths = [[] for _ in pulses]
    max_decisions = max(p.decisions for p in pulses)
    age = 0.0
    for decision in range(max_decisions):
        commands = np.array([p.commands[min(decision, p.decisions - 1)]
                             for p in pulses])
        delta = -cc.MAX_RUDDER_RAD * commands[:, 0]
        for _ in range(cc.SUBSTEPS):
            pending.append(delta.copy())
            state = cc.dyn.rk4_step(state, np.maximum(commands[:, 1], 0),
                                    pending.pop(0), params, cc.PRED_DT)
            duration = (cc.PRED_DT if age >= delay else 0.0) if legacy else active_duration(age, delay, cc.PRED_DT)
            state[0] = np.maximum(0, state[0] - acceleration * duration)
            age += cc.PRED_DT
        for j, pulse in enumerate(pulses):
            if decision < pulse.decisions:
                paths[j].append(state[:3, j].copy())
    return [np.array(path) for path in paths]


def residuals(pulses, acceleration, delay=0.0, initial=None):
    predictions = predict(pulses, acceleration, delay, initial)
    return np.concatenate([((pred - p.measured[1:]) / SIGMA).ravel()
                           for p, pred in zip(pulses, predictions)])


def fit(pulses, delay=0.0, latent=False):
    # A broad physical parameter search, not a controller threshold sweep.
    # The upper bound is checked explicitly, so a bound-hitting fit is rejected.
    scalar = minimize_scalar(lambda a: float(np.sum(residuals(pulses, a, delay) ** 2)),
                             bounds=(0., 3.), method="bounded",
                             options={"xatol": 1e-6})
    if scalar.x > 2.99:
        raise ValueError("Acceleration fit reached declared search bound")
    if not latent:
        return float(scalar.x), float(scalar.fun), None
    raw = np.array([p.measured[0] for p in pulses])
    initial = raw.copy()
    initial[:, 0] = np.maximum(initial[:, 0], 1e-10)
    n = len(pulses)

    def objective(theta):
        start = theta[1:].reshape(n, 3)
        # Latent initial state is penalized by its measured sensor likelihood;
        # target truth is deliberately absent from this objective.
        return np.r_[((start - raw) / SIGMA).ravel(),
                     residuals(pulses, theta[0], delay, start)]

    sparsity = lil_matrix((3 * n + sum(p.decisions * 3 for p in pulses), 1 + 3 * n))
    for j in range(n):
        for k in range(3):
            sparsity[3 * j + k, 1 + 3 * j + k] = 1
    offset = 3 * n
    for j, p in enumerate(pulses):
        size = p.decisions * 3
        sparsity[offset:offset + size, 0] = 1
        sparsity[offset:offset + size, 1 + 3 * j:4 + 3 * j] = 1
        offset += size
    lower = np.r_[0., np.tile([0., -np.inf, -np.inf], n)]
    upper = np.r_[3., np.full(3 * n, np.inf)]
    result = least_squares(objective, np.r_[scalar.x, initial.ravel()],
                           bounds=(lower, upper), jac_sparsity=sparsity.tocsr(),
                           max_nfev=100, ftol=1e-7, xtol=1e-7, gtol=1e-7)
    return float(result.x[0]), float(np.sum(result.fun ** 2)), {
        "success": bool(result.success), "nfev": result.nfev,
        "initial_adjustment_rms_sigma": float(np.sqrt(np.mean(
            ((result.x[1:].reshape(n, 3) - raw) / SIGMA) ** 2)))}


def score(pulses, models):
    rows = []
    for name, acceleration, delay, legacy in models:
        predictions = predict(pulses, acceleration, delay, legacy=legacy)
        for pulse, path in zip(pulses, predictions):
            for j, pred in enumerate(path):
                row = dict(source=pulse.source, mode=pulse.mode, case=pulse.case,
                           step=pulse.step, endpoint=j + 1,
                           split="held" if pulse.case in EXCLUDED else "fit",
                           model=name, acceleration_mps2=acceleration,
                           delay_s=delay, duration_s=(j + 1) * cfg.UPDATE_RATE)
                for k, component in enumerate(("u", "v", "r")):
                    row[f"predicted_{component}"] = pred[k]
                    row[f"measured_{component}"] = pulse.measured[j + 1, k]
                    row[f"true_{component}"] = pulse.truth[j + 1, k]
                    row[f"measured_error_{component}"] = pred[k] - pulse.measured[j + 1, k]
                    row[f"true_error_{component}"] = pred[k] - pulse.truth[j + 1, k]
                rows.append(row)
    return pd.DataFrame(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--preliminary", action="store_true",
                        help="Fit raw-initial immediate response only; zero episodes")
    args = parser.parse_args()
    if Path(args.tag).name != args.tag:
        raise ValueError("Tag must be a single directory name")
    output = ROOT / "results/safety_dev/brake_calibration" / args.tag
    output.mkdir(parents=True, exist_ok=False)
    paths = [ROOT / "results/safety_dev" / name for name in SOURCES]
    pulses, dropped = read_pulses(paths)
    training = [p for p in pulses if p.case not in EXCLUDED]
    metadata = dict(utc=datetime.now(timezone.utc).isoformat(), episodes_run=0,
                    measured_only_fit=True, truth_scoring_only=True,
                    excluded_fit_cases=sorted(EXCLUDED), sensor_sigma=SIGMA.tolist(),
                    pulses=len(pulses), fit_pulses=len(training),
                    fit_cases=sorted({p.case for p in training}),
                    pulse_decision_counts={str(d): sum(p.decisions == d for p in training)
                                           for d in sorted({p.decisions for p in training})},
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths},
                    model_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                                  [Path(__file__), ROOT / "src/classical/common.py",
                                   Path(cc.dyn.__file__), ROOT / "src/constants.py"]},
                    dropped=dropped,
                    citation="Ljung (2002), Prediction Error Estimation Methods, Section 1; DOI 10.1007/BF01211648")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    acceleration, objective, _ = fit(training)
    print(json.dumps(dict(fit_pulses=len(training), fit_cases=len(metadata["fit_cases"]),
                          acceleration_mps2=acceleration,
                          first_decision_impulse_mps=acceleration * cfg.UPDATE_RATE,
                          objective=objective)), flush=True)
    profiles = [dict(initial="measured", delay_s=0., acceleration_mps2=acceleration,
                     first_decision_impulse_mps=acceleration * cfg.UPDATE_RATE,
                     objective=objective)]
    # The weak/delayed baseline is computed from the EXISTING onboard predictor
    # and its declared efficiency, never simulator-default reverse parameters.
    baseline_a = cc.IDENTIFIED["T12"] * (24. / 12.) ** 2 * v2.BRAKE_EFFICIENCY / cc.dyn.M11
    models = [("existing_weak_delayed", baseline_a, v2.BRAKE_DELAY_S, True),
              ("measured_initial_immediate", acceleration, 0., False)]
    cv = []
    if not args.preliminary:
        # Profile onset delay; no outcome-based selection. Report impulse and
        # sensitivity, not a uniquely identified thrust/delay decomposition.
        for delay in (0., .125, .25, .375, .5, .75):
            a, obj, details = fit(training, delay=delay, latent=True)
            profiles.append(dict(initial="latent_sensor_penalty", delay_s=delay,
                                 acceleration_mps2=a,
                                 first_decision_impulse_mps=a * max(0., cfg.UPDATE_RATE - delay),
                                 objective=obj, **details))
            print(json.dumps(profiles[-1]), flush=True)
            if delay == 0.:
                models.append(("latent_initial_immediate", a, delay, False))
        cv_predictions = []
        for case in metadata["fit_cases"]:
            subset = [p for p in training if p.case != case]
            for latent in (False, True):
                a, obj, details = fit(subset, latent=latent)
                name = "latent" if latent else "measured"
                cv.append(dict(held_case=case, initial=name,
                               acceleration_mps2=a,
                               first_decision_impulse_mps=a * cfg.UPDATE_RATE,
                               objective=obj, **(details or {})))
                predictions = score([p for p in training if p.case == case],
                                    [(f"leave_case_out_{name}", a, 0., False)])
                predictions["split"] = "leave_case_out"
                cv_predictions.append(predictions)
    scores = score(pulses, models)
    if not args.preliminary:
        scores = pd.concat([scores, *cv_predictions], ignore_index=True)
    scores.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(profiles).to_csv(output / "delay_profiles.csv", index=False)
    pd.DataFrame(cv).to_csv(output / "leave_case_out.csv", index=False)
    summaries = []
    for (split, model), frame in scores.groupby(["split", "model"], sort=False):
        row = dict(split=split, model=model, endpoints=len(frame), cases=frame.case.nunique())
        for prefix in ("measured", "true"):
            for component in ("u", "v", "r"):
                errors = frame[f"{prefix}_error_{component}"].to_numpy()
                scale = 180 / np.pi if component == "r" else 1.
                row[f"{prefix}_{component}_mae"] = float(np.mean(np.abs(errors)) * scale)
                row[f"{prefix}_{component}_bias"] = float(np.mean(errors) * scale)
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    print(pd.DataFrame(summaries).round(6).to_string(index=False), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
