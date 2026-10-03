"""Saved-only continuous-objective physical-model selection; no episodes.

Select once at decision 11 from identified, bootstrap mean and all 28 existing
whole bootstrap vectors. Initialize once from the first measured body state and
forecast all ten past issued commands continuously. Score measured endpoints
2--11 using the same u/v/r sensor-noise weights as the one-step bank audit.
Intermediate measurements do not reset the training predictor. Every
candidate carries its own estimated servo and rounded transport-delay queue
from reset using issued commands. Freeze the selected vector before validation.

The prediction-error objective is related to Ljung (2002), Section 1:
https://doi.org/10.1007/BF01211648 . This finite-bank engineering diagnostic is
not an IMM, Bayesian posterior, uncertainty bound, physical parameter recovery
or controller. No target/truth/outcome data enters selection or validation.
Future commands are conditional validation inputs, never selection inputs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import os

# Keep this saved-data diagnostic to one numerical CPU thread.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).resolve().parent)]
from audit_causal_yaw_response import clean, measured
from tools.diagnostics.safety.suite_status import read_shared_bytes


class ModelBank:
    """Same RK4/body model for each complete parameter vector; no plant reads."""

    def __init__(self, cc, parameter_order, vectors):
        self.cc = cc
        self.vectors = np.asarray(vectors, dtype=float).copy()
        self.count = len(self.vectors)
        self.parameters = {k: self.vectors[:, i] for i, k in enumerate(parameter_order)}
        self.delay_counts = np.maximum(0, np.rint(self.parameters["rud_delay"] / cc.PRED_DT).astype(int))

    def reset_actuators(self):
        return dict(servo=np.zeros(self.count), pending=None)

    def _pending(self, actuators, command):
        delta = -self.cc.MAX_RUDDER_RAD * float(command[0])
        pending = actuators["pending"]
        return ([([delta] * int(n)) for n in self.delay_counts] if pending is None
                else copy.deepcopy(pending))

    def _delay(self, pending, command):
        delta = -self.cc.MAX_RUDDER_RAD * float(command[0])
        delayed = np.empty(self.count)
        for i, queue in enumerate(pending):
            queue.append(delta)
            delayed[i] = queue.pop(0)
        return delayed

    def advance_actuators(self, actuators, command):
        pending = self._pending(actuators, command)
        servo = actuators["servo"].copy()
        for _ in range(self.cc.SUBSTEPS):
            delayed = self._delay(pending, command)
            servo = self.cc.dyn.advance_rudder(servo, delayed, self.parameters, self.cc.PRED_DT)
        return dict(servo=servo, pending=pending)

    def initial(self, item, actuators, kind):
        state = np.zeros((7, self.count))
        body = item[kind].copy()
        body[0] = max(0., body[0])
        state[:3] = body[:, None]
        state[3] = item["heading"]
        state[4] = actuators["servo"]
        state[5:7] = item["position"][:, None]
        return state

    def forecast(self, item, actuators, commands, kind="raw"):
        state = self.initial(item, actuators, kind)
        pending = None if actuators["pending"] is None else copy.deepcopy(actuators["pending"])
        output = []
        for command in commands:
            if command[1] < 0.:
                raise ValueError("Body forecast excludes reverse braking")
            if pending is None:
                pending = self._pending(actuators, command)
            for _ in range(self.cc.SUBSTEPS):
                delayed = self._delay(pending, command)
                state = self.cc.dyn.rk4_step(state, np.full(self.count, command[1]), delayed,
                                             self.parameters, self.cc.PRED_DT)
                output.append(state.copy())
        return np.asarray(output)


def select_from_past(bank, past, noise):
    """One continuous rollout over exactly ten already observed transitions.

    Only the first measured body state initializes the rollout. Intermediate
    measurements enter the scoring residual, never a predictor state reset.
    Candidate-specific actuator histories start at reset and advance over the
    same ten actual commands; decision 11's command is not used for selection.
    """
    if len(past) != 11:
        raise ValueError("Exactly ten transitions are required")
    act = bank.reset_actuators()
    commands = [item["command"] for item in past[:10]]
    prediction = bank.forecast(past[0], act, commands)[bank.cc.SUBSTEPS-1::bank.cc.SUBSTEPS, :3]
    if prediction.shape != (10, 3, bank.count):
        raise ValueError("Continuous training endpoints are misaligned")
    observed = np.asarray([item["raw"] for item in past[1:11]])
    residuals = (prediction - observed[:, :, None]).transpose(0, 2, 1)
    scores = np.sum((residuals / noise[None, None, :]) ** 2, axis=(0, 2))
    if not np.isfinite(scores).all():
        raise ValueError("A fixed-bank training score is nonfinite")
    selected = int(np.argmin(scores))
    for command in commands:
        act = bank.advance_actuators(act, command)
    return selected, scores, residuals, act


def pair_metrics(errors):
    errors = np.asarray(errors)
    return {"rmse": np.sqrt(np.mean(errors ** 2, axis=0)),
            "mae": np.mean(np.abs(errors), axis=0)} if len(errors) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="conditional_prefix32,motion_axis_probe5")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--case", help="Optional exact canonical case for an initial diagnostic")
    args = parser.parse_args()
    if Path(args.tag).name != args.tag:
        raise ValueError("Use a simple new output tag")
    output = ROOT / "results/safety_dev/v10_iterations/audits" / args.tag
    if output.exists():
        raise FileExistsError(output)
    inputs = {}
    def read(path):
        raw = read_shared_bytes(path)
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        return raw
    import constants as cfg
    manifests = {}
    dependencies = ("src/constants.py", "src/constant_temp.py", "src/classical/common.py",
                    "src/ship.py", "bluefin/dynamics.py", "bluefin/ship_model_v3.py", "src/safety_v3.py")
    for run in args.runs.split(","):
        directory = ROOT / "results/safety_dev/v10_iterations" / run
        manifest = json.loads(read(directory / "manifest.json"))
        completed = json.loads(read(directory / "completion.json"))
        if (completed["source_drift"] or completed["completed_runs"] != manifest["planned_runs"]
                or not completed["checkpoint_unchanged"] or not completed["selection_unchanged"]):
            raise ValueError("Require a completed intact run: " + run)
        for name in dependencies:
            if hashlib.sha256(read(ROOT / name)).hexdigest() != manifest["source_sha256"][name]:
                raise ValueError("Frozen model source mismatch: " + name)
        if manifests and manifest["constants"] != next(iter(manifests.values()))["constants"]:
            raise ValueError("Different effective constants")
        manifests[run] = manifest
    for key, value in next(iter(manifests.values()))["constants"].items():
        try:
            setattr(cfg, key, ast.literal_eval(value))
        except (ValueError, SyntaxError):
            pass
    from classical import common as cc
    import safety_v3 as v3
    import ship
    order = list(cc.dyn.PARAM_NAMES)
    bootstrap = np.asarray(ship.BOOTSTRAP, dtype=float)
    if bootstrap.shape != (28, len(order)) or order != list(ship._v3.PARAM_ORDER):
        raise ValueError("The specified fixed 28-vector bank changed")
    vectors = np.vstack(([cc.IDENTIFIED[k] for k in order], bootstrap.mean(axis=0), bootstrap))
    names = ["identified", "bootstrap_mean"] + [f"bootstrap_{i:02d}" for i in range(28)]
    bank = ModelBank(cc, order, vectors)
    noise = np.array([cfg.EGO_SPEED_NOISE, cfg.EGO_SPEED_NOISE, np.radians(cfg.EGO_YAW_RATE_NOISE_DPS)])
    if not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Need positive recorded measurement noise scales")
    records = []
    for run, manifest in manifests.items():
        directory = ROOT / "results/safety_dev/v10_iterations" / run
        results = [json.loads(read(p)) for p in sorted((directory / "attempts").glob("*_result.json"))]
        if len(results) != manifest["planned_runs"]:
            raise ValueError("Missing committed results")
        for result in results:
            if args.case and result["case"] != args.case:
                continue
            rows = [json.loads(line) for line in read(directory / "traces" /
                    f"{result['attempt']:03d}_{result['mode']}.jsonl").splitlines() if line]
            if len(rows) != result["steps"] or [r["step"] for r in rows] != list(range(1, len(rows) + 1)):
                raise ValueError("Require a complete contiguous committed trace")
            case = next(c for c in manifest["cases"] if c["case"] == result["case"])
            record = dict(run=run, case=result["case"], seed=case["seed"], scenario_sha256=case["scenario_sha256"],
                          steps=len(rows), freeze_decision=11)
            records.append(record)
            if len(rows) < 12:
                record["status"] = "short_trace_before_first_validation_endpoint"
                continue
            if any("diagnostic_decision" not in r for r in rows):
                record["status"] = "missing_snapshots"
                continue
            # measured() strips away every truth, target, result and scene field.
            data = [measured(r) for r in rows]
            if data[0]["actuators"]["servo"] != 0. or data[0]["actuators"]["buffer"] is not None:
                record["status"] = "unknown_initial_actuator_history"
                continue
            rejected = Counter()
            for a, b in zip(data[:10], data[1:11]):
                if not a["fresh"] or not b["fresh"]:
                    rejected["stale_endpoint"] += 1
                if a["command"][1] < 0:
                    rejected["braking_command"] += 1
                if not np.isfinite(a["raw"]).all() or not np.isfinite(b["raw"]).all():
                    rejected["nonfinite_measurement"] += 1
            if rejected:
                record.update(status="first_ten_transitions_not_all_eligible", training_exclusions=dict(rejected))
                continue
            selected, scores, residuals, frozen_act = select_from_past(bank, copy.deepcopy(data[:11]), noise)
            record.update(status="selected", selected_model=names[selected], selected_index=selected,
                          training_scores=scores, training_rmse_uvr_si=np.sqrt(np.mean(residuals**2, axis=0)),
                          training_transition_count=10, training_command_steps=list(range(1, 11)),
                          training_objective="continuous rollout from first raw state; no intermediate resets",
                          selection_uses_future_or_truth=False)
            # Rebuild candidate histories causally from issued commands. Nominal
            # history must agree with the recorded predictor, including repairs.
            histories = []
            act = bank.reset_actuators()
            max_servo_error = 0.
            for item in data[:27]:
                histories.append(copy.deepcopy(act))
                saved = item["actuators"]
                max_servo_error = max(max_servo_error, abs(act["servo"][0] - saved["servo"]))
                np.testing.assert_allclose(act["servo"][0], saved["servo"], atol=1e-13, rtol=0)
                if act["pending"] is not None:
                    np.testing.assert_array_equal(act["pending"][0], saved["buffer"])
                act = bank.advance_actuators(act, item["command"])
            np.testing.assert_array_equal(frozen_act["servo"], histories[10]["servo"])
            assert frozen_act["pending"] == histories[10]["pending"]
            # Numerical baseline parity against the original rollout at decision11.
            item = data[10]
            positive = np.array([[.37, cfg.CRUISE_RPM]])
            bstates = bank.forecast(item, frozen_act, positive, kind="filtered")
            snap = cc.Snapshot(*item["position"], item["heading"], *item["filtered"], np.array([0., 1.]),
                               np.array([1., 0.]), np.zeros(2), 0., 0., 0., np.empty((0, 2)), [])
            original_act = cc.Actuators()
            original_act.servo = frozen_act["servo"][0]
            original_act.buffer = copy.deepcopy(frozen_act["pending"][0])
            reference = v3.rollout_seq(snap, original_act, np.array([[[.37, 0.]]]))
            np.testing.assert_allclose(bstates[:, 5:7, 0], reference.positions[:, 0], atol=1e-13, rtol=0)
            np.testing.assert_allclose(bstates[:, 3, 0], reference.headings[:, 0], atol=1e-13, rtol=0)
            record["nominal_history_max_servo_error_rad"] = max_servo_error
            labels = {"identified": 0, "bootstrap_mean": 1, "selected": selected}
            available = list(zip(data[10:26], data[11:27]))
            held = [(10+i, a, b) for i, (a, b) in enumerate(available)
                    if a["fresh"] and b["fresh"] and a["command"][1] >= 0.]
            one_errors = {label: [] for label in labels}
            for index, a, b in held:
                prediction = bank.forecast(a, histories[index], [a["command"]])[-1, :3]
                for label, model in labels.items():
                    error = prediction[:, model] - b["raw"]
                    one_errors[label].append([error[0], error[1], np.degrees(error[2])])
            record["one_step"] = {label: pair_metrics(value) for label, value in one_errors.items()}
            record["one_step_components"] = ["u_mps", "v_mps", "yaw_deg_s"]
            record["held_forward_command_steps"] = [a["step"] for _, a, _ in held]
            length = 0
            for a, b in available:
                if a["command"][1] < 0.:
                    break
                length += 1
            record["continuous_horizon_s"] = length * cfg.UPDATE_RATE
            record["continuous_stop_reason"] = ("first_braking_command" if length < len(available)
                else "eight_seconds" if length == 16 else "trace_end")
            record["continuous"] = {}
            if length:
                commands = [a["command"] for a in data[10:10+length]]
                observed = data[11:11+length]
                for kind in ("raw", "filtered"):
                    state = bank.forecast(data[10], frozen_act, commands, kind)[cc.SUBSTEPS-1::cc.SUBSTEPS]
                    record["continuous"][kind] = {}
                    for label, model in labels.items():
                        errors = [dict(horizon_s=(i+1)*cfg.UPDATE_RATE, fresh=b["fresh"],
                            u_error_mps=s[0, model]-b["raw"][0], v_error_mps=s[1, model]-b["raw"][1],
                            yaw_error_deg_s=np.degrees(s[2, model]-b["raw"][2]),
                            heading_error_deg=np.degrees(cc.wrap_pi(s[3, model]-b["heading"])),
                            position_error_m=np.linalg.norm(s[5:7, model]-b["position"]))
                            for i, (s, b) in enumerate(zip(state, observed))]
                        valid = [e for e in errors if e["fresh"]]
                        stats = {k.replace("error", "rmse"): float(np.sqrt(np.mean([e[k]**2 for e in valid])))
                                 if valid else None for k in ("u_error_mps", "v_error_mps", "yaw_error_deg_s",
                                                            "heading_error_deg", "position_error_m")}
                        record["continuous"][kind][label] = dict(errors=errors, scored_endpoints=len(valid), **stats)
            print(run, result["case"], names[selected], "horizon", length*cfg.UPDATE_RATE, flush=True)
    def comparison(eligible, getter):
        pairs = [dict(case=r["case"], identified=getter(r, "identified"),
                      bootstrap_mean=getter(r, "bootstrap_mean"), selected=getter(r, "selected")) for r in eligible]
        result = dict(n=len(pairs), cases=pairs)
        for label in ("bootstrap_mean", "selected"):
            result[label] = dict(improved=sum(p[label] < p["identified"] for p in pairs),
                degraded=sum(p[label] > p["identified"] for p in pairs),
                equal=sum(p[label] == p["identified"] for p in pairs),
                mean=float(np.mean([p[label] for p in pairs])) if pairs else None)
        result["mean_identified"] = float(np.mean([p["identified"] for p in pairs])) if pairs else None
        return result
    summary = {}
    for run in manifests:
        group = [r for r in records if r["run"] == run]
        valid = [r for r in group if r["status"] == "selected"]
        item = dict(records=len(group), statuses=dict(Counter(r["status"] for r in group)),
                    selected_models=dict(Counter(r["selected_model"] for r in valid)))
        item["one_step"] = {name: comparison([r for r in valid if r["one_step"]["identified"]],
            lambda r, k: r["one_step"][k]["rmse"][i]) for i, name in enumerate(("u_mps", "v_mps", "yaw_deg_s"))}
        item["continuous"] = {}
        for subset in ("available", "full_eight_seconds"):
            item["continuous"][subset] = {}
            for kind in ("raw", "filtered"):
                eligible = [r for r in valid if kind in r["continuous"]
                    and r["continuous"][kind]["identified"]["scored_endpoints"]
                    and (subset == "available" or r["continuous_horizon_s"] == 8.)]
                item["continuous"][subset][kind] = {metric: comparison(eligible,
                    lambda r, k: r["continuous"][kind][k][metric]) for metric in
                    ("u_rmse_mps", "v_rmse_mps", "yaw_rmse_deg_s", "heading_rmse_deg", "position_rmse_m")}
        summary[run] = item
    artifact = dict(created_utc=datetime.now(timezone.utc).isoformat(), method=__doc__, episode_calls=0,
        bank=dict(names=names, parameter_order=order, vectors=vectors, delay_counts=bank.delay_counts,
                  prediction_dt_s=cc.PRED_DT), measurement_scales_uvr_si=noise,
        score="sum at ten past endpoints2..11 and u/v/r of (continuous prediction initialized at decision1 minus raw measurement)^2 / sigma^2",
        initialization="Known empty command queue and zero estimated servo at reset; candidate-specific history from issued commands only.",
        limitations=["Selected development cases; repeated controller exposures are not independent scenarios.",
            "Ten noisy transitions; the initial state and scoring endpoints are noisy, and residuals are correlated.",
            "Only the training objective changed from the prior bank audit; no window, weights, candidates or validation endpoints were retuned.",
            "Recorded sensor scales weight components; this is not a calibrated likelihood or uncertainty envelope.",
            "Candidate integration/delay discretization stays at the existing predictor .125s; no hidden actuator measurements.",
            "Actual randomized plant may be a blend/jitter outside this finite bank; no sampled vector is known to be true.",
            "Later commands and measurements are conditional validation only, with no refit; braking forecasts are excluded.",
            "No controller change, policy continuation scoring, target geometry, simulator truth or outcome optimization."],
        records=records, summary=summary, provenance=dict(inputs=inputs,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            parent_one_step_audit_script_sha256=hashlib.sha256(Path(__file__).with_name("audit_bootstrap_model_bank.py").read_bytes()).hexdigest(),
            numerical_thread_environment={key: os.environ[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")},
            measurement_helper_sha256=hashlib.sha256(Path(__file__).with_name("audit_causal_yaw_response.py").read_bytes()).hexdigest()))
    output.mkdir(exist_ok=False)
    (output / "audit.json").write_text(json.dumps(clean(artifact), indent=2, allow_nan=False)+"\n", newline="\n")
    (output / "reproducer_evaluated_bytes.py").write_bytes(Path(__file__).read_bytes())
    print("OUTPUT", output)
    print(json.dumps({k: {x: y for x, y in v.items() if x not in ("one_step", "continuous")}
                      for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
