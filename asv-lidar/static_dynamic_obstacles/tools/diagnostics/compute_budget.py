"""Compute budget of the baseline-v3 campaign under ideal conditions (2026-10-17).

    python tools/diagnostics/compute_budget.py      # writes results/baseline_v3/compute_budget.md

Ideal conditions: on mains power, no other training or evaluation running, no OneDrive
sync or other heavy process. The ideal speed of a learner in each curriculum stage is the
85th percentile of its sustained 20,000-step training speeds in that stage, pooled over its
seeds' logs (the slow stretches -- battery, sync, concurrent evaluations -- fall below it).
A run's ideal time is then

  training      sum over the seven stages of stage steps / ideal speed (3.0 M steps; stage
                boundaries 0.16, 0.36, 0.64, 1.0, 1.5, 2.0 M), plus the validation-set
                evaluations during training (every 200,000 steps and at the end; SAC and TQC
                also at 2.5 M) at the learner's median evaluation time;
  evaluations   Tier 1, frozen suite, reference layouts and test set v4 on the best checkpoint
                (each at the fastest time measured for it in results/train_seed.log), plus the
                final-model test set run (decision, 2026-10-15) at the test-set time.

Actual times are the runs' recorded wall clock ("done in" lines; seed 0 trained in two phases,
2.5 M then the extension to 3.0 M; aborted attempts are not counted) and the evaluation
durations in results/train_seed.log. The remaining work is costed at ideal speeds from each
run's current position.
"""
from __future__ import annotations

import datetime as dt
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "results" / "train_seed.log"
OUT = ROOT / "results" / "baseline_v3" / "compute_budget.md"
LEARNERS = [("ppo", "PPO"), ("recurrent_ppo", "RecurrentPPO"), ("sac", "SAC"), ("tqc", "TQC")]
SEEDS = (0, 1, 2)
STAGES = [0, 160_000, 360_000, 640_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000]
EVAL_POINTS = list(range(200_000, 3_000_001, 200_000))
PCT, WINDOW = 0.85, 20_000


def run_log(algo, seed):
    return ROOT / "runs" / f"{algo}_formulation_seed{seed}_bl3.log"


def pairs(path):
    out, t = [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"\|\s+time_elapsed\s+\|\s+(\d+)", line)
        if m:
            t = int(m.group(1))
        m = re.search(r"\|\s+total_timesteps\s+\|\s+(\d+)", line)
        if m and t is not None:
            out.append((int(m.group(1)), t))
    return out


def segment_speeds(path):
    p, speeds = pairs(path), {k: [] for k in range(7)}
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and p[j][0] - p[i][0] < WINDOW:
            j += 1
        if j >= len(p):
            break
        ds, dtime = p[j][0] - p[i][0], p[j][1] - p[i][1]
        if dtime > 0 and ds > 0:              # a resume restarts the clock: skip that segment
            mid = (p[i][0] + p[j][0]) / 2
            speeds[max(k for k in range(7) if mid >= STAGES[k])].append(ds / dtime)
        i = j
    return speeds


def eval_seconds(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    return [float(x) for x in re.findall(r"\[EVAL\] t=[\d,]+ (?:safety|supervisor) off .*?\(([\d.]+) s\)", text)]


def done_hours(path):
    """Wall clock of the phases that make up the finished run: the last "done in" line for a run
    trained straight to 3.0 M (its config records a curriculum budget), the last two (2.5 M, then
    the extension) for seed 0.  Earlier lines in the same log belong to aborted attempts."""
    import json
    hours = [float(x) for x in re.findall(r"done in ([\d.]+) h", open(path, encoding="utf-8", errors="replace").read())]
    cfg = path.with_suffix("") / "config.json"
    straight = cfg.exists() and json.loads(cfg.read_text()).get("curriculum_budget")
    return hours[-1:] if straight else hours[-2:]


def latest_steps(path):
    p = pairs(path)
    return p[-1][0] if p else 0


def stamp(line):
    return dt.datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")


def evaluation_times():
    """Per run id: minutes of each post-training step, from the milestone log."""
    lines = [l.rstrip("\n") for l in open(LOG, encoding="utf-8", errors="replace") if re.match(r"\d{4}-\d\d-\d\d ", l)]
    out = {}
    for algo, _ in LEARNERS:
        for s in SEEDS:
            rid = f"{algo}s{s}_bl3"
            starts = [i for i, l in enumerate(lines) if f"== {algo} seed {s}: already trained" in l]
            if not starts:
                continue
            i0 = starts[-1]
            prev, parts = stamp(lines[i0]), {}
            keys = (("tier1", f"tier 1 {rid} supervisor off"), ("frozen", f"frozen suite {rid} "),
                    ("paper2", f"Paper 2 set {rid} "), ("test", f"test set v4, {rid} "),
                    ("final", f"test set v4 (final model), {rid} "))
            for l in lines[i0 + 1:]:
                for key, pat in keys:
                    if pat in l and key not in parts:
                        gap = (stamp(l) - prev).total_seconds() / 60
                        if gap <= 120:            # a step run later on its own is not timed here
                            parts[key] = gap
                        prev = stamp(l)
                if f"== {rid} run done" in l:
                    break
            out[rid] = parts
    return out


def status(algo, seed, rid, evals):
    run = ROOT / "runs" / f"{algo}_formulation_seed{seed}_bl3"
    log = run_log(algo, seed)
    if not log.exists():
        return "to do"
    if not (run / "final_model.zip").exists():
        age = dt.datetime.now().timestamp() - log.stat().st_mtime
        return "training" if age < 1800 else "stopped"
    if (ROOT / "results" / "test_set" / "v4" / rid / "summary.txt").exists():
        return "done" if "final" in evals.get(rid, {}) else "done (final-model check to run)"
    return "trained, evaluations held"


def main() -> int:
    ev_times = evaluation_times()
    minimum = {}
    for parts in ev_times.values():
        for k, v in parts.items():
            minimum[k] = min(minimum.get(k, v), v)
    minimum.setdefault("final", minimum.get("test", 10.0))
    ideal_eval_h = sum(minimum[k] for k in ("tier1", "frozen", "paper2", "test", "final")) / 60

    rows, speeds_by_learner, totals = [], {}, {}
    for algo, name in LEARNERS:
        pooled = {k: [] for k in range(7)}
        evs = []
        for s in SEEDS:
            log = run_log(algo, s)
            if log.exists():
                for k, v in segment_speeds(log).items():
                    pooled[k] += v
                evs += eval_seconds(log)
        speed = {k: (sorted(v)[int(PCT * (len(v) - 1))] if v else None) for k, v in pooled.items()}
        speeds_by_learner[name] = speed
        ev_s = st.median(evs) if evs else 200.0
        extra = [2_500_000] if algo in ("sac", "tqc") else []
        n_evals = len(EVAL_POINTS) + 1 + len(extra)

        def train_hours(from_steps=0):
            h = sum(max(0, STAGES[k + 1] - max(STAGES[k], from_steps)) / speed[k] / 3600 for k in range(7))
            left = sum(1 for t in EVAL_POINTS + extra if t > from_steps) + 1
            return h + left * ev_s / 3600

        ideal_train = train_hours()
        for s in SEEDS:
            rid = f"{algo}s{s}_bl3"
            log = run_log(algo, s)
            stat = status(algo, s, rid, ev_times)
            actual_train = sum(done_hours(log)) if log.exists() and stat != "training" else None
            if stat == "training":
                started = [stamp(l) for l in open(LOG, encoding="utf-8", errors="replace")
                           if f"== {rid}: training" in l]
                actual_train = (dt.datetime.now() - started[-1]).total_seconds() / 3600 if started else None
            parts = ev_times.get(rid, {})
            actual_eval = sum(parts.values()) / 60 if parts else None
            if stat == "to do":
                remaining = ideal_train + ideal_eval_h
            elif stat == "training":
                remaining = train_hours(latest_steps(log)) + ideal_eval_h
            elif stat == "stopped":
                remaining = train_hours(latest_steps(log)) + ideal_eval_h
            elif stat == "trained, evaluations held":
                remaining = ideal_eval_h
            elif stat.startswith("done ("):
                remaining = minimum["final"] / 60
            else:
                remaining = 0.0
            rows.append(dict(name=name, seed=s, status=stat, ideal_train=ideal_train, ideal_eval=ideal_eval_h,
                             actual_train=actual_train, actual_eval=actual_eval, remaining=remaining))
            t = totals.setdefault(name, dict(ideal=0.0, actual=0.0, remaining=0.0))
            t["ideal"] += ideal_train + ideal_eval_h
            t["actual"] += (actual_train or 0) + (actual_eval or 0)
            t["remaining"] += remaining

    def f(x, d=1):
        return "—" if x is None else f"{x:.{d}f}"

    now = dt.datetime.now()
    L = ["# Compute budget of the baseline-v3 campaign (ideal conditions)", "",
         f"Generated {now:%Y-%m-%d %H:%M} by `tools/diagnostics/compute_budget.py`; the method is in its docstring.",
         "Ideal conditions: on mains power, nothing else training or evaluating, no OneDrive sync or other heavy process.", "",
         "## Ideal training speed by curriculum stage (steps/s)", "",
         "| Learner | S1 | S2 | S3 | S4 | S5 | S6 | S7 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, sp in speeds_by_learner.items():
        L.append(f"| {name} | " + " | ".join(f(sp[k]) for k in range(7)) + " |")
    L += ["", f"Post-training evaluations, ideal (fastest measured): Tier 1 {minimum['tier1']:.0f} min, frozen suite "
          f"{minimum['frozen']:.0f} min, reference layouts {minimum['paper2']:.0f} min, test set v4 {minimum['test']:.0f} min, "
          f"final-model test set {minimum['final']:.0f} min: **{ideal_eval_h:.2f} h per run**.", "",
          "## By run (hours)", "",
          "| Learner | Seed | Status | Ideal training | Ideal evaluations | **Ideal total** | Actual training | Actual evaluations | Remaining, ideal |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['seed']} | {r['status']} | {f(r['ideal_train'])} | {f(r['ideal_eval'])} | "
                 f"**{f(r['ideal_train'] + r['ideal_eval'])}** | {f(r['actual_train'])} | {f(r['actual_eval'])} | {f(r['remaining'])} |")
    L += ["", "## By learner, three seeds (hours)", "",
          "| Learner | **Ideal total** | Actual so far | Remaining, ideal |", "|---|---:|---:|---:|"]
    gi = ga = gr = 0.0
    for name, t in totals.items():
        L.append(f"| {name} | **{t['ideal']:.1f}** | {t['actual']:.1f} | {t['remaining']:.1f} |")
        gi, ga, gr = gi + t["ideal"], ga + t["actual"], gr + t["remaining"]
    L += [f"| **All four** | **{gi:.1f}** ({gi / 24:.1f} days) | {ga:.1f} ({ga / 24:.1f} days) | "
          f"**{gr:.1f}** ({gr / 24:.1f} days) |", "",
          "Actual training excludes aborted attempts; actual evaluations are missing where the milestone log does not "
          "hold the full sequence (seed 0's older evaluation runs) or the evaluations are still held. Remaining work "
          f"from {now:%Y-%m-%d %H:%M} at ideal speeds would end around "
          f"{(now + dt.timedelta(hours=gr)):%a %d %b %H:%M} if run back to back."]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(("\n".join(L) + "\n").encode("utf-8"))      # LF on every platform
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
