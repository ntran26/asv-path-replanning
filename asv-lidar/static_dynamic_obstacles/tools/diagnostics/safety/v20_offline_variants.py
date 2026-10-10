"""Offline evidence for V20 design variants at saved V16 decision states.

Saved development data only; no episode, policy call or environment. Two
analyses, both without allowance tables ("none"), the setting under which the
default G3 survival gate holds:

* ``margin``: from a merged G2/G3 replay, one-step survival of a committed
  contingency and checked-decision availability when a new commitment requires
  slack of at least m (robust-MPC constraint tightening, Chisci et al. 2001,
  https://doi.org/10.1016/S0005-1098(00)00203-1); the recheck of a committed
  tail stays at zero. Cheap; reads the replay's per-episode files.
* ``options``: at every saved decision where V16's issued command has slack
  below ``--max-margin`` with the default stop family, and at every unchecked
  decision, the best slack over the stop family (TAILS) and the extended family
  (EXTENDED_TAILS) for V16's command, SAC's command and V2's projection grid
  (V20's candidate set), with the nearest-to-V16 certified projection at each
  margin. Answers whether a gatekeeper that enforces the certificate at checked
  decisions has a certified command near V16's, and whether the extended
  family adds availability. Expensive; per-episode files, resumable, never
  overwritten; ``--part/--parts`` split; ``--merge`` summarises.

Results describe single decisions along V16's recorded trajectory, not a
closed loop. Truth is not read. Usage (project root):
    python -B tools/diagnostics/safety/v20_offline_variants.py margin --replay g2g3_v3 --tag variants_v1
    python -B tools/diagnostics/safety/v20_offline_variants.py options --replay g2g3_v3 --tag variants_v1 --part 1 --parts 3
    python -B tools/diagnostics/safety/v20_offline_variants.py options --replay g2g3_v3 --tag variants_v1 --merge --parts 3
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
import safety_v20 as v20  # noqa: E402
import safety_v20_contingency as sc  # noqa: E402
import v20_replay_saved_decisions as replay  # noqa: E402

DEV = ROOT / "results/safety_dev/v20_development"
MARGINS = (0.0, 0.05, 0.10, 0.127, 0.15, 0.20, 0.30)
FAMILIES = {"stop": sc.TAILS, "extended": sc.EXTENDED_TAILS}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k, n):
    return replay.wilson(k, n)


def episode_rows(replay_tag):
    files = sorted((DEV / "saved_state_replay" / replay_tag / "episodes").glob("*.json"))
    return [json.loads(f.read_text()) for f in files], files


def margin_analysis(replay_tag):
    episodes, files = episode_rows(replay_tag)
    out = {"replay": replay_tag, "episodes": len(episodes), "margins": []}
    for m in MARGINS:
        st, avail = Counter(), Counter()
        for ep in episodes:
            rows = ep["rows"]
            for a in rows:
                if not a["unchecked"]:
                    avail[bool(a["none_v16_certified"] and a["none_v16_slack"] >= m)] += 1
            for a, b in zip(rows, rows[1:]):
                if b["step"] != a["step"] + 1 or "none_survived" not in b:
                    continue
                if a["none_v16_certified"] and a["none_v16_slack"] >= m:
                    st[("traffic" if b["traffic"] else "no_traffic", bool(b["none_survived"]))] += 1
        row = {"margin_m": m, "checked_available": [avail[True], avail[True] + avail[False]]}
        for stratum in ("traffic", "no_traffic"):
            k, n = st[(stratum, True)], st[(stratum, True)] + st[(stratum, False)]
            row[f"survival_{stratum}"] = {"survived": k, "rechecks": n, "rate": k / n if n else None,
                                          "wilson95": wilson(k, n)}
        out["margins"].append(row)
    out["input_sha256"] = {f.relative_to(ROOT).as_posix(): sha(f) for f in files}
    return out


def option_rows(case, label, trace, wanted, why_by_step):
    rows = []
    for rec in replay.stream_records(trace):
        step = rec["step"]
        if step not in wanted:
            continue
        dd = rec["diagnostic_decision"]
        s, a = dd["snapshot"], dd["actuators_before_decision"] or {}
        if s is None or a.get("servo") is None:
            continue
        t0 = time.perf_counter()
        snap = replay.snapshot(s)
        act = SimpleNamespace(servo=a["servo"], buffer=a["buffer"])
        v16 = replay.issued(rec)
        sac = np.clip(np.asarray(rec["policy_action"], float), -1, 1)
        reference = v16 if np.isfinite(v16[1]) else np.array([v16[0], -1.0])
        grid = sc.projection_candidates(reference)
        firsts = np.vstack([v16[None], sac[None], grid])
        row = {"case": case, "step": step, "why": why_by_step[step],
               "unchecked": wanted[step]}
        for family, bank in FAMILIES.items():
            checker = sc.ContingencyChecker(snap, act, tails=bank)
            slack, _, _ = checker.evaluate(sc.sequences_for(firsts, bank))
            best = np.where(np.isfinite(slack), slack, -np.inf).reshape(len(firsts), -1).max(axis=1)
            row[f"{family}_v16"] = float(best[0])
            row[f"{family}_sac"] = float(best[1])
            row[f"{family}_projection_best"] = float(best[2:].max())
            nearest = {}
            for m in MARGINS:
                hit = np.flatnonzero(best[2:] >= m)
                nearest[str(m)] = None if not len(hit) else {
                    "index": int(hit[0]), "command": grid[hit[0]].tolist(),
                    "distance_to_v16": float(sc.distance_to(reference, grid[hit[0]][None])[0])}
            row[f"{family}_nearest_projection"] = nearest
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
    return rows


def options_part(replay_tag, tag, part, parts, max_margin):
    out = DEV / "offline_variants" / tag
    episodes_dir = out / "options_episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    saved = {ep["episode"]["case"]: ep for ep in episode_rows(replay_tag)[0]}
    todo = [s for s in replay.sources({}) if s[0] in saved][part - 1::parts]
    for case, label, trace, _ in todo:
        name = episodes_dir / (case.replace(":", "_") + ".json")
        if name.exists():
            print(f"{case} reused completed episode file", flush=True)
            continue
        rows = saved[case]["rows"]
        wanted = {r["step"]: bool(r["unchecked"]) for r in rows
                  if r["unchecked"] or not (r["none_v16_certified"] and r["none_v16_slack"] >= max_margin)}
        why = {r["step"]: r["why"] for r in rows}
        t0 = time.perf_counter()
        er = option_rows(case, label, trace, wanted, why)
        episode = {"case": case, "source": label, "category": saved[case]["episode"]["category"],
                   "contact": saved[case]["episode"]["contact"], "selected": len(er),
                   "trace_sha256": sha(trace)}
        with name.open("x", encoding="utf-8") as stream:
            json.dump({"episode": episode, "rows": er}, stream)
        print(f"{case} {len(er)} decisions {time.perf_counter() - t0:.0f}s", flush=True)


def options_merge(replay_tag, tag):
    out = DEV / "offline_variants" / tag
    files = sorted((out / "options_episodes").glob("*.json"))
    episodes = [json.loads(f.read_text()) for f in files]
    replay_rows = {ep["episode"]["case"]: {r["step"]: r for r in ep["rows"]} for ep in episode_rows(replay_tag)[0]}
    summary = {"episodes": len(episodes), "by_margin": []}
    for m in MARGINS:
        counts = defaultdict(Counter)
        distances = defaultdict(list)
        for ep in episodes:
            success = ep["episode"]["category"] in ("rescue", "both_goal")
            for r in ep["rows"]:
                base = replay_rows[ep["episode"]["case"]][r["step"]]
                v16_ok = bool(base["none_v16_certified"] and base["none_v16_slack"] >= m)
                if not r["unchecked"] and v16_ok:
                    continue
                group = ("unchecked" if r["unchecked"] else "checked_uncertified") + (
                    "_v16_success" if success else "_v16_failure")
                c = counts[group]
                c["decisions"] += 1
                for family in FAMILIES:
                    c[f"{family}_v16"] += r[f"{family}_v16"] >= m
                    c[f"{family}_sac"] += r[f"{family}_sac"] >= m
                    near = r[f"{family}_nearest_projection"][str(m)]
                    c[f"{family}_projection"] += near is not None
                    c[f"{family}_any"] += (r[f"{family}_v16"] >= m or r[f"{family}_sac"] >= m
                                           or near is not None)
                    if near is not None:
                        distances[(group, family)].append(near["distance_to_v16"])
        summary["by_margin"].append({
            "margin_m": m, "counts": {g: dict(c) for g, c in counts.items()},
            "nearest_projection_distance_median": {f"{g}:{f}": float(np.median(d))
                                                   for (g, f), d in distances.items() if d}})
    result = {"schema": 1, "replay": replay_tag, "summary": summary,
              "episodes": [ep["episode"] for ep in episodes],
              "script_sha256": sha(Path(__file__)),
              "module_sha256": {m: sha(ROOT / "src" / m) for m in ("safety_v20_contingency.py",)}}
    target = out / "options.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1)
        stream.write("\n")
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=("margin", "options"))
    parser.add_argument("--replay", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--part", type=int, default=1)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--max-margin", type=float, default=max(MARGINS))
    args = parser.parse_args()
    if Path(args.tag).name != args.tag:
        parser.error("Use a simple tag")
    out = DEV / "offline_variants" / args.tag
    if args.analysis == "margin":
        out.mkdir(parents=True, exist_ok=True)
        target = out / "margin.json"
        if target.exists():
            parser.error("Existing result; results are never overwritten")
        result = margin_analysis(args.replay)
        result["script_sha256"] = sha(Path(__file__))
        target.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        for row in result["margins"]:
            print(json.dumps(row))
    elif args.merge:
        options_merge(args.replay, args.tag)
    else:
        options_part(args.replay, args.tag, args.part, args.parts, args.max_margin)


if __name__ == "__main__":
    main()
