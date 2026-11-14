"""
Live UDP bridge for Paper 3: the deployed SAC policy (baseline-v3) on the Bluefin vessel, with
the E1 safety layer in shadow (FIELD_TRIALS_PLAN.md sections 3 and 5, decision D-4).

Shadow session against the fake vessel (bench):
  python fake_vessel_replay.py --log 2026-07-03/1.log
  python udp_live_p3.py --test-case 1 --run-id bench-0703-1 --no-send

Live (calibration session 2, field trials):
  python udp_live_p3.py --server-ip 10.201.205.110 --layout L1 --run-id T1-A-L1-r1

Geometry (one of): --case <test set v4 ID> (a simulated twin's map and path), --layout L1|L2|L3
(Paper 2's basin layouts), --start x,y --goal x,y (a basin leg), --test-case N (the July
bridge's `test_run.TestCase`, for replaying the July logs).

Per LiDAR frame (2 Hz):
- the frame is paired with its own pose line (`udp_live_rl.PoseSync`) and mapped into the
  simulator's frame (`p3_bridge.frame_to_sensor`); the first pose is anchored at the path
  start, as the July bridge did, unless --origin map;
- the 70-value observation is built by the simulator's perception (`p3_bridge.LiveEnv`), the
  policy acts, the command is resolved as `ASVLidarEnv.step` resolves it, and `$CMD` goes out;
- the decision then goes to the shadow layer, in a worker process by default, so its run time
  never delays a command (--shadow-inline runs it here, after the command has gone);
- everything is logged on the frame's clock under `<out-dir>/<run-id>/`: `decisions.jsonl`
  (sensor input, observation, action, command, tracks and encounter contexts, timings),
  `shadow.jsonl` (every shadow decision), `telemetry.log` (every received line as received)
  and `summary.json`.

Operator stop (the only physical stop in shadow mode): E latches full astern (S2 = -100) until
the measured surge reads stopped, then zero propulsion; R hands control back after the minimum
hold.  The latch is `static_dynamic_obstacles/src/emergency_stop.py`, through the July bridge's
`apply_emergency_stop`.  The key acts only while this program runs and has its window.

--safety active puts E1 in the command path (its brake is full astern).  Not for the water
before the ESC reverses (FIELD_CALIBRATION_PLAN.md P-8) and the layer's run time meets T-14.
"""

from __future__ import annotations

import p3_bridge as pb  # first: puts the simulator's modules ahead of this directory's

import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import os
import queue
import socket
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import constants as cfg
import emergency_stop as estop_mod
import scenario as scn
import tracking
from log_parser import BluefinStreamDecoder


def _paper2_bridge():
    """`udp_live_rl` (the July bridge, kept as flown), for `PoseSync` and the stop latch.

    It imports this directory's `asv_lidar` and `lidar_pooling`, names the simulator's modules
    also carry.  They stay bound inside it and are taken out of `sys.modules` again, so every
    later import resolves to the simulator's.
    """
    clash = ("asv_lidar", "lidar_pooling")
    saved = {n: sys.modules.pop(n) for n in clash if n in sys.modules}
    sys.path.insert(0, pb.HERE)
    try:
        import udp_live_rl
    finally:
        sys.path.pop(0)
        for n in clash:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    for n in clash:
        mod = sys.modules.get(n)
        if mod is not None and os.path.dirname(os.path.abspath(mod.__file__)) != os.path.abspath(pb.SIM_SRC):
            raise ImportError(f"{n} resolves to {mod.__file__}")
    return udp_live_rl


p2 = _paper2_bridge()

RECOGNISED = (BluefinStreamDecoder._re_rc, BluefinStreamDecoder._re_hdg, BluefinStreamDecoder._re_pose,
              BluefinStreamDecoder._re_lidar)


def _xy(text: str) -> Tuple[float, float]:
    x, y = (float(v) for v in text.split(","))
    return x, y


def scenario_from_args(args) -> Tuple[object, Optional[int], str]:
    if args.case:
        built, seed = pb.load_test_case(args.case)
        return built, seed, f"test set v4 {args.case}"
    if args.layout:
        return pb.layout_scenario(args.layout), None, f"layout {args.layout.upper()}"
    if args.test_case is not None:
        return test_case_scenario(args.test_case), None, f"test_run case {args.test_case}"
    if args.start and args.goal:
        return pb.field_scenario(_xy(args.start), _xy(args.goal), case_id="leg"), None, "leg"
    raise SystemExit("give --case, --layout, --test-case, or --start and --goal")


def test_case_scenario(number: int):
    """The July bridge's start, goal and panels (`field_deployment/test_run.py`)."""
    sys.path.insert(0, pb.HERE)
    try:
        from test_run import TestCase
    finally:
        sys.path.pop(0)
    case = TestCase()
    sx, sy, gx, gy = case.position(test_case=number)
    return pb.field_scenario((sx, sy), (gx, gy), panels=case.obstacles(test_case=number),
                             case_id=f"test_run-{number}")


def path_start(built) -> Tuple[float, float]:
    """Where `ASVLidarEnv._load_generated` starts the path (the own ship's start)."""
    points = built.channel.reference_path_points(start_s=scn.own_start_s(built.encounter_class, built.channel))
    return float(points[0][0]), float(points[0][1])


def make_latch(args) -> estop_mod.EmergencyStop:
    return estop_mod.EmergencyStop(
        stop_speed=args.estop_stop_speed, min_hold_s=args.estop_min_hold_s,
        max_hold_s=args.estop_auto_release_s if args.estop_auto_release_s > 0 else float("inf"),
        max_brake_s=args.estop_max_brake_s)


# ---------------------------------------------------------------------------
# Shadow layer sinks
# ---------------------------------------------------------------------------
class InlineShadow:
    """The shadow layer in this process, on the bridge's own environment."""

    def __init__(self, bridge: pb.Bridge, path: str, run_id: str, compact: bool) -> None:
        self.bridge, self.run_id, self.compact = bridge, run_id, compact
        self.layer = pb.SafetyLayer("shadow")
        self.out = open(path, "a", encoding="utf-8", buffering=1)
        self.last: Dict = {}

    def __call__(self, decision: pb.Decision) -> None:
        c = decision.command
        sd = pb.shadow_inline(self.layer, self.bridge.env, decision.action, c.rudder_pct, c.rpm)
        record = {"type": "shadow", "run_id": self.run_id, "k": decision.k, "t_frame": decision.sensor.t,
                  "obs_matches_bridge": True, "seconds": {"layer": sd.seconds},
                  "safety": pb.safety_record(sd, compact=self.compact)}
        self.out.write(json.dumps(pb.jsonable(record), allow_nan=False) + "\n")
        self.last = {"k": decision.k, "why": sd.why, "changed": sd.changed, "committed": sd.committed,
                     "t_star": sd.details.get("estop_t_star"), "seconds": sd.seconds}

    def poll(self) -> None:
        pass

    def close(self) -> Dict:
        self.out.close()
        return {"decisions": self.layer.decisions, "changes": self.layer.changes,
                "first_change": self.layer.first_change, "committed": self.layer.filter.estop_committed_decision}


class RunnerShadow(InlineShadow):
    """The worker's code path (`ShadowRunner`, its own environment) in this process: the bench replay."""

    def __init__(self, built, seed, path: str, run_id: str, compact: bool) -> None:
        self.runner = pb.ShadowRunner(built, seed=seed)
        self.layer = self.runner.layer
        self.run_id, self.compact = run_id, compact
        self.out = open(path, "a", encoding="utf-8", buffering=1)
        self.last = {}
        self.mismatches = 0

    def __call__(self, decision: pb.Decision) -> None:
        c = decision.command
        sd, same, perceive = self.runner.process(decision.sensor, decision.executed_prev, decision.action,
                                                 c.rudder_pct, c.rpm, obs=decision.obs)
        self.mismatches += int(not same)
        record = {"type": "shadow", "run_id": self.run_id, "k": decision.k, "t_frame": decision.sensor.t,
                  "obs_matches_bridge": bool(same), "seconds": {"perception": perceive, "layer": sd.seconds},
                  "safety": pb.safety_record(sd, compact=self.compact)}
        self.out.write(json.dumps(pb.jsonable(record), allow_nan=False) + "\n")
        self.last = {"k": decision.k, "why": sd.why, "changed": sd.changed, "committed": sd.committed,
                     "t_star": sd.details.get("estop_t_star"), "seconds": sd.seconds}

    def close(self) -> Dict:
        return dict(super().close(), obs_mismatches=self.mismatches)


class WorkerShadow:
    """The shadow layer in a worker process (`p3_bridge.shadow_worker`)."""

    def __init__(self, built, seed, path: str, run_id: str, compact: bool, *, timeout_s: float = 180.0) -> None:
        ctx = mp.get_context("spawn")
        self.inbox, self.outbox = ctx.Queue(), ctx.Queue()
        self.process = ctx.Process(target=pb.shadow_worker, daemon=True,
                                   args=(self.inbox, path, built, seed, {"run_id": run_id, "compact": compact},
                                         self.outbox))
        self.process.start()
        self.last: Dict = {}
        self.done: Optional[Dict] = None
        self.mismatches = 0
        try:
            ready = self.outbox.get(timeout=timeout_s)
        except queue.Empty:
            raise SystemExit("the shadow worker did not start")
        if ready.get("type") != "ready":
            raise SystemExit(f"unexpected message from the shadow worker: {ready}")

    def __call__(self, decision: pb.Decision) -> None:
        self.inbox.put(pb.shadow_item(decision))

    def _take(self, msg: Dict) -> None:
        if msg.get("type") == "shadow":
            self.last = msg
            self.mismatches += int(not msg.get("obs_matches_bridge", True))
        elif msg.get("type") == "done":
            self.done = msg

    def poll(self) -> None:
        while True:
            try:
                self._take(self.outbox.get_nowait())
            except queue.Empty:
                return

    def close(self, drain_s: float = 900.0) -> Dict:
        self.inbox.put(None)
        deadline = time.monotonic() + drain_s
        while self.done is None and self.process.is_alive() and time.monotonic() < deadline:
            try:
                self._take(self.outbox.get(timeout=1.0))
            except queue.Empty:
                pass
        self.process.join(timeout=10.0)
        return dict(self.done or {"error": "the shadow worker did not finish"}, obs_mismatches=self.mismatches)


# ---------------------------------------------------------------------------
# One frame, from pose sync to logged decision
# ---------------------------------------------------------------------------
class FrameHandler:
    """Per-frame processing shared by the live loop and the bench replay (`p3_shadow_replay.py`)."""

    def __init__(self, *, bridge: pb.Bridge, policy: pb.Policy, run_id: str, run_dir: str, start_xy,
                 transform: Optional[pb.PoseTransform], map_rotation: float, lidar_index0_deg: float,
                 latch: estop_mod.EmergencyStop, safety_mode: str, compact: bool, config: Dict,
                 shadow=None, pose_point: str = "origin", min_interval_s: float = 0.4) -> None:
        self.bridge, self.policy, self.run_id = bridge, policy, run_id
        self.start_xy, self.transform, self.map_rotation = start_xy, transform, map_rotation
        self.lidar_index0_deg = lidar_index0_deg
        self.latch, self.safety_mode, self.compact, self.config = latch, safety_mode, compact, config
        self.shadow = shadow
        self.pose_point, self.min_interval_s = pose_point, float(min_interval_s)
        self.skipped = 0
        self.decisions = open(os.path.join(run_dir, "decisions.jsonl"), "a", encoding="utf-8", buffering=1)
        self.raw = open(os.path.join(run_dir, "telemetry.log"), "a", encoding="utf-8", buffering=1)
        self.raw.write("#P3CONFIG," + json.dumps(pb.jsonable(config)) + "\n")
        self.header_written = False
        self.t_prev: Optional[float] = None
        self.request: Optional[str] = None
        self.release = False
        self.other_lines: List[Dict] = []
        self.trail: List[Tuple[float, float]] = []
        self.frames = 0
        self.errors = 0
        self.last: Optional[pb.Decision] = None

    def note_line(self, line: str, now: float) -> None:
        self.raw.write(line + "\n")
        stripped = line.strip()
        if stripped and not any(rx.match(stripped) for rx in RECOGNISED):
            self.other_lines.append({"t_wall": now, "line": stripped})        # IMU, or anything new

    def handle(self, frame, *, fresh: bool, pose_age_s: float, pose_wait_s: float, t_release: float,
               send: Callable[[str], bool]) -> Optional[pb.Decision]:
        try:
            return self._handle(frame, fresh, pose_age_s, pose_wait_s, t_release, send)
        except Exception as exc:            # a frame the bridge cannot use: stop propulsion, carry on
            self.errors += 1
            print(f"[P3 ERROR] {type(exc).__name__}: {exc}")
            self.decisions.write(json.dumps({"type": "error", "run_id": self.run_id, "t_frame": frame.t_sec,
                                             "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            send("$CMD,0.00,0.00")
            return None

    def _handle(self, frame, fresh, pose_age_s, pose_wait_s, t_release, send):
        # One decision per decision interval: a faster telemetry stream (P-1) is decimated.
        if self.t_prev is not None and frame.t_sec - self.t_prev < self.min_interval_s:
            self.skipped += 1
            return None
        if self.transform is None:
            self.transform = pb.PoseTransform.anchored((frame.x_m, frame.y_m), self.start_xy, self.map_rotation)
        sensor = pb.frame_to_sensor(frame, self.transform, lidar_index0_deg=self.lidar_index0_deg,
                                    pose_stale=not fresh, missed_cycles=pb.missed_cycles(frame.t_sec, self.t_prev),
                                    pose_point=self.pose_point)
        dt_cmd = cfg.UPDATE_RATE if self.t_prev is None else max(1e-3, frame.t_sec - self.t_prev)
        self.t_prev = frame.t_sec
        before = self.latch.state
        s2_latch, state = p2.apply_emergency_stop(self.latch, 0.0, t=frame.t_sec, dt=dt_cmd, u=sensor.ego[0],
                                                  request=self.request, release=self.release)
        self.request = None
        if state != estop_mod.HOLDING:
            self.release = False
        if state != before:
            print(f"[E-STOP] {before} -> {state} at t={frame.t_sec:.2f}s")
        override = s2_latch if self.latch.active else None

        decision = self.bridge.decide(sensor, override_s2=override)
        c = decision.command
        sent = send(c.message())
        t_sent = time.perf_counter()
        if not self.header_written:
            header = pb.run_header(run_id=self.run_id, policy=self.policy, env=self.bridge.env,
                                   safety=self.safety_mode, extra=self.config)
            self.decisions.write(json.dumps(header, allow_nan=False) + "\n")
            self.header_written = True
        if self.shadow is not None:
            self.shadow(decision)
        t_shadow = time.perf_counter()

        record = pb.decision_record(decision, run_id=self.run_id)
        record["perception"] = pb.perception_record(self.bridge.env)
        if decision.safety is not None:
            record["safety"] = pb.safety_record(decision.safety, compact=self.compact)
        record["telemetry"] = {
            "ts": frame.ts_str, "seq": frame.seq, "x": frame.x_m, "y": frame.y_m, "yaw": frame.yaw_deg,
            "u": frame.u_body_mps, "v_port": frame.v_body_mps, "yaw_rate": frame.yaw_rate, "speed": frame.speed_mps,
            "s1": frame.s1, "s2": frame.s2, "hdg_ref": frame.hdg_ref_deg, "pose_fresh": bool(fresh),
            "pose_age_s": pose_age_s, "pose_wait_s": pose_wait_s, "imu": None}
        record["other_lines"], self.other_lines = self.other_lines, []
        record["operator_stop"] = {"state": self.latch.state,
                                   "reason": self.latch.events[-1].reason if self.latch.active else ""}
        record["sent"] = bool(sent)
        record["clock"] = {"t_wall_released": t_release, "t_wall_sent": t_sent}
        record["seconds"].update(release_to_sent=t_sent - t_release,
                                 lidar_line_to_sent=(t_sent - t_release) + float(pose_wait_s),
                                 shadow_handoff=t_shadow - t_sent)
        self.decisions.write(json.dumps(pb.jsonable(record), allow_nan=False) + "\n")
        self.raw.write(f"#P3,k={decision.k},ts={frame.ts_str},a0={decision.action[0]:+.6f},"
                       f"a1={decision.action[1]:+.6f},cmd='{c.message()}',sent={int(bool(sent))},"
                       f"source={c.source},estop={self.latch.state}\n")
        x, y, _ = self.bridge.env.estimated_pose()
        self.trail.append((x, y))
        self.frames += 1
        self.last = decision
        return decision

    def close(self) -> None:
        self.decisions.close()
        self.raw.close()


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
class Display:
    """A plain operator panel: map, scan, tracks, command, stop and shadow state."""

    def __init__(self, title: str) -> None:
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((1280, 720))
        self.font = pygame.font.SysFont("consolas", 17) or pygame.font.Font(None, 17)
        self.clock = pygame.time.Clock()

    def events(self) -> List[str]:
        out = []
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                out.append("quit")
            elif event.type == self.pg.KEYDOWN:
                out.append({self.pg.K_ESCAPE: "quit", self.pg.K_e: "stop", self.pg.K_r: "release"}.get(event.key, ""))
        return out

    def draw(self, lines: List[str], env, trail, banner) -> None:
        pg, screen = self.pg, self.screen
        screen.fill((20, 20, 25))
        y = 10
        if banner is not None:
            screen.blit(self.font.render(banner[0], True, banner[1]), (10, y))
            y += 26
        for s in lines:
            screen.blit(self.font.render(s, True, (235, 235, 245)), (10, y))
            y += 21
        if env is not None:
            rect = pg.Rect(840, 0, 440, 720)
            scale = min(rect.width / (cfg.MAP_WIDTH + 1.0), rect.height / (cfg.MAP_HEIGHT + 1.0))

            def w2s(p):
                return (int(rect.x + (float(p[0]) + 0.5) * scale), int(rect.bottom - (float(p[1]) + 0.5) * scale))

            pg.draw.polygon(screen, (160, 80, 80), [w2s(p) for p in env.boundary_polygon], 2)
            for poly in env.obstacles:
                pg.draw.polygon(screen, (200, 60, 60), [w2s(p) for p in poly], 0)
            pg.draw.lines(screen, (80, 220, 120), False, [w2s(p) for p in env.path.points], 2)
            if len(trail) > 1:
                pg.draw.lines(screen, (90, 160, 255), False, [w2s(p) for p in trail], 2)
            x, y_, h = env.estimated_pose()
            ranges = np.asarray(env.raw_ranges)
            mask = ranges < cfg.LIDAR_RANGE - 1e-5
            sx, sy = tracking.sensor_origin(x, y_, h)
            ang = np.radians(np.asarray(env.lidar.bearings)[mask] + h)
            for px, py in zip(sx + ranges[mask] * np.sin(ang), sy + ranges[mask] * np.cos(ang)):
                q = w2s((px, py))
                if rect.collidepoint(q):
                    screen.set_at(q, (230, 230, 120))
            for t in env.tracker.tracks:
                colour = (255, 120, 40) if t.is_dynamic else (120, 120, 120)
                pg.draw.circle(screen, colour, w2s(t.position), 6, 2)
                pg.draw.line(screen, colour, w2s(t.position), w2s(t.position + 2.0 * t.velocity), 2)
            a = math.radians(h)
            pg.draw.circle(screen, (255, 255, 255), w2s((x, y_)), 5)
            pg.draw.line(screen, (255, 255, 255), w2s((x, y_)), w2s((x + 0.9 * math.sin(a), y_ + 0.9 * math.cos(a))), 2)
        pg.display.flip()
        self.clock.tick(60)

    def close(self) -> None:
        self.pg.quit()


def status_lines(handler: FrameHandler, *, run_id, geometry, safety, send, stats, pose_sync) -> List[str]:
    d = handler.last
    lines = [f"run {run_id}  {geometry}  safety={safety}  send={'yes' if send else 'no'}",
             f"lines={stats['lines']} decisions={handler.frames} backlog={stats['backlog_frames']} "
             f"stale={pose_sync.stale}/{pose_sync.released} errors={handler.errors}"]
    if d is None:
        return lines + ["waiting for the first LiDAR frame ..."]
    env, c, s = handler.bridge.env, d.command, d.sensor
    x, y, h = env.estimated_pose()
    lines += [f"t={s.t:.2f}s  pose x={x:+.2f} y={y:+.2f} hdg={h:6.1f}  "
              f"ego u={s.ego[0]:+.2f} v={s.ego[1]:+.2f} r={s.ego[2]:+.1f}",
              f"action a0={d.action[0]:+.3f} a1={d.action[1]:+.3f}",
              f"command rudder={c.rudder_pct:+.1f}% (vessel {c.vessel_rudder:+.1f}) rpm={c.rpm:.2f} "
              f"S2={c.s2:.1f} [{c.source}]",
              f"tracks={len(env.tracker.tracks)} dynamic={len(env.tracks)} "
              f"contexts={[(k, v.cls, v.state) for k, v in env.encounter_contexts.items()]}",
              f"timing obs={1e3 * d.seconds['observation']:.0f} ms policy={1e3 * d.seconds['policy']:.1f} ms",
              f"OPERATOR STOP: {handler.latch.state.upper()}   (E stop, R release)"]
    if d.safety is not None:
        lines.append(f"E1 ACTIVE: {d.safety.why} changed={d.safety.changed} brake={d.safety.brake} "
                     f"stop={d.safety.committed} {1e3 * d.safety.seconds:.0f} ms")
    elif handler.shadow is not None and handler.shadow.last:
        last = handler.shadow.last
        lines.append(f"E1 shadow k={last['k']} (lag {d.k - int(last['k'])}): {last['why']} "
                     f"would change={last['changed']} stop={last['committed']} T*={last.get('t_star')} "
                     f"{1e3 * last['seconds']:.0f} ms")
    return lines


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    # Network options, their defaults and the shared flags keep the July bridge's (udp_live_rl.py)
    # names.  No abbreviations: a July option this bridge lacks must fail, not match a longer one.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--bind-ip", default="0.0.0.0")
    ap.add_argument("--local-port", type=int, default=5000)
    ap.add_argument("--server-ip", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=5050)
    ap.add_argument("--run-id", default=None, help="run identifier from the run sheet (default: time stamp)")
    ap.add_argument("--out-dir", default=os.path.join(pb.HERE, "p3_runs"))
    g = ap.add_argument_group("geometry")
    g.add_argument("--case", default=None, help="test set v4 ID, e.g. P2-L1-HO-FIX-11")
    g.add_argument("--layout", default=None, help="L1, L2 or L3")
    g.add_argument("--test-case", type=int, default=None, help="the July bridge's test_run case")
    g.add_argument("--start", default=None, help="x,y in the map frame")
    g.add_argument("--goal", default=None, help="x,y in the map frame")
    g.add_argument("--origin", choices=("start", "map"), default="start",
                   help="start: the first pose is placed at the path start (the July rule); "
                        "map: telemetry is in the map frame up to --map-rotation and --map-offset")
    g.add_argument("--map-rotation", type=float, default=0.0, help="deg, clockwise, telemetry -> map")
    g.add_argument("--map-offset", default="0,0", help="dx,dy, m, applied after the rotation (--origin map)")
    ap.add_argument("--lidar-index0-deg", type=float, default=0.0,
                    help="bearing of the decoder's beam 0, deg clockwise from the bow (July logs: 0)")
    ap.add_argument("--pose-point", choices=("origin", "sensor"), default="origin",
                    help="the point the telemetry pose belongs to: the vessel reference point (the July "
                         "assumption) or the LiDAR, LIDAR_OFFSET_M ahead of it")
    ap.add_argument("--min-decision-interval", type=float, default=0.4,
                    help="s; frames closer than this to the last decision are logged but not acted on")
    ap.add_argument("--model-path", default=pb.DEFAULT_MODEL)
    ap.add_argument("--safety", choices=("shadow", "active", "off"), default="shadow")
    ap.add_argument("--shadow-inline", action="store_true",
                    help="run the shadow layer in this process after each command, not in a worker")
    ap.add_argument("--full-filter-trace", action="store_true", help="keep the layer's bulky diagnostics")
    ap.add_argument("--no-send", "--shadow", dest="no_send", action="store_true",
                    help="compute and log, but send no $CMD (--shadow: the July bridge's name)")
    ap.add_argument("--rudder-sign", type=float, default=pb.RUDDER_SIGN)
    ap.add_argument("--no-pose-sync", action="store_true")
    ap.add_argument("--pose-wait-s", type=float, default=p2.POSE_WAIT_S)
    ap.add_argument("--estop-stop-speed", type=float, default=0.05)
    ap.add_argument("--estop-min-hold-s", type=float, default=2.0)
    ap.add_argument("--estop-max-brake-s", type=float, default=8.0)
    ap.add_argument("--estop-auto-release-s", type=float, default=0.0)
    ap.add_argument("--headless", action="store_true", help="no window (bench only: no E key)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many decisions (bench)")
    ap.add_argument("--idle-exit-s", type=float, default=0.0,
                    help="stop when no line has arrived for this long after the first line (bench)")
    return ap


def setup(args, *, shadow_kind: Optional[str] = None):
    """Everything but the socket and the window: shared with the bench replay."""
    pb.RUDDER_SIGN = float(args.rudder_sign)
    table = pb.use_stopping_table()
    if not table["installed"]:
        print(f"[P3] WARNING: stopping-profile table not used ({table['reason']}); each new surge value "
              "costs ~0.5 s once.  Build it: python p3_bridge.py --build-stopping-table")
    run_id = args.run_id or time.strftime("p3-%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=False)
    built, seed, geometry = scenario_from_args(args)
    policy = pb.Policy(args.model_path)
    bridge = pb.Bridge(built, policy, seed=seed,
                       layer=pb.SafetyLayer("active") if args.safety == "active" else None)
    compact = not args.full_filter_trace
    shadow_path = os.path.join(run_dir, "shadow.jsonl")
    kind = shadow_kind or ("inline" if args.shadow_inline else "worker")
    shadow = None
    if args.safety == "shadow":
        shadow = (InlineShadow(bridge, shadow_path, run_id, compact) if kind == "inline"
                  else RunnerShadow(built, seed, shadow_path, run_id, compact) if kind == "runner"
                  else WorkerShadow(built, seed, shadow_path, run_id, compact))
    transform = None
    if args.origin == "map":
        dx, dy = _xy(args.map_offset)
        transform = pb.PoseTransform(args.map_rotation, dx, dy)
    config = {"bridge": "udp_live_p3", "geometry_source": geometry, "seed": seed, "args": vars(args),
              "shadow": kind if shadow is not None else None, "start_xy": path_start(built),
              "wall_utc_at_start": time.time(), "perf_counter_at_start": time.perf_counter(),
              "git_head": pb.git_head(), "source_sha256": pb.source_hashes(),
              "pose_match_s": p2.POSE_MATCH_S, "rudder_sign": pb.RUDDER_SIGN, "compact": compact,
              "stopping_table": table}
    handler = FrameHandler(bridge=bridge, policy=policy, run_id=run_id, run_dir=run_dir,
                           start_xy=path_start(built), transform=transform, map_rotation=args.map_rotation,
                           lidar_index0_deg=args.lidar_index0_deg, latch=make_latch(args),
                           safety_mode=args.safety, compact=compact, config=config, shadow=shadow,
                           pose_point=args.pose_point, min_interval_s=args.min_decision_interval)
    return handler, run_dir, geometry


def finish(handler: FrameHandler, run_dir: str, *, args, geometry: str, stats: Dict, pose_sync) -> Dict:
    shadow = handler.shadow.close() if handler.shadow is not None else None
    layer = handler.bridge.layer
    summary = {"run_id": handler.run_id, "geometry": geometry, "safety": args.safety, **stats,
               "decisions": handler.frames, "frames_skipped": handler.skipped, "errors": handler.errors,
               "pose_sync": {"released": pose_sync.released, "stale": pose_sync.stale,
                             "superseded": pose_sync.superseded},
               "shadow": shadow,
               "active_layer": None if layer is None else {"decisions": layer.decisions, "changes": layer.changes,
                                                           "first_change": layer.first_change,
                                                           "committed": layer.filter.estop_committed_decision},
               "operator_stops": [dataclasses.asdict(e) for e in handler.latch.events]}
    handler.close()
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(pb.jsonable(summary), fh, indent=1)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    handler, run_dir, geometry = setup(args)
    print(f"[P3] run {handler.run_id}: {geometry}, safety={args.safety}, send={'no' if args.no_send else 'yes'}")
    if args.safety == "active":
        print("[P3] WARNING: E1 is in the command path; its stop commands full astern.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind_ip, args.local_port))
    server = (args.server_ip, args.server_port)
    sock.sendto(b"START\n", server)
    print(f"[UDP] START sent to {server[0]}:{server[1]}; listening on {args.bind_ip}:{args.local_port}")

    def send(message: str) -> bool:
        if args.no_send:
            return False
        sock.sendto(message.encode(), server)
        return True

    decoder = BluefinStreamDecoder(lidar_out_beams=cfg.LIDAR_BEAMS)
    pose_sync = p2.PoseSync(decoder, enabled=not args.no_pose_sync, wait_s=args.pose_wait_s)
    display = None if args.headless else Display(f"Paper 3 bridge [{handler.run_id}] safety={args.safety}")
    stats = {"lines": 0, "backlog_frames": 0, "backlog_max": 0}
    last_line: Optional[float] = None
    running = True
    try:
        while running:
            if display is not None:
                for ev in display.events():
                    if ev == "quit":
                        running = False
                    elif ev == "stop":
                        handler.request = "manual (E key)"
                        print("[E-STOP] requested: full astern at the next frame")
                    elif ev == "release":
                        handler.release = True
            # Drain every waiting datagram, then act on every frame released, in order: a
            # backlog (the first decision loads the map) is caught up, never dropped, because
            # the tracker and the observation assume each decision interval's scan.
            released_frames = []
            sock.settimeout(0.02)
            while True:
                try:
                    msg, _ = sock.recvfrom(65535)
                except (socket.timeout, BlockingIOError):
                    break
                except ConnectionResetError:          # Windows: an earlier datagram found no listener
                    continue
                now = time.perf_counter()
                line = msg.decode("utf-8", errors="replace").rstrip("\r\n")
                stats["lines"] += 1
                last_line = now
                handler.note_line(line, now)
                released = pose_sync.push(line, now)
                if released is not None:
                    released_frames.append((released, now, pose_sync.last_fresh, pose_sync.last_pose_age_s,
                                            pose_sync.last_wait_s))
                sock.settimeout(0.0)
            if len(released_frames) > 1:
                stats["backlog_frames"] += len(released_frames) - 1
                stats["backlog_max"] = max(stats["backlog_max"], len(released_frames))
            for frame, t_release, fresh, age, wait in released_frames:
                handler.handle(frame, fresh=fresh, pose_age_s=age, pose_wait_s=wait, t_release=t_release, send=send)
            if handler.shadow is not None:
                handler.shadow.poll()
            if display is not None:
                banner = None
                if handler.latch.active:
                    banner = ("OPERATOR STOP ENGAGED", (255, 80, 80))
                elif last_line is not None and time.perf_counter() - last_line > 1.5:
                    banner = ("NO TELEMETRY FOR > 1.5 s", (255, 200, 60))
                display.draw(status_lines(handler, run_id=handler.run_id, geometry=geometry, safety=args.safety,
                                          send=not args.no_send, stats=stats, pose_sync=pose_sync),
                             handler.bridge.env if handler.frames else None, handler.trail, banner)
            if args.max_frames and handler.frames >= args.max_frames:
                running = False
            if args.idle_exit_s and last_line is not None and time.perf_counter() - last_line > args.idle_exit_s:
                running = False
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        summary = finish(handler, run_dir, args=args, geometry=geometry, stats=stats, pose_sync=pose_sync)
        if display is not None:
            display.close()
        print(f"[P3] {summary['decisions']} decisions logged to {run_dir}; shadow: {summary['shadow']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
