"""Search for a full backup after exactly one current policy decision.

The existing fixed-policy search holds SAC for COMMIT_S (1 s), although the
environment requests another action after UPDATE_RATE (.5 s). Here only the
first decision is fixed; later commands are explicit optimized backups, not
predicted future SAC actions. The horizon and supplied hard checker stay the
same. The default minimum clearance is the existing trigger margin.

Method: Wabersich & Zeilinger (2021), Sec. 4.1, Eq. (5a)-(5f), motivates
retaining the learning input when a feasible backup exists:
https://arxiv.org/html/1812.05506v4#S4.SS1 . Zheng et al. (2022) supplies the
CEM sampling/elite-refitting optimizer inspiration:
https://arxiv.org/abs/2102.12124v3 . This utility implements neither paper's
uncertainty treatment, terminal safety set or CLF/CBF guarantees. A finite
search may miss a feasible tail; an accepted sampled model trajectory does
not guarantee real collision avoidance, recursive feasibility or goals.

Standalone only: no environment reads, policy calls or controller mutation.
Rollout/evaluate callbacks must use the same onboard model, complete hard
checks and pre-command actuator history as the caller's other alternatives.
Callbacks are expected not to mutate their inputs. The caller retains the
returned exact plan if it elects to issue the policy action.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

import constants as cfg
import safety_v2 as v2
from safety_v7 import replace_first_action


MAX_PLANS = 192  # Includes warm/primitive seeds, not just random samples.
ROUNDS = 3
ELITES = 8
TAIL_BLOCK_DECISIONS = 2


@dataclass
class PrefixSearchResult:
    plan: Optional[np.ndarray]
    clearance: float
    diagnostics: dict


def search(snap, act, action, backup, rollout, evaluate, rng, *, traffic: bool,
           minimum_clearance=None) -> PrefixSearchResult:
    """Check <=192 full plans with exactly one fixed current policy action.

    ``backup`` is None or a possibly shortened (decisions, 2) stored plan.
    Its exact remaining tail is checked after existing continuation padding.
    Random plans use two-decision blocks starting at decision 1; the final
    block may contain one decision. Seeds are checked without block rounding.
    NaN throttle means full astern and is allowed only when traffic=True;
    finite throttle -1 remains an ordinary policy-space stop command.

    The default required clearance is v2.TRIGGER_MARGIN_M. An explicitly
    supplied clearance must be finite and nonnegative. Failure returns
    plan=None and diagnostics; it never authorizes an unchecked policy pass.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("Pass a caller-owned numpy.random.Generator")
    action = np.array(action, dtype=float, copy=True)
    if action.shape != (2,) or not np.isfinite(action).all() or (np.abs(action) > 1.).any():
        raise ValueError("Policy action must contain two finite values in [-1, 1]")
    margin = float(v2.TRIGGER_MARGIN_M if minimum_clearance is None else minimum_clearance)
    if not np.isfinite(margin) or margin < 0.:
        raise ValueError("Minimum clearance must be finite and nonnegative")
    decisions = int(np.ceil(v2.HORIZON_S / cfg.UPDATE_RATE))
    if decisions < 2:
        raise ValueError("Horizon must leave at least one decision for the backup")
    seed_rows = []
    dropped_brake_backup = False
    if backup is not None:
        warm = replace_first_action(backup, action)
        dropped_brake_backup = bool(not traffic and np.isnan(warm[:, 1]).any())
        if not dropped_brake_backup:
            seed_rows.append(warm)
    seed_rows.append(np.tile(action, (decisions, 1)))
    # Small existing-control-alphabet family: full left/centre/right with
    # stop/cruise/full ahead, plus astern only in the caller's traffic mode.
    for rudder in (-1., 0., 1.):
        for throttle in ((-1., 0., 1., np.nan) if traffic else (-1., 0., 1.)):
            plan = np.tile((rudder, throttle), (decisions, 1))
            plan[0] = action
            seed_rows.append(plan)
    seeds = np.asarray(seed_rows)
    if MAX_PLANS < len(seeds) + ROUNDS or ROUNDS < 1 or ELITES < 1:
        raise ValueError("Search budget must fit seeds and at least one sample per round")
    blocks = int(np.ceil((decisions - 1) / TAIL_BLOCK_DECISIONS))
    lower = np.tile((-1., -2. if traffic else -1.), (blocks, 1))
    upper = np.ones((blocks, 2))

    def encode(plans):
        return np.nan_to_num(plans[:, 1::TAIL_BLOCK_DECISIONS], nan=-2.)

    def decode(values):
        values = np.clip(values, lower, upper)
        plans = np.empty((len(values), decisions, 2))
        plans[:, 0] = action
        plans[:, 1:] = np.repeat(values, TAIL_BLOCK_DECISIONS, axis=1)[:, :decisions - 1]
        if traffic:
            plans[:, 1:, 1] = np.where(plans[:, 1:, 1] < -1., np.nan, plans[:, 1:, 1])
        return plans

    best_plan, best_clearance = None, -np.inf
    evaluated = accepted_count = hard_passing_count = 0
    maximum_clearance = maximum_hard_passing_clearance = -np.inf

    def check(plans):
        nonlocal best_plan, best_clearance, evaluated, accepted_count, hard_passing_count
        nonlocal maximum_clearance, maximum_hard_passing_clearance
        first, clear = evaluate(snap, rollout(snap, act, plans))
        first, clear = np.asarray(first, dtype=float), np.asarray(clear, dtype=float)
        if first.shape != (len(plans),) or clear.shape != first.shape:
            raise ValueError("Evaluator must return one first-contact time and clearance per plan")
        room = np.where(np.isnan(clear), -np.inf, clear)
        hard_passing = np.isposinf(first) & (room >= 0.)
        accepted = hard_passing & (room >= margin)
        order = np.lexsort((-room, ~accepted))
        evaluated += len(plans)
        accepted_count += int(accepted.sum())
        hard_passing_count += int(hard_passing.sum())
        maximum_clearance = max(maximum_clearance, float(room.max()))
        if hard_passing.any():
            maximum_hard_passing_clearance = max(
                maximum_hard_passing_clearance, float(room[hard_passing].max()))
        if accepted.any() and room[order[0]] > best_clearance:
            best_plan, best_clearance = plans[order[0]].copy(), float(room[order[0]])
        return order

    order = check(seeds)
    mean = encode(seeds[order[:ELITES]]).mean(axis=0)
    std = (upper - lower) / 2.
    remaining = MAX_PLANS - len(seeds)
    for iteration in range(ROUNDS):
        count = int(np.ceil(remaining / (ROUNDS - iteration)))
        samples = decode(rng.normal(mean, std, size=(count, blocks, 2)))
        order = check(samples)
        elite = encode(samples[order[:ELITES]])
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), np.sqrt(np.finfo(float).eps))
        remaining -= count
    return PrefixSearchResult(best_plan, best_clearance, {
        "accepted": best_plan is not None, "fixed_policy_decisions": 1,
        "fixed_policy_seconds": float(cfg.UPDATE_RATE), "horizon_decisions": decisions,
        "minimum_clearance": margin, "rounds": ROUNDS, "seed_plans": len(seeds),
        "sampled_plans": evaluated - len(seeds), "evaluated_plans": evaluated,
        "maximum_plans": MAX_PLANS, "accepted_plans": accepted_count,
        "hard_passing_plans": hard_passing_count,
        "maximum_clearance": maximum_clearance,
        "maximum_hard_passing_clearance": maximum_hard_passing_clearance,
        "dropped_brake_backup": dropped_brake_backup,
        "backup_seed_checked": backup is not None and not dropped_brake_backup,
    })
