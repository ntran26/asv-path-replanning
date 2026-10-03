"""Finite-budget cross-entropy search for a checked backup trajectory.

Optimizer motivation: Zheng et al., "Safe Learning-based Gradient-free Model
Predictive Control Based on Cross-entropy Method", arXiv:2102.12124v3 (2022),
https://arxiv.org/abs/2102.12124 . We adapt only the sampling/elite-refitting
optimizer, not that paper's Gaussian-process model, CLF/CBF construction or
safety claims. A finite search can fail to find an existing feasible plan.

All safety decisions remain in the supplied rollout/evaluate callbacks. A
returned plan passed both the complete hard check and the unchanged trigger
margin. None of these finite-horizon checks establishes recursive feasibility.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import constants as cfg
import safety_v2 as v2

ROUNDS = 3
SAMPLES = 64
ELITES = 8
BLOCK_DECISIONS = 2


@dataclass
class SearchResult:
    plan: Optional[np.ndarray]
    clearance: float
    diagnostics: dict


def search(snap, act, action, seed_sequences, rollout, evaluate, rng,
           *, fixed_policy: bool, traffic: bool) -> SearchResult:
    """Search with a caller-owned Generator; no environment or global RNG.

    Budget: one evaluation of every supplied seed plus ROUNDS*SAMPLES new
    samples. Valid seeds keep their exact tail (including a shifted warm plan);
    their first commit is repeated to match the command that will be issued.
    Sampled plans are constant within two-decision blocks. In fixed-policy
    mode, the first existing COMMIT_S seconds equal ``action`` exactly.
    Failure returns a result with ``plan=None``, retaining diagnostic counts.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("Pass a caller-owned numpy.random.Generator")
    action = np.array(action, dtype=float, copy=True)
    if action.shape != (2,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
        raise ValueError("Policy action must contain two finite values in [-1, 1]")
    decisions = int(round(v2.HORIZON_S / cfg.UPDATE_RATE))
    commit = int(round(v2.COMMIT_S / cfg.UPDATE_RATE))
    if not 0 < commit <= decisions or commit % BLOCK_DECISIONS or decisions % BLOCK_DECISIONS:
        raise ValueError("Horizon and commit must align with two-decision blocks")
    seeds = np.array(seed_sequences, dtype=float, copy=True)
    if seeds.size == 0:
        seeds = np.empty((0, decisions, 2))
    if seeds.ndim != 3 or seeds.shape[1:] != (decisions, 2):
        raise ValueError("Seeds must have shape (count, horizon decisions, 2)")
    if (not np.isfinite(seeds[:, :, 0]).all()
            or np.isinf(seeds[:, :, 1]).any()
            or (np.abs(np.nan_to_num(seeds)) > 1).any()):
        raise ValueError("Seed controls must be in [-1, 1], with NaN only for brake throttle")
    # A stale/warm astern plan is excluded, not silently changed into coasting.
    dropped_brake_seeds = int(np.isnan(seeds[:, :, 1]).any(axis=1).sum()) if not traffic else 0
    if not traffic:
        seeds = seeds[~np.isnan(seeds[:, :, 1]).any(axis=1)]
    if len(seeds):
        seeds[:, :commit] = action if fixed_policy else seeds[:, :1]
    blocks = decisions // BLOCK_DECISIONS
    lower = np.tile([-1., -2. if traffic else -1.], (blocks, 1))
    upper = np.ones((blocks, 2))

    def encode(plans):
        return np.nan_to_num(plans[:, ::BLOCK_DECISIONS], nan=-2.)

    def decode(values):
        values = np.clip(values, lower, upper)
        plans = np.repeat(values, BLOCK_DECISIONS, axis=1)
        if traffic:
            plans[:, :, 1] = np.where(plans[:, :, 1] < -1., np.nan, plans[:, :, 1])
        if fixed_policy:
            plans[:, :commit] = action
        else:
            plans[:, :commit] = plans[:, :1]
        return plans

    evaluated = feasible_count = 0
    best_plan, best_clear, best_distance = None, -np.inf, np.inf
    maximum_clearance = -np.inf

    def check(plans):
        nonlocal evaluated, feasible_count, best_plan, best_clear, best_distance, maximum_clearance
        first, clear = evaluate(snap, rollout(snap, act, plans))
        first, clear = np.asarray(first), np.asarray(clear)
        if first.shape != (len(plans),) or clear.shape != first.shape:
            raise ValueError("Evaluator must return one first-contact time and clearance per plan")
        room = np.where(np.isnan(clear), -np.inf, clear)
        accepted = np.isposinf(first) & (room >= v2.TRIGGER_MARGIN_M)
        throttle = np.where(np.isnan(plans[:, 0, 1]), -1.5, plans[:, 0, 1])
        distance = ((plans[:, 0, 0] - action[0]) ** 2
                    + v2.W_THROTTLE * (throttle - action[1]) ** 2)
        # Feasible plans minimize issued-action deviation, with clearance as
        # tie-breaker. Infeasible plans rank only by worst constraint clearance.
        order = np.lexsort((-room, np.where(accepted, distance, 0.), ~accepted))
        evaluated += len(plans)
        feasible_count += int(accepted.sum())
        maximum_clearance = max(maximum_clearance, float(room.max()))
        if accepted.any():
            i = int(order[0])
            if distance[i] < best_distance or (distance[i] == best_distance and room[i] > best_clear):
                best_plan, best_clear = plans[i].copy(), float(room[i])
                best_distance = float(distance[i])
        return order

    mean = np.tile(action, (blocks, 1))
    std = (upper - lower) / 2.
    if len(seeds):
        order = check(seeds)
        mean = encode(seeds[order[:ELITES]]).mean(axis=0)
    for _ in range(ROUNDS):
        samples = decode(rng.normal(mean, std, size=(SAMPLES, blocks, 2)))
        order = check(samples)
        # Best accepted seeds/earlier samples remain eligible above; each new
        # generation refits its distribution without any repeated evaluations.
        encoded = encode(samples[order[:ELITES]])
        mean = encoded.mean(axis=0)
        std = np.maximum(encoded.std(axis=0), np.finfo(float).eps ** .5)
    return SearchResult(best_plan, best_clear, {
        "fixed_policy": bool(fixed_policy), "rounds": ROUNDS,
        "sampled_plans": ROUNDS * SAMPLES, "seed_plans": len(seeds),
        "dropped_brake_seeds": dropped_brake_seeds,
        "evaluated_plans": evaluated, "accepted_plans": feasible_count,
        "maximum_clearance": maximum_clearance,
        "accepted": best_plan is not None,
        "policy_distance": best_distance,
    })
