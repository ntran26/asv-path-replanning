"""Fresh measured yaw with the inherited model observer for other components.

This diagnostic ablation changes the fresh-frame yaw correction gain to one:
``r_hat = r_measured``. Surge and sway retain EgoObserver's existing update.
Stale frames, initialization and issued-command prediction retain its rules.
The supplied measurement is already onboard; this module reads no environment
or sensors. It changes no rollout dynamics, margins or collision checks.

Luenberger (1971), An introduction to observers, Section II.B, Eq. (2.6),
https://doi.org/10.1109/TAC.1971.1099826, is related prior work for model dynamics
plus measurement-error correction. The gain-one yaw choice is our engineering
ablation, not that paper's design, a Kalman filter or an uncertainty bound.
Saved development measurements motivate testing model-versus-sensor weighting;
they do not establish improved episode outcomes or observer convergence.

Only the current u/v correction formula is unchanged. A changed yaw estimate
feeds the next coupled model prediction and can change later u/v priors too.
"""
from __future__ import annotations

import numpy as np

from safety_observer import EgoObserver


class FreshYawObserver(EgoObserver):
    """Use raw yaw on fresh frames and inherited model priors on stale frames."""

    def __init__(self):
        super().__init__()
        self.last = {}

    def reset(self):
        super().reset()
        self.last = {}

    def update(self, raw, *, fresh=True):
        raw = np.asarray(raw, dtype=float).reshape(3)
        initial = self.estimate is None
        prior = self.estimate if self.prior is None else self.prior
        prior_yaw = None if prior is None else float(prior[2])
        # Delegate validation, initialization, u/v correction, stale handling
        # and prior consumption. Do not apply a second u/v sensor correction.
        super().update(raw, fresh=fresh)
        if fresh:
            self.estimate[2] = raw[2]
        self.last = {
            "fresh": bool(fresh),
            "yaw_source": ("initial_measurement" if initial else
                           "fresh_measurement" if fresh else "model_or_held_prior"),
            "raw_yaw_rate_rad_s": float(raw[2]),
            "prior_yaw_rate_rad_s": prior_yaw,
            "estimated_yaw_rate_rad_s": float(self.estimate[2]),
        }
        return self.estimate.copy()
