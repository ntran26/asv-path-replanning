"""SB3 features extractor: per-branch encoders with presence-gated target slots.

**Revision 2** — plain concatenation.  The shared-encoder-plus-DeepSets
aggregation comparison from Revision 1 is **not built**: superseded decision D3.

01 §6.3 settles the architecture: plain concatenation of the five branches into
the SAC `MultiInputPolicy`.  Permutation invariance is meaningless at one
target, and the measured advantages of attention in the literature come from
high-density regimes -- dozens of aircraft, eight-ship encounters -- that this
scope deliberately does not enter.  The scaling question is pre-empted **by
scope**: restricted waterway, sequential encounters, single target in
deployment, `N_MAX_TARGETS` configurable.

So why a custom extractor at all, when SB3's `CombinedExtractor` would
concatenate the branches for us?  Two reasons:

* **The presence bit has to gate the slot.**  Zero is a legitimate value for
  bearing and for relative speed, so an unmasked empty slot reads as a target
  sitting on top of the vessel on a matching course.  Gating must happen before
  the slot encoder, not after, or the encoder's bias terms let an empty slot
  through anyway.
* **The target branch keeps a small encoder so the multi-vessel extension path
  exists** (S1).  The encoder is applied per slot with shared weights, which is
  all the extension needs; there is no aggregation choice to defend.
"""

from __future__ import annotations

from typing import Dict

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import constants as cfg
from observation import PRESENCE_INDEX


def _mlp(in_dim: int, hidden, out_dim: int) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, out_dim), nn.ReLU()]
    return nn.Sequential(*layers)


class ASVFeaturesExtractor(BaseFeaturesExtractor):
    """Encodes the five-branch Dict observation into one flat feature vector.

    `lidar`, `boundary`, `ego` and `path` are concatenated and passed through a
    small MLP.  `target` is reshaped to (batch, n_slots, 16), gated by its
    presence bit, encoded slot-by-slot with shared weights, and concatenated.
    """

    def __init__(self, observation_space: gym.spaces.Dict, *,
                 slot_hidden=cfg.SLOT_ENCODER_HIDDEN,
                 slot_embed: int = cfg.SLOT_EMBED_DIM,
                 scene_hidden: int = cfg.SCENE_ENCODER_HIDDEN,
                 n_slots: int = cfg.N_MAX_TARGETS,
                 n_features: int = cfg.TARGET_FEATURES) -> None:
        self.n_slots = int(n_slots)
        self.n_features = int(n_features)

        scene_dim = sum(
            int(observation_space[k].shape[0])
            for k in ("lidar", "boundary", "ego", "path")
        )
        super().__init__(observation_space,
                         features_dim=scene_hidden + slot_embed * self.n_slots)

        self.scene_net = _mlp(scene_dim, (scene_hidden,), scene_hidden)
        # One encoder applied to every slot -- the weight sharing that makes the
        # multi-vessel extension a retrain rather than a redesign.
        self.slot_encoder = _mlp(self.n_features, slot_hidden, slot_embed)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        scene = torch.cat([
            observations["lidar"],
            observations["boundary"],
            observations["ego"],
            observations["path"],
        ], dim=1)
        scene_feat = self.scene_net(scene)

        slots = observations["target"].reshape(-1, self.n_slots, self.n_features)
        presence = slots[..., PRESENCE_INDEX].unsqueeze(-1)

        # Gate the slot's inputs before encoding, so an absent target cannot
        # reach the encoder at all.  Gating only the output would still let the
        # bias terms of an empty slot contribute.
        embedded = self.slot_encoder(slots * presence) * presence
        target_feat = embedded.reshape(embedded.shape[0], -1)

        return torch.cat([scene_feat, target_feat], dim=1)


def policy_kwargs(**kwargs) -> dict:
    """Ready-made `policy_kwargs` for SB3.

        model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs())
    """
    return {
        "features_extractor_class": ASVFeaturesExtractor,
        "features_extractor_kwargs": dict(kwargs),
    }
