"""Pure selector checks; no vessel, environment, policy or episode calls."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from tools.diagnostics.safety.audit_bootstrap_continuous_bank import select_from_past


class ToyBank:
    count = 2
    cc = SimpleNamespace(SUBSTEPS=4)

    def __init__(self):
        self.calls = []

    def reset_actuators(self):
        return {"commands": []}

    def forecast(self, initial, actuators, commands):
        self.calls.append((copy.deepcopy(initial), copy.deepcopy(commands)))
        state = np.repeat(initial["raw"][:, None], 2, axis=1)
        output = []
        for command in commands:
            # Two persistent deterministic dynamics; no measurement reset.
            for _ in range(self.cc.SUBSTEPS):
                state = state + np.array([[1., 2.], [0., 0.], [0., 0.]]) / self.cc.SUBSTEPS
                output.append(state.copy())
        return np.asarray(output)

    def advance_actuators(self, actuators, command):
        return {"commands": actuators["commands"] + [command.copy()]}


def past():
    rows = [{"raw": np.array([2.*i, 0., 0.]),
             "command": np.array([i/100., 6.])} for i in range(11)]
    rows[-1]["command"] = np.array([np.nan, np.nan])  # future command forbidden
    return rows


def test_single_initialization_and_exact_past_endpoint_alignment():
    bank = ToyBank()
    rows = past()
    before = copy.deepcopy(rows)
    selected, scores, residuals, actuators = select_from_past(bank, rows, np.ones(3))
    assert selected == 1 and scores[1] == 0.
    np.testing.assert_array_equal(residuals[:, 0, 0], -np.arange(1., 11.))
    assert len(bank.calls) == 1 and len(bank.calls[0][1]) == 10
    np.testing.assert_array_equal(actuators["commands"], [r["command"] for r in rows[:10]])
    for a, b in zip(rows, before):
        np.testing.assert_array_equal(a["raw"], b["raw"])
        np.testing.assert_array_equal(a["command"], b["command"])


def test_intermediate_measurements_score_but_never_initialize_forecast():
    rows = past()
    bank = ToyBank()
    _, scores, residuals, _ = select_from_past(bank, rows, np.ones(3))
    rows[5]["raw"][0] += 100.
    changed_bank = ToyBank()
    _, changed_scores, changed_residuals, _ = select_from_past(changed_bank, rows, np.ones(3))
    np.testing.assert_array_equal(changed_bank.calls[0][0]["raw"], bank.calls[0][0]["raw"])
    np.testing.assert_array_equal(changed_bank.calls[0][1], bank.calls[0][1])
    keep = np.arange(10) != 4
    np.testing.assert_array_equal(changed_residuals[keep], residuals[keep])
    np.testing.assert_array_equal(changed_residuals[4, :, 0], residuals[4, :, 0]-100.)
    assert not np.array_equal(changed_scores, scores)


@pytest.mark.parametrize("count", [10, 12])
def test_selector_rejects_changed_training_window(count):
    rows = past()
    with pytest.raises(ValueError, match="Exactly ten transitions"):
        select_from_past(ToyBank(), (rows+rows)[:count], np.ones(3))
