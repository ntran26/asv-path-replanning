"""Guard descriptive plotting counts and unmistakable incomplete denominators."""
import csv
import importlib.util
from pathlib import Path
import sys

import pytest


_PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/plot_suite_outcomes.py"
sys.path.insert(0, str(_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("outcome_plot_under_test", _PATH)
plot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plot)


def _row(mode="off", **updates):
    return dict(level="group", scope="dev", mode=mode, goals=7, collision_obstacle=1,
                collision_boundary=1, collision_target=1, timeouts=0, collisions=3,
                completed_cases=10, expected_cases=10, complete=True, **updates)


def _write(tmp_path, rows):
    path = tmp_path / "outcomes.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_complete_counts_accept_and_keep_declared_denominators(tmp_path):
    path = _write(tmp_path, [_row(mode) for mode in ("off", "v4", "v5")])
    scopes, rows, partial = plot.load_rows(path, "group", ["off", "v4", "v5"])
    assert scopes == ["dev"] and not partial
    assert rows["dev", "v5"]["completed_cases"] == rows["dev", "v5"]["expected_cases"] == 10


@pytest.mark.parametrize("change", ["unfinished", "missing_mode"])
def test_incomplete_or_absent_mode_requires_explicit_partial_flag(tmp_path, change):
    rows = [_row(mode) for mode in ("off", "v4", "v5")]
    if change == "unfinished":
        rows[-1].update(complete=False, expected_cases=20)
    else:
        rows.pop()
    path = _write(tmp_path, rows)
    with pytest.raises(ValueError, match="Incomplete or missing"):
        plot.load_rows(path, "group", ["off", "v4", "v5"])
    assert plot.load_rows(path, "group", ["off", "v4", "v5"], True)[2]


@pytest.mark.parametrize("updates", [{"goals": 8}, {"collisions": 2}, {"goals": -1},
                                    {"expected_cases": 9}, {"complete": False}])
def test_inconsistent_counts_are_rejected_even_for_partial_progress(tmp_path, updates):
    row = _row()
    row.update(updates)
    path = _write(tmp_path, [row])
    with pytest.raises(ValueError):
        plot.load_rows(path, "group", ["off"], True)


def test_duplicate_mode_scope_is_not_summed(tmp_path):
    path = _write(tmp_path, [_row(), _row()])
    with pytest.raises(ValueError, match="Duplicate"):
        plot.load_rows(path, "group", ["off"])


def _full_table():
    rows = []
    for component, count in plot.FULL_BENCHMARK_COUNTS.items():
        for mode in plot.FULL_BENCHMARK_MODES:
            row = _row(mode)
            row.update(level="component", scope=component, goals=count - 3,
                       completed_cases=count, expected_cases=count)
            rows.append(row)
    for level, scopes in (("group", set(plot.GROUPS.values())), ("all", {"all"})):
        for scope in scopes:
            for mode in plot.FULL_BENCHMARK_MODES:
                members = [row for row in rows if row["level"] == "component" and row["mode"] == mode
                           and (level == "all" or plot.GROUPS[row["scope"]] == scope)]
                row = dict(members[0], level=level, scope=scope)
                for field in (*plot.FIELDS, "completed_cases", "expected_cases", "collisions"):
                    row[field] = sum(member[field] for member in members)
                rows.append(row)
    return rows


def test_full_plot_gate_accepts_8670_consistent_counts(tmp_path):
    path = _write(tmp_path, _full_table())
    data = path.read_bytes()
    plot.require_full_benchmark_rows(path, list(plot.FULL_BENCHMARK_MODES), data)


@pytest.mark.parametrize("fault", ["missing_component", "wrong_group_total", "missing_mode"])
def test_full_plot_gate_rejects_subset_or_inconsistent_aggregate(tmp_path, fault):
    rows = _full_table()
    modes = list(plot.FULL_BENCHMARK_MODES)
    if fault == "missing_component":
        rows = [row for row in rows if row["scope"] != "frozen_a"]
    elif fault == "wrong_group_total":
        row = next(row for row in rows if row["level"] == "group")
        row["goals"] -= 1
        row["timeouts"] += 1
    else:
        modes.remove("v5")
    path = _write(tmp_path, rows)
    with pytest.raises(ValueError):
        plot.require_full_benchmark_rows(path, modes, path.read_bytes())
