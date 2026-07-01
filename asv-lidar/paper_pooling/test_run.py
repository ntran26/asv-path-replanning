import json

"""
Deterministic test cases for ASV path-following and obstacle-avoidance evaluation.

Coordinate convention:
- These cases are authored in the original 10 x 25 local task frame.
- Existing cases 0-7 are kept for backward compatibility.

Existing cases:
Test case None: random start & goal points, random obstacles
Test case 0: centered straight path, no obstacles
Test case 1: field Scenario A, narrow gate + recovery (3 obstacles)
Test case 2: field Scenario B, three-obstacle slalom (3 obstacles)
Test case 3: field Scenario C, boundary-constrained bypass (3 obstacles)
Test case 4: centered straight path, 3 scattered obstacles
Test case 5: diagonal path, horizontal blockage
Test case 6: no obstacles, goal to left
Test case 7: no obstacles, goal to right

Test case 99: load setup from recorded random scenario file

Note:
- Cases 8, 9, and 13 include optional curved-path waypoints through path_waypoints().
  Your environment must call TestCase.path_waypoints(test_case) to actually use the curved path.
  If the environment only calls position(), those cases will still run, but with a straight
  start-to-goal reference path.
"""

ENV_DATA = "data/env_setup/survival_pool/env_0.json"
EVAL_SUITE_DATA = "data/env_setup/eval_suite/asv_eval_suite_100_harder.json"

OBS_LENGTH = 1.0


class TestCase:
    def __init__(self):
        self.obs = []
        self.start_x = None
        self.start_y = None
        self.goal_x = None
        self.goal_y = None
        self.obs_size = OBS_LENGTH / 2.0
        self.env_data = ENV_DATA

    # -------------------------------
    # Small geometry helpers
    # -------------------------------
    def _box(self, x, y, half_size=None):
        """Axis-aligned square obstacle centered at (x, y)."""
        hs = self.obs_size if half_size is None else float(half_size)
        return [(x - hs, y - hs), (x + hs, y - hs), (x + hs, y + hs), (x - hs, y + hs)]

    def _rect(self, x, y, half_x, half_y):
        """Axis-aligned rectangular obstacle centered at (x, y)."""
        return [(x - half_x, y - half_y), (x + half_x, y - half_y),
                (x + half_x, y + half_y), (x - half_x, y + half_y)]

    def obstacles(self, test_case):
        """Return obstacle polygons for a deterministic test case."""
        # IMPORTANT: always clear old obstacles. The same TestCase instance is reused.
        self.obs = []

        if 1000 <= int(test_case) < 1100:
            case = self._load_suite_case(test_case)
            self.obs = case["obstacles"]
            return self.obs

        # Existing no-obstacle cases
        if test_case in [0, 6, 7, 8, 9, 19, 20]:
            return self.obs

        # Existing cases 1-5, kept compatible with your previous evaluations
        if test_case == 1:      # Scenario A: narrow gate + recovery
            # Two obstacles form a gate; the third obstacle forces path recovery and a second correction.
            for x, y in [(2.0, 11.0), (7.0, 11.0), (4.35, 16.5)]:
                self.obs.append(self._box(x, y))

        elif test_case == 2:    # Scenario B: three-obstacle slalom
            # Alternating offsets test sequential avoidance without excessive oscillation.
            for x, y in [(4.1, 8.0), (5.9, 13.0), (2.0, 18.0)]:
                self.obs.append(self._box(x, y))

        elif test_case == 3:    # Scenario C: boundary-constrained bypass
            # Side obstacles and a centerline obstacle test obstacle/boundary trade-offs.
            for x, y in [(1.5, 9.0), (8.5, 14.0), (5.0, 17.5)]:
                self.obs.append(self._box(x, y))

        elif test_case == 4:    # 3 scattered obstacles
            for x, y in [(5, 8), (8, 15), (3, 18)]:
                self.obs.append(self._box(x, y))

        elif test_case == 5:    # horizontal blockage on diagonal path
            self.obs.append(self._rect(3.5, 15, half_x=3.0, half_y=self.obs_size))
            self.obs.append(self._box(9, 15))

        # -------------------------------
        # Paper-replica cases 8+
        # -------------------------------
        elif test_case == 10:
            # Scenario C style: two obstacles close to/on the center path.
            # The gap between them encourages a close-but-controlled pass.
            self.obs.append(self._box(4.45, 11.0))
            self.obs.append(self._box(5.55, 14.0))

        elif test_case == 11:
            # Scenario C style: obstacles near, but not exactly on, the path.
            # A path-adherent policy should make only a small lateral correction.
            self.obs.append(self._box(3.8, 10.5))
            self.obs.append(self._box(6.2, 14.5))
            self.obs.append(self._box(4.1, 18.0))

        elif test_case == 12:
            # Narrow centered gate. The vessel should pass between the obstacles.
            # Gap center is x=5, with a moderate gap for a 0.5 m beam vessel.
            self.obs.append(self._box(3.7, 12.5))
            self.obs.append(self._box(6.3, 12.5))

        elif test_case == 13:
            # Curved/S-bend path with one obstacle near the bend.
            # Requires both path following and a local correction around the obstacle.
            self.obs.append(self._box(5.8, 13.0))

        elif test_case == 14:
            # Single obstacle early on the center path.
            self.obs.append(self._box(5.0, 7.5))

        elif test_case == 15:
            # Single obstacle late on the center path.
            self.obs.append(self._box(5.0, 17.5))

        elif test_case == 16:
            # Controlled slalom: alternating obstacles along the center path.
            for x, y in [(4.0, 8.0), (6.0, 13.0), (4.0, 18.0)]:
                self.obs.append(self._box(x, y))

        elif test_case == 17:
            # Off-path left obstacle. This should not trigger a very wide detour.
            self.obs.append(self._box(2.7, 12.5))

        elif test_case == 18:
            # Off-path right obstacle. This should not trigger a very wide detour.
            self.obs.append(self._box(7.3, 12.5))

        elif test_case == 99:
            with open(self.env_data, "r") as f:
                data = json.load(f)
            self.obs = data["obstacles"]

        else:
            raise ValueError(f"Invalid test case: {test_case}")

        return self.obs

    def position(self, test_case):
        """Return start_x, start_y, goal_x, goal_y for a deterministic test case."""
        if 1000 <= int(test_case) < 1100:
            case = self._load_suite_case(test_case)
            self.start_x, self.start_y = case["start"]
            self.goal_x, self.goal_y = case["goal"]
            return self.start_x, self.start_y, self.goal_x, self.goal_y

        # Existing centered straight-path cases and most paper-replica cases
        if test_case in [0, 1, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
            self.start_x = 5
            self.start_y = 2
            self.goal_x = 5
            self.goal_y = 22
        
        elif test_case == 2:
            self.start_x = 3
            self.start_y = 2
            self.goal_x = 7
            self.goal_y = 22

        elif test_case == 3:
            self.start_x = 7
            self.start_y = 2
            self.goal_x = 3
            self.goal_y = 22

        elif test_case == 5:
            self.start_x = 2
            self.start_y = 2
            self.goal_x = 7
            self.goal_y = 22

        elif test_case == 6:
            self.start_x = 5
            self.start_y = 2
            self.goal_x = 3
            self.goal_y = 22

        elif test_case == 7:
            self.start_x = 5
            self.start_y = 2
            self.goal_x = 7
            self.goal_y = 22

        elif test_case == 8:
            # S-bend path-following case. Start/goal stay centered; actual curve is in path_waypoints().
            self.start_x = 5
            self.start_y = 2
            self.goal_x = 5
            self.goal_y = 22

        elif test_case == 9:
            # Smooth right bend. Start/goal stay centered; actual curve is in path_waypoints().
            self.start_x = 5
            self.start_y = 2
            self.goal_x = 5
            self.goal_y = 22

        elif test_case == 19:
            self.start_x = 3
            self.start_y = 2
            self.goal_x = 7
            self.goal_y = 22

        elif test_case == 20:
            self.start_x = 7
            self.start_y = 2
            self.goal_x = 3
            self.goal_y = 22

        elif test_case == 99:
            with open(self.env_data, "r") as f:
                data = json.load(f)
            self.start_x = data["start"][0]
            self.start_y = data["start"][1]
            self.goal_x = data["goal"][0]
            self.goal_y = data["goal"][1]

        else:
            raise ValueError(f"Invalid test case: {test_case}")

        return self.start_x, self.start_y, self.goal_x, self.goal_y

    def path_waypoints(self, test_case):
        """
        Optional reference path waypoints for curved paper-replica cases.

        Return:
            None for normal straight-line path generation, or a list of (x, y) waypoints.

        Environment integration idea:
            wpts = self.scenario.path_waypoints(test_case)
            if wpts is None:
                path = self._generate_path(start_x, start_y, goal_x, goal_y)
            else:
                path = self._generate_path_from_waypoints(wpts)
        """
        if test_case == 8:
            # Scenario B analogue: S-bend, no obstacles.
            return [(5.0, 2.0), (3.4, 7.5), (6.6, 15.5), (5.0, 22.0)]

        if test_case == 9:
            # Scenario B analogue: one-sided smooth bend, no obstacles.
            return [(5.0, 2.0), (7.0, 8.0), (7.2, 15.0), (5.0, 22.0)]

        if test_case == 13:
            # Scenario C/D analogue: S-bend with nearby obstacle.
            return [(5.0, 2.0), (3.6, 8.0), (6.4, 15.0), (5.0, 22.0)]

        return None

    def paper_replica_cases(self):
        """Convenience list for paper-style evaluation."""
        return [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    
    def _load_suite_case(self, test_case):
        suite_idx = int(test_case) - 1000
        with open(EVAL_SUITE_DATA, "r") as f:
            suite = json.load(f)

        cases = suite["cases"]
        if suite_idx < 0 or suite_idx >= len(cases):
            raise ValueError(f"Invalid eval-suite test case: {test_case}")

        return cases[suite_idx]
