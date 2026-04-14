# RL LiDAR Navigation Spec for Codex

## Goal

Train a stable RL policy for ASV navigation using:

- Paper-style reward (path following + obstacle avoidance + living penalty)
- LiDAR observations with blind-zone-aware preprocessing
- Geometry-based collision detection instead of LiDAR threshold collision

Key design principle:

- The reward should shape behavior.
- Sensor limitations such as LiDAR blind zones should be handled mainly in the observation design, not forced into the reward.

---

## 1. Observation Design

Keep the current observation entries:

- `lidar`
- `pos`
- `hdg`
- `dhdg`
- `speed`
- `tgt`
- `target_heading`

Add the following LiDAR-derived features:

- `front_min`
- `front_p10`
- `left_min`
- `right_min`
- `near_flag`

### Recommended LiDAR sector helper

```python
def _lidar_features(self):
    lidar_d = self.lidar.ranges.astype(np.float32)
    angles = self.lidar.angles.astype(np.float32)

    # Valid range based on real sensor behavior
    valid = (lidar_d >= 1.0) & (lidar_d <= LIDAR_RANGE)

    def sector(mask, mode="min"):
        mask = mask & valid
        if not np.any(mask):
            return float(LIDAR_RANGE)
        vals = lidar_d[mask]
        if mode == "min":
            return float(np.min(vals))
        elif mode == "p10":
            return float(np.percentile(vals, 10))
        else:
            raise ValueError("Unknown mode")

    front = np.abs(angles) <= 30.0
    left = (angles > 30.0) & (angles <= 90.0)
    right = (angles < -30.0) & (angles >= -90.0)

    front_min = sector(front, "min")
    front_p10 = sector(front, "p10")
    left_min = sector(left, "min")
    right_min = sector(right, "min")

    near_flag = 1.0 if front_min < 2.0 else 0.0

    return {
        "front_min": np.array([front_min], dtype=np.float32),
        "front_p10": np.array([front_p10], dtype=np.float32),
        "left_min": np.array([left_min], dtype=np.float32),
        "right_min": np.array([right_min], dtype=np.float32),
        "near_flag": np.array([near_flag], dtype=np.float32),
    }
```

Then inside `_get_obs()`:

```python
obs.update(self._lidar_features())
```

### Why

This keeps raw LiDAR in the observation, but also gives the policy a compact summary of near-field risk. That is a practical way to handle a LiDAR blind-zone limitation without overcomplicating the reward.

---

## 2. Reward Function

Use a paper-style reward with fixed lambda:

```python
LAMBDA_REWARD = 0.5
```

### Total reward

Use:

\[
r =
\begin{cases}
(1-\lambda)R_{collision}, & \text{if collided} \\
\lambda r_{pf} + (1-\lambda)r_{oa} + r_{exist}, & \text{otherwise}
\end{cases}
\]

---

## 2.1 Path-following reward

Use:

\[
r_{pf} = -1 + (U_{norm}\cos\tilde{\chi} + 1)(e^{-\gamma_e |y_e|} + 1)
\]

### Variable meanings

- `ye = abs(self.tgt)` is cross-track error
- `U = self.speed_mps` is pose-derived speed magnitude
- `U_norm = U / U_MAX`
- `chi_tilde` is the course error relative to the desired path direction

### Important heading convention

The ship model uses:

```python
dx = d * np.sin(h)
dy = d * np.cos(h)
```

So heading `0 deg` means facing `+y`. Therefore the correct angle calculations for the reward are:

```python
if U > 1e-6:
    course_deg = np.degrees(np.arctan2(dx_pos, dy_pos))
else:
    course_deg = self.asv_h

path_dx = self.goal_x - self.start_x
path_dy = self.goal_y - self.start_y
path_course_deg = np.degrees(np.arctan2(path_dx, path_dy))

chi_tilde_deg = (course_deg - path_course_deg + 180.0) % 360.0 - 180.0
cos_chi = np.cos(np.radians(chi_tilde_deg))
```

### Code block

```python
lam = float(LAMBDA_REWARD)

ye = float(abs(self.tgt))
U = float(self.speed_mps)
U_norm = U / max(U_MAX, 1e-6)

if U > 1e-6:
    course_deg = float(np.degrees(np.arctan2(dx_pos, dy_pos)))
else:
    course_deg = float(self.asv_h)

path_dx = float(self.goal_x - self.start_x)
path_dy = float(self.goal_y - self.start_y)
path_course_deg = float(np.degrees(np.arctan2(path_dx, path_dy)))

chi_tilde_deg = (course_deg - path_course_deg + 180.0) % 360.0 - 180.0
cos_chi = float(np.cos(np.radians(chi_tilde_deg)))

r_pf = float(-1.0 + (U_norm * cos_chi + 1.0) * (np.exp(-GAMMA_E * ye) + 1.0))
```

---

## 2.2 Obstacle-avoidance reward

Use:

\[
r_{oa} = -\frac{\sum_i w_i \cdot (\gamma_x \max(x_i, \epsilon_x)^2)^{-1}}{\sum_i w_i}
\]

with:

\[
w_i = \frac{1}{1 + |\gamma_\theta \theta_i|}
\]

### Important design choice

For training, use the simulated LiDAR distances in the reward before clipping them to match the real blind-zone behavior.

That means:
- Keep real-sensor clipping behavior in the observation features.
- Use true simulated distances for the reward so the OA term still has a useful gradient near obstacles.

### Code block

```python
lidar_d = self.lidar.ranges.astype(np.float32)
theta = np.radians(self.lidar.angles.astype(np.float32))

w = 1.0 / (1.0 + np.abs(GAMMA_THETA * theta))

# Training reward uses true simulated distances
x = np.clip(lidar_d, EPSILON_X, LIDAR_RANGE)

pen = 1.0 / (GAMMA_X * x**2)
r_oa = -float(np.sum(w * pen) / (np.sum(w) + 1e-6))
```

---

## 2.3 Living penalty

Use:

\[
r_{exist} = -\lambda(2\alpha_r + 1)
\]

### Code

```python
r_exist = float(-lam * (2.0 * ALPHA_R + 1.0))
```

---

## 2.4 Total reward block

```python
if collided:
    reward = float((1.0 - lam) * R_COLLISION)
else:
    reward = float(lam * r_pf + (1.0 - lam) * r_oa + r_exist)
```

Optional goal bonus is not necessary at first.

---

## 3. Collision and termination

Use geometry-based collision detection, not LiDAR threshold collision.

Recommended termination:

```python
GOAL_RADIUS = VESSEL_LENGTH / 2
reached_goal = bool(self.distance_to_goal <= GOAL_RADIUS)
terminated = bool(collided or reached_goal)
```

Use the same goal threshold consistently everywhere.

---

## 4. Recommended info dictionary

Expose the following so the training logger can analyze learning properly:

```python
lidar_feat = self._lidar_features()

info = {
    "lam": float(lam),
    "r_pf": float(r_pf),
    "r_oa": float(r_oa),
    "r_exist": float(r_exist),
    "reward": float(reward),

    "ye": float(ye),
    "U": float(U),
    "U_norm": float(U_norm),
    "course_deg": float(course_deg),
    "path_course_deg": float(path_course_deg),
    "chi_tilde_deg": float(chi_tilde_deg),
    "cos_chi": float(cos_chi),

    "front_min": float(lidar_feat["front_min"][0]),
    "front_p10": float(lidar_feat["front_p10"][0]),
    "left_min": float(lidar_feat["left_min"][0]),
    "right_min": float(lidar_feat["right_min"][0]),
    "near_flag": float(lidar_feat["near_flag"][0]),

    "speed_mps": float(self.speed_mps),
    "rpm": float(rpm),
    "rudder_deg": float(rudder_deg),

    "distance_to_goal": float(self.distance_to_goal),
    "tgt": float(self.tgt),
    "collided": bool(collided),
    "reached_goal": bool(reached_goal),
}
```

---

## 5. Training progression

Do not jump straight into the full random environment.

### Stage 1
- fixed start/goal
- 1 obstacle
- deterministic scenario
- verify the policy can reliably solve it

### Stage 2
- fixed start/goal
- random obstacle position
- still 1 obstacle

### Stage 3
- random start/goal
- multiple obstacles

---

## 6. Key metrics to monitor

From the training/evaluation logger, monitor:

- success rate
- collision rate
- mean distance to goal
- mean speed
- mean absolute `tgt`
- front-sector clearance (`front_min`, `front_p10`)
- reward components: `r_pf`, `r_oa`, `r_exist`

---

## 7. Important implementation notes

### Keep
- Geometry-based collision
- LiDAR in observation
- Paper-style reward structure

### Do not do
- Do not use LiDAR threshold as terminal collision
- Do not clip LiDAR to max range inside the reward if that removes the near-obstacle gradient
- Do not mix angle conventions; use the ship model's heading convention consistently

---

## 8. If training still fails

Tune in this order:

1. `GAMMA_X` (obstacle penalty scale)
2. `EPSILON_X` (safety distance floor)
3. `ALPHA_R` (living penalty / incentive to move)

Do not tune everything at once.

---

## 9. Minimal patch checklist

1. Add `_lidar_features()`
2. Extend `observation_space` with the five LiDAR summary features
3. Update `_get_obs()` to include them
4. Replace reward block with the paper-style version above
5. Use `terminated = bool(collided or reached_goal)`
6. Return the extended `info` dict
7. Train first on a single-obstacle deterministic scenario

