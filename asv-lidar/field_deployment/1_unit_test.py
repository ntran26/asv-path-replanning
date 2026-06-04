import numpy as np
import matplotlib.pyplot as plt

MAX_RUD_ANGLE = 40.0
MAX_RUD_RATE_DPS = 20.0
RUDDER_SCALE = 100.0

def ship_model_style_limit(prev_cmd, raw_cmd, dt):
    max_cmd_rate_per_s = RUDDER_SCALE * MAX_RUD_RATE_DPS / MAX_RUD_ANGLE
    cmd_dot = np.clip(raw_cmd - prev_cmd, -max_cmd_rate_per_s, +max_cmd_rate_per_s)
    return float(prev_cmd + cmd_dot * dt)

dt = 0.1
t = np.arange(0.0, 12.0, dt)

raw = np.zeros_like(t)
raw[(t >= 1.0) & (t < 5.0)] = +100.0
raw[(t >= 5.0) & (t < 9.0)] = -100.0
raw[t >= 9.0] = +40.0

limited = []
cmd = 0.0
for r in raw:
    cmd = ship_model_style_limit(cmd, float(r), dt)
    limited.append(cmd)

limited = np.array(limited)
rate = np.diff(limited) / dt

print("Max command rate:", np.max(np.abs(rate)), "percent/s")
print("Expected <=", RUDDER_SCALE * MAX_RUD_RATE_DPS / MAX_RUD_ANGLE, "percent/s")

plt.figure()
plt.plot(t, raw, label="raw command")
plt.plot(t, limited, label="limited command")
plt.xlabel("Time [s]")
plt.ylabel("Rudder command [%]")
plt.legend()
plt.grid(True)
plt.show()