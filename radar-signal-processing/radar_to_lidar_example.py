import numpy as np
import matplotlib.pyplot as plt

# --- Simulated radar detections (range in meters, angle in degrees) ---
# Imagine these came from CFAR peak detection on a radar heatmap
radar_detections = [
    (30, 10),   # obstacle at 30m, 10°
    (50, 45),   # obstacle at 50m, 45°
    (20, 90),   # obstacle at 20m, 90°
    (70, 200),  # obstacle at 70m, 200°
    (40, 350),  # obstacle at 40m, 350°
]

# --- Parameters ---
max_range = 100  # radar max range (like lidar)
resolution = 1   # degrees per bin
num_beams = int(360 / resolution)

# Initialize lidar-like array: 100 = max range (no obstacle)
lidar_like = np.ones(num_beams) * max_range

# Fill in with radar detections (nearest obstacle per angle bin)
for r, angle in radar_detections:
    bin_index = int(angle // resolution)
    lidar_like[bin_index] = min(lidar_like[bin_index], r)

# --- Visualization ---
angles = np.deg2rad(np.arange(0, 360, resolution))
x = lidar_like * np.cos(angles)
y = lidar_like * np.sin(angles)

plt.figure(figsize=(6,6))
plt.plot(x, y, ".", label="Lidar-like output")
plt.scatter(
    [r*np.cos(np.deg2rad(a)) for r,a in radar_detections],
    [r*np.sin(np.deg2rad(a)) for r,a in radar_detections],
    c="red", marker="x", label="Radar detections"
)
plt.title("Radar → Lidar-like Scan")
plt.axis("equal")
plt.legend()
plt.show()

# Print example array
print("Lidar-like array (first 40 values):")
print(lidar_like[:40])
