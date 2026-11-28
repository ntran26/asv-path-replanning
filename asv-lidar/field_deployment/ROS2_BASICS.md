# ROS 2 basics (field laptop: Ubuntu 24.04, ROS 2 Jazzy)

Practice uses the vessel recording of 2026-11-26. Connecting to the real vessel needs the operator's settings (end of page).

## Terminal

| Action | Keys or command |
|---|---|
| Open a terminal / new tab | Ctrl+Alt+T / Ctrl+Shift+T |
| Paste / copy | Ctrl+Shift+V / Ctrl+Shift+C |
| Stop the running program | Ctrl+C |
| Complete a name / previous command | Tab / up arrow |
| Current folder, list, move | `pwd`, `ls -l`, `cd ~`, `cd ..` |
| Network addresses (coloured) | `ip -c addr` |
| Is a machine reachable? (address of the machine) | `ping 192.168.0.1` |

`sudo` runs a command as administrator; the password stays invisible while typed.

## Set up EVERY new tab (practice on the recording)

```bash
source /opt/ros/jazzy/setup.bash
```

```bash
export ROS_DOMAIN_ID=42
```

```bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
```

Check with `printenv | grep ROS_`. LOCALHOST keeps playback off the lab or basin network.

## Recording

```bash
ros2 bag info ~/rosbag2_2026_11_26-11_08_31
```

```bash
ros2 bag play ~/rosbag2_2026_11_26-11_08_31 --loop --clock
```

Space pauses, Ctrl+C stops. 28 min, MCAP.

## Inspect (second tab, same setup)

```bash
ros2 topic list
```

```bash
ros2 topic hz /scan
```

```bash
ros2 topic hz /imu/data_raw
```

```bash
ros2 topic hz /map_odom
```

```bash
ros2 topic echo --once --no-arr /map_odom
```

```bash
ros2 topic echo --once /tf_static
```

| Topic | Type | Rate (Hz) | Content |
|---|---|---|---|
| /scan | sensor_msgs/LaserScan | 10 | 720 ranges, 0.5° step, 0.15 to 16 m, frame lidar_frame |
| /imu/data_raw | sensor_msgs/Imu | 100 | turn rate, acceleration, frame imu_frame |
| /map_odom | nav_msgs/Odometry | 86 | pose of base_link in map, velocity |
| /odometry/filtered | nav_msgs/Odometry | 86 | EKF estimate |
| /tf, /tf_static | tf2_msgs/TFMessage | 145, fixed | changing and fixed frame links |

Mounting: lidar_frame 0.63 m forward of base_link, yaw 180° (faces aft). imu_frame at x 0.30, y −0.13 m.

## RViz

```bash
rviz2 -d ~/.rviz2/vessel.rviz --ros-args -p use_sim_time:=true
```

`use_sim_time:=true` is for playback with `--clock` only. Fixed Frame `map`. Displays: LaserScan `/scan` (Size 0.05), Odometry `/map_odom` (Covariance off), TF, optional Map `/map`. View: TopDownOrtho, Target Frame `base_link` to follow the vessel.

## Terms

- Node: one running program. Topic: a named data channel; nodes publish to it or subscribe.
- QoS: delivery rules. A best-effort publisher and a reliable subscriber exchange nothing. Recorded topics were offered as reliable; /tf_static and /map are transient local (kept for late joiners).
- ROS_DOMAIN_ID: only nodes with the same number see each other (default 0).

## Nothing shows up

1. The list holds only /parameter_events and /rosout: wait a few seconds, repeat.
2. `printenv | grep ROS_` in both tabs; confirm that the player or sender is running.
3. `ros2 daemon stop`, then `ros2 topic list`.

## Still needed from the operator before a basin day

ROS_DOMAIN_ID; RMW (DDS vendor); discovery range; which computer runs the ROS nodes. LOCALHOST must not be set when listening to another computer. Before any network test, check that no field tool is running: `pgrep -af "udp_raw_logger|udp_manoeuvre|udp_live_p3"`.
