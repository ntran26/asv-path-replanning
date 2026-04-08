"""
Test ship model and plot trajectory
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ship_model_bluefin_v2 import ShipModel, MAX_RUD_ANGLE

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "trajectory_plots"

def run_test(model, rpm, rudder_deg, duration_s=50.0, dt=0.1):
    rudder_percent = (rudder_deg / MAX_RUD_ANGLE) * 100
    t = []
    x = []
    y = []
    u = []
    yaw = []
    yaw_rate = []
    for i in range(int(round(duration_s/dt)) + 1):
        time_s = i * dt
        t.append(time_s)
        st = model.state_dict()
        x.append(st["x_m"])
        y.append(st["y_m"])
        u.append(st["u_body_mps"])
        yaw.append(st["heading_deg"])
        yaw_rate.append(st["yaw_rate_degps"])
        if i <= 10:
            model.update(rpm, 0, dt)
        elif 10 < i < int(round(duration_s/dt)):
            model.update(rpm, rudder_percent, dt)
            # if i % 10 == 0:
            #     print(f"t={0.1*(i+1):4.1f}s vel={model._v:5.2f} hdg={np.rad2deg(model._h)%360:6.2f} dhdg={np.rad2deg(model._w):6.2f}")

    return {
        "t": np.array(t),
        "x": np.array(x),
        "y": np.array(y),
        "u": np.array(u),
        "yaw": np.array(yaw),
        "yaw_rate": np.array(yaw_rate),
        "rpm": rpm,
        "rudder_deg": rudder_deg
    }

def main():
    OUT_DIR.mkdir(exist_ok=True)
    model = ShipModel()
    straight_rpm = 15
    turn_rpm = 24
    turn_rudder_deg = 40

    # configure speed and turning test
    straight = run_test(model=model, rpm=straight_rpm, rudder_deg=0, duration_s=40, dt=0.1)
    turn = run_test(model=model, rpm=turn_rpm, rudder_deg=turn_rudder_deg, duration_s=50, dt=0.1)

    # plot turning trajectory
    plt.figure(figsize=(6,6))
    plt.plot(turn["x"], turn["y"], lw=2)
    plt.scatter(turn["x"][0], turn["y"][0], s=40, label="start")
    plt.scatter(turn["x"][-1], turn["y"][-1], s=40, label="end")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(f"Turning circle: RPM={turn['rpm']}, Rudder={turn['rudder_deg']} deg")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "turning_circle_trajectory.png", dpi=200)
    plt.close()

    # plot yaw / yaw rate
    fig, ax = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    ax[0].plot(turn["t"], turn["yaw"], lw=2)
    ax[0].set_ylabel("Yaw [deg]")
    ax[0].set_title("Turning test: yaw and yaw rate over time")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(turn["t"], turn["yaw_rate"], lw=2)
    ax[1].set_ylabel("Yaw rate [deg/s]")
    ax[1].set_xlabel("Time [s]")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "turning_circle_yaw.png", dpi=200)
    plt.close(fig)

    # plot speed test
    plt.figure(figsize=(8, 4.5))
    plt.plot(straight["t"], straight["u"], lw=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Forward speed [m/s]")
    plt.title(f"Speed test: rpm={straight['rpm']}, rudder=0 deg")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "speed_test_velocity.png", dpi=200)
    plt.close()

if __name__=="__main__":
    main()