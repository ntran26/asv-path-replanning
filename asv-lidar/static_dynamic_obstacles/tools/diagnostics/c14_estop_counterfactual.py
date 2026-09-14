"""A18: the C14 head-on set with the emergency-stop supervisor on and off.

Found (F54): switching the supervisor off cut target collisions from 0.43 to
0.24 at 5 m, 0.25 to 0.15 at 6 m and 0.24 to 0.14 at 7 m, with 10 saved and
1 added; 10 m was unchanged.

    python tools/diagnostics/c14_estop_counterfactual.py
"""
import pandas as pd

from c14_common import OUT, WIDTH_BINS, head_on_scenarios, load_run2_model
from env import ASVLidarEnv


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scen = head_on_scenarios()
    model = load_run2_model()
    rows = []
    for estop in (True, False):
        env = ASVLidarEnv(render_mode=None)
        for i, b in enumerate(scen):
            env.forced_num_obs = 0
            obs, _ = env.reset(seed=700_000 + i, options={"generated": b})
            env.estop_enabled = estop
            speeds = []
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                speeds.append(info["speed_mps"])
                if term or trunc:
                    break
            rows.append(dict(estop="on" if estop else "off", idx=i, width=b.nominal_width, dcpa=b.dcpa_m,
                             coll_target=info["collision_kind"] == "target",
                             outcome=info["collision_kind"] or ("goal" if info["reached_goal"] else "timeout"),
                             estops=int(info["estop/events"]), u_min=min(speeds)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "c14_estop.csv", index=False)
    d["wbin"] = pd.cut(d.width, *WIDTH_BINS)
    pd.set_option("display.width", 250)
    print(d.pivot_table(index="wbin", columns="estop", values=["coll_target", "estops", "u_min"],
                        aggfunc="mean", observed=True).round(2))
    on, off = d[d.estop == "on"].set_index("idx"), d[d.estop == "off"].set_index("idx")
    print("saved by switching the supervisor off:", int((on.coll_target & ~off.coll_target).sum()),
          " newly collided:", int((~on.coll_target & off.coll_target).sum()))


if __name__ == "__main__":
    main()
