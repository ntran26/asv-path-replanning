"""Draw a test set as one light PDF: every episode's starting scene, grouped by cell, up to
twenty panels to a page, each panel named by its test ID (2026-11-23).

The scene geometry and drawing come from `test_set_images.py`: each episode is reset with its
own seed (decoupled-episode panels are placed at reset, so the cache alone does not hold them)
and drawn with the map boundary, static panels, reference path, start and goal, the own hull,
and the target hull with its next 10 s at constant velocity.  Panels carry the test ID, the
variant code and the panel count from `definition.csv`; no outcome is shown.  Pages are vector
graphics with embedded TrueType text, so a test ID can be searched in a PDF reader.

    python tools/tiers/test_set_pdf.py [--version 4] [--processes 2] [--geometry-cache FILE]

Writes `results/test_set/v<version>/test_set_v<version>_scenes.pdf`: a contents page (cells,
episode counts, first page of each cell), then the cells in `definition.csv` order, the cells
of one map family (basin, channel, L1, L2, L3, FS) sharing pages.
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers")]

import numpy as np
import pandas as pd

import test_set_images as tsi

PER_PAGE = 20
LEGEND = ("Blue hull: own ship at the start; green circle and star: start and goal; dashed blue line: "
          "reference path; grey: static panels; red hull: target ship, dotted line its next "
          f"{tsi.TRACK_S:.0f} s at constant velocity; black outline: map boundary.")
SOURCE = {"frozen": "decoupled", "paper2": "coupled"}


def _job(job):
    return tsi._geometry(job)


def _grid(g):
    b = np.asarray(g["boundary"])
    w, h = np.ptp(b[:, 0]), np.ptp(b[:, 1])
    if h > 1.3 * w:
        return 2, 10            # tall maps (basin): two rows of ten
    if w > 1.3 * h:
        return 5, 4             # wide maps: five rows of four
    return 4, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="4")
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--geometry-cache", type=Path, default=None,
                    help="Optional pickle of the reset geometry, reused when it exists")
    a = ap.parse_args()
    base = ROOT / "results" / "test_set" / f"v{a.version}"
    definition = pd.read_csv(base / "definition.csv", keep_default_na=False).set_index("test_id")

    if a.geometry_cache is not None and a.geometry_cache.exists():
        geoms = pickle.loads(a.geometry_cache.read_bytes())
    else:
        with open(base / f"set_v{a.version}.0.pkl", "rb") as fh:
            kept, _ = pickle.load(fh)
        jobs = [(it["test_id"], it["built"], int(it["episode_seed"]), it["cell"], it["variant"]) for it in kept]
        with ProcessPoolExecutor(max_workers=a.processes, initializer=tsi._init) as pool:
            geoms = list(pool.map(_job, jobs, chunksize=8))
        if a.geometry_cache is not None:
            a.geometry_cache.write_bytes(pickle.dumps(geoms))
    print(f"{len(geoms)} scenes", flush=True)

    mismatch = [g["test_id"] for g in geoms if len(g["obstacles"]) != int(definition.loc[g["test_id"], "panels"])]
    if mismatch:
        raise SystemExit(f"panel count differs from definition.csv for {len(mismatch)} episodes: {mismatch[:5]}")
    if set(g["test_id"] for g in geoms) != set(definition.index):
        raise SystemExit("drawn episodes differ from definition.csv")

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 6})
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    order = {c: i for i, c in enumerate(dict.fromkeys(definition.cell))}

    def family(cell):
        stratum = definition[definition.cell == cell].stratum.iloc[0]
        return stratum if stratum in ("basin", "channel") else cell.split("-")[0]

    families = {}
    for g in sorted(geoms, key=lambda g: (order[g["cell"]], g["test_id"])):
        families.setdefault(family(g["cell"]), []).append(g)
    plan, first_page, page = [], {}, 2                  # page 1 is the contents page
    for fam, gs in families.items():
        for i, g in enumerate(gs):
            first_page.setdefault(g["cell"], page + i // PER_PAGE)
        pages = math.ceil(len(gs) / PER_PAGE)
        row = definition[definition.cell == gs[0]["cell"]].iloc[0]
        plan.append((fam, gs, page, pages, SOURCE.get(row.source, row.source), row.stratum))
        page += pages
    total_pages = page - 1

    out = base / f"test_set_v{a.version}_scenes.pdf"
    with PdfPages(out, metadata={"Title": f"Test set v{a.version}: starting scenes",
                                 "Subject": "Starting scene of every test-set episode, by cell"}) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        n_dec = sum(1 for g in geoms if definition.loc[g["test_id"], "source"] == "frozen")
        fig.text(0.05, 0.94, f"Test set v{a.version}: starting scenes", fontsize=16, weight="bold")
        fig.text(0.05, 0.905, f"{len(geoms)} episodes ({n_dec} decoupled, {len(geoms) - n_dec} coupled) in "
                              f"{len(order)} cells; up to {PER_PAGE} episodes per page, cells of one map family "
                              f"on shared pages. Source: results/test_set/v{a.version}/definition.csv and "
                              f"set_v{a.version}.0.pkl.", fontsize=8)
        fig.text(0.05, 0.88, LEGEND, fontsize=7)
        fig.text(0.05, 0.86, "Panel labels: test ID (which names the cell); variant code, panel count and leg, "
                             "as in definition.csv.", fontsize=7)
        rows = []
        for cell in order:
            d = definition[definition.cell == cell]
            rows.append([cell, d["class"].iloc[0], d.stratum.iloc[0], SOURCE.get(d.source.iloc[0], d.source.iloc[0]),
                         str(len(d)), str(first_page[cell])])
        ax = fig.add_axes([0.05, 0.04, 0.9, 0.8])
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=["Cell", "Class", "Stratum", "Part", "Episodes", "First page"],
                         loc="upper left", cellLoc="left", colLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 0.95)
        fig.text(0.95, 0.02, f"1 / {total_pages}", ha="right", fontsize=6)
        pdf.savefig(fig)
        plt.close(fig)

        for fam, gs, first, pages, src, stratum in plan:
            nrows, ncols = _grid(gs[0])
            for p in range(pages):
                chunk = gs[p * PER_PAGE:(p + 1) * PER_PAGE]
                fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 8.27))
                axes = np.atleast_1d(axes).ravel()
                for ax, g in zip(axes, chunk):
                    tsi.draw(ax, g, {}, {}, small=True)
                    ax.set_frame_on(False)          # the map boundary is the frame
                    d = definition.loc[g["test_id"]]
                    ax.set_title(f"{g['test_id']}\n{d.variant or '-'}, {int(d.panels)} panel"
                                 f"{'' if int(d.panels) == 1 else 's'}, {d.leg}", fontsize=5.5, color="black")
                for ax in axes[len(chunk):]:
                    ax.axis("off")
                lo, hi = p * PER_PAGE + 1, p * PER_PAGE + len(chunk)
                cells_here = ", ".join(dict.fromkeys(g["cell"] for g in chunk))
                name = f"{stratum} stratum" if fam == stratum else f"{fam} cells, {stratum} stratum"
                fig.suptitle(f"Test set v{a.version}, {name} ({src}): episodes {lo}-{hi} of {len(gs)}; "
                             f"cells {cells_here}", fontsize=9, x=0.02, ha="left")
                fig.text(0.02, 0.01, LEGEND, fontsize=5.5)
                fig.text(0.98, 0.01, f"{first + p} / {total_pages}", ha="right", fontsize=6)
                fig.tight_layout(rect=(0, 0.025, 1, 0.965))
                pdf.savefig(fig)
                plt.close(fig)
            print(f"{fam}: {len(gs)} episodes, {pages} page(s)", flush=True)
    print(f"written: {out} ({out.stat().st_size / 1e6:.2f} MB, {total_pages} pages)", flush=True)


if __name__ == "__main__":
    main()
