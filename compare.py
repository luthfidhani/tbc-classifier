"""Side-by-side table of every experiment in runs/: internal test vs each external hospital vs Grad-CAM localization."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

COLS = ["test", "external", "DA/DB", "Montgomery", "Shenzhen", "pointing", "energy"]


def site(path):
    return "Montgomery" if "MCUCXR" in path else "Shenzhen" if "CHNCXR" in path else "DA/DB"


def boot_delta(y, pa, pb, n=1000, seed=0):
    """Paired, label-stratified bootstrap 95% CI of AUC(pa) - AUC(pb) on the same images."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    return np.percentile(d, [2.5, 97.5])


def main():
    print(f"{'run':16s}" + "".join(f"{c:>11s}" for c in COLS))
    for run in sorted(Path("runs").glob("*/pred_external.npz")):
        d = run.parent
        t, e = np.load(d / "pred_test.npz"), np.load(d / "pred_external.npz")
        sites = np.array([site(p) for p in e["path"]])
        row = [roc_auc_score(t["y"], t["p"]), roc_auc_score(e["y"], e["p"])]
        row += [roc_auc_score(e["y"][sites == s], e["p"][sites == s]) for s in COLS[2:5]]
        pg = json.loads((d / "pointing_game.json").read_text()) if (d / "pointing_game.json").exists() else {}
        row += [pg.get("pointing_game", np.nan), pg.get("cam_energy_in_box", np.nan)]
        print(f"{d.name:16s}" + "".join(f"{v:11.3f}" for v in row))
    print("AUC columns: test..Shenzhen | Grad-CAM vs radiologist boxes: pointing (random ~0.106), energy-in-box")

    # Is a run's external gain real or noise? CI excluding 0 = difference unlikely to be chance (single seed though).
    if not Path("runs/baseline/pred_external.npz").exists():  # e.g. on Colab, where only the remote runs live
        return
    base = np.load("runs/baseline/pred_external.npz")
    sites = np.array([site(p) for p in base["path"]])
    groups = {"external": np.ones(len(sites), bool)} | {s: sites == s for s in COLS[2:5]}
    print(f"\n{'Δ AUC vs baseline':18s}" + "".join(f"{g:>24s}" for g in groups))
    for run in sorted(Path("runs").glob("*/pred_external.npz")):
        if run.parent.name == "baseline":
            continue
        e = np.load(run)
        assert (e["path"] == base["path"]).all(), f"{run}: external images differ from baseline"
        cells = []
        for m in groups.values():
            y, pa, pb = e["y"][m], e["p"][m], base["p"][m]
            lo, hi = boot_delta(y, pa, pb)
            star = "*" if lo > 0 or hi < 0 else " "
            cells.append(f"{roc_auc_score(y, pa) - roc_auc_score(y, pb):+.3f} [{lo:+.3f},{hi:+.3f}]{star}")
        print(f"{run.parent.name:18s}" + "".join(f"{c:>24s}" for c in cells))
    print("[95% bootstrap CI]  * = CI excludes 0")


if __name__ == "__main__":
    main()
