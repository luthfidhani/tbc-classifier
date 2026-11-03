"""Per-source confusion matrices, ROC and precision-recall curves at each run's val-chosen threshold.

Writes runs/<name>/report.png and runs/<name>/report.csv, and prints one summary table for all runs.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from compare import site

SOURCES = ["TBX11K test", "DA/DB", "Montgomery", "Shenzhen"]
COLS = ["run", "source", "threshold", "n", "tp", "fn", "fp", "tn", "sensitivity", "specificity",
        "ppv", "npv", "f1", "balanced_acc", "auc", "avg_precision"]


def groups(d):
    """(source name, y, p) for the internal test set and each external hospital."""
    t, e = np.load(d / "pred_test.npz"), np.load(d / "pred_external.npz")
    out = [("TBX11K test", t["y"], t["p"])]
    sites = np.array([site(p) for p in e["path"]])
    return out + [(s, e["y"][sites == s], e["p"][sites == s]) for s in SOURCES[1:]]


def row(run, name, y, p, thr):
    pred = p >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")
    sens, spec, ppv = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    return {"run": run, "source": name, "threshold": thr, "n": len(y), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": div(tn, tn + fn),
            "f1": div(2 * ppv * sens, ppv + sens), "balanced_acc": (sens + spec) / 2,
            "auc": roc_auc_score(y, p), "avg_precision": average_precision_score(y, p)}


def figure(run, d, gs, rows, thr):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2))
    for ax, (name, y, p), r in zip(axes[0], gs, rows):
        cm = np.array([[r["tp"], r["fn"]], [r["fp"], r["tn"]]])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)  # row-normalized: rate within each true class
        for (i, j), v in np.ndenumerate(cm):
            rate = v / cm[i].sum()
            ax.text(j, i, f"{v}\n{rate:.0%}", ha="center", va="center", fontsize=12, color="white" if rate > 0.55 else "black")
        ax.set_xticks([0, 1], ["pred TB", "pred non-TB"])
        ax.set_yticks([0, 1], ["true TB", "true non-TB"])
        ax.set_title(f"{name} (n={r['n']})\nsens {r['sensitivity']:.0%} · spec {r['specificity']:.0%}", fontsize=10)
    for name, y, p in gs:
        fpr, tpr, _ = roc_curve(y, p)
        axes[1, 0].plot(fpr, tpr, lw=1.8, label=f"{name} {roc_auc_score(y, p):.3f}")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1, 1].plot(rec, prec, lw=1.8, label=f"{name} {average_precision_score(y, p):.3f}")
        axes[1, 2].hist(p[y == 0], bins=20, range=(0, 1), weights=np.full((y == 0).sum(), 1 / (y == 0).sum()), histtype="step", lw=1.5, label=f"{name} non-TB")
        axes[1, 3].hist(p[y == 1], bins=20, range=(0, 1), weights=np.full((y == 1).sum(), 1 / (y == 1).sum()), histtype="step", lw=1.5, label=f"{name} TB")
    axes[1, 0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1, 0].set(title="ROC (AUC)", xlabel="1 − specificity", ylabel="sensitivity")
    axes[1, 1].set(title="Precision–recall (average precision)", xlabel="recall (sensitivity)", ylabel="precision (PPV)")
    for ax, cls in [(axes[1, 2], "non-TB"), (axes[1, 3], "TB")]:
        ax.axvline(thr, color="k", ls=":", lw=1)
        ax.set(title=f"Scores of true {cls} (threshold {thr:.3f})", xlabel="P(TB)", ylabel="fraction of source")
    for ax in axes[1]:
        ax.legend(fontsize=7.5)
    fig.suptitle(f"{run}: metrics at val-chosen threshold {thr:.3f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(d / "report.png", dpi=110)
    plt.close(fig)


def main():
    allrows = []
    for d in sorted(p.parent for p in Path("runs").glob("*/pred_external.npz")):
        thr = json.loads((d / "metrics.json").read_text())["threshold"]
        gs = groups(d)
        rows = [row(d.name, name, y, p, thr) for name, y, p in gs]
        figure(d.name, d, gs, rows, thr)
        with open(d / "report.csv", "w", newline="") as f:
            w = csv.DictWriter(f, COLS)
            w.writeheader()
            w.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
        allrows += rows
    print(f"{'run':18s}{'source':13s}{'thr':>6s}{'TP':>5s}{'FN':>5s}{'FP':>5s}{'TN':>5s}"
          + "".join(f"{c:>7s}" for c in ["sens", "spec", "PPV", "NPV", "F1"]))
    for r in allrows:
        print(f"{r['run']:18s}{r['source']:13s}{r['threshold']:6.3f}{r['tp']:5d}{r['fn']:5d}{r['fp']:5d}{r['tn']:5d}"
              + "".join(f"{r[c]:7.1%}" for c in ["sensitivity", "specificity", "ppv", "npv", "f1"]))
    print("wrote runs/<run>/report.png and report.csv")


if __name__ == "__main__":
    main()
