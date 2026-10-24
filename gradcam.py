"""Grad-CAM heatmaps for runs/<name>/best.pt, plus pointing-game score against TBX11K radiologist boxes."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from train import DEVICE, build_model, load_split, to_input

SCALE = 256 / 512  # boxes are in 512px coords


def target_layer(net):
    if hasattr(net, "layer4"):  # resnet
        return net.layer4[-1]
    if hasattr(net, "conv_head"):  # efficientnet
        return net.conv_head
    return net.features.denseblock4  # xrv densenet (norm5 output is relu'd in place, so hook the block before)


def cams_for(model, x, bs=32):
    layer = target_layer(model[1])
    with GradCAM(model=model, target_layers=[layer]) as cam:
        return np.concatenate([cam(input_tensor=to_input(x[i:i + bs]).to(DEVICE),
                                   targets=[ClassifierOutputTarget(0)] * len(x[i:i + bs]))
                               for i in range(0, len(x), bs)])


def box_mask(bxs):
    m = np.zeros((256, 256), bool)
    for bx, by, bw, bh in np.array(bxs) * SCALE:
        m[int(by):int(np.ceil(by + bh)), int(bx):int(np.ceil(bx + bw))] = True
    return m


def grid(x, cams, titles, boxes, path, cols=4):
    rows = int(np.ceil(len(x) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, img, c, t, bxs in zip(np.ravel(axes), x, cams, titles, boxes):
        ax.imshow(img, cmap="gray")
        ax.imshow(c, cmap="jet", alpha=0.4)
        for bx, by, bw, bh in np.array(bxs) * SCALE:
            ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec="lime", lw=1.5))
        ax.set_title(t, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline")
    runs = Path("runs") / ap.parse_args().name
    ck = torch.load(runs / "best.pt", map_location="cpu")
    cfg = ck.get("cfg", {"model": ck.get("model"), "clahe": False})  # first baseline run saved no cfg
    model = build_model(cfg["model"], pretrained=False).to(DEVICE)
    model[1].load_state_dict(ck["state"])
    model.eval()
    thr = json.loads((runs / "metrics.json").read_text())["threshold"]
    boxes = json.loads(open("data/boxes.json").read())

    # 1) Pointing game on test TB images that have radiologist boxes
    x, y, paths, _ = load_split("test", cfg["clahe"])
    p = np.load(runs / "pred_test.npz")["p"]
    tb = [i for i in range(len(y)) if paths[i] in boxes]
    cams = cams_for(model, x[tb])
    masks = [box_mask(boxes[paths[i]]) for i in tb]
    hits = [m.flat[c.argmax()] for c, m in zip(cams, masks)]
    energy = [c[m].sum() / c.sum() for c, m in zip(cams, masks)]
    chance = [m.mean() for m in masks]  # hit rate of a random point = box area fraction
    score = {"n": len(tb), "pointing_game": round(float(np.mean(hits)), 3),
             "random_point_baseline": round(float(np.mean(chance)), 3),
             "cam_energy_in_box": round(float(np.mean(energy)), 3)}
    print(json.dumps(score, indent=2))
    (runs / "pointing_game.json").write_text(json.dumps(score, indent=2))
    show = tb[:12]
    grid(x[show], cams[:12], [f"{paths[i].split('/')[-1]} p={p[i]:.2f} {'HIT' if h else 'miss'}"
                              for i, h in zip(show, hits)],
         [boxes[paths[i]] for i in show], runs / "gradcam_tb.png")

    # 2) Mistakes: missed TB (false negatives) and false alarms (false positives)
    fn = [i for i in np.argsort(p) if y[i] == 1 and p[i] < thr][:4]
    fp = [i for i in np.argsort(-p) if y[i] == 0 and p[i] >= thr][:4]
    idx = fn + fp
    if idx:
        grid(x[idx], cams_for(model, x[idx]),
             [f"{'FN' if y[i] else 'FP'} {paths[i].split('/')[-1]} p={p[i]:.2f}" for i in idx],
             [boxes.get(paths[i], []) for i in idx], runs / "gradcam_errors.png")

    # 3) External hospitals: does attention stay on the lungs on unseen scanners?
    x, y, paths, src = load_split("external", cfg["clahe"])
    pe = np.load(runs / "pred_external.npz")["p"]
    idx = [i for s in np.unique(src) for i in np.flatnonzero(src == s)[np.argsort(-pe[src == s])][:4]]
    grid(x[idx], cams_for(model, x[idx]),
         [f"{src[i]} y={y[i]} p={pe[i]:.2f}" for i in idx], [[]] * len(idx), runs / "gradcam_external.png")


if __name__ == "__main__":
    main()
