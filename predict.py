"""Score new chest X-rays with a trained run and save a Grad-CAM overlay for each.

    uv run python predict.py path/to/xray.png [more.jpg ...] [--name B_xrv]

Preprocessing matches prepare.py: grayscale, center-crop to square, resize to 256px (+ CLAHE if the run used it).
The score is the model's raw sigmoid output, not a calibrated probability; it shifts between hospitals.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from gradcam import cams_for
from train import DEVICE, build_model, predict

SIZE = 256


def load_xray(path, clahe):
    img = Image.open(path).convert("L")
    s = min(img.size)
    left, top = (img.width - s) // 2, (img.height - s) // 2
    x = np.asarray(img.crop((left, top, left + s, top + s)).resize((SIZE, SIZE), Image.BILINEAR))
    if clahe:
        x = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--name", default="B_xrv", help="run folder under runs/")
    args = ap.parse_args()
    runs = Path("runs") / args.name
    ck = torch.load(runs / "best.pt", map_location="cpu")
    cfg = ck.get("cfg", {"model": ck.get("model"), "clahe": False})
    model = build_model(cfg["model"], pretrained=False).to(DEVICE)
    model[1].load_state_dict(ck["state"])
    model.eval()
    thr = json.loads((runs / "metrics.json").read_text())["threshold"]

    x = torch.from_numpy(np.stack([load_xray(p, cfg["clahe"]) for p in args.images]))
    probs, cams = predict(model, x), cams_for(model, x)
    print(f"run {args.name} · threshold {thr:.3f} (chosen on TBX11K val)")
    for path, img, p, c in zip(args.images, x.numpy(), probs, cams):
        verdict = "TB" if p >= thr else "non-TB"
        out = Path(path).with_name(Path(path).stem + f"_gradcam_{args.name}.png")
        fig, ax = plt.subplots(1, 2, figsize=(8, 4.3))
        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(img, cmap="gray")
        ax[1].imshow(c, cmap="jet", alpha=0.4)
        for a in ax:
            a.axis("off")
        fig.suptitle(f"{Path(path).name}: score {p:.3f} → {verdict} (threshold {thr:.3f})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"{p:7.3f}  {verdict:6s}  {path}  →  {out}")


if __name__ == "__main__":
    main()
