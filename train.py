"""Fine-tune a pretrained CNN for TB vs non-TB on data/*.npz (from prepare.py). Tuned for a 16GB MacBook Air (MPS); also runs on CUDA (Colab)."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch import nn
from torchvision.transforms import v2
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_split(name, clahe=False):
    z = np.load(f"data/{name}.npz")
    x = z["x"]
    if clahe:  # local contrast equalization: evens out scanner/processing differences between hospitals
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        x = np.stack([c.apply(img) for img in x])
    return torch.from_numpy(x), torch.from_numpy(z["y"]), z["path"], z["source"]


class Scale(nn.Module):
    """Maps [0,1] pixels to whatever range the backbone was pretrained on."""

    def __init__(self, mul, add):
        super().__init__()
        self.mul, self.add = mul, add

    def forward(self, x):
        return x * self.mul + self.add


def build_model(name, pretrained=True):
    """Returns Sequential(Scale, net); checkpoints store net only, so model[1] is the backbone."""
    if name == "xrv-densenet":  # DenseNet121 pretrained on ~8 public chest X-ray datasets from many hospitals
        net = xrv.models.DenseNet(weights="densenet121-res224-all") if pretrained else xrv.models.DenseNet(num_classes=1)
        net.op_threshs = None
        net.classifier = nn.Linear(net.classifier.in_features, 1)
        net.input_resolution = 224  # xrv resizes internally to its native resolution
        return nn.Sequential(Scale(2048, -1024), net)  # xrv expects [-1024, 1024]
    mean, std = 0.449, 0.226  # ImageNet RGB stats averaged, since timm folds RGB weights into 1 channel
    net = timm.create_model(name, pretrained=pretrained, num_classes=1, in_chans=1)
    return nn.Sequential(Scale(1 / std, -mean / std), net)


def to_input(x_uint8):
    return (x_uint8.float() / 255).unsqueeze(1)


# No horizontal flip: heart/aorta side is anatomical signal, flipping makes fake anatomy.
AUGMENT = v2.Compose([
    v2.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
])
# Simulates other scanners: exposure curve (gamma), contrast, softness/sharpness of the detector.
STRONG_AUGMENT = v2.Compose([
    v2.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.85, 1.15)),
    v2.Lambda(lambda x: v2.functional.adjust_gamma(x, float(np.random.uniform(0.7, 1.5)))),
    v2.ColorJitter(brightness=0.4, contrast=0.4),
    v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
    v2.RandomAdjustSharpness(2, p=0.3),
])


@torch.no_grad()
def predict(model, x, bs=128):
    model.eval()
    return torch.cat([torch.sigmoid(model(to_input(x[i:i + bs]).to(DEVICE))).squeeze(1).cpu()
                      for i in range(0, len(x), bs)]).numpy()


def metrics(y, p, thr):
    tn, fp, fn, tp = confusion_matrix(y, p >= thr, labels=[0, 1]).ravel()
    return {"auc": round(roc_auc_score(y, p), 4), "sensitivity": round(tp / (tp + fn), 4),
            "specificity": round(tn / (tn + fp), 4), "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline", help="experiment folder under runs/")
    ap.add_argument("--model", default="resnet18", help="any timm model, or xrv-densenet")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--strong-aug", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)  # 32 @256px fits 16GB unified memory comfortably
    args = ap.parse_args()
    cfg = vars(args)
    runs = Path("runs") / args.name
    runs.mkdir(parents=True, exist_ok=True)
    print(cfg)

    torch.manual_seed(0)
    np.random.seed(0)
    xtr, ytr, _, _ = load_split("train", args.clahe)
    xva, yva, _, _ = load_split("val", args.clahe)
    augment = STRONG_AUGMENT if args.strong_aug else AUGMENT
    model = build_model(args.model).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = args.epochs * (len(xtr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.1)
    pos_weight = ((ytr == 0).sum() / (ytr == 1).sum()).float().to(DEVICE)  # ~10x: TB is rare
    best = -1

    for ep in range(args.epochs):
        model.train()
        t0, perm, total = time.time(), torch.randperm(len(xtr)), 0.0
        bar = tqdm(range(0, len(perm) - args.batch + 1, args.batch), desc=f"ep {ep + 1}/{args.epochs}", leave=False)
        for i in bar:  # drop last partial batch
            idx = perm[i:i + args.batch]
            xb = torch.stack([augment(img[None]) for img in xtr[idx]]).squeeze(1)
            xb, yb = to_input(xb).to(DEVICE), ytr[idx].float().to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(xb).squeeze(1), yb, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
            bar.set_postfix(loss=f"{loss.item():.3f}")
        auc = roc_auc_score(yva.numpy(), predict(model, xva))
        if auc > best:
            best = auc
            torch.save({"cfg": cfg, "state": model[1].state_dict()}, runs / "best.pt")
        print(f"ep {ep + 1:2d}  loss {total / len(xtr):.4f}  val_auc {auc:.4f}  {time.time() - t0:.0f}s"
              + ("  *best" if auc == best else ""))

    # Evaluate best checkpoint. Threshold picked on val (Youden's J), then frozen for test/external.
    model[1].load_state_dict(torch.load(runs / "best.pt", map_location="cpu")["state"])
    pva = predict(model, xva)
    fpr, tpr, thr = roc_curve(yva.numpy(), pva)
    t = float(thr[np.argmax(tpr - fpr)])
    out = {"threshold": round(t, 4), "val": metrics(yva.numpy(), pva, t)}
    for name in ["test", "external"]:
        x, y, paths, src = load_split(name, args.clahe)
        p = predict(model, x)
        out[name] = metrics(y.numpy(), p, t)
        np.savez(runs / f"pred_{name}.npz", p=p, y=y.numpy(), path=paths, source=src)
    (runs / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
