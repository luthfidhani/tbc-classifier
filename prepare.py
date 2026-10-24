"""Download TBX11K, resize to 256px grayscale, write data/{train,val,test,external}.npz + data/boxes.json."""
import json
from collections import defaultdict
from pathlib import Path

import kagglehub
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

SIZE = 256
OUT = Path("data")
ROOT = Path(kagglehub.dataset_download("usmanshams/tbx-11")) / "TBX11K"


def tbx_list(name):
    rels = (ROOT / "lists" / name).read_text().split()
    return [(ROOT / "imgs" / r, int(r.startswith("tb/")), "tbx11k") for r in rels]


def extra_label(p):
    if p.parent.parent.name == "mc+shenzhen":
        return int(p.stem.endswith("_1"))  # MCUCXR_0023_0 / CHNCXR_0636_1
    return int(p.stem.startswith("p"))  # DA/DB: n*/nx* normal, p*/px* TB


val_all = tbx_list("TBX11K_val.txt")
val, test = train_test_split(val_all, test_size=0.5, stratify=[y for _, y, _ in val_all], random_state=0)
extra = [(p, extra_label(p), p.parent.parent.name) for p in sorted((ROOT / "imgs/extra").glob("*/*/*.*"))]
splits = {"train": tbx_list("TBX11K_train.txt"), "val": val, "test": test, "external": extra}


def thumbs(x):
    t = x.reshape(len(x), 32, SIZE // 32, 32, SIZE // 32).mean((2, 4)).reshape(len(x), -1)
    return t.astype(np.float32)


def min_dist(a, b):
    """Per-pixel RMS distance from each row of a to its nearest row in b."""
    d2 = (a**2).sum(1)[:, None] + (b**2).sum(1)[None] - 2 * a @ b.T
    return np.sqrt(np.maximum(d2.min(1), 0) / a.shape[1])


OUT.mkdir(exist_ok=True)
# Official TBX11K splits contain (near-)identical copies across train/val; drop later-split copies to avoid leakage.
DUP_RMS = 8.0  # 0-255 scale; observed: dup pairs <= 3.6, nearest distinct pair ~14
seen = np.zeros((0, 32 * 32), np.float32)
for name, items in splits.items():
    x = np.stack([np.asarray(Image.open(p).convert("L").resize((SIZE, SIZE), Image.BILINEAR)) for p, _, _ in items])
    t = thumbs(x)
    keep = min_dist(t, seen) > DUP_RMS if len(seen) else np.ones(len(x), bool)
    seen = np.concatenate([seen, t[keep]])
    items = [it for it, k in zip(items, keep) if k]
    y = np.array([y for _, y, _ in items])
    np.savez(OUT / f"{name}.npz", x=x[keep], y=y,
             path=[str(p.relative_to(ROOT / "imgs")) for p, _, _ in items],
             source=[s for _, _, s in items])
    print(f"{name:9s} n={len(y):5d}  TB={y.sum():4d}  dropped_dups={(~keep).sum()}")

# COCO bboxes (512px coords, xywh) for TB images, keyed by relative path
boxes = defaultdict(list)
for f in ["TBX11K_train.json", "TBX11K_val.json"]:
    j = json.loads((ROOT / "annotations/json" / f).read_text())
    names = {im["id"]: im["file_name"] for im in j["images"]}
    for a in j["annotations"]:
        boxes[names[a["image_id"]]].append(a["bbox"])
(OUT / "boxes.json").write_text(json.dumps(boxes))
print(f"bbox images: {len(boxes)}")
