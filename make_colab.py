"""Build a Colab notebook from the local scripts, so Colab runs exactly the code in this folder. Rerun after editing code.

Usage: python make_colab.py ["<name> <train args>" ...]   e.g. "B_xrv --model xrv-densenet"
"""
import hashlib  # noqa: F401  (used by eval(FP))
import json
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (used by eval(FP))

RUNS = sys.argv[1:] or [
    "B_xrv --model xrv-densenet",
    "AB_xrv_clahe_aug --model xrv-densenet --clahe --strong-aug",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


cells = [
    md("# TB classifier: remote experiments\nRuntime → Change runtime type → **T4 GPU**, then Runtime → **Run all**. "
       "The last cell downloads `runs_<names>.zip`; unzip it in the project folder on the Mac."),
    code("!nvidia-smi -L\n!pip -q install timm grad-cam kagglehub torchxrayvision opencv-python-headless"),
]
cells += [code(f"%%writefile {f}\n" + Path(f).read_text()) for f in ["prepare.py", "train.py", "gradcam.py", "compare.py"]]
cells.append(code("!python prepare.py  # downloads TBX11K (~3GB) and builds the same deterministic splits as the Mac"))
# Fingerprint each split's file list so Colab provably trains/tests on the same images as the Mac.
FP = "hashlib.md5('|'.join(np.load(f'data/{s}.npz')['path']).encode()).hexdigest()"
splits = ["train", "val", "test", "external"]
local = {s: eval(FP) for s in splits}
cells.append(code(f"import hashlib, numpy as np\nlocal = {local!r}\n"
                  f"colab = {{s: {FP} for s in local}}\n"
                  "assert colab == local, f'split mismatch vs Mac: {colab}'\nprint('splits identical to Mac')"))
for r in RUNS:
    name = r.split()[0]
    cells.append(code(f"!python -u train.py --name {r}\n!python gradcam.py --name {name}"))
tag = "_".join(r.split()[0] for r in RUNS)
names = " ".join(f"runs/{r.split()[0]}" for r in RUNS)
cells.append(code(f"!python compare.py\n!zip -qr runs_{tag}.zip {names}\n"
                  f"from google.colab import files\nfiles.download('runs_{tag}.zip')"))

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
Path(f"colab_{tag}.ipynb").write_text(json.dumps(nb, indent=1))
print(f"wrote colab_{tag}.ipynb")
