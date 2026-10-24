#!/usr/bin/env bash
# Train + Grad-CAM each experiment into runs/<name>/, then print the comparison table.
# Live progress: tail -f runs/<name>/train.log
set -euo pipefail
cd "$(dirname "$0")"

run() {
  local name=$1; shift
  mkdir -p "runs/$name"
  uv run python -u train.py --name "$name" "$@" > "runs/$name/train.log" 2>&1
  uv run python gradcam.py --name "$name" >> "runs/$name/train.log" 2>&1
}

run A_clahe_aug      --model resnet18     --clahe --strong-aug
run B_xrv            --model xrv-densenet
run AB_xrv_clahe_aug --model xrv-densenet --clahe --strong-aug
uv run python compare.py
