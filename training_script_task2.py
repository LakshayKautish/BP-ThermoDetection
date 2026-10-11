
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: Lakshay Kautish (20002772)
# Last Modified: 2026-10-01
#
# TASK 2 training: single-class digit detector (YOLOv8n) that runs on a
# rectified LCD image and returns one box per seven-segment digit.
#
# Requires prepare_data.py to have produced <work>/bpm_upright (the Roboflow
# dataset with its rotation/mirror augmentation undone). Each sample is the
# LCD region (union of all labelled boxes + a random margin) cropped from an
# upright image, so training images look like task 1 outputs / task 2 inputs.
#
# Usage: python training_script_task2.py [--epochs 80] [--batch 32] [--device 0]

import os
import glob
import json
import shutil
import random
import argparse

import cv2

from common import DIGIT_DET_WEIGHTS, DATA_DIR
from training_utils import DEFAULT_WORK, DOCS_DIR, NUMBER_CLASS, seed_everything


def read_upright_labels(path):
    out = []
    for line in open(path):
        p = line.split()
        if len(p) == 5:
            out.append((int(p[0]), *map(float, p[1:])))
    return out


def build_dataset(work, max_side=416, copies=2):
    src = os.path.join(work, "bpm_upright")
    root = os.path.join(work, "digit_det")
    if os.path.isdir(root):
        shutil.rmtree(root)
    stats = {}
    for split_in, split_out in (("train", "train"), ("valid", "val"), ("test", "test")):
        os.makedirs(os.path.join(root, split_out, "images"))
        os.makedirs(os.path.join(root, split_out, "labels"))
        n = 0
        for img_path in sorted(glob.glob(os.path.join(src, split_in, "images", "*.jpg"))):
            labels = read_upright_labels(img_path.replace("images", "labels")[:-4] + ".txt")
            digits = [l for l in labels if l[0] != NUMBER_CLASS]
            if len(digits) < 3:
                continue
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            x0 = min(l[1] for l in labels); y0 = min(l[2] for l in labels)
            x1 = max(l[3] for l in labels); y1 = max(l[4] for l in labels)
            for c in range(copies if split_out == "train" else 1):
                # random margin so the model sees tight and loose LCD crops
                m = [random.uniform(0.03, 0.25) for _ in range(4)]
                cx0 = int(max(0, x0 - m[0] * (x1 - x0))); cy0 = int(max(0, y0 - m[1] * (y1 - y0)))
                cx1 = int(min(w, x1 + m[2] * (x1 - x0))); cy1 = int(min(h, y1 + m[3] * (y1 - y0)))
                crop = img[cy0:cy1, cx0:cx1]
                s = min(1.0, max_side / max(crop.shape[:2]))
                crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                ch, cw = crop.shape[:2]
                name = f"{os.path.basename(img_path)[:-4]}_{c}"
                cv2.imwrite(os.path.join(root, split_out, "images", name + ".jpg"), crop)
                with open(os.path.join(root, split_out, "labels", name + ".txt"), "w") as f:
                    for _, a, b, cc, d in digits:
                        a, b, cc, d = (a - cx0) * s, (b - cy0) * s, (cc - cx0) * s, (d - cy0) * s
                        a, b, cc, d = max(0, a), max(0, b), min(cw, cc), min(ch, d)
                        if cc - a < 2 or d - b < 2:
                            continue
                        f.write(f"0 {(a + cc) / 2 / cw:.6f} {(b + d) / 2 / ch:.6f} {(cc - a) / cw:.6f} {(d - b) / ch:.6f}\n")
                n += 1
        stats[split_out] = n
    yaml_path = os.path.join(root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {root}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: digit\n")
    print(f"  digit detector samples: {stats}")
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=DEFAULT_WORK)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    seed_everything(0)
    if not os.path.isdir(os.path.join(args.work, "bpm_upright")):
        raise SystemExit("Run prepare_data.py first (creates <work>/bpm_upright).")

    yaml_path = os.path.join(args.work, "digit_det", "data.yaml")
    if not args.skip_build or not os.path.exists(yaml_path):
        print("[1/3] Building digit-detector dataset from upright LCD crops")
        yaml_path = build_dataset(args.work, max_side=args.imgsz)

    print("[2/3] Fine-tuning YOLOv8n digit detector")
    from ultralytics import YOLO, settings
    settings.update({"wandb": False})          # no experiment-tracking login needed
    model = YOLO("yolov8n.pt")
    run_dir = os.path.join(args.work, "runs")
    # NO left-right flips: a mirrored seven-segment digit is a different digit.
    model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                project=run_dir, name="digit_det", exist_ok=True, seed=0, patience=25,
                fliplr=0.0, flipud=0.0, degrees=5.0, shear=5.0, perspective=0.0005, scale=0.4,
                translate=0.1, mosaic=0.5, hsv_s=0.5, plots=True)
    best = os.path.join(run_dir, "digit_det", "weights", "best.pt")
    os.makedirs(DATA_DIR, exist_ok=True)
    shutil.copy(best, DIGIT_DET_WEIGHTS)

    print("[3/3] Evaluating on the held-out Roboflow test split")
    metrics = YOLO(DIGIT_DET_WEIGHTS).val(data=yaml_path, split="test", imgsz=args.imgsz, device=args.device,
                                          project=run_dir, name="digit_det_test", exist_ok=True)
    out_docs = os.path.join(DOCS_DIR, "task2_digit_detector_training")
    os.makedirs(out_docs, exist_ok=True)
    for f in glob.glob(os.path.join(run_dir, "digit_det", "*.png")) + glob.glob(os.path.join(run_dir, "digit_det", "*.csv")):
        shutil.copy(f, out_docs)
    json.dump({"test_mAP50": float(metrics.box.map50), "test_mAP50_95": float(metrics.box.map),
               "test_precision": float(metrics.box.mp), "test_recall": float(metrics.box.mr)},
              open(os.path.join(out_docs, "test_metrics.json"), "w"), indent=1)
    print(f"Saved digit detector to {DIGIT_DET_WEIGHTS}; test mAP50={metrics.box.map50:.3f}")


if __name__ == "__main__":
    main()
