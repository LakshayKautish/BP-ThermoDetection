
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
# STEP 0 of training (run once, before the training_script_task*.py files).
#
# The public Roboflow "Blood-Pressure-monitor-digit-reader" dataset was exported
# with random 90-degree rotations and mirror flips, but its boxes are stored as
# axis-aligned rectangles, so the labels do not tell us which images are
# rotated/mirrored. A mirrored '2' looks like a '5', so using those images
# directly would poison the digit classifier and the digit detector.
#
# This script "de-augments" the dataset:
#   1. train a seed digit CNN purely on synthetic seven-segment digits;
#   2. for every dataset image try all 8 rotations/mirrors, classify every
#      labelled digit crop and keep the orientation whose predictions agree
#      best with the ground-truth labels;
#   3. write the upright images + labels to <work>/bpm_upright/<split>/.
#
# Usage: python prepare_data.py [--roboflow DIR] [--work DIR]

import os
import json
import argparse
import random

import cv2
import numpy as np

from common import preprocess_digit
from training_utils import (DEFAULT_ROBOFLOW, DEFAULT_WORK, DOCS_DIR, NUMBER_CLASS, ROBOFLOW_NAMES,
                            roboflow_items, read_roboflow_labels, digit_label, dihedral_image,
                            dihedral_box, render_sevenseg, train_digit_cnn, predict_digit_cnn,
                            seed_everything)


def train_seed_model(n_per_class=2500, epochs=6):
    print(f"[1/3] Training seed CNN on {n_per_class * 10} synthetic seven-segment digits")
    xs, ys = [], []
    for d in range(10):
        for _ in range(n_per_class):
            xs.append(render_sevenseg(d))
            ys.append(d)
    vx, vy = [], []
    for d in range(10):
        for _ in range(100):
            vx.append(render_sevenseg(d))
            vy.append(d)
    model, hist = train_digit_cnn(xs, ys, vx, vy, epochs=epochs, lr=2e-3)
    return model, hist


def crop(img, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    return img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]


def deaugment(model, roboflow, work):
    import torch
    print("[2/3] De-augmenting Roboflow images (trying 8 orientations per image)")
    out_root = os.path.join(work, "bpm_upright")
    report = {}
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(out_root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_root, split, "labels"), exist_ok=True)
        kept, dropped, chosen = 0, 0, [0] * 8
        for n, (img_path, lab_path) in enumerate(roboflow_items(roboflow, split)):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            labels = read_roboflow_labels(lab_path, w, h)
            digits = [(digit_label(c), b) for c, *b in labels if digit_label(c) is not None]
            numbers = [b for c, *b in labels if c == NUMBER_CLASS]
            if len(digits) < 2:
                dropped += 1
                continue
            crops, ks = [], []
            for k in range(8):
                timg = dihedral_image(img, k)
                for _, b in digits:
                    tb, _, _ = dihedral_box(b, k, w, h)
                    c = crop(timg, tb)
                    crops.append(preprocess_digit(c) if c.size else np.zeros((48, 48), np.float32))
            prob = predict_digit_cnn(model, torch.from_numpy(np.stack(crops)[:, None]), return_prob=True)
            prob = prob.reshape(8, len(digits), 10)
            y = np.array([d for d, _ in digits])
            logp = np.log(prob[:, np.arange(len(digits)), y] + 1e-6).mean(1)       # (8,)
            acc = (prob.argmax(2) == y[None]).mean(1)
            # prior: number boxes such as "135" are wider than tall when upright
            if numbers:
                wide = np.mean([(b[2] - b[0]) > (b[3] - b[1]) for b in numbers])
                for k in range(8):
                    rot90 = (k % 2) == 1
                    logp[k] += 0.5 * ((1 - wide) if not rot90 else wide)
            k = int(np.argmax(logp))
            if acc[k] < 0.6:                           # nothing explains the labels -> drop
                dropped += 1
                continue
            chosen[k] += 1
            kept += 1
            timg = dihedral_image(img, k)
            th, tw = timg.shape
            name = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(out_root, split, "images", name + ".jpg"), timg)
            with open(os.path.join(out_root, split, "labels", name + ".txt"), "w") as f:
                for c, *b in labels:
                    tb, _, _ = dihedral_box(b, k, w, h)
                    f.write(f"{c} {tb[0]:.2f} {tb[1]:.2f} {tb[2]:.2f} {tb[3]:.2f}\n")   # pixel xyxy
            if n % 500 == 0:
                print(f"  {split}: {n} images processed")
        report[split] = {"kept": kept, "dropped": dropped, "orientation_counts": chosen}
        print(f"  {split}: kept {kept}, dropped {dropped}, orientation histogram {chosen}")
    return out_root, report


def preview(out_root, path, n=12):
    names = sorted(os.listdir(os.path.join(out_root, "train", "images")))
    random.shuffle(names)
    tiles = []
    for name in names[:n]:
        im = cv2.cvtColor(cv2.imread(os.path.join(out_root, "train", "images", name), 0), cv2.COLOR_GRAY2BGR)
        for line in open(os.path.join(out_root, "train", "labels", name[:-4] + ".txt")):
            c, *b = line.split()
            b = [int(float(v)) for v in b]
            cv2.rectangle(im, b[:2], b[2:], (0, 200, 0) if int(c) != NUMBER_CLASS else (255, 0, 0), 2)
            if int(c) != NUMBER_CLASS:
                cv2.putText(im, ROBOFLOW_NAMES[int(c)], (b[0], b[1] - 3), 0, 0.8, (0, 0, 255), 2)
        tiles.append(cv2.resize(im, (320, 320)))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles) - len(tiles) % 6, 6)]
    cv2.imwrite(path, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roboflow", default=DEFAULT_ROBOFLOW)
    ap.add_argument("--work", default=DEFAULT_WORK)
    args = ap.parse_args()
    seed_everything(0)
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    model, hist = train_seed_model()
    out_root, report = deaugment(model, args.roboflow, args.work)
    print("[3/3] Writing report and preview")
    report["seed_model_history"] = hist
    json.dump(report, open(os.path.join(DOCS_DIR, "prepare_data_report.json"), "w"), indent=1)
    preview(out_root, os.path.join(DOCS_DIR, "deaugmented_preview.jpg"))
    print(f"Done. Upright dataset in {out_root}")


if __name__ == "__main__":
    main()
