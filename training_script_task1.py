
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
# TASK 1 training: device detector (YOLOv8n fine-tuned from COCO weights)
# with two classes: 0 = lcd (BPM display panel), 1 = thermometer.
#
# Training data (built automatically into <work>/detector):
#   * Roboflow BPM images: the LCD box is the union of all digit/number boxes
#     plus a margin (the dataset has no LCD labels of its own).
#   * Our 143 thermometer photos (boxes in docs/thermometer_boxes.json; desk
#     photos auto-labelled with manual review or by hand; 114 varied-background
#     photos added 2 Oct, labelled with a model-in-the-loop) with rotation/scale
#     augmentation.
#   * Composite scenes: a BPM image pasted, small, into a thermometer photo,
#     so the detector learns small LCDs inside a large cluttered scene.
#   * Negative crops (no device) with empty label files.
# Part of the colour images are converted to greyscale because the Roboflow
# images are all greyscale - otherwise "grey image" would become a shortcut.
#
# Usage: python training_script_task1.py [--epochs 80] [--batch 32] [--device 0]

import os
import glob
import json
import shutil
import random
import argparse

import cv2
import numpy as np

from common import DETECTOR_WEIGHTS, DETECTOR_CLASSES, DATA_DIR
from training_utils import (DEFAULT_ROBOFLOW, DEFAULT_THERMO, DEFAULT_WORK, DOCS_DIR, roboflow_items,
                            read_roboflow_labels, load_thermo_boxes, seed_everything)

LCD, THERMO = 0, 1


def yolo_line(cls, box, w, h):
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    return f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n"


def save_sample(root, split, name, img, objects):
    cv2.imwrite(os.path.join(root, split, "images", name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    h, w = img.shape[:2]
    with open(os.path.join(root, split, "labels", name + ".txt"), "w") as f:
        for cls, box in objects:
            f.write(yolo_line(cls, box, w, h))


def lcd_box_from_labels(labels, w, h, margin=0.08):
    """Union of all digit/number boxes, expanded by a margin."""
    xs0, ys0, xs1, ys1 = zip(*[b[1:] for b in labels])
    x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)


def maybe_gray(img, p=0.3):
    if random.random() < p:
        return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return img


def rotate_with_box(img, box, angle):
    """Rotate image (no clipping) and return the axis-aligned box of the rotated box corners."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(img, M, (nw, nh), borderMode=cv2.BORDER_REFLECT)
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    pts = (M[:, :2] @ pts.T).T + M[:, 2]
    return out, (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())


def build_dataset(roboflow, thermo_dir, work, n_bpm_train=2500, n_bpm_val=300, n_composite=700, n_neg=200):
    root = os.path.join(work, "detector")
    if os.path.isdir(root):
        shutil.rmtree(root)
    for sp in ("train", "val"):
        os.makedirs(os.path.join(root, sp, "images"))
        os.makedirs(os.path.join(root, sp, "labels"))
    stats = {"train": {}, "val": {}}

    # --- Roboflow BPM images --------------------------------------------------
    bpm_train = list(roboflow_items(roboflow, "train"))
    random.shuffle(bpm_train)
    bpm_val = list(roboflow_items(roboflow, "valid"))[:n_bpm_val]
    for split, items, n in (("train", bpm_train, n_bpm_train), ("val", bpm_val, n_bpm_val)):
        k = 0
        for img_path, lab_path in items:
            if k >= n:
                break
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            labels = read_roboflow_labels(lab_path, w, h)
            if len(labels) < 3:
                continue
            save_sample(root, split, f"bpm_{k:05d}", img, [(LCD, lcd_box_from_labels(labels, w, h))])
            k += 1
        stats[split]["bpm"] = k
    print(f"  Roboflow LCD samples: {stats['train']['bpm']} train / {stats['val']['bpm']} val")

    # --- thermometer photos ----------------------------------------------------
    boxes = load_thermo_boxes()
    names = sorted(boxes)
    rnd = random.Random(1)
    val_names = set(rnd.sample(names, max(5, len(names) // 10)))
    therm = {}
    for name in names:
        img = cv2.imread(os.path.join(thermo_dir, name))
        s = 1280 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        b = boxes[name]
        therm[name] = (img, (b[0] * w, b[1] * h, b[2] * w, b[3] * h))
    nt = 0
    for name in names:
        img, box = therm[name]
        stem = os.path.splitext(name)[0]
        if name in val_names:
            save_sample(root, "val", f"th_{stem}", img, [(THERMO, box)])
            continue
        save_sample(root, "train", f"th_{stem}_0", img, [(THERMO, box)])
        nt += 1
        for j in range(1, 7):
            ang = random.choice([random.uniform(-20, 20), 90 + random.uniform(-15, 15), -90 + random.uniform(-15, 15)])
            rimg, rbox = rotate_with_box(img, box, ang)
            # zoom out: shrink and paste at a random place on a plain canvas
            # filled with the photo's median colour (smaller apparent size)
            if random.random() < 0.5:
                sc = random.uniform(0.4, 0.8)
                small = cv2.resize(rimg, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                canvas = np.empty_like(rimg)
                canvas[:] = np.median(rimg.reshape(-1, 3), axis=0).astype(np.uint8)
                canvas = cv2.add(canvas, np.random.randint(0, 12, canvas.shape, dtype=np.uint8))
                oy = random.randint(0, rimg.shape[0] - small.shape[0])
                ox = random.randint(0, rimg.shape[1] - small.shape[1])
                canvas[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
                rimg = canvas
                rbox = (rbox[0] * sc + ox, rbox[1] * sc + oy, rbox[2] * sc + ox, rbox[3] * sc + oy)
            save_sample(root, "train", f"th_{stem}_{j}", maybe_gray(rimg), [(THERMO, rbox)])
            nt += 1
    stats["train"]["thermometer"] = nt
    stats["val"]["thermometer"] = len(val_names)
    print(f"  thermometer samples: {nt} train / {len(val_names)} val")

    # --- composites: BPM pasted into thermometer scenes ------------------------
    train_names = [n for n in names if n not in val_names]
    for i in range(n_composite):
        split = "val" if i < n_composite // 10 else "train"
        name = random.choice(sorted(val_names) if split == "val" else train_names)
        scene, tbox = therm[name]
        scene = scene.copy()
        H, W = scene.shape[:2]
        img_path, lab_path = random.choice(bpm_train)
        bpm = cv2.imread(img_path)
        labels = read_roboflow_labels(lab_path, bpm.shape[1], bpm.shape[0])
        if len(labels) < 3:
            continue
        lbox = lcd_box_from_labels(labels, bpm.shape[1], bpm.shape[0])
        s = random.uniform(0.12, 0.45) * min(H, W) / bpm.shape[0]
        bpm = cv2.resize(bpm, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        bh, bw = bpm.shape[:2]
        objects = []
        for _ in range(30):                      # find a spot not covering the thermometer
            px, py = random.randint(0, W - bw), random.randint(0, H - bh)
            if px > tbox[2] or px + bw < tbox[0] or py > tbox[3] or py + bh < tbox[1]:
                break
        else:
            continue
        # soft-edged paste so the square border is less of a cue
        mask = np.zeros((bh, bw), np.float32)
        cv2.rectangle(mask, (4, 4), (bw - 5, bh - 5), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)[..., None]
        roi = scene[py:py + bh, px:px + bw].astype(np.float32)
        scene[py:py + bh, px:px + bw] = (mask * bpm + (1 - mask) * roi).astype(np.uint8)
        objects.append((LCD, (lbox[0] * s + px, lbox[1] * s + py, lbox[2] * s + px, lbox[3] * s + py)))
        objects.append((THERMO, tbox))
        save_sample(root, split, f"comp_{i:04d}", maybe_gray(scene, 0.2), objects)
    print(f"  composite scenes: {n_composite}")

    # --- negatives: crops of the scenes that do not contain the thermometer ----
    k = 0
    for i in range(n_neg * 5):
        if k >= n_neg:
            break
        name = random.choice(train_names)
        scene, (tx0, ty0, tx1, ty1) = therm[name]
        H, W = scene.shape[:2]
        cw, ch = random.randint(W // 3, W // 2), random.randint(H // 3, H // 2)
        x, y = random.randint(0, W - cw), random.randint(0, H - ch)
        if x > tx1 or x + cw < tx0 or y > ty1 or y + ch < ty0:
            save_sample(root, "train", f"neg_{k:04d}", maybe_gray(scene[y:y + ch, x:x + cw]), [])
            k += 1
    stats["train"]["negatives"] = k
    print(f"  negative crops: {k}")

    yaml_path = os.path.join(root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {root}\ntrain: train/images\nval: val/images\n")
        f.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(DETECTOR_CLASSES)))
    json.dump(stats, open(os.path.join(root, "stats.json"), "w"), indent=1)
    return yaml_path, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roboflow", default=DEFAULT_ROBOFLOW)
    ap.add_argument("--thermo", default=DEFAULT_THERMO)
    ap.add_argument("--work", default=DEFAULT_WORK)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="e.g. 0 for the first GPU, cpu for CPU")
    ap.add_argument("--skip-build", action="store_true", help="reuse an existing <work>/detector dataset")
    args = ap.parse_args()
    seed_everything(0)

    yaml_path = os.path.join(args.work, "detector", "data.yaml")
    if not args.skip_build or not os.path.exists(yaml_path):
        print("[1/2] Building detector dataset")
        yaml_path, stats = build_dataset(args.roboflow, args.thermo, args.work)

    print("[2/2] Fine-tuning YOLOv8n (COCO pre-trained)")
    from ultralytics import YOLO, settings
    settings.update({"wandb": False})          # no experiment-tracking login needed
    model = YOLO("yolov8n.pt")
    run_dir = os.path.join(args.work, "runs")
    model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                project=run_dir, name="detector", exist_ok=True, seed=0, patience=25,
                degrees=10.0, scale=0.6, translate=0.15, fliplr=0.5, mosaic=1.0, hsv_s=0.6, plots=True)
    best = os.path.join(run_dir, "detector", "weights", "best.pt")
    os.makedirs(DATA_DIR, exist_ok=True)
    shutil.copy(best, DETECTOR_WEIGHTS)
    # keep training evidence for the report
    out_docs = os.path.join(DOCS_DIR, "task1_detector_training")
    os.makedirs(out_docs, exist_ok=True)
    for f in glob.glob(os.path.join(run_dir, "detector", "*.png")) + glob.glob(os.path.join(run_dir, "detector", "*.csv")) \
            + glob.glob(os.path.join(run_dir, "detector", "*.yaml")):
        shutil.copy(f, out_docs)
    print(f"Saved detector weights to {DETECTOR_WEIGHTS}; training curves in {out_docs}")


if __name__ == "__main__":
    main()
