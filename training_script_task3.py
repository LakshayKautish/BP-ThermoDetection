
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
# TASK 3 training:
#   (a) digit CNN (common.build_digit_cnn) for seven-segment digits 0-9,
#       trained on real digit crops from the de-augmented Roboflow dataset
#       (prepare_data.py) plus synthetic seven-segment digits. Reports
#       per-class accuracy and a confusion matrix on the held-out test split
#       and on the assignment validation digits.
#   (b) printed-number classifier for the thermometer scale (HOG + linear SVM):
#       numbers 50, 40, ..., -20 are cut automatically from our rectified
#       thermometer photos (8 numbers top-to-bottom on the Celsius side).
#
# Usage: python training_script_task3.py [--epochs 15] [--skip-digits] [--skip-thermo]

import os
import glob
import json
import random
import argparse

import cv2
import numpy as np

from common import (BASE_DIR, DATA_DIR, DIGIT_CLS_WEIGHTS, THERMO_NUM_WEIGHTS, preprocess_digit)
from training_utils import (DEFAULT_WORK, DEFAULT_THERMO, DOCS_DIR, NUMBER_CLASS, ROBOFLOW_NAMES, render_sevenseg,
                            augment_crop, jitter_box, train_digit_cnn, predict_digit_cnn, load_thermo_boxes,
                            seed_everything)


# ---------------------------------------------------------------------------
# (a) digit CNN
# ---------------------------------------------------------------------------

def load_digit_regions(work, split, margin=0.2):
    """Return [(region, inner_box, label)] where region is the digit plus a
    margin (so random jitter can enlarge the box) and inner_box the digit
    box inside the region."""
    out = []
    for img_path in sorted(glob.glob(os.path.join(work, "bpm_upright", split, "images", "*.jpg"))):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        for line in open(img_path.replace("images", "labels")[:-4] + ".txt"):
            c, x0, y0, x1, y1 = line.split()
            c = int(c)
            if c == NUMBER_CLASS:
                continue
            x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))
            if x1 - x0 < 3 or y1 - y0 < 6:
                continue
            mx, my = margin * (x1 - x0) + 2, margin * (y1 - y0) + 2
            rx0, ry0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
            rx1, ry1 = int(min(w, x1 + mx)), int(min(h, y1 + my))
            out.append((img[ry0:ry1, rx0:rx1], (x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0), int(ROBOFLOW_NAMES[c])))
    return out


def tight(region, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    return region[max(0, y0):y1, max(0, x0):x1]


def make_augment():
    def aug(sample):
        region, box = sample
        h, w = region.shape[:2]
        x0, y0, x1, y1 = jitter_box(box, w, h, amount=0.12)
        crop = region[y0:y1, x0:x1]
        if crop.size == 0:
            crop = region
        return augment_crop(crop)
    return aug


def confusion(y_true, y_pred, n=10):
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_confusion_png(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def train_digits(work, epochs, out_docs):
    import torch
    print("[a] Loading digit crops from the de-augmented dataset")
    train = load_digit_regions(work, "train")
    val = load_digit_regions(work, "valid")
    test = load_digit_regions(work, "test")
    print(f"    real digits: train {len(train)}, val {len(val)}, test {len(test)}")
    train_x = [(r, b) for r, b, _ in train]
    train_y = [y for _, _, y in train]
    # synthetic digits balance rare classes and add style variety
    for d in range(10):
        for _ in range(1500):
            s = render_sevenseg(d)
            train_x.append((s, (0, 0, s.shape[1], s.shape[0])))
            train_y.append(d)
    val_x = [tight(r, b) for r, b, _ in val]
    val_y = [y for _, _, y in val]
    print(f"    training CNN on {len(train_x)} crops (real + synthetic) for {epochs} epochs")
    model, hist = train_digit_cnn(train_x, train_y, val_x, val_y, epochs=epochs, lr=2e-3,
                                  augment_fn=make_augment())
    os.makedirs(DATA_DIR, exist_ok=True)
    torch.save(model.state_dict(), DIGIT_CLS_WEIGHTS)
    print(f"    saved {DIGIT_CLS_WEIGHTS}")

    report = {"history": hist}
    # held-out Roboflow test split
    tx = torch.from_numpy(np.stack([preprocess_digit(tight(r, b)) for r, b, _ in test])[:, None])
    ty = np.array([y for _, _, y in test])
    pred = predict_digit_cnn(model, tx)
    cm = confusion(ty, pred)
    report["test_accuracy"] = float((pred == ty).mean())
    report["test_per_class_accuracy"] = {str(d): float(cm[d, d] / max(1, cm[d].sum())) for d in range(10)}
    report["test_confusion"] = cm.tolist()
    save_confusion_png(cm, f"Digit CNN - Roboflow test split (acc {report['test_accuracy']:.3f})",
                       os.path.join(out_docs, "digit_confusion_test.png"))
    print(f"    held-out test accuracy: {report['test_accuracy']:.4f}")

    # assignment validation digits (task 3 input + expected output)
    vdir = os.path.join(BASE_DIR, "validation", "task3")
    edir = os.path.join(BASE_DIR, "expected_validation_output", "output", "task3")
    vx, vy = [], []
    for f in sorted(glob.glob(os.path.join(vdir, "lcd*", "d*.png"))):
        exp = os.path.join(edir, os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-4] + ".txt")
        if os.path.exists(exp):
            vx.append(preprocess_digit(cv2.imread(f)))
            vy.append(int(open(exp).read().strip()))
    if vx:
        p = predict_digit_cnn(model, torch.from_numpy(np.stack(vx)[:, None]))
        report["assignment_validation_accuracy"] = float((p == np.array(vy)).mean())
        report["assignment_validation_predictions"] = [[int(a), int(b)] for a, b in zip(vy, p)]
        print(f"    assignment validation digits: {int((p == np.array(vy)).sum())}/{len(vy)} correct")
    json.dump(report, open(os.path.join(out_docs, "digit_classifier_report.json"), "w"), indent=1)
    _plot_history(hist, os.path.join(out_docs, "digit_classifier_curves.png"))


def _plot_history(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(ep, [h["train_loss"] for h in hist]); ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(ep, [h["train_acc"] for h in hist], label="train")
    if "val_acc" in hist[0]:
        ax[1].plot(ep, [h["val_acc"] for h in hist], label="val")
    ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# (b) thermometer printed-number classifier
# ---------------------------------------------------------------------------

def collect_thermo_numbers(thermo_dir, out_docs):
    from thermo import rectify_thermometer, find_tube_x, number_blobs, pad_box, celsius_number_run, THERMO_LABELS
    boxes = load_thermo_boxes()
    samples = []                       # (photo name, crop, value)
    used = []
    for name, b in sorted(boxes.items()):
        img = cv2.imread(os.path.join(thermo_dir, name))
        H, W = img.shape[:2]
        try:
            rect = rectify_thermometer(img, (b[0] * W, b[1] * H, b[2] * W, b[3] * H), verbose=False)
        except Exception as e:                   # e.g. tiny, distant thermometers
            print(f"    {name}: rectification failed ({e}) -> skipped")
            continue
        tube_x, _ = find_tube_x(rect)
        groups, _ = number_blobs(rect, tube_x, "left")
        groups = sorted(groups, key=lambda g: (g[1] + g[3]) / 2)
        n = len(THERMO_LABELS)
        if len(groups) < n:
            print(f"    {name}: only {len(groups)} number blobs -> skipped")
            continue
        # the most evenly spaced run of 8 blobs whose narrowest member (the
        # lone "0") sits on the 0 label; this stops the evenly spaced "degC"
        # symbol from shifting every label by one
        best = celsius_number_run(groups, max_cv=0.12)
        if best is None:
            print(f"    {name}: no evenly spaced run of numbers with the 0 in place -> skipped")
            continue
        groups = best[1]
        for g, v in zip(groups, THERMO_LABELS):
            x0, y0, x1, y1 = pad_box(g, rect.shape)
            samples.append((name, rect[y0:y1, x0:x1].copy(), v))
        used.append(name)
    print(f"    collected {len(samples)} number crops from {len(used)} photos")
    # contact sheet for checking the automatic labels
    if samples:
        tiles = []
        for _, c, v in samples[:80]:
            t = cv2.resize(c, (96, 48))
            cv2.putText(t, str(v), (2, 12), 0, 0.4, (0, 0, 255), 1)
            tiles.append(t)
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(os.path.join(out_docs, "thermo_number_samples.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
    return samples


def augment_number(crop):
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-4, 4), random.uniform(0.9, 1.1))
    M[:, 2] += [random.uniform(-0.05, 0.05) * w, random.uniform(-0.08, 0.08) * h]
    out = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    s = random.uniform(0.3, 1.0)                      # simulate lower resolution
    out = cv2.resize(cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (w, h))
    out = np.clip(out.astype(np.float32) * random.uniform(0.75, 1.2) + random.uniform(-25, 25), 0, 255).astype(np.uint8)
    return out


def train_thermo_numbers(thermo_dir, out_docs, n_aug=40):
    from sklearn.svm import LinearSVC
    from thermo import number_hog
    print("[b] Building thermometer number dataset")
    samples = collect_thermo_numbers(thermo_dir, out_docs)
    if not samples:
        print("    no samples -> skipped")
        return
    photos = sorted({s[0] for s in samples})

    def features(subset, aug):
        X, y = [], []
        for _, c, v in subset:
            X.append(number_hog(c)); y.append(v)
            for _ in range(aug):
                X.append(number_hog(augment_number(c))); y.append(v)
        return np.array(X, np.float32), np.array(y)

    # grouped 5-fold estimate of accuracy: all crops of one photo stay in the
    # same fold, so the score measures generalisation to unseen photos
    folds = [set(photos[i::5]) for i in range(min(5, len(photos)))]
    correct, total = 0, 0
    for fold in folds:
        tr = [s for s in samples if s[0] not in fold]
        te = [s for s in samples if s[0] in fold]
        X, y = features(tr, 10)
        mean, std = X.mean(0), X.std(0) + 1e-6
        clf = LinearSVC(C=0.5, max_iter=5000).fit((X - mean) / std, y)
        Xt, yt = features(te, 0)
        correct += int((clf.predict((Xt - mean) / std) == yt).sum())
        total += len(yt)
    loo = correct / max(1, total)
    print(f"    photo-grouped {len(folds)}-fold accuracy: {correct}/{total} = {loo:.3f}")

    X, y = features(samples, n_aug)
    mean, std = X.mean(0), X.std(0) + 1e-6
    clf = LinearSVC(C=0.5, max_iter=10000).fit((X - mean) / std, y)
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(THERMO_NUM_WEIGHTS, W=clf.coef_.astype(np.float32), b=clf.intercept_.astype(np.float32),
             classes=clf.classes_, mean=mean, std=std)
    json.dump({"photos_used": photos, "n_crops": len(samples), "photo_grouped_cv_accuracy": loo},
              open(os.path.join(out_docs, "thermo_number_report.json"), "w"), indent=1)
    print(f"    saved {THERMO_NUM_WEIGHTS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=DEFAULT_WORK)
    ap.add_argument("--thermo", default=DEFAULT_THERMO)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--skip-digits", action="store_true")
    ap.add_argument("--skip-thermo", action="store_true")
    args = ap.parse_args()
    seed_everything(0)
    out_docs = os.path.join(DOCS_DIR, "task3_training")
    os.makedirs(out_docs, exist_ok=True)
    if not args.skip_thermo:
        train_thermo_numbers(args.thermo, out_docs)
    if not args.skip_digits:
        if not os.path.isdir(os.path.join(args.work, "bpm_upright")):
            raise SystemExit("Run prepare_data.py first (creates <work>/bpm_upright).")
        train_digits(args.work, args.epochs, out_docs)


if __name__ == "__main__":
    main()
