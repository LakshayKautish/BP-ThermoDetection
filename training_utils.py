
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
# Helpers used ONLY by the training scripts (not needed at inference time):
#   * default locations of the raw training data and the work directory
#   * Roboflow YOLO-OBB label reading
#   * a synthetic seven-segment digit renderer
#   * a small torch training loop for the digit CNN

import os
import glob
import json
import random

import cv2
import numpy as np

from common import BASE_DIR, build_digit_cnn, preprocess_digit, DIGIT_SIZE

ASSIGNMENT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_ROBOFLOW = os.path.join(ASSIGNMENT_ROOT, "Blood-Pressure-monitor-digit-reader.v3-yolov8-with-640x640.yolov8-obb")
DEFAULT_THERMO = os.path.join(ASSIGNMENT_ROOT, "Thermometer")
DEFAULT_WORK = os.path.join(ASSIGNMENT_ROOT, "work")          # generated datasets / training runs (NOT submitted)
THERMO_BOXES = os.path.join(BASE_DIR, "docs", "thermometer_boxes.json")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Roboflow class index -> name. Index 2 ("10") is a whole-number box (e.g. "135"),
# all others are single digits.
ROBOFLOW_NAMES = ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9']
NUMBER_CLASS = 2


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Roboflow dataset
# ---------------------------------------------------------------------------

def read_roboflow_labels(label_path, w, h):
    """Return list of (class_idx, x0, y0, x1, y1) in pixels. The export is
    YOLO-OBB (4 corners) but all boxes are axis aligned."""
    out = []
    if not os.path.exists(label_path):
        return out
    for line in open(label_path):
        p = line.split()
        if len(p) < 9:
            continue
        xy = np.array(p[1:9], float).reshape(4, 2) * [w, h]
        x0, y0 = xy.min(0)
        x1, y1 = xy.max(0)
        out.append((int(p[0]), float(x0), float(y0), float(x1), float(y1)))
    return out


def roboflow_items(root, split):
    """Yield (image_path, label_path) for one split (train/valid/test)."""
    for img in sorted(glob.glob(os.path.join(root, split, "images", "*.jpg"))):
        lab = os.path.join(root, split, "labels", os.path.splitext(os.path.basename(img))[0] + ".txt")
        yield img, lab


def digit_label(cls_idx):
    """Roboflow class index -> digit 0-9, or None for the number class."""
    if cls_idx == NUMBER_CLASS:
        return None
    return int(ROBOFLOW_NAMES[cls_idx])


def load_thermo_boxes():
    data = json.load(open(THERMO_BOXES))
    return {k: v["box"] for k, v in data["boxes"].items()}


# ---------------------------------------------------------------------------
# The 8 dihedral transforms (used to undo Roboflow's flip/rotate augmentation)
# ---------------------------------------------------------------------------

def dihedral_image(img, k):
    """k in 0..7: rotate by 90*(k%4) degrees clockwise, then mirror horizontally if k>=4."""
    out = np.rot90(img, -(k % 4))
    if k >= 4:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def dihedral_box(box, k, w, h):
    """Transform an axis-aligned box (x0,y0,x1,y1) of a w x h image like dihedral_image."""
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0], [x1, y1]], float)
    W, H = w, h
    for _ in range(k % 4):                       # 90 deg clockwise: (x,y) -> (H-y, x)
        pts = np.stack([H - pts[:, 1], pts[:, 0]], 1)
        W, H = H, W
    if k >= 4:
        pts[:, 0] = W - pts[:, 0]
    (a, b), (c, d) = pts
    return (min(a, c), min(b, d), max(a, c), max(b, d)), W, H


# ---------------------------------------------------------------------------
# Synthetic seven-segment digits
# ---------------------------------------------------------------------------

#   aaa
#  f   b
#   ggg
#  e   c
#   ddd
SEGMENTS = {0: "abcdef", 1: "bc", 2: "abged", 3: "abgcd", 4: "fgbc",
            5: "afgcd", 6: "afgedc", 7: "abc", 8: "abcdefg", 9: "abcdfg"}
VARIANTS = {6: ["fgedc"], 7: ["abcf"], 9: ["abcfg"]}   # alternative glyph styles


def _segment_polygons(w, h, t, gap):
    """Hexagonal segment polygons for a digit cell of size w x h, thickness t."""
    ht = t / 2.0

    def hseg(y, x0, x1):
        return np.array([[x0 + gap, y], [x0 + gap + ht, y - ht], [x1 - gap - ht, y - ht],
                         [x1 - gap, y], [x1 - gap - ht, y + ht], [x0 + gap + ht, y + ht]])

    def vseg(x, y0, y1):
        return np.array([[x, y0 + gap], [x + ht, y0 + gap + ht], [x + ht, y1 - gap - ht],
                         [x, y1 - gap], [x - ht, y1 - gap - ht], [x - ht, y0 + gap + ht]])

    l, r, top, mid, bot = ht, w - ht, ht, h / 2.0, h - ht
    return {"a": hseg(top, l, r), "g": hseg(mid, l, r), "d": hseg(bot, l, r),
            "f": vseg(l, top, mid), "b": vseg(r, top, mid),
            "e": vseg(l, mid, bot), "c": vseg(r, mid, bot)}


def render_sevenseg(digit, rng=random):
    """Render one synthetic LCD seven-segment digit crop (greyscale uint8)."""
    segs = SEGMENTS[digit]
    if digit in VARIANTS and rng.random() < 0.3:
        segs = rng.choice(VARIANTS[digit])
    S = 4                                            # supersampling factor
    w = rng.uniform(55, 80) * S
    h = rng.uniform(1.6, 2.1) * w
    t = rng.uniform(0.13, 0.24) * w
    gap = rng.uniform(0.01, 0.05) * w
    slant = rng.uniform(-0.02, 0.15)                 # LCD digits lean right
    polys = _segment_polygons(w, h, t, gap)

    pad = 0.6 * w
    cw, ch = int(w + 2 * pad), int(h + 2 * pad)
    bg = rng.uniform(110, 225)
    fg = bg - rng.uniform(55, min(bg, 170))
    ghost = bg - rng.uniform(0, 18)                  # faintly visible unlit segments
    canvas = np.full((ch, cw), bg, np.float32)
    # smooth illumination gradient
    gx, gy = np.meshgrid(np.linspace(-1, 1, cw), np.linspace(-1, 1, ch))
    canvas += rng.uniform(-20, 20) * gx + rng.uniform(-20, 20) * gy

    def place(poly):
        p = poly.copy()
        p[:, 0] += pad - slant * (p[:, 1] - h / 2)
        p[:, 1] += pad
        return p.astype(np.int32)

    for name, poly in polys.items():
        if name not in segs and rng.random() < 0.8:
            cv2.fillPoly(canvas, [place(poly)], ghost)
    lit = [place(polys[s]) for s in segs]
    for p in lit:
        cv2.fillPoly(canvas, [p], fg + rng.uniform(-8, 8))

    # crop: either tight around the lit segments (like a narrow '1') or the whole cell
    pts = np.concatenate(lit)
    if rng.random() < 0.6:
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
    else:
        allp = np.concatenate([place(p) for p in polys.values()])
        x0, y0 = allp.min(0)
        x1, y1 = allp.max(0)
    mx, my = rng.uniform(-0.03, 0.18) * w, rng.uniform(-0.03, 0.1) * h
    x0, x1 = int(max(0, x0 - mx * rng.uniform(0.5, 1.5))), int(min(cw, x1 + mx * rng.uniform(0.5, 1.5)))
    y0, y1 = int(max(0, y0 - my)), int(min(ch, y1 + my))
    crop = canvas[y0:y1, x0:x1]

    # camera effects
    crop = cv2.GaussianBlur(crop, (0, 0), rng.uniform(0.5, 3.0) * S / 2)
    out_h = rng.randint(24, 120)
    crop = cv2.resize(crop, (max(4, int(crop.shape[1] * out_h / crop.shape[0])), out_h), interpolation=cv2.INTER_AREA)
    crop += np.random.normal(0, rng.uniform(0, 8), crop.shape)
    crop = np.clip(crop, 0, 255).astype(np.uint8)
    if rng.random() < 0.15:                           # backlit / inverted LCDs exist in the dataset
        crop = 255 - crop
    return crop


def augment_crop(img, rng=random):
    """Light photometric/geometric augmentation for real digit crops."""
    h, w = img.shape[:2]
    # random rotation / shear
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-6, 6), rng.uniform(0.92, 1.08))
    M[0, 1] += rng.uniform(-0.1, 0.1)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    img = img.astype(np.float32)
    img = img * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.5))
    img += np.random.normal(0, rng.uniform(0, 6), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.1:                            # both LCD polarities
        img = 255 - img
    return img


def jitter_box(box, w, h, rng=random, amount=0.08):
    """Randomly grow/shrink a box a little so the classifier tolerates
    segmentation crops that are slightly loose or tight."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    x0 += rng.uniform(-amount, amount * 0.5) * bw
    x1 += rng.uniform(-amount * 0.5, amount) * bw
    y0 += rng.uniform(-amount, amount * 0.5) * bh
    y1 += rng.uniform(-amount * 0.5, amount) * bh
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Torch training loop for the digit CNN
# ---------------------------------------------------------------------------

def train_digit_cnn(train_x, train_y, val_x, val_y, epochs=10, lr=1e-3, batch_size=128,
                    augment_fn=None, log=print):
    """train_x: list of uint8 crops; returns (model, history). If augment_fn is
    given it is applied to every training crop on every epoch."""
    import torch
    import torch.nn as nn

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = build_digit_cnn().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(train_x) + batch_size - 1) // batch_size))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    vx = torch.from_numpy(np.stack([preprocess_digit(im) for im in val_x])[:, None]) if len(val_x) else None
    vy = torch.tensor(val_y) if len(val_y) else None
    history = []
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(train_x))
        tot, correct, loss_sum = 0, 0, 0.0
        for b in range(0, len(idx), batch_size):
            bi = idx[b:b + batch_size]
            ims = [augment_fn(train_x[i]) if augment_fn else train_x[i] for i in bi]
            x = torch.from_numpy(np.stack([preprocess_digit(im) for im in ims])[:, None]).to(device)
            y = torch.tensor([train_y[i] for i in bi]).to(device)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(bi)
            correct += (out.argmax(1) == y).sum().item()
            tot += len(bi)
        rec = {"epoch": ep + 1, "train_loss": loss_sum / tot, "train_acc": correct / tot}
        if vx is not None:
            pred = predict_digit_cnn(model, vx)
            rec["val_acc"] = float((pred == vy.numpy()).mean())
        history.append(rec)
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
    return model.cpu().eval(), history


def predict_digit_cnn(model, x, batch_size=512, return_prob=False):
    """x: float tensor N x 1 x 48 x 48 (already preprocessed)."""
    import torch
    device = next(model.parameters()).device
    model.eval()
    outs = []
    with torch.no_grad():
        for b in range(0, len(x), batch_size):
            outs.append(torch.softmax(model(x[b:b + batch_size].to(device)), 1).cpu())
    prob = torch.cat(outs).numpy() if outs else np.zeros((0, 10))
    return prob if return_prob else prob.argmax(1)
