
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
# Shared helpers used by every task: paths, image/text I/O, model loading
# (cached so task4 loads each model only once) and the digit CNN definition.

import os
import re
import glob
import shutil

import cv2
import numpy as np

# All paths are resolved relative to this file so the programs work no matter
# which directory the marker launches them from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

DETECTOR_WEIGHTS = os.path.join(DATA_DIR, "device_detector.pt")    # task 1 (YOLOv8n: lcd / thermometer)
DIGIT_DET_WEIGHTS = os.path.join(DATA_DIR, "digit_detector.pt")    # task 2 (YOLOv8n: digit)
DIGIT_CLS_WEIGHTS = os.path.join(DATA_DIR, "digit_classifier.pt")  # task 3 (CNN: 0-9)
THERMO_NUM_WEIGHTS = os.path.join(DATA_DIR, "thermo_numbers.npz")  # task 3 (HOG + linear SVM)

DETECTOR_CLASSES = ["lcd", "thermometer"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def task_output_dir(task_name, clean=True):
    """Return output/<task_name>, creating it. Stale results are removed so
    the folder only ever holds the results of the latest run."""
    path = os.path.join(OUTPUT_DIR, task_name)
    if clean and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


def save_output(output_path, content, output_type='txt'):
    """Framework-style saver for text or OpenCV images."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")


def natural_key(path):
    """Sort key so img2 < img10 and d2 < d10."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_images(folder, patterns=("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")):
    files = set()
    for p in patterns:
        files.update(glob.glob(os.path.join(folder, p)))
    return sorted(files, key=natural_key)


def image_number(path):
    """'img3.jpg' -> '3', 'lcd12.png' -> '12'. Any other name keeps its whole
    stem ('20260815_103024.jpg' -> '20260815_103024') so that photos sharing
    a date prefix do not overwrite each other's output."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.fullmatch(r"(?:img|lcd|thermo)[_-]?(\d+)", stem, re.IGNORECASE)
    return m.group(1) if m else stem


def read_image(path):
    """cv2.imread applies the EXIF orientation flag for JPEGs, so phone photos
    come out the right way up."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img


# ---------------------------------------------------------------------------
# Models (lazy-loaded, cached)
# ---------------------------------------------------------------------------

_MODEL_CACHE = {}


def _device():
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_yolo(weights):
    if weights not in _MODEL_CACHE:
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Model weights not found: {weights}. Run the training scripts first.")
        from ultralytics import YOLO
        print(f"Loading YOLO model: {os.path.relpath(weights, BASE_DIR)}")
        _MODEL_CACHE[weights] = YOLO(weights)
    return _MODEL_CACHE[weights]


def build_digit_cnn(num_classes=10):
    """Small VGG-style CNN for 48x48 greyscale seven-segment digit crops."""
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

    return nn.Sequential(
        block(1, 32), block(32, 64), block(64, 128),          # 48 -> 24 -> 12 -> 6
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.3), nn.Linear(128, num_classes))


DIGIT_SIZE = 48


def preprocess_digit(img):
    """Greyscale -> contrast-normalise -> letterbox to a 48x48 square while
    keeping the aspect ratio (so a narrow '1' stays narrow) -> float array in
    [-1, 1]. The same function is used in training and inference."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip((img - lo) / max(hi - lo, 1e-3), 0, 1)
    h, w = img.shape
    s = DIGIT_SIZE / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    # pad with the median (background) value
    bg = float(np.median(img))
    out = np.full((DIGIT_SIZE, DIGIT_SIZE), bg, np.float32)
    y0, x0 = (DIGIT_SIZE - nh) // 2, (DIGIT_SIZE - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = img
    # NOTE: polarity (dark-on-light vs light-on-dark) is not normalised here;
    # the CNN sees both during training instead (median-based flipping fails
    # on bold digits such as 8/9 whose segments fill most of the crop).
    return (out - 0.5) / 0.5


def load_digit_classifier():
    if "digit_cls" not in _MODEL_CACHE:
        import torch
        if not os.path.exists(DIGIT_CLS_WEIGHTS):
            raise FileNotFoundError(f"Model weights not found: {DIGIT_CLS_WEIGHTS}. Run training_script_task3.py first.")
        print(f"Loading digit classifier: {os.path.relpath(DIGIT_CLS_WEIGHTS, BASE_DIR)}")
        model = build_digit_cnn()
        model.load_state_dict(torch.load(DIGIT_CLS_WEIGHTS, map_location="cpu"))
        model.eval()
        _MODEL_CACHE["digit_cls"] = model
    return _MODEL_CACHE["digit_cls"]


def classify_digits(images):
    """Return (labels, confidences) for a list of BGR/grey digit crops."""
    import torch
    if not images:
        return [], []
    model = load_digit_classifier()
    x = torch.from_numpy(np.stack([preprocess_digit(im) for im in images])[:, None])
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1).numpy()
    return [int(p.argmax()) for p in prob], [float(p.max()) for p in prob]
