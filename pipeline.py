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
# Last Modified: 2026-10-02
#
# The detection + extraction chain shared by task 1 (which saves the
# corrected device image) and task 4 (which runs everything in memory).

import numpy as np

from common import DETECTOR_WEIGHTS, DETECTOR_CLASSES, load_yolo
from bpm import rectify_lcd, segment_digits, looks_like_bpm
from thermo import rectify_thermometer

DETECT_CONF = 0.4


def detect_devices(img, conf=DETECT_CONF, verbose=True):
    """Return [(class name, (x0,y0,x1,y1), conf)], best first, keeping the
    single most confident box of each class."""
    model = load_yolo(DETECTOR_WEIGHTS)
    res = model.predict(img, imgsz=640, conf=conf, verbose=False)[0]
    best = {}
    for (x0, y0, x1, y1), c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                      res.boxes.cls.cpu().numpy().astype(int)):
        name = DETECTOR_CLASSES[k]
        if name not in best or c > best[name][2]:
            best[name] = (name, (float(x0), float(y0), float(x1), float(y1)), float(c))
    dets = sorted(best.values(), key=lambda d: -d[2])
    if verbose:
        if dets:
            for name, box, c in dets:
                print(f"  detector: {name} conf={c:.2f} box={tuple(int(v) for v in box)}")
        else:
            print("  detector: no device found")
    return dets


def extract_device(img, verbose=True):
    """Detect and extract the device in a photo.

    Returns ("lcd", corrected LCD image, (digit crops, rows)),
            ("thermo", upright thermometer image, None), or
            (None, None, None) for a negative image.
    Detections are tried in order of confidence; an LCD is only accepted if
    the digit layout looks like a blood-pressure reading, which rejects
    other displays such as digital clocks."""
    for name, box, c in detect_devices(img, verbose=verbose):
        try:
            if name == "lcd":
                lcd = rectify_lcd(img, box, verbose)
                crops, rows = segment_digits(lcd, verbose)
                if looks_like_bpm(rows):
                    return "lcd", lcd, (crops, rows)
                if verbose:
                    print(f"  rejected: digit rows {[len(r) for r in rows]} are not a BPM reading")
            else:
                return "thermo", rectify_thermometer(img, box, verbose), None
        except Exception as e:                       # keep going with the next candidate
            if verbose:
                print(f"  {name} extraction failed: {e}")
    return None, None, None


def rows_to_values(rows, labels):
    """Join per-digit labels (in reading order) back into one integer per row."""
    values, i = [], 0
    for r in rows:
        values.append(int("".join(str(d) for d in labels[i:i + len(r)])))
        i += len(r)
    return values
