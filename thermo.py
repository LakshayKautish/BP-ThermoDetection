
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
# Last Modified: 2026-10-05
#
# Wall-thermometer processing:
#   rectify_thermometer()  - task 1: rotate the detected thermometer upright (bulb down)
#   segment_thermometer()  - task 2: find the fluid top + scale ticks, crop around it
#   read_thermometer()     - task 3: compute the temperature from the crop
#
# Geometry conventions after rectification: the thermometer is vertical, the
# bulb is at the bottom, the Celsius scale is on the LEFT of the glass tube.

import os

import cv2
import numpy as np

from common import THERMO_NUM_WEIGHTS

THERMO_LABELS = [50, 40, 30, 20, 10, 0, -10, -20]   # printed Celsius numbers, top to bottom


# ---------------------------------------------------------------------------
# Low-level cues
# ---------------------------------------------------------------------------

def redness(img):
    """Per-pixel 'redness' R - max(G, B) as float32 (alcohol column is red)."""
    b, g, r = cv2.split(img.astype(np.float32))
    return r - np.maximum(g, b)


def dominant_axis_angle(gray):
    """Angle (degrees, OpenCV rotation convention) of the dominant long straight
    edges in the image: the thermometer body edges and the glass tube."""
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    min_len = int(0.25 * max(gray.shape))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60, minLineLength=min_len, maxLineGap=10)
    if lines is None:
        return None
    hist = np.zeros(180)
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = np.hypot(x2 - x1, y2 - y1)
        hist[int(ang) % 180] += length
    # circular smoothing, then take the peak
    sm = np.convolve(np.concatenate([hist[-3:], hist, hist[:3]]), np.ones(7), "same")[3:-3]
    peak = int(np.argmax(sm))
    # refine with a weighted mean around the peak
    idx = (np.arange(peak - 3, peak + 4)) % 180
    w = hist[idx]
    if w.sum() == 0:
        return float(peak)
    unwrapped = peak + ((idx - peak + 90) % 180 - 90)
    return float((unwrapped * w).sum() / w.sum())


def rotate_bound(img, angle):
    """Rotate by `angle` degrees (counter-clockwise) without clipping corners.
    Returns the rotated image and the 2x3 affine matrix used."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), M


def find_tube_x(img, x_lo=None, x_hi=None):
    """x position of the vertical glass tube: the narrow column whose redness
    stands out from its neighbourhood (a 1-D top-hat on the column profile).
    The search is limited to [x_lo, x_hi) (default: the central 40% of the
    width, because the tube runs along the middle of the thermometer body)."""
    red = redness(img)
    h, w = red.shape
    x_lo = int(0.3 * w) if x_lo is None else max(0, int(x_lo))
    x_hi = int(0.7 * w) if x_hi is None else min(w, int(x_hi))
    prof = np.clip(red, 0, None).mean(0)
    k = max(5, (w // 10) | 1)
    background = cv2.morphologyEx(prof.reshape(1, -1).astype(np.float32), cv2.MORPH_OPEN,
                                  np.ones((1, k), np.uint8)).ravel()
    tophat = np.convolve(prof - background, np.ones(3) / 3, "same")
    if x_hi <= x_lo:
        x_lo, x_hi = 0, w
    x = x_lo + int(np.argmax(tophat[x_lo:x_hi]))
    return x, float(tophat[x])


def fluid_profile(img, x):
    """Redness along the tube minus redness of the plastic just beside it.
    Subtracting the neighbours cancels colour casts from the lighting, so the
    profile is ~0 where the tube is empty and clearly positive in the fluid."""
    h, w = img.shape[:2]
    half = max(3, int(0.025 * w))                      # covers the whole glass tube
    red = redness(img)
    # in bright photos the capillary is a red line only 1-2 pixels wide that
    # may drift sideways inside the tube, so a percentile over the window
    # misses it; smooth along the line (vertically) and take the reddest
    # column of each row, which follows the line up to its true top
    line = cv2.blur(red, (3, max(3, h // 150) | 1))
    tube = line[:, max(0, x - half):x + half + 1].max(axis=1)
    # reference: the plastic either side of the tube. The (green/black) tick
    # marks have strongly negative redness, so take a high percentile per row,
    # which picks the bare plastic between the ticks.
    a, b = max(half + 2, int(0.036 * w)), int(0.12 * w)
    side = np.concatenate([red[:, max(0, x - b):max(1, x - a)], red[:, min(w - 1, x + a):min(w, x + b)]], axis=1)
    ref = np.percentile(side, 80, axis=1) if side.size else np.zeros(h, np.float32)
    prof = tube - ref
    k = max(3, h // 200)
    return np.convolve(prof, np.ones(k) / k, "same")


def fluid_extent(img, x):
    """Locate the top of the fluid column (the reading point).

    The threshold is self-calibrating: halfway between the 15th percentile
    (empty tube / plastic) and the 85th percentile (fluid) of the profile.
    The top is the change point y maximising sum_{y' >= y} (p(y') - t), i.e.
    'mostly empty above, mostly full below' - robust to small gaps and
    reflections inside the column. Returns (top, bulb_end, profile)."""
    prof = fluid_profile(img, x)
    lo, hi = np.percentile(prof, 15), np.percentile(prof, 85)
    if hi - lo < 4:                                    # no visible fluid
        return None, None, prof
    t = 0.5 * (lo + hi)
    suffix = np.cumsum((prof - t)[::-1])[::-1]
    top = int(np.argmax(suffix))
    prefix = np.cumsum(prof - t)
    bottom = int(np.argmax(prefix))                    # last row of the fluid (bulb end)
    if bottom <= top:
        bottom = len(prof) - 1
    return refine_fluid_top(prof, top, img.shape[1]), bottom, prof


def refine_fluid_top(prof, top, width, frac=0.2):
    """The fluid gets fainter towards its top (thin capillary), so the global
    halfway threshold lands a few pixels too low, on the ramp below the true
    top. Re-threshold with local levels instead: 'empty' = tube just above the
    coarse top, 'full' = column just below it; then walk up from the coarse
    top while the profile stays above empty + frac * (full - empty). Window
    sizes are fractions of the thermometer width, which is the same in a full
    image and in a crop."""
    h = len(prof)
    k = max(3, int(0.01 * width)) | 1
    ps = np.convolve(prof, np.ones(k) / k, "same")
    ea, eb = max(0, top - int(0.3 * width)), max(0, top - int(0.08 * width))
    empty = np.median(ps[ea:eb]) if eb - ea > 2 else np.percentile(ps, 10)
    full = np.median(ps[top:min(h, top + int(0.2 * width))])
    if full <= empty:
        return top
    t = empty + frac * (full - empty)
    y = top
    while y > 0 and ps[y - 1] > t:
        y -= 1
    # sub-pixel position of the threshold crossing between rows y-1 and y
    # (one pixel is a noticeable fraction of a degree in a small crop)
    if y > 0 and ps[y] > ps[y - 1]:
        return y - float((ps[y] - t) / (ps[y] - ps[y - 1]))
    return y


def fluid_is_below(img, x):
    """True if the fluid (which always fills the bulb) is mainly in the lower
    half, i.e. the thermometer is the right way up."""
    prof = fluid_profile(img, x)
    h = len(prof)
    return prof[h // 2:].mean() >= prof[:h // 2].mean()


# ---------------------------------------------------------------------------
# Task 1: rectification
# ---------------------------------------------------------------------------

def rectify_thermometer(img, box, verbose=True):
    """img: full BGR photo, box: (x0,y0,x1,y1) detector box of the whole
    thermometer. Returns the rotated, upright, tightly cropped thermometer."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    m = 0.12
    cx0, cy0 = int(max(0, x0 - m * bw)), int(max(0, y0 - m * bh))
    cx1, cy1 = int(min(W, x1 + m * bw)), int(min(H, y1 + m * bh))
    crop = img[cy0:cy1, cx0:cx1]

    # 1) orientation of the long axis from Hough lines (on a downscaled copy)
    s = 900.0 / max(crop.shape[:2])
    small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop.copy()
    angle = dominant_axis_angle(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    if angle is None:
        angle = 90.0 if bh >= bw else 0.0
    # rotate so that the axis becomes vertical (90 degrees)
    rot = angle - 90.0
    if verbose:
        print(f"  thermometer axis at {angle:.1f} deg -> rotating by {rot:.1f} deg")
    rimg, M = rotate_bound(crop, rot)

    # region covered by the detector box after rotation
    corners = np.array([[x0 - cx0, y0 - cy0], [x1 - cx0, y0 - cy0], [x1 - cx0, y1 - cy0], [x0 - cx0, y1 - cy0]], float)
    rc = (M[:, :2] @ corners.T).T + M[:, 2]

    # 2) body extent: strong straight vertical edges left/right of the tube,
    #    strongest horizontal edge responses at the two ends.
    gray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    rh, rw = gray.shape
    # rotated box is a diamond for diagonal thermometers; the body lies around its centre line
    bx0, bx1 = int(max(0, rc[:, 0].min())), int(min(rw, rc[:, 0].max()))
    by0, by1 = int(max(0, rc[:, 1].min())), int(min(rh, rc[:, 1].max()))
    # the box centre lies on the thermometer axis; the tube is close to it
    cxr = float(rc[:, 0].mean())
    length_guess = by1 - by0
    tube_x, _ = find_tube_x(rimg[by0:by1], cxr - 0.06 * length_guess, cxr + 0.06 * length_guess)
    # vertical extent: use the central band of rows where the body is surely present
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    mid0, mid1 = int(by0 + 0.3 * length_guess), int(by0 + 0.7 * length_guess)
    colprof = np.median(gx[mid0:mid1], axis=0)       # straight long edges survive the median
    max_half = int(0.16 * length_guess)
    min_half = int(0.035 * length_guess)
    left_rng = colprof[max(0, tube_x - max_half):max(1, tube_x - min_half)]
    right_rng = colprof[tube_x + min_half:tube_x + max_half]
    left = tube_x - max_half + int(np.argmax(left_rng)) if left_rng.size else tube_x - max_half
    left = max(0, left)
    right = tube_x + min_half + int(np.argmax(right_rng)) if right_rng.size else tube_x + max_half
    # horizontal edges along the body (top/bottom rounded ends)
    body_w = right - left
    rowprof = np.median(gy[:, left + body_w // 4:right - body_w // 4], axis=1)
    rowprof = np.convolve(rowprof, np.ones(5) / 5, "same")
    search = int(0.15 * length_guess)
    top_rng = rowprof[max(0, by0 - search):by0 + search]
    bot_rng = rowprof[max(0, by1 - search):min(rh, by1 + search)]
    top = max(0, by0 - search) + int(np.argmax(top_rng)) if top_rng.size else by0
    bottom = max(0, by1 - search) + int(np.argmax(bot_rng)) if bot_rng.size else by1
    if bottom - top < 3 * body_w:                     # implausible -> fall back to box extents
        top, bottom = by0, by1
    pad = int(0.02 * body_w)
    out = rimg[max(0, top - pad):min(rh, bottom + pad), max(0, left - pad):min(rw, right + pad)]

    # 3) bulb must be at the bottom. Primary cue: the printed Celsius scale
    #    layout (see celsius_layout_on_left) holds in exactly one of the two
    #    orientations. Fallback when the numbers are unreadable: the fluid,
    #    which always fills the bulb, should be mainly in the lower half.
    flipped = cv2.rotate(out, cv2.ROTATE_180)
    up, down = celsius_layout_on_left(out), celsius_layout_on_left(flipped)
    if up != down:
        if down:
            out = flipped
        if verbose:
            print(f"  scale layout -> {'rotated 180 deg' if down else 'already upright'}")
    else:
        tx, _ = find_tube_x(out)
        if not fluid_is_below(out, tx):
            out = flipped
            if verbose:
                print("  bulb was at the top -> rotated 180 deg")
    return out


# ---------------------------------------------------------------------------
# Scale analysis shared by tasks 2 and 3
# ---------------------------------------------------------------------------

def tick_rows(img, tube_x, side="left"):
    """Row darkness profile of the tick strip next to the tube (high-pass
    filtered so only the periodic tick marks remain)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if side == "left":
        xa, xb = max(0, tube_x - int(0.16 * w)), max(1, tube_x - int(0.05 * w))
    else:
        xa, xb = tube_x + int(0.05 * w), min(w, tube_x + int(0.16 * w))
    dark = (255.0 - gray[:, xa:xb]).mean(1)
    dark = dark - cv2.GaussianBlur(dark.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=max(3.0, w / 15)).ravel()
    return dark, (xa, xb)


def tick_period(dark, min_p=4, max_p=None):
    """Dominant period of a 1-D tick profile via autocorrelation."""
    d = dark - dark.mean()
    n = len(d)
    if n < 3 * min_p:
        return None
    ac = np.correlate(d, d, "full")[n - 1:]
    ac /= ac[0] + 1e-6
    max_p = max_p or n // 3
    best, best_v = None, -1
    for p in range(min_p, min(max_p, n - 1)):
        if ac[p] > ac[p - 1] and ac[p] >= ac[p + 1] and ac[p] > best_v:
            best, best_v = p, ac[p]
    if best is None:
        return None
    # sub-pixel refinement with a parabola through the peak
    a, b, c = ac[best - 1], ac[best], ac[best + 1]
    den = a - 2 * b + c
    return best + (0.5 * (a - c) / den if den != 0 else 0.0)


def number_blobs(img, tube_x, side="left"):
    """Find printed scale numbers on one side of the tube. Returns a list of
    (x0, y0, x1, y1) boxes, one per number (digits and minus sign grouped)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if side == "left":
        xa, xb = 0, max(1, tube_x - int(0.12 * w))
    else:
        xa, xb = tube_x + int(0.12 * w), w
    region = gray[:, xa:xb]
    if region.shape[1] < 3:
        return [], []
    block = max(15, (min(h, w) // 8) | 1)
    bw = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 12)
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw, 8)
    comps = []
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        comps.append((x + xa, y, x + xa + cw, y + ch, area))
    if not comps:
        return [], []
    # tick marks are wide & short; characters are taller than wide-ish
    heights = np.array([c[3] - c[1] for c in comps])
    # Components touching the clipped inner edge of the region are cut-off
    # tick ends (thin slivers); components touching the outer image border are
    # background beside the thermometer. Neither can be a printed number.
    def clipped(c):
        return (c[2] >= xb or c[0] <= 0) if side == "left" else (c[0] <= xa or c[2] >= w)
    char_like = [c for c in comps if (c[3] - c[1]) >= 0.02 * max(h, w) * 0.5 and (c[3] - c[1]) > 0.6 * (c[2] - c[0])
                 and c[4] > 8 and not clipped(c)]
    if not char_like:
        return [], comps
    # reference character height from the largest components (the printed
    # digits), so that many small texture specks cannot drag it down
    ch_med = np.median([c[3] - c[1] for c in sorted(char_like, key=lambda c: -c[4])[:6]])
    chars = [c for c in char_like if 0.6 * ch_med < (c[3] - c[1]) < 1.6 * ch_med]
    # minus signs: short wide blobs at character mid-height, to the left of characters
    minus = [c for c in comps if (c[2] - c[0]) > 1.5 * (c[3] - c[1]) and 0.25 * ch_med < (c[2] - c[0]) < 1.2 * ch_med
             and (c[3] - c[1]) < 0.35 * ch_med]
    # group characters on the same text line that are horizontally close
    chars.sort(key=lambda c: (c[1] + c[3]) / 2)
    groups = []
    for c in chars:
        cy = (c[1] + c[3]) / 2
        for g in groups:
            gy = (g[1] + g[3]) / 2
            if abs(cy - gy) < 0.5 * ch_med and (c[0] - g[2] < 0.8 * ch_med and g[0] - c[2] < 0.8 * ch_med):
                g[0], g[1], g[2], g[3] = min(g[0], c[0]), min(g[1], c[1]), max(g[2], c[2]), max(g[3], c[3])
                break
        else:
            groups.append(list(c[:4]))
    for m in minus:
        my = (m[1] + m[3]) / 2
        for g in groups:
            if g[1] < my < g[3] and 0 < g[0] - m[2] < 0.8 * ch_med:
                g[0] = min(g[0], m[0])
    # printed numbers are right-aligned against the tick marks: drop blobs
    # (e.g. the "degC" symbol, specks) whose right edge is off that column
    if len(groups) >= 3:
        right = np.median([g[2] for g in groups])
        groups = [g for g in groups if abs(g[2] - right) < 0.6 * ch_med]
    return [tuple(g) for g in groups], comps


def celsius_number_run(groups, max_cv=0.15):
    """Find the printed Celsius scale among number blobs sorted top to bottom:
    a run of len(THERMO_LABELS) evenly spaced blobs in which the narrowest is
    the lone "0" at its expected position (50 40 30 20 10 0 -10 -20). Returns
    (spacing cv, run) for the most even such run, or None. Uses blob
    geometry only, so it also works before any classifier is trained."""
    n, zero = len(THERMO_LABELS), THERMO_LABELS.index(0)
    best = None
    for i in range(len(groups) - n + 1):
        run = groups[i:i + n]
        if int(np.argmin([g[2] - g[0] for g in run])) != zero:
            continue
        gaps = np.diff([(g[1] + g[3]) / 2 for g in run])
        cv = gaps.std() / gaps.mean()
        if cv < max_cv and (best is None or cv < best[0]):
            best = (cv, run)
    return best


def celsius_layout_on_left(img):
    """True if the Celsius scale reads correctly on the left of the tube."""
    tube_x, _ = find_tube_x(img)
    groups, _ = number_blobs(img, tube_x, "left")
    return celsius_number_run(sorted(groups, key=lambda g: (g[1] + g[3]) / 2)) is not None


# ---------------------------------------------------------------------------
# Printed-number classifier (HOG + linear SVM trained by training_script_task3.py)
# ---------------------------------------------------------------------------

NUM_W, NUM_H = 64, 32
_HOG = None
_NUM_MODEL = None


def number_hog(crop):
    global _HOG
    if _HOG is None:
        _HOG = cv2.HOGDescriptor((NUM_W, NUM_H), (16, 16), (8, 8), (8, 8), 9)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = g.shape
    s = NUM_H / h
    nw = min(NUM_W, max(1, int(round(w * s))))
    g = cv2.resize(g, (nw, NUM_H), interpolation=cv2.INTER_AREA)
    canvas = np.full((NUM_H, NUM_W), 255, np.uint8)
    canvas[:, NUM_W - nw:] = g                     # right-align: units digit always at the same place
    return _HOG.compute(canvas).ravel()


def pad_box(box, img_shape, frac=0.15):
    x0, y0, x1, y1 = box
    h, w = img_shape[:2]
    p = int(frac * (y1 - y0))
    return max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)


def major_tick_y(img, box, tube_x):
    """y of the long tick a printed number labels. The number's own centre can
    sit a few pixels off the tick, which is a sizeable fraction of a degree, so
    use the darkness-weighted centre of the tick line just right of the number
    (where only major ticks reach). Falls back to the box centre."""
    x0, y0, x1, y1 = box
    ch = y1 - y0
    h, w = img.shape[:2]
    centre = (y0 + y1) / 2.0
    xa, xb = x1 + max(2, int(0.1 * ch)), min(tube_x - int(0.16 * w), x1 + int(0.8 * ch))
    ya, yb = max(0, int(y0 - 0.3 * ch)), min(h, int(y1 + 0.3 * ch))
    if xb - xa < 3 or yb - ya < 3:
        return centre
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = (255.0 - gray[ya:yb, xa:xb]).mean(1)
    d = np.clip(d - np.median(d), 0, None)
    if d.max() < 30:                                   # no clear tick line
        return centre
    p = int(np.argmax(d))
    lo, hi = p, p
    while lo > 0 and d[lo - 1] > 0.5 * d[p]:
        lo -= 1
    while hi < len(d) - 1 and d[hi + 1] > 0.5 * d[p]:
        hi += 1
    ys = np.arange(ya + lo, ya + hi + 1)
    y = float((ys * d[lo:hi + 1]).sum() / d[lo:hi + 1].sum())
    return y if abs(y - centre) < 0.5 * ch else centre


def number_scores(img, boxes):
    """SVM scores of every number box: (classes, array [box, class])."""
    global _NUM_MODEL
    if _NUM_MODEL is None:
        if not os.path.exists(THERMO_NUM_WEIGHTS):
            raise FileNotFoundError(f"{THERMO_NUM_WEIGHTS} missing. Run training_script_task3.py first.")
        d = np.load(THERMO_NUM_WEIGHTS)
        _NUM_MODEL = (d["W"], d["b"], d["classes"], d["mean"], d["std"])
    W, b, classes, mean, std = _NUM_MODEL
    scores = []
    for box in boxes:
        x0, y0, x1, y1 = pad_box(box, img.shape)
        f = (number_hog(img[y0:y1, x0:x1]) - mean) / std
        scores.append(W @ f + b)
    return [int(c) for c in classes], np.array(scores, np.float32).reshape(len(boxes), len(classes))


def classify_numbers(img, boxes):
    """Return [(value, score)] for each number box, using the trained SVM."""
    classes, scores = number_scores(img, boxes)
    out = []
    for s in scores:
        srt = np.sort(s)
        out.append((classes[int(np.argmax(s))], float(srt[-1] - srt[-2])))
    return out


def decode_number_run(ys, classes, scores):
    """Read the printed numbers together instead of one by one.

    The labels are a fixed sequence (50, 40, ... -20) printed at equal
    spacing, so numbers one spacing apart must differ by exactly 10 degrees.
    A single number is easily misread (20 as -20, 10 as -10), but only one
    starting value makes the whole run agree with the classifier. ys: tick
    positions sorted top to bottom. Returns one value per number (None for a
    Returns one value per number, or None when the blobs are not evenly
    spaced (then they are not all labels and are read one by one)."""
    n = len(ys)
    if n < 2:
        return None
    gaps = np.diff(ys)
    gap = float(np.median(gaps))
    if gap <= 0:
        return None
    jumps = gaps / gap                                 # 1 = neighbouring labels, 2 = one missing
    if np.any(jumps < 0.8) or np.any(np.abs(jumps - np.round(jumps)) > 0.2):
        return None
    steps = np.concatenate([[0], np.cumsum(np.round(jumps))]).astype(int)
    best = None
    for v0 in classes:
        values = [v0 - 10 * int(s) for s in steps]
        if any(v not in classes for v in values):
            continue
        total = sum(float(scores[i][classes.index(v)]) for i, v in enumerate(values))
        if best is None or total > best[0]:
            best = (total, values)
    return None if best is None else best[1]


def fit_scale(numbers, period=None):
    """numbers: [(y_centre, value, margin)] -> (a, b) for T = a + b*y.
    Uses RANSAC over pairs so a single misread number cannot break the fit;
    a valid model must have negative slope (temperature rises upwards)."""
    if not numbers:
        return None
    if len(numbers) == 1:
        if period is None:
            return None
        y, v, _ = numbers[0]
        return v + y / period, -1.0 / period
    best, best_err = None, None
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            (yi, vi, _), (yj, vj, _) = numbers[i], numbers[j]
            if vi == vj or yi == yj:
                continue
            b = (vi - vj) / (yi - yj)
            if b >= 0:
                continue
            a = vi - b * yi
            err = sorted(abs(a + b * y - v) for y, v, _ in numbers)
            inl = [e for e in err if e < 3.0]
            # the tick period can lock onto a multiple of the true spacing, so
            # it only vetoes a pair that is not backed by a third number
            if period is not None and not (0.6 < (-1.0 / b) / period < 1.6) and len(inl) < 3:
                continue
            key = (-len(inl), sum(inl))
            if best is None or key < best_err:
                best, best_err = (a, b), key
    if best is None:
        return fit_scale(sorted(numbers, key=lambda n: -n[2])[:1], period)
    a, b = best
    inl = [(y, v) for y, v, _ in numbers if abs(a + b * y - v) < 3.0]
    if len(inl) >= 2:
        ys, vs = np.array(inl, float).T
        b, a = np.polyfit(ys, vs, 1)
    return a, b


# ---------------------------------------------------------------------------
# Task 2: segmentation of the reading region
# ---------------------------------------------------------------------------

def analyse_scale(img, verbose=True):
    """Common analysis: tube x, fluid top, tick period (px/deg), numbers."""
    h, w = img.shape[:2]
    tube_x, _ = find_tube_x(img)
    top, bottom, _ = fluid_extent(img, tube_x)
    dark, _ = tick_rows(img, tube_x, "left")
    # one degree is roughly 1/45 .. 1/10 of the body width on this thermometer
    period = tick_period(dark, min_p=max(3, int(w / 45)), max_p=max(6, int(w / 10)))
    boxes, _ = number_blobs(img, tube_x, "left")
    if verbose:
        print(f"  tube x={tube_x}, fluid top y={top if top is None else round(top, 1)}, tick period={period if period is None else round(period, 2)} px/deg, "
              f"{len(boxes)} number blobs")
    return tube_x, top, period, boxes


def segment_thermometer(img, verbose=True):
    """img: upright thermometer (task 1 output). Returns the crop centred on
    the fluid top that contains at least one labelled major tick above and
    below the reading (typically two or more)."""
    h, w = img.shape[:2]
    tube_x, top, period, boxes = analyse_scale(img, verbose)
    if top is None:
        raise RuntimeError("fluid column not found")
    top = int(round(top))
    # vertical spacing between printed numbers = 10 degrees
    ys = sorted((b[1] + b[3]) / 2 for b in boxes)
    gaps = np.diff(ys) if len(ys) > 1 else np.array([])
    label_gap = None
    if gaps.size:
        g = np.median(gaps)
        good = gaps[(gaps > 0.6 * g) & (gaps < 1.4 * g)]
        label_gap = float(np.median(good)) if good.size else float(g)
    if period is not None and (label_gap is None or not (6 < label_gap / period < 14)):
        # trust the tick period when the number spacing looks inconsistent
        if label_gap is None or len(ys) < 3:
            label_gap = 10 * period
    if label_gap is None:
        label_gap = h / 9.0                           # ~8 labelled intervals over the body
    half = int(1.6 * label_gap)

    def window(top):
        y0, y1 = top - half, top + half
        if y0 < 0:
            y1, y0 = y1 - y0, 0
        if y1 > h:
            y0, y1 = max(0, y0 - (y1 - h)), h
        return y0, y1

    y0, y1 = window(top)
    # Over the whole thermometer a faint fluid column can be hidden by the
    # lighting gradient, so the first estimate may be far off. Task 3 looks
    # for the fluid top again inside the crop, with local levels; do the same
    # here and re-centre the crop until the two agree.
    for _ in range(4):
        local = fluid_extent(img[y0:y1], find_tube_x(img[y0:y1])[0])[0]
        if local is None or abs(y0 + local - top) < 0.15 * half:
            break
        top = y0 + int(round(local))
        y0, y1 = window(top)
        if verbose:
            print(f"  fluid top re-detected in the crop -> y={top}")
    if verbose:
        print(f"  label spacing {label_gap:.1f} px (10 deg), crop rows {y0}-{y1}")
    return img[y0:y1], (y0, y1, top)


# ---------------------------------------------------------------------------
# Task 3: temperature calculation
# ---------------------------------------------------------------------------

def read_thermometer(img, verbose=True):
    """img: crop around the fluid top (task 2 output). Returns integer degC."""
    h, w = img.shape[:2]
    tube_x, top, period, boxes = analyse_scale(img, verbose)
    if top is None:
        raise RuntimeError("fluid column not found")
    boxes = sorted(boxes, key=lambda b: b[1] + b[3])
    preds = classify_numbers(img, boxes) if boxes else []
    ys = [major_tick_y(img, box, tube_x) for box in boxes]
    run = decode_number_run(ys, *number_scores(img, boxes)) if len(boxes) >= 2 else None
    numbers = []
    for i, (yc, (v, margin)) in enumerate(zip(ys, preds)):
        note = ""
        if run is not None and run[i] != v:
            note, v = f" (read alone as {v})", run[i]
        numbers.append((yc, v, margin))
        if verbose:
            print(f"  number at y={yc:.0f}: {v} (margin {margin:.2f}){note}")
    model = fit_scale(numbers, period)
    if model is None:
        raise RuntimeError("could not establish the temperature scale")
    a, b = model
    temp = a + b * top
    if verbose:
        print(f"  scale: T = {a:.2f} + {b:.4f} * y  ->  {1 / abs(b):.2f} px/deg; fluid top y={top:.1f} -> {temp:.2f} C")
    return int(round(temp))
