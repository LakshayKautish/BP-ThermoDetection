
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
# Blood pressure monitor (seven-segment LCD) processing:
#   rectify_lcd()      - task 1: find the LCD panel quadrilateral inside the
#                        detector box and perspective-correct it
#   segment_digits()   - task 2: detect digits, group them into rows (SYS, DIA,
#                        PULSE) and return one crop per digit in reading order
#   looks_like_bpm()   - sanity check that rejects LCDs that are not a BPM
#                        (e.g. the digital clock in the negative example)

import cv2
import numpy as np

from common import DIGIT_DET_WEIGHTS, load_yolo


# ---------------------------------------------------------------------------
# Task 1: LCD rectification
# ---------------------------------------------------------------------------

def order_corners(pts):
    """Order 4 points clockwise as top-left, top-right, bottom-right,
    bottom-left (valid for in-plane rotations up to +/-45 degrees; larger
    rotations are resolved later by fix_lcd_orientation)."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])      # image coords: y down
    pts = pts[np.argsort(ang)]                                # clockwise on screen
    # start from the corner closest to the "up-left" direction (-135 deg)
    a = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    start = int(np.argmin(np.abs(np.angle(np.exp(1j * (a + 3 * np.pi / 4))))))
    return np.roll(pts, -start, axis=0)


def warp_quad(img, quad):
    """Perspective-warp the quadrilateral to an upright rectangle whose size
    is the average of opposite side lengths."""
    tl, tr, br, bl = quad
    w = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    h = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
    w, h = max(w, 8), max(h, 8)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def find_lcd_quad(crop, ref_box, target=1.0, lo=0.25, hi=2.2, must_contain=None):
    """Search the crop for the LCD panel outline: a large convex 4-sided
    contour. Tries several binarisations (Otsu, inverse Otsu, Canny) and
    returns the best quadrilateral or None.

    ref_box is the reference rectangle that sizes the search: the quad area
    must lie between lo and hi times its area and should be close to target
    times its area. If must_contain (a list of points) is given the quad has
    to enclose all of them, otherwise its centre has to fall in ref_box."""
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    bx0, by0, bx1, by1 = ref_box
    box_area = (bx1 - bx0) * (by1 - by0)
    cands = []
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = [otsu, 255 - otsu, cv2.dilate(cv2.Canny(gray, 30, 90), np.ones((3, 3), np.uint8))]
    for m in masks:
        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < lo * box_area or area > 0.95 * w * h:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in (0.02, 0.03, 0.05):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4:
                    break
            if len(approx) == 4:
                quad = order_corners(approx)
            else:
                # rounded-corner panel: use the minimum-area rotated rectangle
                quad = order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))
            qarea = cv2.contourArea(quad)
            if must_contain is not None:
                if any(cv2.pointPolygonTest(quad, (float(x), float(y)), False) < 0 for x, y in must_contain):
                    continue
            else:
                cx, cy = quad[:, 0].mean(), quad[:, 1].mean()
                if not (bx0 <= cx <= bx1 and by0 <= cy <= by1):
                    continue
            rect_fill = area / max(qarea, 1)
            size_ratio = qarea / box_area
            if not (lo < size_ratio < hi) or rect_fill < 0.8:
                continue
            # prefer quads close to the expected size and very rectangular
            score = rect_fill - 0.5 * abs(np.log(size_ratio / target))
            cands.append((score, quad))
    if not cands:
        return None
    return max(cands, key=lambda t: t[0])[1]


def rotate_region(img, box, angle, scale=1.0):
    """Cut the region box = (x0,y0,x1,y1) out of img, rotated by angle
    degrees (counter-clockwise) about its centre and resized by scale. The
    canvas is enlarged so that the whole region stays visible, and its
    corners are filled from the surrounding photo rather than padded."""
    x0, y0, x1, y1 = box
    w, h = (x1 - x0) * scale, (y1 - y0) * scale
    M = cv2.getRotationMatrix2D(((x0 + x1) / 2, (y0 + y1) / 2), angle, scale)
    c, s = abs(M[0, 0]) / scale, abs(M[0, 1]) / scale
    nw, nh = int(round(h * s + w * c)), int(round(h * c + w * s))
    M[0, 2] += nw / 2 - (x0 + x1) / 2
    M[1, 2] += nh / 2 - (y0 + y1) / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def row_tilt(rows):
    """Residual tilt of the digit rows in degrees (positive = rows fall
    towards the right), from the line through each row's first and last
    digit centre."""
    angs = []
    for r in rows:
        if len(r) >= 2:
            (ax, ay), (bx, by) = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in (r[0], r[-1])]
            angs.append(np.degrees(np.arctan2(by - ay, bx - ax)))
    return float(np.median(angs)) if angs else 0.0


def orientation_score(rows):
    """How much a set of digit rows looks like an upright BPM reading: the
    detector confidences, a bonus for a valid SYS/DIA/PULSE layout, the pulse
    row (smaller digits) at the bottom rather than at the top, and level
    rows. The pulse-row term separates upright from upside-down displays,
    where digits such as 8, 5, 2 and 0 are detected equally well."""
    score = sum(b[4] for r in rows for b in r)
    if looks_like_bpm(rows):
        score += 2.0
        hs = [np.median([b[3] - b[1] for b in r]) for r in rows]
        if hs[-1] < 0.8 * hs[0]:
            score += 1.0
        elif hs[0] < 0.8 * hs[-1]:
            score -= 1.0
    return score - 0.05 * abs(row_tilt(rows))


SEARCH_SIDE = 480      # the orientation search runs on a reduced copy
SEARCH_STEP = 15       # degrees


def find_lcd_angle(img, box):
    """Try every in-plane rotation of the region and keep the one in which
    the digit detector sees the most convincing upright reading, then level
    the digit rows. Returns (angle, digit rows in the coordinates of
    rotate_region(img, box, angle)), or (0, []) when no digits are found."""
    s = min(1.0, SEARCH_SIDE / max(box[2] - box[0], box[3] - box[1]))
    k = max(1, int(round(1 / s)))
    small = cv2.blur(img, (k, k)) if k > 1 else img      # avoid aliasing in the reduced copy

    def read(ang):
        view = rotate_region(small, box, ang, s)
        rows = select_reading_rows(group_rows(detect_digit_boxes(view, verbose=False)))
        return orientation_score(rows), ang, rows

    best = max((read(a) for a in range(0, 360, SEARCH_STEP)), key=lambda t: t[0])
    for _ in range(2):                                   # the coarse step leaves up to half a step
        tilt = row_tilt(best[2])
        if abs(tilt) < 1.5:
            break
        fine = read((best[1] + tilt) % 360)
        if fine[0] < best[0]:
            break
        best = fine
    rows = [[(b[0] / s, b[1] / s, b[2] / s, b[3] / s, b[4]) for b in r] for r in best[2]]
    return best[1], rows


def rectify_lcd(img, box, verbose=True):
    """img: full photo; box: detector (x0,y0,x1,y1) of the LCD. Returns the
    perspective-corrected LCD image.

    The device may lie at any angle, so the region is first rotated until
    the digits are upright; the panel is then the smallest rectangular
    outline that encloses all of them."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    m = 0.15
    cx0, cy0 = int(max(0, x0 - m * bw)), int(max(0, y0 - m * bh))
    cx1, cy1 = int(min(W, x1 + m * bw)), int(min(H, y1 + m * bh))
    crop = img[cy0:cy1, cx0:cx1]
    try:
        ang, rows = find_lcd_angle(img, (cx0, cy0, cx1, cy1))
    except FileNotFoundError:
        ang, rows = 0, []
    if rows:
        if verbose and abs(ang) > 0.05:
            print(f"  digits upright after rotating the crop by {ang:.1f} deg")
        rot = rotate_region(img, (cx0, cy0, cx1, cy1), ang)
        bs = [b for r in rows for b in r]
        ux0, uy0 = min(b[0] for b in bs), min(b[1] for b in bs)
        ux1, uy1 = max(b[2] for b in bs), max(b[3] for b in bs)
        # the panel must enclose the core (middle half) of every digit box
        cores = [(b[0] + fx * (b[2] - b[0]), b[1] + fy * (b[3] - b[1]))
                 for b in bs for fx in (0.25, 0.75) for fy in (0.25, 0.75)]
        # the panel is 1-2 times the area spanned by the digits; the device
        # body around it is far larger
        quad = find_lcd_quad(rot, (ux0, uy0, ux1, uy1), target=1.5, lo=0.8, hi=3.0, must_contain=cores)
        if quad is not None:
            if verbose:
                print("  LCD panel outline found -> perspective correction")
            return fix_lcd_orientation(warp_quad(rot, quad), verbose)
        if verbose:
            print("  LCD outline not found -> cropping around the levelled digits")
        uw, uh = ux1 - ux0, uy1 - uy0
        a, b = int(max(0, ux0 - 0.2 * uw)), int(min(rot.shape[1], ux1 + 0.1 * uw))
        c, d = int(max(0, uy0 - 0.12 * uh)), int(min(rot.shape[0], uy1 + 0.08 * uh))
        return fix_lcd_orientation(rot[c:d, a:b], verbose)
    quad = find_lcd_quad(crop, (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0))
    if quad is not None:
        if verbose:
            print("  LCD panel outline found -> perspective correction")
        return fix_lcd_orientation(warp_quad(crop, quad), verbose)
    # fallback: deskew the box content using the dominant text-row angle
    if verbose:
        print("  LCD outline not found -> using detector box with deskew")
    roi = img[int(y0):int(y1), int(x0):int(x1)]
    return fix_lcd_orientation(deskew_by_digits(roi), verbose)


def fix_lcd_orientation(lcd, verbose=True):
    """Final check among the 0/90/180/270 degree rotations of the corrected
    panel, in case the corner order of the outline was ambiguous."""
    try:
        cands = [(0, lcd), (90, cv2.rotate(lcd, cv2.ROTATE_90_CLOCKWISE)),
                 (180, cv2.rotate(lcd, cv2.ROTATE_180)),
                 (270, cv2.rotate(lcd, cv2.ROTATE_90_COUNTERCLOCKWISE))]
        best = None
        for ang, im in cands:
            rows = select_reading_rows(group_rows(detect_digit_boxes(im, verbose=False)))
            score = orientation_score(rows)
            if best is None or score > best[0] + 0.5:      # prefer 0 deg unless clearly better
                best = (score, ang, im)
        if verbose and best[1] != 0:
            print(f"  LCD rotated by {best[1]} deg to make digit rows horizontal")
        return best[2]
    except FileNotFoundError:
        return lcd


def deskew_by_digits(roi):
    """Rotate so that the dark digit strokes' principal horizontal structure is level."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi / 180, 40, minLineLength=gray.shape[1] // 6, maxLineGap=5)
    if lines is None:
        return roi
    angs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 30:
            angs.append(a)
    if not angs:
        return roi
    a = float(np.median(angs))
    h, w = roi.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    return cv2.warpAffine(roi, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Task 2: digit segmentation
# ---------------------------------------------------------------------------

def detect_digit_boxes(lcd, conf=0.25, verbose=True):
    """Digit boxes (x0,y0,x1,y1,conf) from the trained YOLO digit detector."""
    model = load_yolo(DIGIT_DET_WEIGHTS)
    res = model.predict(lcd, imgsz=416, conf=conf, iou=0.4, verbose=False)[0]
    boxes = []
    for (x0, y0, x1, y1), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
        boxes.append((float(x0), float(y0), float(x1), float(y1), float(c)))
    if verbose:
        print(f"  digit detector: {len(boxes)} candidate digits")
    return boxes


def group_rows(boxes):
    """Cluster boxes into text rows by vertical overlap, top to bottom; each
    row is sorted left to right."""
    rows = []
    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        cy, bh = (b[1] + b[3]) / 2, b[3] - b[1]
        for r in rows:
            ry0 = np.median([x[1] for x in r])
            ry1 = np.median([x[3] for x in r])
            overlap = min(b[3], ry1) - max(b[1], ry0)
            if overlap > 0.5 * min(bh, ry1 - ry0):
                r.append(b)
                break
        else:
            rows.append([b])
    rows = [sorted(r, key=lambda b: b[0]) for r in rows]
    rows.sort(key=lambda r: np.median([(b[1] + b[3]) / 2 for b in r]))
    return rows


def select_reading_rows(rows):
    """Keep the SYS / DIA / PULSE rows: rows of 2-3 digits whose height is
    not tiny compared with the biggest row (drops icons, memory numbers)."""
    if not rows:
        return []
    heights = [np.median([b[3] - b[1] for b in r]) for r in rows]
    hmax = max(heights)
    keep = [r for r, h in zip(rows, heights) if h >= 0.35 * hmax and 1 <= len(r) <= 4]
    if len(keep) > 3:
        # keep the three tallest rows, preserving top-to-bottom order
        hs = [np.median([b[3] - b[1] for b in r]) for r in keep]
        idx = sorted(np.argsort(hs)[-3:])
        keep = [keep[i] for i in idx]
    # trim rows with more than 3 digits to the 3 most confident
    out = []
    for r in keep:
        if len(r) > 3:
            r = sorted(sorted(r, key=lambda b: -b[4])[:3], key=lambda b: b[0])
        out.append(r)
    return out


def crop_digits(lcd, rows, pad=0.06):
    """Crop each digit with a small margin, never crossing the midpoint to
    the neighbouring digit so no part of an adjacent digit is included."""
    h, w = lcd.shape[:2]
    crops = []
    for r in rows:
        for i, (x0, y0, x1, y1, _) in enumerate(r):
            bw, bh = x1 - x0, y1 - y0
            px, py = pad * max(bw, 0.5 * bh), pad * bh
            left = x0 - px
            right = x1 + px
            if i > 0:
                left = max(left, (r[i - 1][2] + x0) / 2)
            if i < len(r) - 1:
                right = min(right, (x1 + r[i + 1][0]) / 2)
            a, b = int(max(0, left)), int(min(w, right))
            c, d = int(max(0, y0 - py)), int(min(h, y1 + py))
            crops.append(lcd[c:d, a:b])
    return crops


def segment_digits(lcd, verbose=True):
    """Return (digit crops in reading order, rows of boxes)."""
    rows = select_reading_rows(group_rows(detect_digit_boxes(lcd, verbose=verbose)))
    if verbose:
        print(f"  rows: {[len(r) for r in rows]} digits per row")
    return crop_digits(lcd, rows), rows


def looks_like_bpm(rows):
    """A blood-pressure reading has 2-3 rows stacked vertically, each with
    2-3 digits, e.g. 135 / 93 / 65. A clock LCD shows a single row."""
    if len(rows) < 2:
        return False
    counts = [len(r) for r in rows]
    return sum(counts) >= 5 and all(1 <= c <= 3 for c in counts)
