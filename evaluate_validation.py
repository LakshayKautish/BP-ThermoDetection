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
# Runs tasks 1-4 on validation/ exactly as the marker does
# (python assignment.py taskN <dir> ./config.txt), times each task and
# compares the results with expected_validation_output/.
#
#   python evaluate_validation.py [--tasks 1 2 3 4]

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import cv2

from common import BASE_DIR, OUTPUT_DIR, natural_key


def path_key(path):
    """Natural sort on the whole relative path (lcd2/d1 < lcd2/d10 < lcd4/d1)."""
    return [natural_key(part) for part in path.replace("\\", "/").split("/")]

VAL = os.path.join(BASE_DIR, "validation")
EXP = os.path.join(BASE_DIR, "expected_validation_output", "output")
TIME_LIMIT = 300


def run(task):
    t0 = time.time()
    p = subprocess.run([sys.executable, "assignment.py", f"task{task}", os.path.join("validation", f"task{task}"),
                        "./config.txt"], cwd=BASE_DIR, capture_output=True, text=True)
    dt = time.time() - t0
    log = p.stdout + p.stderr
    log_dir = os.path.join(BASE_DIR, "docs", "validation_logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"task{task}_log.txt"), "w", encoding="utf-8") as f:
        f.write(log)
    return dt, p.returncode, log


def read_txt(path):
    with open(path) as f:
        return f.read().strip()


def rel(paths, root):
    return sorted((os.path.relpath(p, root).replace("\\", "/") for p in paths), key=natural_key)


def check_task1():
    exp = rel(glob.glob(os.path.join(EXP, "task1", "*.png")), os.path.join(EXP, "task1"))
    got = rel(glob.glob(os.path.join(OUTPUT_DIR, "task1", "*.png")), os.path.join(OUTPUT_DIR, "task1"))
    for f in got:
        h, w = cv2.imread(os.path.join(OUTPUT_DIR, "task1", f)).shape[:2]
        print(f"    {f}: {w}x{h}" + ("" if f in exp else "   <- unexpected (negative not rejected?)"))
    for f in exp:
        if f not in got:
            print(f"    {f}: MISSING")
    return {"expected": exp, "produced": got, "files_match": exp == got}


def check_task2():
    res = {}
    for d in sorted(glob.glob(os.path.join(EXP, "task2", "*")), key=natural_key):
        name = os.path.basename(d)
        exp = rel(glob.glob(os.path.join(d, "*.png")), d)
        gd = os.path.join(OUTPUT_DIR, "task2", name)
        got = rel(glob.glob(os.path.join(gd, "*.png")), gd)
        ok = exp == got
        res[name] = ok
        print(f"    {name}: expected {len(exp)} crops, got {len(got)}" + ("" if ok else "   <- MISMATCH"))
    return res


def check_task3():
    digits_ok = digits_total = 0
    thermo = {}
    for f in sorted(glob.glob(os.path.join(EXP, "task3", "*", "*.txt")), key=path_key):
        r = os.path.relpath(f, os.path.join(EXP, "task3")).replace("\\", "/")
        g = os.path.join(OUTPUT_DIR, "task3", r)
        exp, got = read_txt(f), (read_txt(g) if os.path.exists(g) else None)
        if r.startswith("lcd"):
            digits_total += 1
            digits_ok += int(exp == got)
            if exp != got:
                print(f"    {r}: expected {exp}, got {got}")
        else:
            err = None if got is None else abs(int(got) - int(exp))
            thermo[r] = {"expected": exp, "got": got, "abs_error": err}
            print(f"    {r}: expected {exp}, got {got}")
    acc = digits_ok / max(1, digits_total)
    print(f"    digit accuracy: {digits_ok}/{digits_total} = {acc:.3f}")
    return {"digit_accuracy": acc, "digits_correct": digits_ok, "digits_total": digits_total, "thermo": thermo}


def check_task4():
    res = {}
    for i in range(1, 6):
        e = os.path.join(EXP, "task4", f"img{i}.txt")
        g = os.path.join(OUTPUT_DIR, "task4", f"img{i}.txt")
        exp = read_txt(e) if os.path.exists(e) else None
        got = read_txt(g) if os.path.exists(g) else None
        res[f"img{i}"] = {"expected": exp, "got": got, "correct": exp == got}
        print(f"    img{i}: expected {exp!r:22} got {got!r:22} {'OK' if exp == got else 'WRONG'}")
    return res


CHECKS = {1: check_task1, 2: check_task2, 3: check_task3, 4: check_task4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = {}
    for t in args.tasks:
        dt, code, log = run(t)
        flag = "" if dt < TIME_LIMIT else "   <- OVER THE 5 MINUTE LIMIT"
        print(f"task{t}: {dt:.1f} s, exit code {code}{flag}")
        if "An error occurred" in log or code != 0:
            print("    " + log.strip().splitlines()[-1])
        report[f"task{t}"] = {"seconds": round(dt, 1), "exit_code": code, "result": CHECKS[t]()}
    out = os.path.join(BASE_DIR, "docs", "validation_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
