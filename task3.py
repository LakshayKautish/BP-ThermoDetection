

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
# Task 3: recognition.
#   lcdX/dZ.png  -> lcdX/dZ.txt   (one digit, trained CNN classifier)
#   thermoX/t.png -> thermoX/t.txt (integer degrees Celsius)

import os
import glob

from common import task_output_dir, save_output, natural_key, read_image, classify_digits
from thermo import read_thermometer


def run_task3(image_path, config):
    out_dir = task_output_dir("task3")
    folders = sorted((d for d in glob.glob(os.path.join(image_path, "*")) if os.path.isdir(d)), key=natural_key)
    print(f"Task 3: {len(folders)} folder(s) in {image_path}")
    for d in folders:
        name = os.path.basename(d)
        print(f"[{name}]")
        try:
            if name.startswith("lcd"):
                files = sorted(glob.glob(os.path.join(d, "d*.png")), key=natural_key)
                labels, confs = classify_digits([read_image(f) for f in files])
                for f, v, c in zip(files, labels, confs):
                    stem = os.path.splitext(os.path.basename(f))[0]
                    print(f"  {stem}: {v} (p={c:.2f})")
                    save_output(os.path.join(out_dir, name, f"{stem}.txt"), str(v), output_type='txt')
            elif name.startswith("thermo"):
                t = read_thermometer(read_image(os.path.join(d, "t.png")))
                save_output(os.path.join(out_dir, name, "t.txt"), str(t), output_type='txt')
            else:
                print("  not an lcd*/thermo* folder -> skipped")
        except Exception as e:
            print(f"  failed: {e}")
