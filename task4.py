

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
# Task 4: the full pipeline (tasks 1-3 in memory) on raw photos.
#   BPM         -> imgX.txt "bpm SYS,DIA,PULSE"
#   thermometer -> imgX.txt "thermo T"
#   negative    -> nothing is written

import os

from common import task_output_dir, save_output, list_images, image_number, read_image, classify_digits
from pipeline import extract_device, rows_to_values
from thermo import segment_thermometer, read_thermometer


def run_task4(image_path, config):
    out_dir = task_output_dir("task4")
    files = list_images(image_path)
    print(f"Task 4: {len(files)} image(s) in {image_path}")
    for f in files:
        print(f"[{os.path.basename(f)}]")
        try:
            kind, device, seg = extract_device(read_image(f))
            if kind == "lcd":
                crops, rows = seg
                labels, _ = classify_digits(crops)
                result = "bpm " + ",".join(str(v) for v in rows_to_values(rows, labels))
            elif kind == "thermo":
                crop, _ = segment_thermometer(device)
                result = f"thermo {read_thermometer(crop)}"
            else:
                print("  no blood pressure monitor or thermometer found -> no output")
                continue
        except Exception as e:
            print(f"  failed: {e}")
            continue
        print(f"  result: {result}")
        save_output(os.path.join(out_dir, f"img{image_number(f)}.txt"), result, output_type='txt')
