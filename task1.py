

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
# Task 1: detect the device (YOLOv8n), then extract and correct it.
#   BPM         -> perspective-corrected LCD saved as lcdX.png
#   thermometer -> upright, bulb-down thermometer saved as thermoX.png
#   negative    -> nothing is saved

import os

from common import task_output_dir, save_output, list_images, image_number, read_image
from pipeline import extract_device


def run_task1(image_path, config):
    out_dir = task_output_dir("task1")
    files = list_images(image_path)
    print(f"Task 1: {len(files)} image(s) in {image_path}")
    for f in files:
        print(f"[{os.path.basename(f)}]")
        try:
            kind, device, _ = extract_device(read_image(f))
        except Exception as e:
            print(f"  failed: {e}")
            continue
        n = image_number(f)
        if kind == "lcd":
            save_output(os.path.join(out_dir, f"lcd{n}.png"), device, output_type='image')
        elif kind == "thermo":
            save_output(os.path.join(out_dir, f"thermo{n}.png"), device, output_type='image')
        else:
            print("  no blood pressure monitor or thermometer found -> no output")
