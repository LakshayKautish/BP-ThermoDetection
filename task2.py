

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
# Task 2: segmentation of the corrected device images from task 1.
#   lcdX.png    -> lcdX/d1.png .. dN.png (rows top to bottom, digits left to right)
#   thermoX.png -> thermoX/t.png (scale region around the top of the fluid)

import os

from common import task_output_dir, save_output, list_images, read_image
from bpm import segment_digits
from thermo import segment_thermometer


def run_task2(image_path, config):
    out_dir = task_output_dir("task2")
    files = list_images(image_path)
    print(f"Task 2: {len(files)} image(s) in {image_path}")
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        print(f"[{os.path.basename(f)}]")
        try:
            img = read_image(f)
            if stem.startswith("lcd"):
                crops, _ = segment_digits(img)
                for i, c in enumerate(crops, 1):
                    save_output(os.path.join(out_dir, stem, f"d{i}.png"), c, output_type='image')
            elif stem.startswith("thermo"):
                crop, _ = segment_thermometer(img)
                save_output(os.path.join(out_dir, stem, "t.png"), crop, output_type='image')
            else:
                print("  not an lcd*/thermo* image -> skipped")
        except Exception as e:
            print(f"  failed: {e}")
