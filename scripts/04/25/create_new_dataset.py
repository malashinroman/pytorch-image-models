"""
train series of mobilenets on 100 subclasses of imagenet
"""

import sys
sys.path.append(".")
from local_config import IMAGENET_PATH, IR_VIS_DATASET_PATH
from script_manager.func.script_boilerplate import do_everything
import os
import random

# weights and biases project name
WANDB_PROJECT_NAME = "python-image-models"
base_tag = os.path.split(__file__)[-1].split('.')[0]
# keys
appendix_keys = ["tag"]
extra_folder_keys = []

# same parameters that are
# used for 1000 classes
default_parameters = {
    "__script_output_arg__": "output",
}

configs = []

test_parameters = {
}

# parser.add_argument('--output', default=None,
#                     type=str, help='output path')
# parser.add_argument('--hours', default=str2list,
#                     type=str, help='hours of the day that we accept (e.g. [night, day])')
# parser.add_argument('--diag', default=16,
#                     type=int, help='minimal diagonal size')
# parser.add_argument('--extend_bbox', default=0.4,
#                     type=float, help='extend bbox by this ratio')
# parser.add_argument('--flir_subfolder', default="video_thermal_test",
#                     type='str', help='flir subfolder name')
# parser.add_argument('--output_folder', default="crops",
#                     type='str', help='subfolder to save crops in flir_subfolder')

MAIN_SCRIPT = "create_flir_dataset.py"
for hours in ["[day,dawn/dusk,night]", "[day]",  "[day,dawn/dusk]"]:
    for flir_subfolders in ["images_rgb_train",  "images_rgb_val",  "images_thermal_train",  "images_thermal_val"]:
        config = {
            "flir_subfolder": flir_subfolders,
            "hours": hours,
            "tag": f"{base_tag}_{flir_subfolders}_{hours.replace('/', '_').replace('[', '').replace(']', '').replace(',', '_')}",
        }
        configs.append([config, None])

# RUN everything
# !normally you don't have to change anything here
if __name__ == "__main__":
    do_everything(
        default_parameters=default_parameters,
        configs=configs,
        extra_folder_keys=extra_folder_keys,
        appendix_keys=appendix_keys,
        main_script=MAIN_SCRIPT,
        test_parameters=test_parameters,
        wandb_project_name=WANDB_PROJECT_NAME,
        script_file=__file__,
    )
