"""
train series of mobilenets on 100 subclasses of imagenet
"""

import os
import sys
sys.path.append(".")
from script_manager.func.script_boilerplate import do_everything

# weights and biases project name
WANDB_PROJECT_NAME = "python-image-models"
base_tag = os.path.split(__file__)[-1].split('.')[0]
# keys
appendix_keys = ["tag"]
extra_folder_keys = []

# same parameters that are
# used for 1000 classes
default_parameters = {
}

configs = []

test_parameters = {

}

MAIN_SCRIPT = f"get_clusters_from_confusion_matrix.py"

config = {
    "tag": "debug_",
    "class-map": f"class_maps/class_map_all.txt",
    "confusion_matrix_file": 'confusion_matrices/mobilenetv3_small_100',
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
