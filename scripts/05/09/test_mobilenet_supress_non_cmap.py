"""
train series of mobilenets on 100 subclasses of imagenet
"""

import sys
sys.path.append(".")
from local_config import IMAGENET_PATH
from script_manager.func.script_boilerplate import do_everything
import os

# weights and biases project name
WANDB_PROJECT_NAME = "python-image-models"
base_tag = os.path.split(__file__)[-1].split('.')[0]
# keys
appendix_keys = ["tag"]
extra_folder_keys = []

# same parameters that are
# used for 1000 classes
default_parameters = {
    "model": "mobilenetv3_small_100",
    "pretrained": True,
    "batch-size": 512,
    "dycs_classes_per_group": 200,
    "dycs_meaning_neurons": 'inplace',
    "dycs_supress_not_mapped_classes": "paramter_without_value",
    "workers": 24,
    "data-dir": os.path.join(IMAGENET_PATH, 'pytorch_models_structure', 'validation'),
}

configs = []

test_parameters = {

}

MAIN_SCRIPT = "validate.py"
for i in range(0, 5):
    config = {
        "tag": f"{base_tag}_class_map{i}",
        "class-map": f"class_maps/class_maps_five_in_place/class_map{i}.txt"
    }
    configs.append([config, None])


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
