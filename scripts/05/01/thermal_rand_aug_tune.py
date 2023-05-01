"""
train series of mobilenets on 100 subclasses of imagenet
"""

import os
import random
import sys
sys.path.append(".")
from script_manager.func.script_boilerplate import do_everything
from local_config import IR_VIS_DATASET_PATH_NEW2 as IR_VIS_DATASET_PATH

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
    "sched": "step",
    "epochs": 200,
    "workers": 24,
    'warmup-epochs': 0,
    "opt": "sgd",
    "warmup-lr": 1e-6,
    "weight-decay": 1e-5,
    "drop": 0.2,
    "drop-path": 0.2,
    "batch-size": 64,
    "decay-rate": 0.1,
    "decay-epochs": 80,
    "lr": 1e-1,
    # "model-ema":  "parameter_without_value",
    # "model-ema-decay": 0.9999,
    # "aa": "rand-m9-mstd0.5",
    "remode": "pixel",
    "reprob": 0.2,
    "amp": "parameter_without_value",
    "pretrained": "parameter_without_value",
    "test-split": "test",
    "data-dir": os.path.join(IR_VIS_DATASET_PATH, 'pytorch_models_structure', 'day'),
    "hflip": 0,
    "disable_geometry_aug": "parameter_without_value",
    "color-jitter": 0,
}

configs = []

test_parameters = {

}

MAIN_SCRIPT = f"torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:{random.randint(0,1000)} --nproc_per_node=1 train.py"


# for color_jitter in [0.4]:
#     config = {
#         "adjust_sharpness": 10,
#         "color-jitter": color_jitter,
#         "force-color-jitter": "parameter_without_value",
#         "random_invert_p": 0.5,
#     }
#     configs.append([config, None])

for color_jitter in [0.4]:
    config = {
        "aa": "rand-m9-mstd0.5",
        "adjust_sharpness": 10,
        "color-jitter": color_jitter,
        "force_color_jitter": "parameter_without_value",
        "random_invert_p": 0.5,
        "tag": f"{base_tag}_day_FORCE_color_jitter_{color_jitter}_aa_random_invert_p_0.5_sharp",
    }
    configs.append([config, None])


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
