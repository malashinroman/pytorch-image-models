"""
train series of mobilenets on 100 subclasses of imagenet
"""

import os
import random
import sys
sys.path.append(".")
from script_manager.func.script_boilerplate import do_everything
from local_config import IR_VIS_DATASET_PATH_NEW as IR_VIS_DATASET_PATH

# weights and biases project name
WANDB_PROJECT_NAME = "python-image-models"
base_tag = os.path.split(__file__)[-1].split('.')[0]
# keys
appendix_keys = ["tag"]
extra_folder_keys = []

default_parameters = {
    "__script_output_arg__": "output",
    "test-split": "test",
    "model": "resnet18",
    "data-dir": os.path.join(IR_VIS_DATASET_PATH, 'pytorch_models_structure'),
    "lr": 0.6,
    "warmup-epochs": 5,
    "epochs": 240,
    "weight-decay": 1e-4,
    "sched": "cosine",
    "reprob": 0.4,
    "recount": 3,
    "remode": "pixel",
    "aa": "rand-m7-mstd0.5-inc1",
    "batch-size": 192,
    "workers": 24,
    "amp": "parameter_without_value",
    "dist-bn": "reduce",
}

configs = []

test_parameters = {

}

MAIN_SCRIPT = f"torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:{random.randint(0,1000)} --nproc_per_node=1 train.py"

for data in ['day', 'day_dawn', 'day_dawn_night', 'all']:
    config = {
        "data-dir": os.path.join(IR_VIS_DATASET_PATH, 'pytorch_models_structure', data),
        "model": 'resnet18',
        "tag": f"{base_tag}_{data}",
        "no-aug": "parameter_without_value",
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
