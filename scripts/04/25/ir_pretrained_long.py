"""
train series of mobilenets on 100 subclasses of imagenet
"""

import os
import random
import sys
sys.path.append(".")
from script_manager.func.script_boilerplate import do_everything
from local_config import IR_VIS_DATASET_PATH2 as IR_VIS_DATASET_PATH

# weights and biases project name
WANDB_PROJECT_NAME = "python-image-models"
base_tag = os.path.split(__file__)[-1].split('.')[0]
# keys
appendix_keys = ["tag"]
extra_folder_keys = []

default_parameters = {
    "__script_output_arg__": "output",
    "model": "mobilenetv3_small_100", 
    "aa": "rand-m9-mstd0.5",
    "amp": "parameter_without_value",
    "batch-size": 64, #-b 512 
    "data-dir": os.path.join(IR_VIS_DATASET_PATH, 'pytorch_models_structure'),
    "decay-epochs": 2.4,
    "decay-rate": 0.973,
    "drop": 0.2,
    "drop-path": 0.2,
    "epochs": 600,
    "lr": 0.064,
    "lr-noise": "0.42 0.9",
    "model-ema":  "parameter_without_value",
    "model-ema-decay": 0.9999,
    "opt": "rmsproptf",
    "opt-eps": 0.001,
    "remode": "pixel",
    "reprob": 0.2,
    "sched": "step", #--sched step 
    "warmup-lr": 1e-6,
    "weight-decay": 1e-5,
    "workers": 1,
    "test-split": "test"
}

configs = []

test_parameters = {

}

MAIN_SCRIPT = f"torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:{random.randint(0,1000)} --nproc_per_node=1 train.py"
for model in ['resnet18']:
    config = {
        "model": model,
        "tag": f"{base_tag}_{model}",
        "pretrained": "parameter_without_value",
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
