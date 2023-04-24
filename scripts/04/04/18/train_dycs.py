"""
train series of mobilenets on 100 subclasses of imagenet
"""

import os
import random
import sys
sys.path.append(".")
from script_manager.func.script_boilerplate import do_everything
from local_config import IMAGENET_PATH

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
    "model": "dycs_mobilenetv3_large_100",
    "batch-size": 64,
    "sched": "step",
    "epochs": 600,
    "decay-epochs": 2.4,
    "workers": 24,
    "decay-rate": 0.973,
    "opt": "rmsproptf",
    "opt-eps": 0.001,
    "warmup-lr": 1e-6,
    "weight-decay": 1e-5,
    "drop": 0.2,
    "drop-path": 0.2,
    "model-ema":  "parameter_without_value",
    "model-ema-decay": 0.9999,
    "aa": "rand-m9-mstd0.5",
    "remode": "pixel",
    "reprob": 0.2,
    "amp": "parameter_without_value",
    "lr": 0.064,
    "lr-noise": "0.42 0.9",
    "data-dir": os.path.join(IMAGENET_PATH, 'pytorch_models_structure'),
}

configs = []

test_parameters = {

}

# "    --rdzv_backend=c10d" \
# "    --rdzv_endpoint=localhost:{random.randint(0,1000)}" \
# "   train.py"
MAIN_SCRIPT = f"torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:{random.randint(0,1000)} --nproc_per_node=3 train.py"
config = {
    "tag": f"{base_tag}_",
    "initial-checkpoint": 'checkpoints/10_networks/'
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