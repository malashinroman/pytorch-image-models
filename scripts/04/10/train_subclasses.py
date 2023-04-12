sys.path.append(".")
from local_config import IMAGENET_PATH
from script_manager.func.script_parse_args import get_script_args
from script_manager.func.script_boilerplate import do_everything
import os
import sys

args = get_script_args()

# weights and biases project name
wandb_project_name = "lac_ppo"
base_tag = os.path.split(__file__)[-1].split('.')[0]
# keys
appendix_keys = ["tag"]
extra_folder_keys = []
# ./distributed_train.sh 2 /imagenet -b 64
# --aa rand-m9-mstd0.5-inc1 --amp --aug-splits 3 --dist-bn reduce --epochs 200 --jsd --lr 0.05 --model resnet50 --remode pixel --reprob 0.6 --resplit --sched cosine --split-bn

default_parameters = {
    "aa": "rand-m9-mstd0.5-inc1",
    "amp": "parameter_without_value",
    "aug-splits": 3,
    "batch-size": 64,
    "dist-bn": "reduce",
    "epochs": 200,
    "jsd": True,
    "lr": 0.05,
    "model": "resnet50s",
    "remode": "pixel",
    "reprob": 0.6,
    "resplit": True,
    "sched": "cosine",
    "split-bn": True,
}

configs = []

test_parameters = {
    # "epochs": 1,
    # "train_set_size": 1000,
    # "test_set_size": 1000,
    # "skip_validation": True
}

main_script = f"torchrun --nproc_per_node=2 train.py {os.path.join(IMAGENET_PATH, 'pytorch_models_structure')}"

for i in range(0, 10):
    config = {
        "tag": f"{base_tag}_class_map{i}",
        "class_map": f"class_maps\/class_map{i}.txt"
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
        main_script=main_script,
        test_parameters=test_parameters,
        wandb_project_name=wandb_project_name,
        script_file=__file__,
    )
