"""Module to train every type of model. Has different options.
"""

"""Script to train the stroke estimation model.
"""

import argparse
import os

from derenderer.common import load_yaml
from derenderer.train_ddp_strokes import main as main_ddp
from derenderer.train_binarize import main as main_binarize
from derenderer.train_strokes import main as main_strokes

import torch
import torch.multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", "--config", 
                        default="./configs/train_strokes.yaml",
                        help="Config .yaml file.")
    
    desc = """
    Choose the training task: 
    - 'train_binary': Train the binarization model. Use train_binarize.yaml.
    - 'train_ddp': Train the strokes estimation model with multiple GPUs.
    - 'train_strokes': Train the strokes estimation model with a single GPU.
    """
    parser.add_argument("-task", "-task", 
                        required=True, 
                        default="train_binary",
                        help=desc)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    config_path = args.config
    configs = load_yaml(config_path)

    # Restricts the CUDA devices available to these:
    visible_devices = configs["visible_devices"]
    visible_str = ', '.join([str(x) for x in visible_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_str

    task = args.task
    if task == "train_ddp":
        # Use allocated available devices:
        world_size = torch.cuda.device_count()
        # All arguments that go to main:
        mp.spawn(main_ddp, args=(world_size, configs), nprocs=world_size)
    
    elif task == "train_binary":
        output = main_binarize(configs)
    
    elif task == "train_strokes":
        output = main_strokes(configs)
