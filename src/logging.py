import os

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from typing import List
from argparse import Namespace
from dataclasses import is_dataclass

def call_backs(output_dir, save_ckpts=False):
    lrmonitor_callback = LearningRateMonitor(logging_interval='step')

    #Example 2: other options
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename="model-{epoch:03d}-{val_loss:.4f}",
        save_top_k=2,
        mode='min',
        save_last=True,
    )

    if save_ckpts:
        callbacks = [checkpoint_callback, lrmonitor_callback]
    else:
        callbacks = [lrmonitor_callback]
    
    return callbacks

def combine_args_and_configs(args: Namespace, dataclasses: List):
    if not isinstance(args, dict):
        args = vars(args).items()
    else:
        args = args.items()
    for name, value in args:
        if value is not None:
            for obj in dataclasses:
                if is_dataclass(obj) and hasattr(obj, name):
                    print("overwriting default", name, value)
                    setattr(obj, name, value)