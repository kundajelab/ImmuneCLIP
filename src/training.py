import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from lightning.pytorch.accelerators import find_usable_cuda_devices
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning.loggers as log
from lightning.pytorch.strategies import DDPStrategy

import argparse
import os
import wandb

from .alignment import CLIPModel
from .data_module import EpitopeReceptorDataModule
from .configs import get_lightning_config, get_projection_config
from .logging import call_backs, combine_args_and_configs


def setup_parser():

    # Command line interface arguments and parsing
    parser = argparse.ArgumentParser(description='argument parser for training')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--grad-accum', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay parameter for AdamW algorithm')
    parser.add_argument('--random-seed', type=int, default=14, help='Random seed for reproducibility')

    # WandB configuration:
    parser.add_argument('--entity', type=str, default='lordim', help='entity name')
    parser.add_argument('--project', type=str, default='clip_antibody', help='project name')
    parser.add_argument('--group', type=str, default='clipbody_test', help='group name')
    parser.add_argument('--run-id', type=str, default='clipbody_test', help='run id')
    parser.add_argument('--use-wandb', default=False, action='store_true', help='use wandb for logging')

    # Training and Data configuration
    parser.add_argument('--receptor-model-name', type=str, default='ablang', help='name of the receptor foundation model')
    parser.add_argument('--receptor-type', type=str, default='TCR', help='Is the receptor BCR or TCR')
    parser.add_argument('--include-mhc', default=False, action='store_true', help='include MHC sequences alongside epitope in the training data')
    parser.add_argument('--mhc-groove-only', default=False, action='store_true', help='only include A1-A2 domains for class I MHC, A1-B1 domains for class II MHC')
    parser.add_argument('--unique-epitopes', default=False, action='store_true', help='split the data based on unique epitopes')
    parser.add_argument('--no-lora', default=False, action='store_true', help='do not use LoRA adapter matrices for the models')
    parser.add_argument('--regular-ft', default=False, action='store_true', help='use regular fine-tuning')
    parser.add_argument('--mask-seqs', default=False, action='store_true', help='mask the sequences for training')
    parser.add_argument('--mask-prob', type=float, default=0.15, help='probability of masking a residue')
    parser.add_argument('--mse-weight', type=float, default=0., help='weight for the MSE loss')
    parser.add_argument('--weigh-epitope-count', default=False, action='store_true', help='weight the epitope count in the clip loss')
    parser.add_argument('--swe-pooling', default=False, action='store_true', help='use SWE pooling for sequence embeddings')
    parser.add_argument('--hidden-dim', type=int, default=None, help='dimension of the hidden layer')
    parser.add_argument('--projection-dim', type=int, default=None, help='dimension of the projection layer')
    parser.add_argument('--lightning-config-name', type=str, default='default')
    parser.add_argument('--dataset-path', type=str, required=True, help='path to the dataset')
    parser.add_argument('--mhc-path', type=str, default=None, help='path to file with MHC sequence info. Required if --include-mhc is set to True')
    parser.add_argument('--oversample', default=False, action='store_true', help='oversample the epitopes with few receptor data')
    parser.add_argument('--fewshot-ratio', type=float, default=None, help='ratio of few-shot data to the total data')
    parser.add_argument('--lr-scheduler', type=str, default='cos_anneal', help='learning rate scheduler')

    # PyTorch Lightning configuration
    parser.add_argument('--torch-device', type=str, default='gpu')
    parser.add_argument('--output-dir', type=str, required=True, help='wandb and checkpoint output')
    parser.add_argument('--num-gpus', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--max-epochs', type=int, default = 1, required=False)
    parser.add_argument('--gpus-used', type=int, nargs='+', required=False, help='which GPUs used for env variable CUDA_VISIBLE_DEVICES')
    parser.add_argument('--stage', type=str, default='fit', help='stage of training')
    parser.add_argument('--check-val-every-n-epoch', type=int, default=1, help='check validation every n epochs')
    parser.add_argument('--val-check-interval', type=float, default=1.0, help='validation check interval')
    parser.add_argument('--save-ckpts', default=False, action='store_true', help='save checkpoints')
    parser.add_argument('--from-checkpoint', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--save-embed-path', type=str, default=None, help='path to save embeddings for eval')

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    # utilizing Tensor Cores:
    torch.set_float32_matmul_precision('high')

    # setting up the environment variables:
    os.environ["TOKENIZERS_PARALLELISM"] = "true" # resolving tokenizers parallelism issue

    args = setup_parser()

    # retrieve the configs:
    lightning_config = get_lightning_config()
    model_config = get_projection_config(args.receptor_model_name)

    # update configs based on input arguments:
    combine_args_and_configs(args, [lightning_config, model_config])

    # setup callbacks:
    if args.stage == 'fit' and args.output_dir is not None:
        if not os.path.exists(os.path.join(args.output_dir, args.run_id)):
            os.makedirs(os.path.join(args.output_dir, args.run_id))
        
        output_dir = os.path.join(args.output_dir, args.run_id)

        # get callbacks
        callbacks = call_backs(output_dir, args.save_ckpts)
    else:
        callbacks = None

    # construct PyTorch Lightning Module:
    if args.receptor_type == 'TCR':
        print("Using TCR data!")
    else:
        print("Using BCR data!")
    tsv_file_path = args.dataset_path
    mhc_file_path = args.mhc_path
    
    if args.mask_seqs:
        print("WARNING: Partially making sequence residues during training")
    
    if args.unique_epitopes:
        print("WARNING: Splitting data based on unique epitopes")

    pl_datamodule = EpitopeReceptorDataModule(tsv_file_path, mhc_file=mhc_file_path, ln_cfg=lightning_config,
                                              batch_size=lightning_config.batch_size, include_mhc=lightning_config.include_mhc,
                                              model_config=model_config, random_seed=args.random_seed)

    # construct the CLIP model:
    clip_model = CLIPModel(lightning_config, model_config)

    if rank_zero_only.rank == 0:
        # initalize wandb:
        if args.use_wandb:
            run = wandb.init(project=args.project,
                            entity=args.entity,
                            group=args.group,
                            dir=output_dir,
                            name=args.run_id,
                            id=args.run_id,
                            resume=True if args.from_checkpoint else None,
                            )
            
            run_output_dir = run.dir
            wandb_logger = log.WandbLogger(save_dir=run_output_dir, log_model=False)
            wandb_logger.watch(clip_model)

        if len(args.gpus_used) > 1:
            strat = 'ddp'
            if args.regular_ft:
                strat = 'ddp_find_unused_parameters_true'
        else:
            strat = 'auto'
        # build PyTorch Lightning Trainer:
        trainer = Trainer(max_epochs=args.max_epochs,
                        logger=wandb_logger if args.use_wandb else None,
                        accelerator=args.torch_device,
                        devices=args.gpus_used if args.gpus_used else 1, #TODO: smooth CPU/GPU conversion
                        enable_progress_bar=True,
                        callbacks=callbacks if callbacks is not None else None,
                        accumulate_grad_batches=args.grad_accum,
                        reload_dataloaders_every_n_epochs=1 if args.oversample else 0,
                        strategy=strat,
                        )
    else:
        strat = 'ddp'
        if args.regular_ft:
            strat = 'ddp_find_unused_parameters_true'
        # build PyTorch Lightning Trainer:
        trainer = Trainer(max_epochs=args.max_epochs,
                        logger=None,
                        accelerator=args.torch_device,
                        devices=args.gpus_used if args.gpus_used else 1, #TODO: smooth CPU/GPU conversion
                        enable_progress_bar=True,
                        callbacks=callbacks if callbacks is not None else None,
                        accumulate_grad_batches=args.grad_accum,
                        reload_dataloaders_every_n_epochs=1 if args.oversample else 0,
                        strategy=strat,
                        )

    # run the model:
    if args.stage == 'fit':
        print('Start Training...')
        trainer.fit(model=clip_model, datamodule=pl_datamodule, ckpt_path=args.from_checkpoint)
    else:
        print("**********************")
        print("* Inference Mode...  *")
        print("**********************")
        trainer.test(model=clip_model, datamodule=pl_datamodule, ckpt_path=args.from_checkpoint)
    