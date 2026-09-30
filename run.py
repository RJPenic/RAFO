import argparse
from pathlib import Path
from datetime import timedelta

import torch

import lightning.pytorch as pl

from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from rafo.utils.arg_parsing import (
    is_multiple_devices, update_args_with_parser_default_vals
)

from rafo.data.datamodule import RAFODataModule
from rafo.lightning.model_wrapper import RNAStructurePredictionWrapper
from rafo.config import get_config


def main(args):
    # Initialize random number generation
    if args.seed:
        pl.seed_everything(args.seed, workers=True)

    # Create output directory if it doesn't exist
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Model
    config = get_config(args.config, args.finetune)
    model = RNAStructurePredictionWrapper(config=config)

    # Set performance optimization parameters (for evaluation)
    model.set_chunk_size(args.chunk_size)
    model.set_use_lma(args.use_lma)

    # Activate dropout during inference
    model.set_inference_dropout(args.dropout)

    # Override the number of recycling cycles at inference
    # (None = use config)
    model.set_inference_no_cycles(args.no_cycles)

    # Load initial parameters
    if args.init_params:
        checkpoint = torch.load(args.init_params, map_location="cpu")

        if "state_dict" in checkpoint:
            # PL checkpoint
            if "ema" in checkpoint:
                # EMA weights used both as the live weights and as the
                # initial EMA state. Only trainable parameters are averaged,
                # hence the non-strict.
                model.model.load_state_dict(
                    checkpoint["ema"]["params"], strict=False
                )
            else:
                model.load_state_dict(checkpoint["state_dict"])
        else:
            # PT file
            model.model.load_state_dict(checkpoint, strict=False)

        if model.ema is not None:
            model.ema.reset(model.model)

    # Datamodule
    datamodule = RAFODataModule(
        train_data_dir=args.train_data_dir,
        val_data_dirs=args.val_data_dirs,
        test_data_dir=args.test_data_dir,
        max_train_residues=config.training.max_train_residues,
        pred_fasta=args.fasta_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # Callbacks
    callbacks = []

    if args.checkpoint_every_epoch:
        epoch_ckpt_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename='train-rna-epoch-{epoch}-{step}',
            every_n_epochs=1,
            save_top_k=-1
        )
        callbacks.append(epoch_ckpt_callback)

    if args.checkpoint_every_n_steps:
        step_ckpt_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename='train-rna-step-{epoch}-{step}',
            every_n_train_steps=args.checkpoint_every_n_steps,
            save_top_k=-1
        )
        callbacks.append(step_ckpt_callback)

    if args.checkpoint_every_hour:
        time_ckpt_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename='train-rna-hourly-{epoch}-{step}',
            train_time_interval=timedelta(hours=1.0),
            save_top_k=1
        )
        callbacks.append(time_ckpt_callback)

    # Loggers
    loggers = []

    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.wandb_experiment_name,
            save_dir=args.output_dir,
            project=args.wandb_project,
            entity=args.wandb_entity,
            offline=args.wandb_offline,
            save_code=True,
            config=config,
        )
        loggers.append(wandb_logger)

    if loggers:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    strategy = "auto"
    if is_multiple_devices(args.devices):
        strategy = DDPStrategy(find_unused_parameters=False)

    # Training
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_algorithm=config.training.gradient_clip_algorithm,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        strategy=strategy,
        detect_anomaly=args.detect_anomaly,
    )

    if args.command == "fit":
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=args.ckpt_path
        )
    elif args.command == "test":
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=args.ckpt_path
        )
    elif args.command == "predict":
        model.set_inference_seed(args.seed)

        trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=args.ckpt_path
        )
    else:
        raise ValueError("Unrecognized command!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ribonucleic Acid Folding Oracle (RAFO)"
    )

    # - - Command-agnostic arguments - -
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parent_parser.add_argument(
        "--output_dir", type=str, default=None,
        help="""
            Directory for all the output files (checkpoints, logs, temporary
            files, etc.)
        """
    )

    # Model initialization
    parent_parser.add_argument(
        "--config", type=str, default="frodo",
        help="Model and loss configuration setting"
    )
    parent_parser.add_argument(
        "--init_params", type=str, default=None,
        help="""
            Path to the '.pt' or '.ckpt' file containing model weights
            that will be used as the starting point for the training/evaluation
        """
    )
    parent_parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="""
            Path of the checkpoint from which the training is resumed
            or that is used for evaluation/prediction
        """
    )

    # Data loading
    parent_parser.add_argument(
        "--num_workers", type=int, default=0,
        help="How many subprocesses to use for data loading"
    )
    parent_parser.add_argument(
        "--pin_memory", action="store_true", default=False,
        help="""
            If activated, the data loader will copy Tensors into
            device/CUDA pinned memory before returning them
        """
    )

    # Memory optimizations
    parent_parser.add_argument(
        "--chunk_size", type=int, default=None,
        help="""
            Use chunking memory optimization described in AlphaFold2 paper
            with the given chunk size.
        """
    )
    parent_parser.add_argument(
        "--use_lma", action="store_true", default=False,
        help="Whether to use low-memory attention during evaluation."
    )
    parent_parser.add_argument(
        "--no_cycles", type=int, default=None,
        help="""
            Number of recycling iterations to use at inference (validation,
            test, predict). If unset, the value from the model config is used.
            Ignored during training.
        """
    )

    # Trainer
    parent_parser.add_argument(
        "--accelerator", type=str, default='auto',
        help="""
            Supports passing different accelerator types (“cpu”, “gpu”, “tpu”,
            “ipu”, “hpu”, “mps”, “auto”)
        """
    )
    parent_parser.add_argument(
        "--devices", type=str, default='auto',
        help="The devices to use"
    )
    parent_parser.add_argument(
        "--precision", type=str, default='bf16-mixed',
        help="""
            Double precision, full precision, 16bit mixed precision or
            bfloat16 mixed precision
        """
    )

    # - - Command specification - -
    subparsers = parser.add_subparsers(
        dest="command", title="Available commands", required=True
    )

    # - - Fit - -
    fit_parser = subparsers.add_parser(
        "fit",
        help="Train the model", parents=[parent_parser]
    )

    fit_parser.add_argument(
        "train_data_dir", type=str, default=None,
        help="Directory containing training data structures"
    )
    fit_parser.add_argument(
        "--val_data_dirs", type=str, nargs="+", default=None,
        help="Directories containing validation data structures"
    )

    # Configuration
    fit_parser.add_argument(
        "--finetune", action="store_true", default=False,
        help="Use fine-tuning configuration setting"
    )

    # Checkpointing
    fit_parser.add_argument(
        "--checkpoint_every_epoch", action="store_true", default=False,
        help="Whether to checkpoint at the end of every training epoch"
    )
    fit_parser.add_argument(
        "--checkpoint_every_n_steps", type=int, default=None,
        help="Number of training steps between checkpoints"
    )
    fit_parser.add_argument(
        "--checkpoint_every_hour", action="store_true", default=False,
        help="""
            Whether to checkpoint every hour during the training (each
            checkpoint overwrites the last one)
        """
    )

    # W&B
    fit_parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Whether to log metrics to Weights & Biases"
    )
    fit_parser.add_argument(
        "--wandb_offline", action="store_true", default=False,
        help="Run logging offline"
    )
    fit_parser.add_argument(
        "--wandb_experiment_name", type=str, default=None,
        help="Name of the current experiment. Used for wandb logging"
    )
    fit_parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Name of the wandb project to which this run will belong"
    )
    fit_parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="Wandb username or team name to which runs are attributed"
    )
    fit_parser.add_argument(
        "--log_every_n_steps", type=int, default=50,
        help="How often to log within steps"
    )

    # Trainer
    fit_parser.add_argument(
        "--max_steps", type=int, default=-1,
        help="Stop training after this number of steps"
    )
    fit_parser.add_argument(
        "--max_epochs", type=int, default=-1,
        help=" Stop training once this number of epochs is reached"
    )
    fit_parser.add_argument(
        "--detect_anomaly", action="store_true", default=False,
        help="Enable anomaly detection for the autograd engine"
    )
    fit_parser.add_argument(
        "--check_val_every_n_epoch", type=int, default=1,
        help="Perform validation loop after every N training epochs"
    )

    # Data loading
    fit_parser.add_argument(
        "--batch_size", type=int, default=1,
        help="How many samples per batch to load"
    )
    fit_parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1,
        help="""
        Accumulate gradients over this many batches before each optimizer step
        """
    )

    # - - Test - -
    test_parser = subparsers.add_parser(
        "test",
        help="Evaluate the model on the test set", parents=[parent_parser]
    )
    test_parser.add_argument(
        "test_data_dir", type=str, default=None,
        help="Directory containing test data structures"
    )

    # - - Predict - -
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict structures for given RNAs", parents=[parent_parser]
    )
    predict_parser.add_argument(
        "fasta_file", type=str, default=None,
        help="FASTA file with RNA sequences"
    )
    predict_parser.add_argument(
        "--dropout", action="store_true", default=False,
        help="Activate dropout layers"
    )

    args = parser.parse_args()
    args = update_args_with_parser_default_vals(
        args, fit_parser, predict_parser, test_parser
    )

    main(args)
