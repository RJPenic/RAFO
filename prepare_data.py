import argparse
from pathlib import Path

from rafo.data.data_access import DataAccess
from rafo.data.preparation.pipelines import (
    TrainingDataPreparationPipeline, SeqTestDataPreparationPipeline
)
from rafo.utils.arg_parsing import update_args_with_parser_default_vals

import logging


def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Logging
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )

    # Data output paths
    data_access = DataAccess(
        data_root_dir=args.output_dir,
    )

    # Create and run data preparation pipeline

    if args.command == "train":
        pipeline = TrainingDataPreparationPipeline(
            num_workers=args.num_workers,
            rna3dhub_version=args.rna3dhub_version,
            max_resolution=args.max_resolution,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            min_max_chain_len=args.min_max_chain_len,
            max_undef_seq_perc=args.max_undef_seq_perc,
            min_modeled_res=args.min_modeled_res,
            min_struct_depth=args.min_struct_depth,
            lrc_dist_threshold=args.lrc_dist_threshold,
            lrc_ignore_neighbors=args.lrc_ignore_neighbors,
            lrc_min_coverage=args.lrc_min_coverage,
            backbone_break_dist_threshold=args.backbone_break_dist_threshold,
            cutoff_date=args.cutoff_date,
            dup_min_seq_id=args.dup_min_seq_id,
            dup_coverage=args.dup_coverage,
        )
    elif args.command == "seq-test":
        pipeline = SeqTestDataPreparationPipeline(
            num_workers=args.num_workers,
            rna3dhub_version=args.rna3dhub_version,
            max_resolution=args.max_resolution,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            min_max_chain_len=args.min_max_chain_len,
            max_undef_seq_perc=args.max_undef_seq_perc,
            min_modeled_res=args.min_modeled_res,
            min_struct_depth=args.min_struct_depth,
            lrc_dist_threshold=args.lrc_dist_threshold,
            lrc_ignore_neighbors=args.lrc_ignore_neighbors,
            lrc_min_coverage=args.lrc_min_coverage,
            backbone_break_dist_threshold=args.backbone_break_dist_threshold,
            cutoff_date=args.cutoff_date,
            dup_min_seq_id=args.dup_min_seq_id,
            dup_coverage=args.dup_coverage,
            train_structs_dir=args.train_structs_dir,
            overlap_min_seq_id=args.overlap_min_seq_id,
            overlap_coverage=args.overlap_coverage,
        )
    else:
        raise ValueError("Unrecognized command!")

    pipeline(data_access)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation")

    # - - Command-agnostic arguments - -
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument(
        "output_dir", type=str, default=None,
        help="""
            Directory in which the loadable training and evaluation datasets
            will be created
        """
    )
    parent_parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of worker processes used during data preparation"
    )
    parent_parser.add_argument(
        "--log_file", type=str, default=None,
        help="Open the specified file and use it as stream for logging",
    )

    # - - Command specification - -
    subparsers = parser.add_subparsers(
        dest="command", title="Available commands", required=True
    )

    # - - Train / test shared arguments parser - -
    train_test_parser = argparse.ArgumentParser(add_help=False)

    train_test_parser.add_argument(
        "--rna3dhub_version", type=str, default=None,
        help="RNA 3D Hub version",
    )
    train_test_parser.add_argument(
        "--max_resolution", type=float, default=20.0,
        help="Maximum structure resolution",
    )
    train_test_parser.add_argument(
        "--min_seq_len", type=int, default=0,
        help="Minimum number of residues",
    )
    train_test_parser.add_argument(
        "--max_seq_len", type=int, default=999_999,
        help="Maximum number of residues"
    )
    train_test_parser.add_argument(
        "--max_undef_seq_perc", type=float, default=1.0,
        help="Maximum percentage of undefined/unknown residues",
    )
    train_test_parser.add_argument(
        "--min_modeled_res", type=int, default=3,
        help="Minimum number of modeled residues"
    )
    train_test_parser.add_argument(
        "--min_struct_depth", type=float, default=0.0,
        help="Minimum structure depth",
    )
    train_test_parser.add_argument(
        "--lrc_dist_threshold", type=float, default=999.0,
        help="Distance threshold for long-range contact coverage",
    )
    train_test_parser.add_argument(
        "--lrc_ignore_neighbors", type=int, default=0,
        help="""
            Number of neighboring residues to ignore for
            long-range contact coverage
        """,
    )
    train_test_parser.add_argument(
        "--lrc_min_coverage", type=float, default=0.0,
        help="Minimum long-range contact coverage",
    )
    train_test_parser.add_argument(
        "--backbone_break_dist_threshold", type=float, default=9999.0,
        help="Distance threshold for backbone break filtering",
    )
    train_test_parser.add_argument(
        "--cutoff_date", type=str, required=True,
        help="Cut-off release date",
    )
    train_test_parser.add_argument(
        "--dup_min_seq_id", type=float, default=1.0,
        help=" Minimum sequence identity for MMseqs2 duplicates clustering",
    )
    train_test_parser.add_argument(
        "--dup_coverage", type=float, default=1.0,
        help="Coverage for MMseqs2 duplicates clustering",
    )
    train_test_parser.add_argument(
        "--min_max_chain_len", type=int, default=0,
        help="Minimum maximum chain length in an IFE"
    )

    # - - Training data preparation - -
    train_parser = subparsers.add_parser(
        "train",
        help="Prepare training data",
        parents=[parent_parser, train_test_parser]
    )

    # - - Abstract test command (parent for seq-test) - -
    test_parser = argparse.ArgumentParser(
        add_help=False,
        parents=[parent_parser, train_test_parser]
    )

    test_parser.add_argument(
        "--train_structs_dir", type=str, required=True,
        help="Directory with parsed training structures"
    )

    # - - Sequence-based test data preparation - -
    seq_test_parser = subparsers.add_parser(
        "seq-test",
        help="Prepare sequence-based test data",
        parents=[test_parser]
    )

    seq_test_parser.add_argument(
        "--overlap_min_seq_id", type=float, default=1.0,
        help="Minimum sequence identity for filtering sequence overlap"
    )
    seq_test_parser.add_argument(
        "--overlap_coverage", type=float, default=1.0,
        help="Coverage for filtering sequence overlap"
    )

    args = parser.parse_args()
    args = update_args_with_parser_default_vals(
        args, train_parser, test_parser, seq_test_parser
    )

    main(args)
