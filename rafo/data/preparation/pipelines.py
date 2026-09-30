from pathlib import Path

from rafo.data.data_access import DataAccess
from rafo.data.preparation.steps import (
    DataPreparationStep,
    Download,
    Parse,
    ExtractIFEs,
    CalculateSecondaryStructures,
    SaveAsPDB,
    CreateSeqsFASTA,
    ClusterSequences,
    RemoveSequenceSimilarityOverlap,
)
from rafo.data.preparation.filters import (
    create_meta_filter,
    create_quality_filter,
    create_struct_depth_filter,
    create_long_range_contact_coverage_filter,
    create_chain_clash_filter,
    create_backbone_break_filter,
)


class DataPreparationPipeline:
    def __init__(self, *steps: DataPreparationStep):
        super().__init__()
        self.steps = steps

    def __call__(self, data_access: DataAccess):
        for step in self.steps:
            step(data_access)


class TrainingDataPreparationPipeline(DataPreparationPipeline):
    def __init__(
        self,
        num_workers: int,
        rna3dhub_version: str,
        max_resolution: float,
        min_seq_len: int,
        max_seq_len: int,
        min_max_chain_len: int,
        max_undef_seq_perc: float,
        min_modeled_res: int,
        min_struct_depth: float,
        lrc_dist_threshold: float,
        lrc_ignore_neighbors: int,
        lrc_min_coverage: float,
        backbone_break_dist_threshold: float,
        cutoff_date: str,
        dup_min_seq_id: float,
        dup_coverage: float,
    ):
        super().__init__(
            Download(
                num_workers=num_workers,
                rna3dhub_version=rna3dhub_version,
            ),
            Parse(num_workers=num_workers),
            create_meta_filter(
                max_resolution=max_resolution,
                cutoff_date=cutoff_date,
                is_training=True
            ),
            ExtractIFEs(),
            create_quality_filter(
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                min_max_chain_len=min_max_chain_len,
                max_undef_seq_perc=max_undef_seq_perc,
                min_modeled_res=min_modeled_res,
            ),
            create_struct_depth_filter(
                min_depth=min_struct_depth,
            ),
            create_long_range_contact_coverage_filter(
                dist_threshold=lrc_dist_threshold,
                ignore_neighbors=lrc_ignore_neighbors,
                min_coverage=lrc_min_coverage,
            ),
            create_chain_clash_filter(),
            create_backbone_break_filter(
                break_dist_threshold=backbone_break_dist_threshold
            ),
            ClusterSequences(
                min_seq_id=dup_min_seq_id,
                coverage=dup_coverage,
                threads=num_workers,
                split="train",
            ),
            CalculateSecondaryStructures(
                num_workers=num_workers,
            ),
            SaveAsPDB(),
            CreateSeqsFASTA(),
        )


class SeqTestDataPreparationPipeline(DataPreparationPipeline):
    def __init__(
        self,
        num_workers: int,
        rna3dhub_version: str,
        max_resolution: float,
        min_seq_len: int,
        max_seq_len: int,
        min_max_chain_len: int,
        max_undef_seq_perc: float,
        min_modeled_res: int,
        min_struct_depth: float,
        lrc_dist_threshold: float,
        lrc_ignore_neighbors: int,
        lrc_min_coverage: float,
        backbone_break_dist_threshold: float,
        cutoff_date: str,
        dup_min_seq_id: float,
        dup_coverage: float,
        train_structs_dir: str,
        overlap_min_seq_id: float,
        overlap_coverage: float,
    ):
        super().__init__(
            Download(
                num_workers=num_workers,
                rna3dhub_version=rna3dhub_version,
            ),
            Parse(num_workers=num_workers),
            create_meta_filter(
                max_resolution=max_resolution,
                cutoff_date=cutoff_date,
                is_training=False
            ),
            ExtractIFEs(),
            create_quality_filter(
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                min_max_chain_len=min_max_chain_len,
                max_undef_seq_perc=max_undef_seq_perc,
                min_modeled_res=min_modeled_res,
            ),
            create_struct_depth_filter(
                min_depth=min_struct_depth,
            ),
            create_long_range_contact_coverage_filter(
                dist_threshold=lrc_dist_threshold,
                ignore_neighbors=lrc_ignore_neighbors,
                min_coverage=lrc_min_coverage,
            ),
            create_chain_clash_filter(),
            create_backbone_break_filter(
                break_dist_threshold=backbone_break_dist_threshold
            ),
            ClusterSequences(
                min_seq_id=dup_min_seq_id,
                coverage=dup_coverage,
                threads=num_workers,
            ),
            RemoveSequenceSimilarityOverlap(
                train_structs_dir=Path(train_structs_dir),
                min_seq_id=overlap_min_seq_id,
                coverage=overlap_coverage,
                threads=num_workers,
            ),
            CalculateSecondaryStructures(
                num_workers=num_workers,
            ),
            SaveAsPDB(),
            CreateSeqsFASTA(),
        )
