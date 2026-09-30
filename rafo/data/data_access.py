from pathlib import Path
from typing import Optional, Union

from rafo.data.default_local_paths import (
    DEFAULT_RAW_DIR,
    DEFAULT_PARSED_DIR,
    DEFAULT_PDBS_DIR,
    DEFAULT_SS_DIR,
    DEFAULT_TRAIN_CLUSTER_JSON,
    DEFAULT_RNA3DHUB_NRLIST_CSV,
    DEFAULT_MAX_TRAIN_SIM_CSV,
    DEFAULT_SEQS_FASTA,
)


class DataAccess:
    def __init__(
        self,
        data_root_dir: Union[str, Path],
        raw_dir: Optional[Union[str, Path]] = None,
        parsed_dir: Optional[Union[str, Path]] = None,
        pdbs_dir: Optional[Union[str, Path]] = None,
        ss_dir: Optional[Union[str, Path]] = None,
        train_cluster_json: Optional[Union[str, Path]] = None,
        rna3dhub_nrlist_csv: Optional[Union[str, Path]] = None,
        max_train_sim_csv: Optional[Union[str, Path]] = None,
        seqs_fasta: Optional[Union[str, Path]] = None,
    ):
        self.data_root_dir = Path(data_root_dir)

        self.raw_dir = Path(
            raw_dir or
            self.data_root_dir / DEFAULT_RAW_DIR
        )

        self.parsed_dir = Path(
            parsed_dir or
            self.data_root_dir / DEFAULT_PARSED_DIR
        )

        self.pdbs_dir = Path(
            pdbs_dir or
            self.data_root_dir / DEFAULT_PDBS_DIR
        )

        self.ss_dir = Path(
            ss_dir or
            self.data_root_dir / DEFAULT_SS_DIR
        )

        self.train_cluster_json = Path(
            train_cluster_json or
            self.data_root_dir / DEFAULT_TRAIN_CLUSTER_JSON
        )

        self.rna3dhub_nrlist_file = Path(
            rna3dhub_nrlist_csv or
            self.data_root_dir / DEFAULT_RNA3DHUB_NRLIST_CSV
        )

        self.max_train_sim_file = Path(
            max_train_sim_csv or
            self.data_root_dir / DEFAULT_MAX_TRAIN_SIM_CSV
        )

        self.seqs_file = Path(
            seqs_fasta or
            self.data_root_dir / DEFAULT_SEQS_FASTA
        )
