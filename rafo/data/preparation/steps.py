from abc import ABC, abstractmethod
from typing import Optional, Callable
from pathlib import Path
import pickle
from collections import defaultdict
import itertools
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
from Bio.PDB import PDBList
import json

from rafo.utils.data_structures import MultiRNA
from rafo.data.data_access import DataAccess
from rafo.utils.structure_parsing import parse_struct_file
from rafo.utils.rna3dhub import (
    download_rna3dhub_nrlist_csv, load_rna3dhub_nrlist_csv,
    get_latest_rna3dhub_nrlist_version
)
from rafo.utils.sec_struct import get_single_seq_ss_feats
from rafo.utils.save import save_multirna_to_pdb

from rafo.utils.external_tools import (
    run_mmseqs_easy_cluster, run_mmseqs_easy_search
)
from rafo.constants import SEQ_DELIMITER

import logging

logger = logging.getLogger(__name__)


class DataPreparationStep(ABC):
    @abstractmethod
    def __call__(self, data_access: DataAccess) -> None:
        pass


class Download(DataPreparationStep):
    MAX_ATTEMPTS = 3
    MAX_WORKERS = 10

    def __init__(
        self,
        num_workers: int,
        rna3dhub_version: Optional[str] = None,
        pdb_ids: Optional[list[str]] = None,
        skip_rna3dhub_download: bool = False,
    ):
        super().__init__()

        # Download starts behaving weirdly when there's too many workers
        self.num_workers = min(num_workers, self.MAX_WORKERS)

        assert (
            not (
                rna3dhub_version is None and
                pdb_ids is None and
                skip_rna3dhub_download
            )
        ), "Please provide the source of relevant PDB codes!"

        self.rna3dhub_version = rna3dhub_version
        self.pdb_ids = pdb_ids

        self.skip_rna3dhub_download = skip_rna3dhub_download

    def __call__(self, data_access: DataAccess) -> None:
        if (
            self.rna3dhub_version is None and
            not self.skip_rna3dhub_download
        ):
            logger.warning(
                "BGSU 'RNA 3D Hub' NR list version has not been "
                "defined! The latest available version will be "
                "used by default."
            )
            self.rna3dhub_version = get_latest_rna3dhub_nrlist_version()

        # Log messages
        log_msg = "Downloading relevant PDB entries"

        if not self.skip_rna3dhub_download:
            log_msg += " (BGSU 'RNA 3D Hub' NR list version:"
            log_msg += f" {self.rna3dhub_version})"

        log_msg += "..."

        logger.info(log_msg)

        # Download and load RNA3DHub non-redundant list CSV
        if not self.skip_rna3dhub_download:
            download_rna3dhub_nrlist_csv(
                local_csv_file=data_access.rna3dhub_nrlist_file,
                version=self.rna3dhub_version
            )

        # Find relevant PDB IDs / codes
        pdb_codes = set()

        if self.pdb_ids is None:
            # Load PDB IDs from RNA3DHub
            rna3dhub_nrlist = load_rna3dhub_nrlist_csv(
                data_access.rna3dhub_nrlist_file
            )

            for ife in itertools.chain(*rna3dhub_nrlist.values()):
                pdb_codes.add(ife.split("_")[0])
        else:
            pdb_codes = set(self.pdb_ids)

        # Download mmCIF files
        pdbl = PDBList(verbose=False)
        for attempt_idx in range(self.MAX_ATTEMPTS):
            pdbl.download_pdb_files(
                pdb_codes=pdb_codes,
                pdir=data_access.raw_dir,
                file_format="mmCif",
                max_num_threads=self.num_workers,
            )

            downloaded_pdb_codes = set()
            for cif_file in data_access.raw_dir.glob("**/*.cif"):
                downloaded_pdb_codes.add(cif_file.stem)

            missing_pdb_codes = pdb_codes - downloaded_pdb_codes
            if len(missing_pdb_codes) > 0:
                logger.warning(
                    "Failed to download the following PDB entries: " +
                    ", ".join(missing_pdb_codes)
                )
            else:
                break

            if attempt_idx != (self.MAX_ATTEMPTS - 1):
                logger.info("Retrying to download missing PDB entries...")
                pdb_codes = missing_pdb_codes

        logger.info(
            f"Successfully downloaded {len(downloaded_pdb_codes)} "
            "relevant PDB entries!"
        )


class Parse(DataPreparationStep):
    def __init__(
        self,
        num_workers: int,
    ):
        super().__init__()

        self.num_workers = num_workers

    def _parse_worker(self, struct_file: Path, dump_dir: Path) -> None:
        struct = parse_struct_file(struct_file)

        # Save if at least one atom is modeled
        if any(np.sum(chain.atom_mask) > 1e-5 for chain in struct.chains):
            with open(dump_dir / f"{struct.id}.p", "wb") as f:
                pickle.dump(struct, f)

    def __call__(self, data_access: DataAccess) -> None:
        logger.info("Parsing structure files and pickling RNA structures...")
        data_access.parsed_dir.mkdir(parents=True, exist_ok=True)

        # "Collect" structure files (pdb and mmcif)
        struct_files = \
            list(data_access.raw_dir.glob("**/*.cif")) + \
            list(data_access.raw_dir.glob("**/*.pdb"))

        # Parse and pickle
        with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
            _ = list(
                tqdm(
                    ex.map(
                        self._parse_worker,
                        struct_files,
                        len(struct_files) * [data_access.parsed_dir]
                    ),
                    total=len(struct_files)
                )
            )

        logger.info("Structure parsing done!")


class Filter(DataPreparationStep):
    def __init__(
        self,
        filter_funcs: Callable[[MultiRNA], bool],
        filter_descs: list[str],
    ):
        super().__init__()

        assert len(filter_descs) == len(filter_funcs), \
            "Please provide descriptions of all filters!"

        self.filter_funcs = filter_funcs
        self.filter_descs = filter_descs

    def __call__(self, data_access: DataAccess) -> None:
        logger.info(f"Filtering entries ({', '.join(self.filter_descs)})...")

        # Find parsed files and initialize count variable
        removed_count = 0
        struct_files = list(data_access.parsed_dir.glob("**/*.p"))

        # Iterate through files and apply defined filters
        for struct_file in tqdm(struct_files):
            with open(struct_file, "rb") as f:
                struct = pickle.load(f)

            for filter_func in self.filter_funcs:
                if not filter_func(struct):
                    removed_count += 1
                    struct_file.unlink()
                    break

        logger.info(f"Filtered out {removed_count} entries!")


class ExtractIFEs(DataPreparationStep):
    def __init__(self, ifes_dict: Optional[dict[str, list[str]]] = None):
        super().__init__()

        self.ifes_dict = ifes_dict

    def __call__(self, data_access: DataAccess) -> None:
        logger.info("Extracting IFEs...")

        ifes_dict = self.ifes_dict
        if ifes_dict is None:
            # Load RNA 3D Hub list
            rna3dhub_nrlist = load_rna3dhub_nrlist_csv(
                data_access.rna3dhub_nrlist_file
            )

            # Build ID-to-IFEs mapping (ID-to-set-of-sets)
            ifes_dict = defaultdict(list)
            for ife in itertools.chain(*rna3dhub_nrlist.values()):
                ifes_dict[ife.split("_")[0]].append(
                    [chain_id for chain_id in ife.split("_")[1:]]
                )

        # Extract IFEs
        struct_files = list(data_access.parsed_dir.glob("**/*.p"))
        ife_count = 0

        for struct_file in tqdm(struct_files):
            with open(struct_file, "rb") as f:
                struct = pickle.load(f)

            if struct.id in ifes_dict:
                for ife in struct.split_into_ifes(ifes_dict[struct.id]):
                    ife_count += 1
                    ife_file = data_access.parsed_dir / f"{ife.id}.p"
                    with open(ife_file, "wb") as f:
                        pickle.dump(ife, f)

                struct_file.unlink()

        logger.info(f"Found {ife_count} IFEs!")


class ClusterSequences(DataPreparationStep):
    def __init__(
        self,
        min_seq_id: float,
        coverage: float,
        threads: int,
        split: Optional[str] = None,
    ):
        super().__init__()

        self.min_seq_id = min_seq_id
        self.coverage = coverage
        self.threads = threads

        self.split = split

    def __call__(self, data_access: DataAccess) -> None:
        logger.info(
            "Clustering sequences " +
            f"(min-seq-id = {self.min_seq_id:.2f}, " +
            f"coverage = {self.coverage:.2f})..."
        )

        struct_files = list(data_access.parsed_dir.glob("**/*.p"))
        # Collect chain sequences
        seqs = {}

        for struct_file in struct_files:
            with open(struct_file, "rb") as f:
                struct = pickle.load(f)

            pdb_id = struct.id.split("_")[0]

            for chain in struct.chains:
                seqs[f"{pdb_id}_{chain.id}"] = chain.seq

        # Cluster chain sequences with MMseqs
        clusters = run_mmseqs_easy_cluster(
            seqs=seqs,
            min_seq_id=self.min_seq_id,
            coverage=self.coverage,
            threads=self.threads
        )

        # MMseqs2 sometimes struggles with short sequences
        # Do "manual" clustering of identical sequences
        seq_to_cluster = defaultdict(set)
        for chain_id, cluster_id in clusters.items():
            seq_to_cluster[seqs[chain_id]].add(cluster_id)

        for cluster_ids in seq_to_cluster.values():
            if len(cluster_ids) > 1:
                cluster_ids = list(cluster_ids)
                main_cluster = cluster_ids[0]
                for duplicate_cluster_id in cluster_ids[1:]:
                    for chain_id, cluster_id in clusters.items():
                        if cluster_id == duplicate_cluster_id:
                            clusters[chain_id] = main_cluster

        # Collect clusters
        ife_clusters = defaultdict(list)
        for struct_file in struct_files:
            with open(struct_file, "rb") as f:
                struct = pickle.load(f)

            pdb_id = struct.id.split("_")[0]
            ife_clusters[
                tuple(
                    sorted(
                        [
                            clusters[f"{pdb_id}_{chain.id}"]
                            for chain in struct.chains
                        ]
                    )
                )
            ].append(struct.id)

        logger.info(
            f"{sum(len(ife_ids) for ife_ids in ife_clusters.values())} IFEs "
            f"clustered into {len(ife_clusters)} sequence-based clusters!"
        )

        if self.split == "train":
            # Save training clustering info
            train_clusters = {
                sorted(ife_ids)[0]: ife_ids
                for ife_ids in ife_clusters.values()
            }

            with open(data_access.train_cluster_json, "w") as f:
                json.dump(train_clusters, f, indent=4)
        else:
            # If removing duplicates (not training split)
            # keep structures with best resolution (from each cluster)
            removed_count = 0
            logger.info(
                "Removing sequence duplicates" +
                (f" from the '{self.split}' split" if self.split else "") +
                "..."
            )

            removed_ife_ids = []
            for ife_ids in ife_clusters.values():
                min_resolution = float("inf")
                keep_struct_file = None

                for ife_id in ife_ids:
                    struct_file = data_access.parsed_dir / f"{ife_id}.p"
                    with open(struct_file, "rb") as f:
                        struct = pickle.load(f)

                    if struct.resolution < min_resolution:
                        min_resolution = struct.resolution
                        keep_struct_file = struct_file

                for ife_id in ife_ids:
                    struct_file = data_access.parsed_dir / f"{ife_id}.p"
                    if struct_file != keep_struct_file:
                        struct_file.unlink()
                        removed_ife_ids.append(ife_id)
                        removed_count += 1

            logger.info(f"Removed {removed_count} sequence duplicates!")


class CalculateSecondaryStructures(DataPreparationStep):
    def __init__(
        self,
        num_workers: int,
    ):
        super().__init__()

        self.num_workers = num_workers

    def _calculate_ss_worker(
        self,
        parsed_struct_file: Path,
        ss_dir: Path,
    ) -> None:
        with open(parsed_struct_file, "rb") as f_struct:
            struct = pickle.load(f_struct)

        for chain in struct.chains:
            get_single_seq_ss_feats(
                seq=chain.seq,
                seq_id=f"{struct.id}_{chain.id}",
                ss_dir=ss_dir,
                force_overwrite=False,
            )

    def __call__(self, data_access: DataAccess) -> None:
        logger.info("Predicting secondary structures...")

        # Collect parsed structure files
        parsed_struct_files = list(data_access.parsed_dir.glob("**/*.p"))

        # Calculate SS predictions
        with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
            _ = list(
                tqdm(
                    ex.map(
                        self._calculate_ss_worker,
                        parsed_struct_files,
                        len(parsed_struct_files) * [data_access.ss_dir],
                    ),
                    total=len(parsed_struct_files)
                )
            )

        logger.info("Secondary structure prediction done!")


class RemoveSequenceSimilarityOverlap(DataPreparationStep):
    def __init__(
        self,
        train_structs_dir: Path,
        min_seq_id: float,
        coverage: float,
        threads: int = 1,
    ):
        super().__init__()

        self.train_structs_dir = train_structs_dir

        self.min_seq_id = min_seq_id
        self.coverage = coverage
        self.threads = threads

    def __call__(self, data_access: DataAccess):
        logger.info(
            "Removing structures similar (sequence-wise) to any structure "
            f"in the training dataset (min-seq-id = {self.min_seq_id:.2f}, "
            f"coverage = {self.coverage:.2f}, "
            f"train. dir. = '{self.train_structs_dir}')..."
        )

        # Collect training sequences
        train_seqs = {}
        for train_struct_file in self.train_structs_dir.glob("**/*.p"):
            with open(train_struct_file, "rb") as f:
                struct = pickle.load(f)

            pdb_id = struct.id.split("_")[0]
            for chain in struct.chains:
                train_seqs[f"{pdb_id}_{chain.id}"] = chain.seq

        # Collect test sequences and structure files
        test_seqs = {}
        struct_files = list(data_access.parsed_dir.glob("**/*.p"))

        for struct_file in struct_files:
            with open(struct_file, "rb") as f:
                struct = pickle.load(f)

            pdb_id = struct.id.split("_")[0]
            for chain in struct.chains:
                test_seqs[f"{pdb_id}_{chain.id}"] = chain.seq

        # Search test sequences against training sequences
        hits = run_mmseqs_easy_search(
            query_seqs=test_seqs,
            target_seqs=train_seqs,
            min_seq_id=self.min_seq_id,
            coverage=self.coverage,
            threads=self.threads
        )

        # Track which structures to remove
        removed_count = 0
        for query_id, matches in hits.items():
            if len(matches) > 0:  # Has similar sequences in training set
                pdb_id = query_id.split("_")[0]
                chain_id = query_id.split("_")[1]

                # Mark all structures containing this chain for removal
                for struct_file in struct_files:
                    if (
                        struct_file.stem.startswith(pdb_id) and
                        chain_id in struct_file.stem.split("_")[1:]
                    ):
                        if struct_file.exists():
                            struct_file.unlink()
                            removed_count += 1

        logger.info(f"Removed {removed_count} structures!")


class SaveAsPDB(DataPreparationStep):
    def __call__(self, data_access: DataAccess):
        logger.info("Converting parsed structures to PDB format...")

        struct_files = list(data_access.parsed_dir.glob("**/*.p"))

        for struct_file in tqdm(struct_files):
            with open(struct_file, "rb") as f:
                struct = pickle.load(f)

            save_multirna_to_pdb(
                struct=struct,
                output_file=data_access.pdbs_dir / f"{struct.id}.pdb",
            )

        logger.info("Conversion into PDB format done!")


class CreateSeqsFASTA(DataPreparationStep):
    def __call__(self, data_access: DataAccess):
        logger.info("Saving sequences into FASTA file...")

        struct_files = list(data_access.parsed_dir.glob("**/*.p"))

        with open(data_access.seqs_file, "w") as f:
            for struct_file in tqdm(struct_files):
                with open(struct_file, "rb") as f_struct:
                    struct = pickle.load(f_struct)

                f.write(f">{struct.id}\n")

                seqs = [chain.seq for chain in struct.chains]
                f.write(
                    f"{SEQ_DELIMITER.join(seqs)}\n"
                )

        logger.info("Sequences FASTA file creation done!")
