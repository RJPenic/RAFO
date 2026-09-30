import pickle
from pathlib import Path
from typing import Optional, Iterable
from Bio import SeqIO
import numpy as np
import string
import json
import random
from itertools import chain

import torch

from torch.utils.data import Dataset

from rafo.utils.sec_struct import get_multi_seq_ss_feats
from rafo.utils.seq import tokenize, LM_TOKENIZER
from rafo.utils.spatial.geometry import (
    get_rna_torsion_angles_2d,
    get_rna_frames
)
from rafo.utils.multimer import (
    get_asym_ids, get_sym_ids, get_entity_ids
)
from rafo.utils.data_structures import MultiRNA
from rafo.constants import FRAME_ORDER, ROOT_FRAME, SEQ_DELIMITER

TENSOR_0D_DICT_KEYS = [
    "resolution",
]

TENSOR_1D_DICT_KEYS = [
    "seq_tokens",
    "residue_index",
    "atom_positions",
    "atom_mask",
    "torsion_angles",
    "torsion_angles_mask",
    "all_frames",
    "all_frames_mask",
    "root_frames",
    "root_frames_mask",
    "entity_id",
    "sym_id",
    "asym_id",
]

TENSOR_2D_DICT_KEYS = [
    "secondary_structures"
]


def prepare_meta_bundle(
    struct_id: str,
    chain_ids: list[str],
) -> dict:
    bundle = {
        "entry_id": struct_id,
        "chain_ids": chain_ids,
    }

    return bundle


def prepare_input_bundle(
    seqs: list[str],
    chain_ids: list[str],
    struct_id: str,
    ss_dir: Optional[Path] = None,
) -> dict:
    assert len(chain_ids) == len(seqs), \
        "Please provide chain ID for every sequence!"

    seq_concat = ''.join([seq for seq in seqs])

    # Sequence metadata and inputs
    seq_tokens = tokenize(seq_concat)
    seq_tokens_lm = LM_TOKENIZER.batch_tokenize(seqs)

    # Chain and entity IDs
    entity_id = get_entity_ids(seqs)
    asym_id = get_asym_ids(seqs)
    sym_id = get_sym_ids(seqs)

    # Expand ID arrays to match number of residues
    repeats = [len(seq) for seq in seqs]

    asym_id = np.repeat(asym_id, repeats)
    sym_id = np.repeat(sym_id, repeats)
    entity_id = np.repeat(entity_id, repeats)

    # Secondary structure
    ss_feats = get_multi_seq_ss_feats(
        seqs=seqs,
        seq_ids=[f"{struct_id}_{chain_id}" for chain_id in chain_ids],
        ss_dir=ss_dir,
    )

    # Final bundle
    bundle = {
        "sequence": seq_concat,
        "sequences": seqs,
        "seq_tokens": torch.tensor(seq_tokens),
        "seq_tokens_lm": torch.tensor(seq_tokens_lm),
        "entity_id": torch.tensor(entity_id),
        "asym_id": torch.tensor(asym_id),
        "sym_id": torch.tensor(sym_id),
        "residue_index": torch.arange(sum(len(seq) for seq in seqs)),
        "secondary_structures": torch.from_numpy(ss_feats),
    }

    return bundle


def prepare_label_bundle(struct: MultiRNA) -> dict:
    # Multimer concatenation
    seq_concat = ''.join([chain.seq for chain in struct.chains])

    atom_pos = np.concatenate(
        [chain.atom_pos for chain in struct.chains],
        axis=0
    )
    atom_mask = np.concatenate(
        [chain.atom_mask for chain in struct.chains],
        axis=0
    )

    # Torsion angles and frames
    trsn_angls, trsn_angls_mask = get_rna_torsion_angles_2d(
        atom_pos=atom_pos,
        atom_mask=atom_mask,
        seq=seq_concat
    )
    all_frames, all_frames_mask = get_rna_frames(
        atom_pos=atom_pos,
        atom_mask=atom_mask,
        seq=seq_concat
    )

    if all_frames.shape[-1] == 4:
        root_frames = all_frames[..., FRAME_ORDER[ROOT_FRAME], :, :]
    else:
        root_frames = all_frames[..., FRAME_ORDER[ROOT_FRAME], :]

    root_frames_mask = all_frames_mask[..., FRAME_ORDER[ROOT_FRAME]]

    # Create data bundle
    data_bundle = {
        # Label tensors
        "atom_positions": torch.from_numpy(atom_pos),
        "atom_mask": torch.from_numpy(atom_mask),
        "torsion_angles": torch.from_numpy(trsn_angls),
        "torsion_angles_mask": torch.from_numpy(trsn_angls_mask),
        "all_frames": torch.from_numpy(all_frames),
        "all_frames_mask": torch.from_numpy(all_frames_mask),
        "root_frames": torch.from_numpy(root_frames),
        "root_frames_mask": torch.from_numpy(root_frames_mask),
        # Other tensors
        "resolution": torch.tensor(struct.resolution),
    }

    return data_bundle


def prepare_data_bundle_labeled(struct_file: Path, ss_dir: Path) -> dict:
    data_bundle = {}

    # Load structure label file
    with open(struct_file, "rb") as f_struct:
        struct = pickle.load(f_struct)

    # Inputs
    data_bundle.update(
        prepare_input_bundle(
            seqs=[chain.seq for chain in struct.chains],
            chain_ids=[chain.id for chain in struct.chains],
            struct_id=struct.id,
            ss_dir=ss_dir,
        )
    )

    # Meta info
    data_bundle.update(
        prepare_meta_bundle(
            struct_id=struct.id,
            chain_ids=[chain.id for chain in struct.chains]
        )
    )

    # Labels
    data_bundle.update(
        prepare_label_bundle(struct)
    )

    return data_bundle


def prepare_data_bundle_inference(seqs: list[str], struct_id: str) -> dict:
    data_bundle = {}

    # Default chain IDs
    chain_ids = list(
        string.ascii_uppercase + string.ascii_lowercase
    )[: len(seqs)]

    # Extract chain IDs from structure ID (if possible)
    # NOTE: If you notice weird chain labeling, try changing this
    if "_" in struct_id:
        chain_ids = struct_id.split("_")[1:]

    # Inputs
    data_bundle.update(
        prepare_input_bundle(
            seqs=seqs,
            chain_ids=chain_ids,
            struct_id=struct_id,
        )
    )

    # Meta info
    data_bundle.update(
        prepare_meta_bundle(
            struct_id=struct_id,
            chain_ids=chain_ids,
        )
    )

    return data_bundle


class RNAStructureDataset(Dataset):
    def __init__(
        self,
        struct_dir: Path,
        ss_dir: Optional[Path] = None,
        struct_ids: Optional[Iterable[str]] = None,
        cluster_json: Optional[Path] = None,
        max_residues: Optional[int] = None,
    ):
        super().__init__()

        self.struct_dir = struct_dir
        self.ss_dir = ss_dir

        self.struct_files = list(self.struct_dir.glob("**/*.p"))

        # Consider only specified structures
        if struct_ids is not None:
            struct_ids = set(struct_ids)

            self.struct_files = [
                struct_file
                for struct_file in self.struct_files
                if struct_file.stem in struct_ids
            ]

        struct_id_to_path = {
            struct_file.stem: struct_file
            for struct_file in self.struct_files
        }

        # Generate dummy clusters (1 structure per cluster)
        self.clusters = [[struct_file] for struct_file in self.struct_files]

        # Load clustering info (if provided)
        if cluster_json is not None:
            with open(cluster_json, "r") as f_cluster:
                self.clusters = json.load(f_cluster)

            # Use paths instead of IDs
            self.clusters = {
                cluster_id: [
                    struct_id_to_path[struct_id] for struct_id in cluster
                ]
                for cluster_id, cluster in self.clusters.items()
            }

            # Check cluster JSON and structure IDs list consistency
            if (
                set(self.struct_files) !=
                set(chain.from_iterable(self.clusters.values()))
            ):
                raise ValueError(
                    "Clustering JSON is not compatible with the structure " +
                    "files list!"
                )

            # Convert dictionary to list of lists
            self.clusters = list(self.clusters.values())

        # Filter by maximum number of residues
        if max_residues is not None:
            filtered_clusters = []
            for struct_files in self.clusters:
                filtered_files = []
                for struct_file in struct_files:
                    with open(struct_file, "rb") as f_struct:
                        struct = pickle.load(f_struct)

                    total_len = sum(len(chain.seq) for chain in struct.chains)
                    if total_len <= max_residues:
                        filtered_files.append(struct_file)

                if len(filtered_files) > 0:
                    filtered_clusters.append(filtered_files)

            self.clusters = filtered_clusters

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, idx):
        # Randomly sample a structure from the cluster
        struct_file = random.choice(self.clusters[idx])

        return prepare_data_bundle_labeled(
            struct_file=struct_file,
            ss_dir=self.ss_dir,
        )


class RNASequenceDataset(Dataset):
    def __init__(
        self,
        fasta_file: Path,
    ):
        super().__init__()

        self.seqs, self.seq_ids = [], []
        for record in SeqIO.parse(fasta_file, "fasta"):
            self.seqs.append(str(record.seq))
            self.seq_ids.append(record.id)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        data_bundle = prepare_data_bundle_inference(
            seqs=self.seqs[idx].split(SEQ_DELIMITER),
            struct_id=self.seq_ids[idx],
        )

        return data_bundle
