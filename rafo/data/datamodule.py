from pathlib import Path
from typing import Union, Optional

import lightning.pytorch as pl
import torch

from rafo.data.data_access import DataAccess
from rafo.data.dataset import (
    RNAStructureDataset, RNASequenceDataset,
    TENSOR_0D_DICT_KEYS, TENSOR_1D_DICT_KEYS, TENSOR_2D_DICT_KEYS
)
from rafo.utils.seq import LM_TOKENIZER

from torch.utils.data import DataLoader, ConcatDataset


class RAFODataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: Optional[Union[str, Path]] = None,
        val_data_dirs: Optional[list[Union[str, Path]]] = None,
        test_data_dir: Optional[Union[str, Path]] = None,
        distill_data_dir: Optional[Union[str, Path]] = None,
        pred_fasta: Optional[Union[str, Path]] = None,
        max_train_residues: Optional[int] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.train_data_access = \
            DataAccess(train_data_dir) if train_data_dir else None
        self.val_data_accesses = \
            [
                DataAccess(val_dir) for val_dir in val_data_dirs
            ] if val_data_dirs else []
        self.test_data_access = \
            DataAccess(test_data_dir) if test_data_dir else None
        self.distill_data_access = \
            DataAccess(distill_data_dir) if distill_data_dir else None

        self.pred_fasta = pred_fasta
        self.max_train_residues = max_train_residues

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = RNAStructureDataset(
                struct_dir=self.train_data_access.parsed_dir,
                ss_dir=self.train_data_access.ss_dir,
                cluster_json=self.train_data_access.train_cluster_json,
                max_residues=self.max_train_residues,
            )

            if self.distill_data_access:
                self.train_dataset = ConcatDataset(
                    [
                        self.train_dataset,
                        RNAStructureDataset(
                            struct_dir=self.distill_data_access.parsed_dir,
                            ss_dir=self.distill_data_access.ss_dir,
                        )
                    ]
                )

        if stage in ("fit", "validate"):
            self.val_datasets = [
                RNAStructureDataset(
                    struct_dir=da.parsed_dir,
                    ss_dir=da.ss_dir,
                )
                for da in self.val_data_accesses
            ]

        if stage == "test":
            self.test_dataset = RNAStructureDataset(
                struct_dir=self.test_data_access.parsed_dir,
                ss_dir=self.test_data_access.ss_dir,
            )

        if stage == "predict":
            self.predict_dataset = RNASequenceDataset(
                fasta_file=self.pred_fasta
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=(self.num_workers > 0),
            collate_fn=BatchPreparator(),
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=1,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=BatchPreparator(),
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=BatchPreparator(),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=BatchPreparator(),
        )


class BatchPreparator:
    def __call__(self, batch: list[dict]) -> dict:
        batch_size = len(batch)
        seq_lens = [len(instance["seq_tokens"]) for instance in batch]
        max_len = max(seq_lens)

        seq_mask = torch.zeros(batch_size, max_len)
        for i in range(batch_size):
            seq_mask[i, :seq_lens[i]] = 1.

        padded_batch = {}
        padded_batch["seq_mask"] = seq_mask
        padded_batch["msa_mask"] = seq_mask[..., None, :]

        padded_batch["seq_batch_idx"] = torch.tensor(
            [
                i for i, instance in enumerate(batch)
                for _ in instance["sequences"]
            ]
        )

        dict_keys = batch[0].keys()

        for dict_key in dict_keys:
            if dict_key == "seq_tokens_lm":
                all_seqs = [
                    seq for instance in batch for seq in instance["sequences"]
                ]
                padded_batch["seq_tokens_lm"] = torch.tensor(
                    LM_TOKENIZER.batch_tokenize(all_seqs)
                )
            elif dict_key in TENSOR_1D_DICT_KEYS:
                # 1D tensors (e.g. tokenized sequence)
                padded_batch[dict_key] = torch.zeros(
                    (
                        batch_size, max_len,
                        *batch[0][dict_key].shape[1:]
                    ),
                    dtype=batch[0][dict_key].dtype
                )

                for i in range(batch_size):
                    padded_batch[dict_key][i, : seq_lens[i]] = \
                        batch[i][dict_key]
            elif dict_key in TENSOR_2D_DICT_KEYS:
                # 2D tensors (e.g. secondary structure)
                padded_batch[dict_key] = torch.zeros(
                    (
                        batch_size, max_len, max_len,
                        *batch[0][dict_key].shape[2:]
                    ),
                    dtype=batch[0][dict_key].dtype
                )

                for i in range(batch_size):
                    padded_batch[dict_key][i, :seq_lens[i], :seq_lens[i]] = \
                        batch[i][dict_key]
            elif dict_key in TENSOR_0D_DICT_KEYS:
                # 0D tensors (e.g. resolution)
                padded_batch[dict_key] = torch.zeros(
                    batch_size, *batch[0][dict_key].shape,
                    dtype=batch[0][dict_key].dtype
                )

                for i in range(batch_size):
                    padded_batch[dict_key][i] = batch[i][dict_key]
            else:
                # Non-tensor components (e.g. meta information)
                padded_batch[dict_key] = [
                    instance[dict_key] for instance in batch
                ]

        return padded_batch
