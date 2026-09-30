"""
Structure file parsing utilities. Parts of code taken from
https://github.com/aqlaboratory/openfold/blob/c2f46ce86367689864eac6a954dd9204b2576d3b/openfold/data/mmcif_parsing.py
"""
from typing import Optional
from abc import ABC, abstractmethod

from Bio import PDB
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

import numpy as np
from pathlib import Path

from rafo.constants import (
    ATOM_ORDER,
    ATOM_TYPES,
    NUM_ATOM_TYPES,
    STANDARD_RESIDUE_TOKENS,
    BASE_ATOMS,
)
from rafo.utils.data_structures import MultiRNA, SingleRNA

# TODO: b-factor parsing?

RNA_POLYMER_TYPE = "polyribonucleotide"
UNDEFINED_RES_IDX = -999_999_999


class ParsedStructureFile(ABC):
    @abstractmethod
    def get_rna_structs(self) -> MultiRNA:
        pass


class ParsedMmCIF(ParsedStructureFile):
    def __init__(self, mmcif_file: Path):
        super().__init__()

        self.mmcif_dict = PDB.MMCIF2Dict.MMCIF2Dict(mmcif_file)
        self.xyz_generator = None

    def _get_entry_id(self):
        return self.mmcif_dict["_entry.id"].lower()

    def _get_resolution(self) -> float:
        resolution = None

        for res_key in [
            "_refine.ls_d_res_high",
            "_em_3d_reconstruction.resolution",
            "_reflns.d_resolution_high",
        ]:
            if res_key in self.mmcif_dict:
                try:
                    str_resolution = self.mmcif_dict[res_key][0]
                    resolution = float(str_resolution)
                except ValueError:
                    pass

        return resolution

    def _get_release_date(self) -> Optional[str]:
        key = "_pdbx_audit_revision_history.revision_date"
        if key in self.mmcif_dict:
            return min(self.mmcif_dict[key])
        return None

    def _get_method(self) -> Optional[str]:
        if "_exptl.method" in self.mmcif_dict:
            return ",".join(self.mmcif_dict["_exptl.method"])
        return None

    def _get_chain_ids(self) -> list[str]:
        # Get chain IDs provided by author
        return [
            chain_id
            for chain_list in self.mmcif_dict["_entity_poly.pdbx_strand_id"]
            for chain_id in chain_list.split(",")
        ]

    def _build_xyz_generator(self) -> dict:
        if self.xyz_generator is not None:
            # XYZ generator already built
            return

        # Generator key: (Chain ID, Residue Index, Atom ID, Model ID)
        chain_ids = self.mmcif_dict["_atom_site.auth_asym_id"]
        res_idcs = [
            int(res_idx) if res_idx != "." else UNDEFINED_RES_IDX
            for res_idx in self.mmcif_dict["_atom_site.label_seq_id"]
        ]
        atom_ids = self.mmcif_dict["_atom_site.label_atom_id"]
        model_ids = self.mmcif_dict["_atom_site.pdbx_PDB_model_num"]

        # Generator value: (x, y, z) coordinates
        xs, ys, zs = (
            self.mmcif_dict["_atom_site.Cartn_x"],
            self.mmcif_dict["_atom_site.Cartn_y"],
            self.mmcif_dict["_atom_site.Cartn_z"],
        )

        # Create the generator
        self.xyz_generator = {
            (chain_id, res_idx, atom_id, model_id):
            (float(x), float(y), float(z))
            for chain_id, res_idx, atom_id, model_id, x, y, z
            in zip(chain_ids, res_idcs, atom_ids, model_ids, xs, ys, zs)
        }

    def _get_atom_pos(
        self,
        chain_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._build_xyz_generator()

        seq = self._get_seq(chain_id)
        num_res = len(seq)

        atom_pos = np.zeros((num_res, NUM_ATOM_TYPES, 3), dtype=np.float32)
        atom_mask = np.zeros((num_res, NUM_ATOM_TYPES), dtype=np.float32)

        for res_idx in range(num_res):
            for atom_idx, atom_id in enumerate(ATOM_TYPES):
                if (chain_id, res_idx + 1, atom_id, "1") in self.xyz_generator:
                    # Do not parse base coordinates if residue is not a
                    # standard (A, C, U or G) nucleotide
                    if not (
                        seq[res_idx] not in STANDARD_RESIDUE_TOKENS
                        and atom_id in BASE_ATOMS
                    ):
                        x, y, z = self.xyz_generator[
                            (chain_id, res_idx + 1, atom_id, "1")
                        ]  # Use only first model coordinates

                        atom_pos[res_idx, atom_idx, 0] = x
                        atom_pos[res_idx, atom_idx, 1] = y
                        atom_pos[res_idx, atom_idx, 2] = z

                        atom_mask[res_idx, atom_idx] = 1.0

        return atom_pos, atom_mask

    def _get_entity_idx(self, chain_id: str) -> int:
        entity_chain_lists = [
            [chain_id for chain_id in chain_list.split(",")]
            for chain_list in self.mmcif_dict["_entity_poly.pdbx_strand_id"]
        ]

        for idx, entity_chain_list in enumerate(entity_chain_lists):
            if chain_id in entity_chain_list:
                return idx

        raise ValueError(
            "Given chain ID does not exist in the given mmCIF file!"
        )

    def _get_entity_id(self, chain_id: str) -> str:
        entity_idx = self._get_entity_idx(chain_id)
        entity_ids = self.mmcif_dict["_entity_poly.entity_id"]

        return entity_ids[entity_idx]

    def _get_seq(self, chain_id: str) -> str:
        entity_idx = self._get_entity_idx(chain_id)

        entity_seqs = \
            self.mmcif_dict["_entity_poly.pdbx_seq_one_letter_code_can"]
        entity_seqs = [
            seq.replace("\n", "").upper().replace("T", "U")
            for seq in entity_seqs
        ]  # "Clean up" the sequences

        return entity_seqs[entity_idx]

    def _get_polymer_type(self, chain_id: str) -> str:
        entity_idx = self._get_entity_idx(chain_id)
        polymer_types = self.mmcif_dict["_entity_poly.type"]

        return polymer_types[entity_idx]

    def get_metadata(self) -> dict:
        return {
            "resolution": self._get_resolution(),
            "release_date": self._get_release_date(),
            "method": self._get_method(),
            "chain_ids": self._get_chain_ids(),
        }

    def get_rna_structs(self) -> MultiRNA:
        metadata = self.get_metadata()
        entry_id = self._get_entry_id()

        rnas = []
        for chain_id in metadata["chain_ids"]:
            poly_type = self._get_polymer_type(chain_id)
            if poly_type != RNA_POLYMER_TYPE:
                # Skip non-RNA chains
                continue

            # Get atom positions
            atom_pos, atom_mask = self._get_atom_pos(chain_id)

            # Get chain sequence and entity ID
            chain_seq = self._get_seq(chain_id)

            # "Pack" parsed info into data structure
            rnas.append(
                SingleRNA(
                    id=chain_id,
                    seq=chain_seq,
                    atom_pos=atom_pos,
                    atom_mask=atom_mask,
                )
            )

        multimer = MultiRNA(
            id=entry_id,
            chains=rnas,
            release_date=metadata["release_date"],
            resolution=metadata["resolution"],
            method=metadata["method"]
        )

        return multimer


class ParsedPDB(ParsedStructureFile):
    # Token used to fill gaps in the residue numbering
    GAP_RESIDUE_TOKEN = "N"

    def __init__(self, pdb_file: Path):
        super().__init__()

        self.entry_id = pdb_file.stem
        parser = PDB.PDBParser(QUIET=True)
        self.structure = parser.get_structure(id=self.entry_id, file=pdb_file)

    @staticmethod
    def _is_valid_residue(residue: Residue) -> bool:
        # Are residue atoms labeled as ATOM?
        res_atom_type, *_ = residue.get_id()
        return res_atom_type == " "

    @classmethod
    def _get_valid_residues(cls, chain: Chain) -> list[Residue]:
        return [
            residue for residue in chain if cls._is_valid_residue(residue)
        ]

    @classmethod
    def _get_chain_seq(cls, chain: Chain) -> str:
        residues = cls._get_valid_residues(chain)
        if not residues:
            return ""

        # Sequence numbering should always starts at 1 (if not, prepend)
        seq_len = residues[-1].get_id()[1]

        seq_tokens = [cls.GAP_RESIDUE_TOKEN] * seq_len
        for residue in residues:
            res_idx = residue.get_id()[1] - 1
            seq_tokens[res_idx] = residue.get_resname().strip()

        return "".join(seq_tokens)

    def get_rna_structs(self) -> MultiRNA:
        rnas = []
        for chain in self.structure.get_chains():
            chain_seq = self._get_chain_seq(chain)
            num_res = len(chain_seq)

            atom_pos = np.zeros((num_res, NUM_ATOM_TYPES, 3), dtype=np.float32)
            atom_mask = np.zeros((num_res, NUM_ATOM_TYPES), dtype=np.float32)

            for residue in self._get_valid_residues(chain):
                # Residue 'n' maps to index 'n - 1'
                res_idx = residue.get_id()[1] - 1

                for atom in residue:
                    atom_id = atom.get_name()

                    if atom_id not in ATOM_ORDER:
                        # Skip non-standard atoms
                        continue

                    # Write atom position into array
                    if not (
                        chain_seq[res_idx] not in STANDARD_RESIDUE_TOKENS
                        and atom_id in BASE_ATOMS
                    ):
                        atom_pos[res_idx, ATOM_ORDER[atom_id]] = atom.get_coord()
                        atom_mask[res_idx, ATOM_ORDER[atom_id]] = 1.0

            rnas.append(
                SingleRNA(
                    id=chain.get_id(),
                    seq=chain_seq,
                    atom_pos=atom_pos,
                    atom_mask=atom_mask,
                )
            )

        multimer = MultiRNA(
            id=self.entry_id,
            chains=rnas,
            resolution=99.0,  # TODO: might need change
        )

        return multimer


def parse_mmcif_file(
    mmcif_file: Path,
) -> MultiRNA:
    parsed_mmcif = ParsedMmCIF(mmcif_file=mmcif_file)
    return parsed_mmcif.get_rna_structs()


def parse_pdb_file(
    pdb_file: Path,
) -> MultiRNA:
    parsed_pdb = ParsedPDB(pdb_file=pdb_file)
    return parsed_pdb.get_rna_structs()


def parse_struct_file(
    struct_file: Path,
) -> MultiRNA:
    if struct_file.suffix == ".pdb":
        return parse_pdb_file(struct_file)
    elif struct_file.suffix == ".cif":
        return parse_mmcif_file(struct_file)

    raise NotImplementedError("Unsupported structure file format!")
