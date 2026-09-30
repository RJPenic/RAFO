import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Union
from pathlib import Path
import string
import itertools

from rafo.constants import ATOM_TYPES
from rafo.utils.data_structures import MultiRNA

EPS = 1e-5

DEFAULT_CHAIN_IDS = list(string.ascii_uppercase + string.ascii_lowercase)
DEFAULT_OCCUPANCY = 1.0
DEFAULT_TEMP_FACTOR = 1.0

ATOM_NAME_MAPPINGS = {}
for atom in ATOM_TYPES:
    # width of 4 columns in PDB format, minus the space added at the start
    ATOM_NAME_MAPPINGS[atom] = " " + atom + " " * (4 - 1 - len(atom))

DEFAULT_REMARK_ID = 101


class Atom:
    def __init__(
        self,
        atom_name: str,
        res_name: str,
        atom_id: int,
        res_id: int,
        x: float,
        y: float,
        z: float,
        chain_id: str,
        occupancy: float = DEFAULT_OCCUPANCY,
        temp_factor: float = DEFAULT_TEMP_FACTOR,
    ):
        # Names
        self.atom_name = ATOM_NAME_MAPPINGS[atom_name]
        self.res_name = res_name

        # IDs
        self.atom_id = atom_id
        self.res_id = res_id
        self.chain_id = chain_id

        # Atom coordinates
        self.x = x
        self.y = y
        self.z = z

        # Other
        self.occupancy = occupancy
        self.temp_factor = temp_factor

    def to_pdb(self) -> str:
        # https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
        pdb_row = [" "] * 80
        pdb_row[0: 4] = "ATOM"
        pdb_row[6: 11] = f"{self.atom_id:>5d}"
        pdb_row[12: 16] = f"{self.atom_name:<4s}"
        pdb_row[17: 20] = f"{self.res_name:>3s}"
        pdb_row[21] = f"{self.chain_id[-1]}"  # Use only last character of chain ID
        pdb_row[22: 26] = f"{self.res_id:>4d}"
        pdb_row[30: 38] = f"{self.x:>8.3f}"
        pdb_row[38: 46] = f"{self.y:>8.3f}"
        pdb_row[46: 54] = f"{self.z:>8.3f}"
        pdb_row[54: 60] = f"{self.occupancy:>6.2}"
        pdb_row[60: 66] = f"{self.temp_factor:>6.2}"

        pdb_row = "".join(pdb_row)
        pdb_row = pdb_row.rstrip()
        pdb_row += "\n"

        return pdb_row


def _get_atoms(
    atom_pos: ArrayLike,
    atom_mask: ArrayLike,
    seqs: list[str],
    chain_ids: Optional[list[str]] = None,
    res_temp_factors: Optional[ArrayLike] = None,
) -> list[Atom]:
    # Assertions
    assert atom_pos.shape[:-1] == atom_mask.shape, \
        "Shapes of atom positions and atom mask arrays must match!"
    assert atom_pos.shape[0] == sum(len(seq) for seq in seqs), \
        "Lengths of atom positions array and sequence must match!"
    assert (chain_ids is None) or (len(chain_ids) >= len(seqs)), \
        "Please provide chain ID for each sequence!"

    # Define default chain IDs (if needed)
    if chain_ids is None:
        chain_ids = DEFAULT_CHAIN_IDS

    # Extract atom information
    atoms = []
    seq_concat = "".join(seq for seq in seqs)
    chain_ids = list(
        itertools.chain(
            *[
                len(seq) * [chain_ids[i]]
                for i, seq in enumerate(seqs)
            ]
        )
    )  # "Expand" chain IDs list so it matches the sequence
    
    # Set up per-sequence residue indices
    res_idcs = [i for seq in seqs for i in range(1, len(seq) + 1)]

    for i in range(atom_pos.shape[0]):
        for atom_idx in range(atom_pos.shape[1]):
            if atom_mask[i, atom_idx] > EPS:
                x, y, z = [
                    atom_pos[i][atom_idx][k].item()
                    for k in range(3)
                ]

                temp_factor = DEFAULT_TEMP_FACTOR
                if res_temp_factors is not None:
                    temp_factor = res_temp_factors[i]

                atoms.append(
                    Atom(
                        atom_name=ATOM_TYPES[atom_idx],
                        res_name=seq_concat[i],
                        atom_id=len(atoms) + 1,
                        res_id=res_idcs[i],
                        x=x, y=y, z=z,
                        chain_id=chain_ids[i],
                        temp_factor=temp_factor,
                    )
                )

    return atoms


def create_remark_line(remark: str) -> str:
    line = ""

    line += "REMARK"
    line += f"{DEFAULT_REMARK_ID:>4d}"
    line += f" {remark}"
    line += "\n"

    return line


def save_to_pdb(
    atom_pos: ArrayLike,
    atom_mask: ArrayLike,
    output_file: Union[str, Path],
    seqs: Optional[list[str]] = None,
    chain_ids: Optional[list[str]] = None,
    res_temp_factors: Optional[ArrayLike] = None,
    remarks: list[str] = [],
) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    if seqs is None:
        seqs = ["N" * atom_pos.shape[0]]

    # Atom INFO
    atoms = _get_atoms(
        atom_pos=atom_pos,
        atom_mask=atom_mask,
        seqs=seqs,
        chain_ids=chain_ids,
        res_temp_factors=res_temp_factors
    )
    lines = [atom.to_pdb() for atom in atoms]

    # Remarks/Comments
    lines = [create_remark_line(remark) for remark in remarks] + lines

    # Save into file
    with open(output_file, 'w') as file:
        file.writelines(lines)


def save_multirna_to_pdb(
    struct: MultiRNA,
    output_file: Union[str, Path]
) -> None:
    atom_pos = np.concatenate(
        [chain.atom_pos for chain in struct.chains],
        axis=0,
    )
    atom_mask = np.concatenate(
        [chain.atom_mask for chain in struct.chains],
        axis=0,
    )
    seqs = [chain.seq for chain in struct.chains]

    save_to_pdb(
        atom_pos=atom_pos,
        atom_mask=atom_mask,
        output_file=output_file,
        seqs=seqs,
        chain_ids=[chain.id for chain in struct.chains],
    )
