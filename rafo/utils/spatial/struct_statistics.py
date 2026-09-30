import torch
import numpy as np
from numpy import linalg as LA

from rafo.constants import (
    ATOM_ORDER, BASE_ATOMS
)

from collections import defaultdict
from typing import Callable

from rafo.utils.spatial.geometry import get_bond_angle
from rafo.utils.spatial.transformation import EuclideanTransformation

from rafo.utils.seq import get_pur_pyr_masks

# Bond angle triplets
# Angles "consumed" by the C4'-rooted construction
BB_ANGLE_TRIPLETS = [
    ("C5'", "C4'", "O4'"),   # root frame seed (C5', C4', O4')
    ("C4'", "C5'", "O5'"),   # places O5'
    ("C5'", "O5'", "P"),     # places P
    ("C4'", "O4'", "C1'"),   # places C1'
    ("C5'", "C4'", "C3'"),   # places C3'
    ("C4'", "C3'", "C2'"),   # places C2'
    ("C4'", "C3'", "O3'"),   # places O3'
]
PURINE_BASE_ANGLE_TRIPLET = ("O4'", "C1'", "N9")
PYRIMIDINE_BASE_ANGLE_TRIPLET = ("O4'", "C1'", "N1")

# Bond length pairs
# Lengths "consumed" by the C4'-rooted construction
BB_DIST_PAIRS = [
    ("C5'", "C4'"),   # root frame seed
    ("C4'", "O4'"),   # root frame seed
    ("O5'", "C5'"),   # places O5'
    ("P", "O5'"),     # places P
    ("O4'", "C1'"),   # places C1'
    ("C4'", "C3'"),   # places C3'
    ("C3'", "C2'"),   # places C2'
    ("C3'", "O3'"),   # places O3'
]
PURINE_BASE_DIST_PAIR = ("C1'", "N9")
PYRIMIDINE_BASE_DIST_PAIR = ("C1'", "N1")

# Frame atoms for local positions.
#   O2'    -> in-ring frame [C3', C2', C1']
#   OP1/OP2 -> phosphate-rigid frame [O5', P, OP1]
PURINE_BASE_FRAME_ATOMS = ("C1'", "N9", "C8")
PYRIMIDINE_BASE_FRAME_ATOMS = ("C1'", "N1", "C2")

O2_FRAME_ATOMS = ("C3'", "C2'", "C1'")
OP_FRAME_ATOMS = ("O5'", "P", "OP1")


# Utility methods
def _get_measurement_key(*atoms: str) -> str:
    return "_".join(atom for atom in atoms)


def _mirror_dict_keys(
    dict: dict,
    separator: str = "_"
) -> dict:
    mirrored_dict = {}

    for key, val in dict.items():
        mirrored_dict[key] = val
        mirrored_dict[separator.join(key.split(separator)[::-1])] = val

    return mirrored_dict


def _get_stats_dict(measurements_dict: dict) -> dict:
    stats_dict = {}

    # Calculate mean, median and standard deviation for each key
    for measurement, vals in measurements_dict.items():
        vals = np.array(vals)
        stats_dict[measurement] = \
            {
                "mean": np.nanmean(vals, axis=0).tolist(),
                "median": np.nanmedian(vals, axis=0).tolist(),
                "std": np.nanstd(vals, axis=0).tolist(),
            }

    return stats_dict


# Measurments (distance/angle) calculation methods
def _get_measurement_vals(
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
    seq: str,
    calc_func: Callable,
    bb_atom_tuples: list[tuple[str, ...]],
    pur_base_atom_tuple: tuple[str, ...],
    pyr_base_atom_tuple: tuple[str, ...]
) -> dict[str, list]:
    assert pur_base_atom_tuple[-1][0] == 'N', \
        f"Expected nitrogen atom (got {pur_base_atom_tuple[-1][0]})!"
    assert pyr_base_atom_tuple[-1][0] == 'N', \
        f"Expected nitrogen atom (got {pyr_base_atom_tuple[-1][0]})!"

    assert pur_base_atom_tuple[:-2] == pyr_base_atom_tuple[:-2], \
        """
        Purine and pyrimidine base atom tuples can differ only in
        the last atom!
        """

    assert len(seq) == atom_pos.shape[-3], \
        "Sequence lengths do not match!"

    # Initialize undefined values to NaN
    atom_mask = np.array(atom_mask, dtype=bool)
    atom_pos[~atom_mask] = float("nan")
    measurements_dict = defaultdict(list)

    # Get backbone measurements
    for atoms in bb_atom_tuples:
        relevant_atom_pos = atom_pos[
            ..., [ATOM_ORDER[atom] for atom in atoms], :
        ]
        measurements = calc_func(
            *np.split(
                relevant_atom_pos,
                relevant_atom_pos.shape[-2],
                axis=-2
            )
        ).flatten().tolist()
        measurements_dict[_get_measurement_key(*atoms)].extend(measurements)

    # Get nucleic base measurements
    purine_measurement = calc_func(
        *np.split(
            atom_pos[
                ...,
                [ATOM_ORDER[atom] for atom in pur_base_atom_tuple],
                :
            ],
            len(pur_base_atom_tuple),
            axis=-2
        )
    )
    pyrimidine_measurement = calc_func(
        *np.split(
            atom_pos[
                ...,
                [ATOM_ORDER[atom] for atom in pyr_base_atom_tuple],
                :
            ],
            len(pyr_base_atom_tuple),
            axis=-2
        )
    )

    # Purine/Pyrimidine mask
    purine_mask, pyrimidine_mask = \
        get_pur_pyr_masks(seq)

    # Add base measurments into dictionary
    base_measurement_key = _get_measurement_key(
        *pur_base_atom_tuple[:-1], "N"
    )
    measurements_dict[base_measurement_key].extend(
        purine_measurement[
            purine_mask
        ].flatten().tolist()
    )
    measurements_dict[base_measurement_key].extend(
        pyrimidine_measurement[
            pyrimidine_mask
        ].flatten().tolist()
    )

    return measurements_dict


def get_bond_angles(
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
    seq: str
) -> dict[str, list]:
    return \
        _get_measurement_vals(
            atom_pos=atom_pos,
            atom_mask=atom_mask,
            seq=seq,
            calc_func=get_bond_angle,
            bb_atom_tuples=BB_ANGLE_TRIPLETS,
            pur_base_atom_tuple=PURINE_BASE_ANGLE_TRIPLET,
            pyr_base_atom_tuple=PYRIMIDINE_BASE_ANGLE_TRIPLET
        )


def get_bond_lengths(
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
    seq: str
) -> dict[str, list]:
    # Intra-residue bond lengths
    measurement_vals = _get_measurement_vals(
        atom_pos=atom_pos,
        atom_mask=atom_mask,
        seq=seq,
        calc_func=lambda x1, x2: LA.norm(x1 - x2, axis=-1),
        bb_atom_tuples=BB_DIST_PAIRS,
        pur_base_atom_tuple=PURINE_BASE_DIST_PAIR,
        pyr_base_atom_tuple=PYRIMIDINE_BASE_DIST_PAIR,
    )

    # Inter-residue bond length (P-O3')
    atom_mask = np.array(atom_mask, dtype=bool)
    atom_pos[~atom_mask] = float("nan")

    o3_pos = atom_pos[:-1][..., ATOM_ORDER["O3'"], :]
    p_pos = atom_pos[1:][..., ATOM_ORDER["P"], :]

    inter_res_dists = LA.norm(o3_pos - p_pos, axis=-1)
    measurement_vals[_get_measurement_key("O3'", "P")] = \
        inter_res_dists.flatten().tolist()

    return measurement_vals


def _get_local_pos(
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
    frame_atoms: tuple[str, str, str],
    atoms_to_calculate: list[str],
) -> dict:
    atom_mask = np.array(atom_mask, dtype=bool)
    atom_pos[~atom_mask] = float("nan")

    atom_pos = torch.from_numpy(atom_pos)

    local_pos_dict = defaultdict(list)

    transform = \
        EuclideanTransformation.from_3_points(
            *[atom_pos[..., ATOM_ORDER[atom], :] for atom in frame_atoms]
        )

    for atom in atoms_to_calculate:
        atom_local_pos = transform.invert_apply(
            atom_pos[..., ATOM_ORDER[atom], :]
        )

        local_pos_dict[atom].extend(
            [
                local_pos.numpy()
                for local_pos in atom_local_pos.view(-1, 3).unbind(dim=0)
            ]
        )

    return local_pos_dict


def get_local_o2_pos(
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
) -> dict:
    return \
        _get_local_pos(
            atom_pos,
            atom_mask,
            O2_FRAME_ATOMS,
            ["O2'"]
        )


def get_local_op_pos(
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
) -> dict:
    return \
        _get_local_pos(
            atom_pos,
            atom_mask,
            OP_FRAME_ATOMS,
            ["OP1", "OP2"]
        )


def get_local_side_oxy_pos(
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
) -> dict:
    op_local_pos = get_local_op_pos(atom_pos, atom_mask)
    o2_local_pos = get_local_o2_pos(atom_pos, atom_mask)

    local_pos_dict = {**op_local_pos, **o2_local_pos}
    return local_pos_dict


def get_local_base_pos(
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
    seq: str
) -> dict:
    local_coords_dict = {}

    purine_mask, pyrimidine_mask = \
        get_pur_pyr_masks(seq)

    local_coords_dict["A"] = \
        _get_local_pos(
            atom_pos[purine_mask],
            atom_mask[purine_mask],
            PURINE_BASE_FRAME_ATOMS,
            BASE_ATOMS
        )
    local_coords_dict["G"] =  \
        _get_local_pos(
            atom_pos[purine_mask],
            atom_mask[purine_mask],
            PURINE_BASE_FRAME_ATOMS,
            BASE_ATOMS
        )
    local_coords_dict["C"] =  \
        _get_local_pos(
            atom_pos[pyrimidine_mask],
            atom_mask[pyrimidine_mask],
            PYRIMIDINE_BASE_FRAME_ATOMS,
            BASE_ATOMS
        )
    local_coords_dict["U"] =  \
        _get_local_pos(
            atom_pos[pyrimidine_mask],
            atom_mask[pyrimidine_mask],
            PYRIMIDINE_BASE_FRAME_ATOMS,
            BASE_ATOMS
        )

    return local_coords_dict


def get_struct_stats_dict(
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
    seq: str
) -> dict[str, dict]:
    stats = {}

    # Bond length stats
    bond_lens = get_bond_lengths(atom_pos, atom_mask, seq)
    stats["bond_lengths"] = _mirror_dict_keys(_get_stats_dict(bond_lens))

    # Bond angle stats
    bond_angls = get_bond_angles(atom_pos, atom_mask, seq)
    stats["bond_angles"] = _mirror_dict_keys(_get_stats_dict(bond_angls))

    # "Side"-oxygen local positions stats
    stats["side_oxygen_local"] = {}
    local_oxy_pos = get_local_side_oxy_pos(atom_pos, atom_mask)
    stats["side_oxygen_local"] = _get_stats_dict(local_oxy_pos)

    # Base atoms local positions stats
    stats["base_local"] = {}
    base_local_pos = get_local_base_pos(atom_pos, atom_mask, seq)
    for nuc in base_local_pos.keys():
        stats["base_local"][nuc] = _get_stats_dict(base_local_pos[nuc])

    return stats
