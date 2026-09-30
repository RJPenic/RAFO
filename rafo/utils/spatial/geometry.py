import torch

import numpy as np
from numpy import linalg as LA

from rafo.constants import (
    NUM_TORSIONS, ATOM_ORDER, TORSIONS,
    TORSION_ATOMS, TORSION_ORDER, FRAME_ATOMS,
    FRAMES, NUM_FRAMES, FRAME_ORDER,
)

from rafo.utils.seq import get_pur_pyr_masks
from rafo.utils.spatial.transformation import EuclideanTransformation


def get_angle_2d(angle_rad: np.ndarray) -> np.ndarray:
    return np.stack((np.cos(angle_rad), np.sin(angle_rad)), axis=-1)


def get_torsion_angle(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> np.ndarray:
    # https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    v0 = -1.0 * (p1 - p0)
    v1 = p2 - p1
    v2 = p3 - p2

    v1_norm = LA.norm(v1, axis=-1, keepdims=True)
    v1_norm[np.isclose(v1_norm, 0.0)] = 1.

    v1 /= v1_norm

    v = v0 - np.sum(v0 * v1, axis=-1, keepdims=True) * v1
    w = v2 - np.sum(v2 * v1, axis=-1, keepdims=True) * v1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(v1, v) * w, axis=-1)

    angle = np.arctan2(y, x)

    return angle


def get_torsion_angle_2d(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray
) -> np.ndarray:
    angle = get_torsion_angle(p1, p2, p3, p4)
    return get_angle_2d(angle)


def get_bond_angle(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray
) -> np.ndarray:
    v1 = p1 - p2
    v2 = p3 - p2

    dot_product = np.sum(v1 * v2, axis=-1)
    norm_v1 = LA.norm(v1, axis=-1)
    norm_v2 = LA.norm(v2, axis=-1)

    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(cos_theta)

    return angle


def get_bond_angle_2d(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray
) -> np.ndarray:
    angle = get_bond_angle(p1, p2, p3)
    return get_angle_2d(angle)


def _get_torsion_angle_for_atoms(
    atoms: list[str],
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
):
    # Extract relevant positions and masks
    trsn_atom_pos = atom_pos[
        ..., [
            ATOM_ORDER[atom]
            for atom in atoms
        ], :
    ]
    trsn_atom_mask = atom_mask[
        ..., [
            ATOM_ORDER[atom]
            for atom in atoms
        ]
    ]

    # Calculate torsion angle and mask
    trsn_angl = get_torsion_angle(
        *np.split(trsn_atom_pos, trsn_atom_pos.shape[-2], axis=-2)
    )
    trsn_angl = np.squeeze(trsn_angl, axis=-1)

    trsn_mask = np.prod(
        trsn_atom_mask, axis=-1
    )

    return trsn_angl, trsn_mask


def get_rna_torsion_angles(
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
    seq: str,
) -> tuple[np.ndarray, np.ndarray]:
    seq_len, *_ = atom_pos.shape
    assert seq_len == len(seq), \
        "Array shape and sequence length are not compatible!"

    # Initialize torsion arrays
    trsn_angls = np.zeros((seq_len, NUM_TORSIONS))
    trsn_angls_mask = np.zeros((seq_len, NUM_TORSIONS))

    # Purine/Pyrimidine mask
    pur_mask, pyr_mask = get_pur_pyr_masks(seq)

    # Torsion calculation
    for trsn_angl_id in TORSIONS:
        if type(TORSION_ATOMS[trsn_angl_id]) is list:
            # Backbone torsion
            trsn_angl, trsn_mask = _get_torsion_angle_for_atoms(
                atoms=TORSION_ATOMS[trsn_angl_id],
                atom_pos=atom_pos,
                atom_mask=atom_mask,
            )

            trsn_angls[..., TORSION_ORDER[trsn_angl_id]] = trsn_angl
            trsn_angls_mask[..., TORSION_ORDER[trsn_angl_id]] = trsn_mask
        elif type(TORSION_ATOMS[trsn_angl_id]) is dict:
            # Base torsion
            trsn_pur_angl, trsn_pur_mask = _get_torsion_angle_for_atoms(
                atoms=TORSION_ATOMS[trsn_angl_id]["purine"],
                atom_pos=atom_pos,
                atom_mask=atom_mask,
            )
            trsn_pyr_angl, trsn_pyr_mask = _get_torsion_angle_for_atoms(
                atoms=TORSION_ATOMS[trsn_angl_id]["pyrimidine"],
                atom_pos=atom_pos,
                atom_mask=atom_mask,
            )

            trsn_angl = \
                trsn_pur_angl * pur_mask + \
                trsn_pyr_angl * pyr_mask

            trsn_mask = \
                trsn_pur_mask * pur_mask + \
                trsn_pyr_mask * pyr_mask

            trsn_angls[..., TORSION_ORDER[trsn_angl_id]] = trsn_angl
            trsn_angls_mask[..., TORSION_ORDER[trsn_angl_id]] = trsn_mask
        else:
            raise RuntimeError("Unexpected torsion descriptor!")

    return trsn_angls, trsn_angls_mask


def get_rna_torsion_angles_2d(
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
    seq: str
):
    trsn_angls, trsn_angls_mask = get_rna_torsion_angles(
        atom_pos, atom_mask, seq
    )
    trsn_angls = get_angle_2d(trsn_angls)

    return trsn_angls, trsn_angls_mask


def _get_frame_for_atoms(
    atoms: list[str],
    atom_pos: torch.Tensor,
    atom_mask: torch.Tensor,
):
    frame = EuclideanTransformation.from_3_points(
        *torch.split(
            atom_pos[..., [ATOM_ORDER[atom] for atom in atoms], :],
            split_size_or_sections=1,
            dim=-2
        )
    )

    frame = frame.to_tensor_4x4().numpy()
    frame = np.squeeze(frame, axis=-3)

    frame_mask = \
        atom_mask[..., [ATOM_ORDER[atom] for atom in atoms]].prod(dim=-1)
    frame_mask = frame_mask.numpy()

    return frame, frame_mask


def get_rna_frames(
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
    seq: str,
):
    seq_len, *_ = atom_pos.shape
    assert seq_len == len(seq), \
        "Array shape and sequence length are not compatible!"

    # Numpy to torch conversion
    atom_pos = torch.from_numpy(atom_pos)
    atom_mask = torch.from_numpy(atom_mask)

    # Initialize arrays
    all_frames = np.zeros((seq_len, NUM_FRAMES, 4, 4))
    all_frames_mask = np.zeros((seq_len, NUM_FRAMES))

    # Purine/Pyrimidine mask
    pur_mask, pyr_mask = get_pur_pyr_masks(seq)

    # Get frames (translation + orientation)
    for frame_id in FRAMES:
        atoms = FRAME_ATOMS[frame_id]

        if type(atoms) is list:
            frame, frame_mask = _get_frame_for_atoms(
                atoms=atoms,
                atom_pos=atom_pos,
                atom_mask=atom_mask,
            )

            all_frames[:, FRAME_ORDER[frame_id], ...] = frame
            all_frames_mask[:, FRAME_ORDER[frame_id], ...] = frame_mask
        elif type(atoms) is dict:
            frame_pur, frame_pur_mask = _get_frame_for_atoms(
                atoms=atoms["purine"],
                atom_pos=atom_pos,
                atom_mask=atom_mask,
            )
            frame_pyr, frame_pyr_mask = _get_frame_for_atoms(
                atoms=atoms["pyrimidine"],
                atom_pos=atom_pos,
                atom_mask=atom_mask,
            )

            frame = \
                frame_pur * pur_mask[:, None, None] + \
                frame_pyr * pyr_mask[:, None, None]

            frame_mask = \
                frame_pur_mask * pur_mask + \
                frame_pyr_mask * pyr_mask

            all_frames[:, FRAME_ORDER[frame_id], ...] = frame
            all_frames_mask[:, FRAME_ORDER[frame_id], ...] = frame_mask
        else:
            raise RuntimeError("Unexpected frame descriptor!")

    return all_frames, all_frames_mask
