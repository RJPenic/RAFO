import math

import torch
import torch.nn.functional as F

from typing import Optional

from rafo.constants import (
    NUM_RESIDUE_TOKENS, ATOM_ORDER,
    NUM_ATOM_TYPES, TORSION_ORDER, BASE_ATOMS,
    BASE_TO_ATOMS_MAP, RESIDUE_TOKENS, RESIDUE_TOKEN_ORDER,
    BACKBONE_ATOMS, STRUCT_STATS_PATH
)
from rafo.utils.spatial.transformation import EuclideanTransformation
from rafo.utils.spatial.struct_constants import StructureConstants

from openfold.utils.rigid_utils import Rigid


class StructureConstructor:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        self.struct_constants = StructureConstants.from_json(STRUCT_STATS_PATH)

        # - - - Root frame (C4') seed local positions - - -
        # C5' lies on the -x axis at one bond length.
        self.c5_local_position = torch.tensor(
            [-self.struct_constants.c5_c4_distance, 0.0, 0.0]
        )
        # O4' lies in the +xy half-plane at angle C5'-C4'-O4' from the -x axis.
        self.o4_local_position = torch.tensor([
            (
                -self.struct_constants.c4_o4_distance *
                math.cos(self.struct_constants.c5_c4_o4_angle)
            ),
            (
                self.struct_constants.c4_o4_distance *
                math.sin(self.struct_constants.c5_c4_o4_angle)
            ),
            0.0,
        ])

        # - - - Atom-to-atom transitions - - -
        # C5' -> O5' -> P
        self.c5_to_o5_transform = \
            EuclideanTransformation.rot_z(
                torch.pi - self.struct_constants.o5_c5_c4_angle
            ) @ \
            EuclideanTransformation.trans_x(
                self.struct_constants.o5_c5_distance
            )

        self.o5_to_p_transform = \
            EuclideanTransformation.rot_z(
                torch.pi - self.struct_constants.p_o5_c5_angle
            ) @ \
            EuclideanTransformation.trans_x(
                self.struct_constants.p_o5_distance
            )

        # O4' -> C1'
        self.o4_to_c1_transform = \
            EuclideanTransformation.rot_z(
                torch.pi - self.struct_constants.c4_o4_c1_angle
            ) @ \
            EuclideanTransformation.trans_x(
                self.struct_constants.c1_o4_distance
            )

        # C4' -> C3' -> {C2', O3'}
        self.c4_to_c3_transform = \
            EuclideanTransformation.rot_z(
                torch.pi - self.struct_constants.c5_c4_c3_angle
            ) @ \
            EuclideanTransformation.trans_x(
                self.struct_constants.c4_c3_distance
            )

        self.c3_to_c2_transform = \
            EuclideanTransformation.rot_z(
                torch.pi - self.struct_constants.c4_c3_c2_angle
            ) @ \
            EuclideanTransformation.trans_x(
                self.struct_constants.c3_c2_distance
            )

        self.c3_to_o3_transform = \
            EuclideanTransformation.rot_z(
                torch.pi - self.struct_constants.c4_c3_o3_angle
            ) @ \
            EuclideanTransformation.trans_x(
                self.struct_constants.c3_o3_distance
            )

        # C1' -> N
        self.c1_to_n_transform = \
            EuclideanTransformation.rot_z(
                torch.pi - self.struct_constants.o4_c1_n_angle
            ) @ \
            EuclideanTransformation.trans_x(
                self.struct_constants.c1_n_distance
            )

        # Nucleobase atoms local positions
        self.base_local_positions = torch.tensor(
            self.struct_constants.base_local_positions
        )

        # Oxygen atoms local positions
        self.o2_local_position = torch.tensor(
            self.struct_constants.o2_local_position
        )
        self.op1_local_position = torch.tensor(
            self.struct_constants.op1_local_position
        )
        self.op2_local_position = torch.tensor(
            self.struct_constants.op2_local_position
        )

        # Is-nucleobase-atom mask
        self.base_masks = torch.zeros(NUM_RESIDUE_TOKENS, NUM_ATOM_TYPES)
        for tkn in RESIDUE_TOKENS:
            if tkn in BASE_TO_ATOMS_MAP:
                for atom in BASE_TO_ATOMS_MAP[tkn]:
                    if atom in BASE_ATOMS:
                        self.base_masks[
                            RESIDUE_TOKEN_ORDER[tkn], ATOM_ORDER[atom]
                        ] = 1.

        # Is-backbone-atom mask
        self.backbone_mask = torch.zeros(NUM_ATOM_TYPES)
        for atom in BACKBONE_ATOMS:
            self.backbone_mask[ATOM_ORDER[atom]] = 1.

        # Type/Device handling
        self.device = device
        self.dtype = dtype
        self.to(device, dtype)

    def to(self, device, dtype):
        self.device = device
        self.dtype = dtype

        # Root frame seed positions (tensors)
        self.c5_local_position = self.c5_local_position.to(device, dtype)
        self.o4_local_position = self.o4_local_position.to(device, dtype)

        # Transformations (Rigids)
        self.c5_to_o5_transform = self.c5_to_o5_transform.to(device, dtype)
        self.o5_to_p_transform = self.o5_to_p_transform.to(device, dtype)
        self.o4_to_c1_transform = self.o4_to_c1_transform.to(device, dtype)
        self.c4_to_c3_transform = self.c4_to_c3_transform.to(device, dtype)
        self.c3_to_c2_transform = self.c3_to_c2_transform.to(device, dtype)
        self.c3_to_o3_transform = self.c3_to_o3_transform.to(device, dtype)
        self.c1_to_n_transform = self.c1_to_n_transform.to(device, dtype)

        # Local positions (tensors)
        self.base_local_positions = self.base_local_positions.to(device, dtype)

        self.o2_local_position = self.o2_local_position.to(device, dtype)
        self.op1_local_position = self.op1_local_position.to(device, dtype)
        self.op2_local_position = self.op2_local_position.to(device, dtype)

        # Atom masks (tensor)
        self.base_masks = self.base_masks.to(device, dtype)
        self.backbone_mask = self.backbone_mask.to(device, dtype)

    def _attach_base(
        self,
        atom_positions: torch.Tensor,
        seq_encoded: torch.Tensor,
        n_transform: Rigid,
    ) -> torch.Tensor:
        # Convert local coordinates to global frame
        base_pos = n_transform[..., None, None].apply(
            self.base_local_positions
        )
        seq_oh = F.one_hot(seq_encoded, num_classes=NUM_RESIDUE_TOKENS)
        base_pos = seq_oh[..., None, None] * base_pos

        # Remove residue type dimension
        base_pos = base_pos.sum(dim=-3)

        # Calculate base mask
        base_mask = self.base_masks * seq_oh[..., None]
        base_mask = base_mask.sum(dim=-2).unsqueeze(-1)

        # Set base atom coordinates
        atom_positions = atom_positions * (1 - base_mask)
        atom_positions = atom_positions + base_mask * base_pos

        return atom_positions

    def _attach_oxygen_atoms(
        self,
        atom_positions: torch.Tensor,
        op_frame: Rigid,
        o2_frame: Rigid
    ) -> torch.Tensor:
        # OP1/OP2 are rigid in the phosphate frame [O5', P, OP1] (op_frame),
        # O2' is rigid in the in-ring frame [C3', C2', C1'] (o2_frame).
        o2_pos = o2_frame.apply(self.o2_local_position)
        op1_pos = op_frame.apply(self.op1_local_position)
        op2_pos = op_frame.apply(self.op2_local_position)

        atom_positions[..., ATOM_ORDER["O2'"], :] = o2_pos
        atom_positions[..., ATOM_ORDER["OP1"], :] = op1_pos
        atom_positions[..., ATOM_ORDER["OP2"], :] = op2_pos

        return atom_positions

    def _get_next_transformation(
        self,
        prev_transformation: EuclideanTransformation,
        trsn_angle: torch.Tensor,
        transition_transformation: EuclideanTransformation,
    ) -> EuclideanTransformation:
        # "Move" to the next atom
        return \
            prev_transformation @ \
            EuclideanTransformation.rot_x(
                trsn_angle
            ) @ \
            transition_transformation

    def _get_atom_mask(
        self,
        seq_encoded: torch.Tensor,
    ) -> torch.Tensor:
        seq_oh = F.one_hot(seq_encoded, num_classes=NUM_RESIDUE_TOKENS)

        # Base atoms mask
        base_mask = self.base_masks * seq_oh[..., None]
        base_mask = base_mask.sum(dim=-2)

        # Final predicted atoms mask
        atom_mask = base_mask + self.backbone_mask
        atom_mask = atom_mask.clamp(max=1.0)

        return atom_mask

    def construct_structure(
        self,
        root_transform: Rigid,
        trsn_angls: torch.Tensor,
        seq_encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, EuclideanTransformation]:
        # To prevent dtype discrepancies when using mixed-precision
        trsn_angls = trsn_angls.to(dtype=root_transform.dtype)

        # Move to device or convert to certain data type (if needed)
        if trsn_angls.device != self.device or trsn_angls.dtype != self.dtype:
            self.to(device=trsn_angls.device, dtype=trsn_angls.dtype)

        # Initialize root (C4') transformation
        c4_transform = EuclideanTransformation.from_rigid(root_transform)

        # Initialize atom positions tensor
        atom_positions = trsn_angls.new_zeros(
            trsn_angls.shape[:-2] + (NUM_ATOM_TYPES, 3),
        )

        # - - - Root frame atoms: C4' (origin), C5' (-x), O4' (xy-plane) - - -
        c4_pos = c4_transform.get_trans()
        c5_pos = c4_transform.apply(self.c5_local_position)
        o4_pos = c4_transform.apply(self.o4_local_position)

        atom_positions[..., ATOM_ORDER["C4'"], :] = c4_pos
        atom_positions[..., ATOM_ORDER["C5'"], :] = c5_pos
        atom_positions[..., ATOM_ORDER["O4'"], :] = o4_pos

        # Frames rooted at C5' (+x: C4'->C5') and O4' (+x: C4'->O4')
        c5_transform = EuclideanTransformation.from_3_points(
            c4_pos, c5_pos, o4_pos
        )
        o4_transform = EuclideanTransformation.from_3_points(
            c4_pos, o4_pos, c5_pos
        )

        # - - -  C5' -> O5' -> P - - -
        o5_transform = self._get_next_transformation(
            c5_transform,
            trsn_angls[..., TORSION_ORDER["O4'_C4'_C5'_O5'"], :],
            self.c5_to_o5_transform
        )
        atom_positions[..., ATOM_ORDER["O5'"], :] = o5_transform.get_trans()

        p_transform = self._get_next_transformation(
            o5_transform,
            trsn_angls[..., TORSION_ORDER["C4'_C5'_O5'_P"], :],
            self.o5_to_p_transform
        )
        atom_positions[..., ATOM_ORDER["P"], :] = p_transform.get_trans()

        op_frame = p_transform @ EuclideanTransformation.rot_x(
            trsn_angls[..., TORSION_ORDER["C5'_O5'_P_OP1"], :]
        )

        # - - - O4' -> C1' - - -
        c1_transform = self._get_next_transformation(
            o4_transform,
            trsn_angls[..., TORSION_ORDER["C5'_C4'_O4'_C1'"], :],
            self.o4_to_c1_transform
        )
        atom_positions[..., ATOM_ORDER["C1'"], :] = c1_transform.get_trans()

        # - - - C4' -> C3' -> {C2', O3'} - - -
        c3_transform = self._get_next_transformation(
            c4_transform,
            trsn_angls[..., TORSION_ORDER["O4'_C5'_C4'_C3'"], :],
            self.c4_to_c3_transform
        )
        atom_positions[..., ATOM_ORDER["C3'"], :] = c3_transform.get_trans()

        c2_transform = self._get_next_transformation(
            c3_transform,
            trsn_angls[..., TORSION_ORDER["C5'_C4'_C3'_C2'"], :],
            self.c3_to_c2_transform
        )
        atom_positions[..., ATOM_ORDER["C2'"], :] = c2_transform.get_trans()

        o3_transform = self._get_next_transformation(
            c3_transform,
            trsn_angls[..., TORSION_ORDER["C5'_C4'_C3'_O3'"], :],
            self.c3_to_o3_transform
        )
        atom_positions[..., ATOM_ORDER["O3'"], :] = o3_transform.get_trans()

        # - - - Base atoms (N1, N9, ...) - - -
        n_pre_torsion_transform = self._get_next_transformation(
            c1_transform,
            trsn_angls[..., TORSION_ORDER["C4'_O4'_C1'_N"], :],
            self.c1_to_n_transform
        )

        n_transform = \
            n_pre_torsion_transform @ \
            EuclideanTransformation.rot_x(
                trsn_angls[..., TORSION_ORDER["O4'_C1'_N_C"], :]
            )

        atom_positions = self._attach_base(
            atom_positions, seq_encoded, n_transform
        )

        # - - - Oxygen atoms (O2', OP1, OP2) - - -
        # O2' is placed in the rigid in-ring frame [C3', C2', C1'].
        o2_frame = EuclideanTransformation.from_3_points(
            c3_transform.get_trans(),
            c2_transform.get_trans(),
            c1_transform.get_trans(),
        )
        atom_positions = self._attach_oxygen_atoms(
            atom_positions, op_frame, o2_frame
        )

        # All frames, ordered to match constants.FRAMES:
        # C4', C5', O5', P, C1', C3', C2', O3', N_pre_torsion, N
        all_frames = EuclideanTransformation.cat(
            [
                c4_transform.unsqueeze(-1),
                c5_transform.unsqueeze(-1),
                o5_transform.unsqueeze(-1),
                p_transform.unsqueeze(-1),
                c1_transform.unsqueeze(-1),
                c3_transform.unsqueeze(-1),
                c2_transform.unsqueeze(-1),
                o3_transform.unsqueeze(-1),
                n_pre_torsion_transform.unsqueeze(-1),
                n_transform.unsqueeze(-1),
            ],
            dim=-1
        )

        atom_mask = self._get_atom_mask(seq_encoded=seq_encoded)

        return atom_positions, atom_mask, all_frames
