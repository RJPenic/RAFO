from typing import Optional

import torch
from torch import nn

from rafo.utils.spatial.struct_constructor import StructureConstructor

from openfold.model.structure_module import InvariantPointAttention
from openfold.utils.rigid_utils import Rigid
from openfold.model.primitives import LayerNorm


class BackboneUpdate(nn.Module):
    def __init__(
        self,
        c_s: int,
    ):
        super().__init__()

        self.linear = nn.Linear(c_s, 6)

    def forward(self, s):
        quat_trans = self.linear(s)
        return quat_trans


class AnglePredictorBlock(nn.Module):
    def __init__(
        self,
        c_hidden: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(c_hidden, c_hidden)
        )

    def forward(self, s):
        s = s + self.net(s)
        return s


class AnglePredictor(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_hidden: int,
        no_blocks: int,
        no_angles: int,
    ):
        super().__init__()

        self.linear_in = nn.Linear(c_s, c_hidden)
        self.linear_init = nn.Linear(c_s, c_hidden)

        self.blocks = nn.ModuleList(
            [
                AnglePredictorBlock(c_hidden)
                for _ in range(no_blocks)
            ]
        )

        self.linear_angls_out = nn.Linear(c_hidden, no_angles * 2)

    def forward(self, s, s_init):
        s_init = self.linear_init(torch.relu(s_init))
        s = self.linear_in(torch.relu(s))

        s = s + s_init

        for block in self.blocks:
            s = block(s)

        s = torch.relu(s)

        angles = self.linear_angls_out(s)  # [..., no_angles * 2]
        angles = angles.view(s.shape[:-1] + (-1, 2))  # [..., no_angles, 2]

        return angles


class TransitionBlock(nn.Module):
    def __init__(
        self,
        c: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(c, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, c),
        )

    def forward(self, s):
        s = s + self.net(s)
        return s


class Transition(nn.Module):
    def __init__(
        self,
        c: int,
        no_blocks: int,
        dropout: float,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransitionBlock(c)
                for _ in range(no_blocks)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(c)

    def forward(self, s):
        for block in self.blocks:
            s = block(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


def _merge_dicts(dict_list: list[dict[str, torch.Tensor]]):
    merged = {}
    for key in dict_list[0].keys():
        merged[key] = torch.stack([d[key] for d in dict_list])

    return merged


class StructureModule(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_ipa: int,
        c_angl: int,
        no_blocks: int,
        no_transition_blocks: int,
        no_resnet_blocks: int,
        no_angles: int,
        no_heads: int,
        no_qk_pts: int,
        no_v_pts: int,
        trans_scale_factor: float,
        dropout: float
    ):
        super().__init__()

        self.no_blocks = no_blocks

        # Input modules
        self.single_layer_norm = LayerNorm(c_s)
        self.pair_layer_norm = LayerNorm(c_z)
        self.linear_in = nn.Linear(c_s, c_s)

        # IPA
        self.ipa = InvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_ipa,
            no_heads=no_heads,
            no_qk_points=no_qk_pts,
            no_v_points=no_v_pts
        )
        self.ipa_dropout = nn.Dropout(p=dropout)
        self.ipa_layer_norm = LayerNorm(c_s)

        # Transition projections
        self.transition = Transition(
            c=c_s,
            no_blocks=no_transition_blocks,
            dropout=dropout,
        )

        # Frame "updater" and angle predictor
        self.bb_update = BackboneUpdate(c_s)
        self.trsn_angls_predictor = AnglePredictor(
            c_s=c_s,
            c_hidden=c_angl,
            no_blocks=no_resnet_blocks,
            no_angles=no_angles,
        )

        self.trans_scale_factor = trans_scale_factor

        self.structure_constructor = StructureConstructor()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        seq_encoded: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        if mask is None:
            mask = s.new_ones(s.shape[:-1])

        # Initial normalizations and projections
        s = self.single_layer_norm(s)
        z = self.pair_layer_norm(z)

        s_init = s

        s = self.linear_in(s)

        # Initialize frames
        rigids = Rigid.identity(
            shape=s.shape[:-1],
            dtype=s.dtype,
            device=s.device,
            requires_grad=self.training
        )

        outputs = []
        for _ in range(self.no_blocks):
            # IPA followed by dropout and normalization
            s = s + self.ipa(s, z, rigids, mask=mask)
            s = self.ipa_dropout(s)
            s = self.ipa_layer_norm(s)

            # Transition layer
            s = self.transition(s)

            # Update frames
            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # Get torsion angles (2D representations)
            trsn_angls = self.trsn_angls_predictor(s, s_init)

            # Scale translations and "pack" the outputs into dictionary
            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)
            outputs.append({
                "root_frames": scaled_rigids.to_tensor_7(),
                "torsion_angles": trsn_angls,
            })

            rigids = rigids.stop_rot_gradient()

        # Merge outputs list into a single dictionary
        outputs = _merge_dicts(outputs)

        atom_pos, atom_mask, all_frames = \
            self.structure_constructor.construct_structure(
                root_transform=scaled_rigids,
                trsn_angls=trsn_angls,
                seq_encoded=seq_encoded,
            )
        outputs["atom_positions"] = atom_pos
        outputs["atom_mask"] = atom_mask
        outputs["all_frames"] = all_frames.to_tensor_4x4()

        # Output final single representation
        outputs["single"] = s

        return outputs
