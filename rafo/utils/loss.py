import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from rafo.utils.binning import Binner
from rafo.constants import (
    ATOM_ORDER, STRUCT_STATS_PATH, RNA_VDW_RADII,
    LDDT_CUTOFF, NUM_ATOM_TYPES, RESIDUE_BONDS,
)
from rafo.utils.spatial.struct_constants import StructureConstants
from rafo.utils.tensor import select_atoms

from openfold.utils.loss import lddt, backbone_loss, compute_fape
from openfold.utils.rigid_utils import Rigid


class RNAStructureLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config.loss

        self.fape_loss = FAPE(self.config)
        self.torsion_loss = TorsionAngleLoss(self.config)
        self.distogram_loss = DistogramLoss(self.config)
        self.lddt_loss = LDDTLoss(self.config)
        self.violation_loss = ViolationLoss(self.config)

    def forward(self, model_outputs, batch):
        loss = {}
        loss["total"] = 0.0

        # FAPE
        loss["fape"] = self.fape_loss(
            pred_atom_pos=model_outputs["sm"]["atom_positions"],
            pred_root_frames=model_outputs["sm"]["root_frames"],
            pred_all_frames=model_outputs["sm"]["all_frames"],
            target_atom_pos=batch["atom_positions"],
            target_root_frames=batch["root_frames"],
            target_all_frames=batch["all_frames"],
            atom_mask=batch["atom_mask"],
            root_frames_mask=batch["root_frames_mask"],
            all_frames_mask=batch["all_frames_mask"],
        )
        loss["total"] = \
            loss["total"] + self.config.fape.weight * loss["fape"]

        # Torsion loss
        loss["torsion"] = self.torsion_loss(
            pred_torsions=model_outputs["sm"]["torsion_angles"],
            target_torsions=batch["torsion_angles"],
            torsions_mask=batch["torsion_angles_mask"],
        )
        loss["total"] = \
            loss["total"] + self.config.torsion.weight * loss["torsion"]

        # Distogram loss (one distogram per reference atom)
        loss["distogram"] = self.distogram_loss(
            logits=model_outputs["aux"]["dist_logits"],
            atom_positions=batch["atom_positions"],
            atom_mask=batch["atom_mask"],
            seq_tokens=batch["seq_tokens"],
        )

        loss["total"] = \
            loss["total"] + self.config.distogram.weight * loss["distogram"]

        # LDDT loss
        loss["lddt"] = self.lddt_loss(
            logits=model_outputs["aux"]["lddt_logits"],
            atom_pos_pred=model_outputs["sm"]["atom_positions"],
            atom_pos_target=batch["atom_positions"],
            atom_mask=batch["atom_mask"],
            resolution=batch["resolution"],
        )
        loss["total"] = \
            loss["total"] + self.config.lddt.weight * loss["lddt"]

        # Violation loss
        loss["violation"] = self.violation_loss(
            atom_pos=model_outputs["sm"]["atom_positions"],
            atom_exists_mask=model_outputs["sm"]["atom_mask"],
            res_idx=batch["residue_index"],
            chain_asym_id=batch["asym_id"],
        )
        loss["total"] = \
            loss["total"] + self.config.violation.weight * loss["violation"]

        return loss


class DistogramLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config.distogram

        self.ref_atoms = list(self.config.ref_atoms)
        self.dist_binner = Binner(**self.config.binner)

        self.eps = self.config.eps

    def forward(
        self,
        logits: torch.Tensor,
        atom_positions: torch.Tensor,
        atom_mask: torch.Tensor,
        seq_tokens: torch.Tensor,
    ):
        # Extract reference atoms
        ref_pos, ref_mask = select_atoms(
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            seq_tokens=seq_tokens,
            atom_names=self.ref_atoms,
        )

        # Get RNA distograms (reference atom distances) and bin the values
        dists = torch.sqrt(
            torch.sum(
                (ref_pos[..., :, None, :, :] - ref_pos[..., None, :, :, :])**2,
                dim=-1
            ) + self.eps  # eps for numerical stability
        )
        target_dist_bins = self.dist_binner.val2bin(dists)

        # Create pair mask
        pair_mask = ref_mask[..., :, None, :] * ref_mask[..., None, :, :]

        # Calculate loss
        errs = F.cross_entropy(
            input=logits.view(-1, logits.shape[-1]),
            target=target_dist_bins.view(-1),
            reduction='none'
        )

        # BLLD => B x L x L x D
        errs = errs.view(*logits.shape[:-1])

        # Mask out undefined values
        loss = errs * pair_mask
        loss_num = torch.sum(loss, dim=(-3, -2, -1))
        loss_denom = (self.eps + torch.sum(pair_mask, dim=(-3, -2, -1)))
        loss = loss_num / loss_denom

        # Batch mean
        loss = torch.mean(loss)

        return loss


class LDDTLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config.lddt
        self.lddt_binner = Binner(**self.config.binner)

        self.eps = self.config.eps
        self.min_resolution = self.config.min_resolution
        self.max_resolution = self.config.max_resolution

        self.atom = self.config.atom

    def forward(
        self,
        logits: torch.Tensor,
        atom_pos_pred: torch.Tensor,
        atom_pos_target: torch.Tensor,
        atom_mask: torch.Tensor,
        resolution: torch.Tensor
    ):
        # Extract reference atom coordinates and mask
        ref_pos_pred = atom_pos_pred[..., ATOM_ORDER[self.atom], :]
        ref_pos_target = atom_pos_target[..., ATOM_ORDER[self.atom], :]
        ref_mask = atom_mask[..., ATOM_ORDER[self.atom]]

        # Get LDDT (for reference atoms) and bin the values
        lddt_ref = lddt(
            ref_pos_pred,
            ref_pos_target,
            ref_mask.unsqueeze(-1),
            per_residue=True,
            cutoff=LDDT_CUTOFF,
        )
        target_lddt_bins = self.lddt_binner.val2bin(lddt_ref)

        # Calculate loss
        errs = F.cross_entropy(
            input=logits.view(-1, logits.shape[-1]),
            target=target_lddt_bins.view(-1),
            reduction='none'
        )
        errs = errs.view(*logits.shape[:-1])

        loss = errs * ref_mask
        loss = \
            torch.sum(loss, dim=-1) / (self.eps + torch.sum(ref_mask, dim=-1))

        # Ignore bad resolution structures during loss calculation
        resolution_mask = \
            (resolution >= self.min_resolution) & \
            (resolution <= self.max_resolution)

        # Batch average
        loss = loss * resolution_mask
        loss = loss.sum() / (self.eps + resolution_mask.sum())

        return loss


class TorsionAngleLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config.torsion

        self.angle_weight = self.config.angle_weight
        self.norm_weight = self.config.norm_weight
        self.eps = self.config.eps

    def forward(
        self,
        pred_torsions: torch.Tensor,
        target_torsions: torch.Tensor,
        torsions_mask: torch.Tensor,
    ):
        pred_torsions_norm = pred_torsions.norm(dim=-1)
        pred_torsions_normalized = (
            pred_torsions /
            (pred_torsions_norm[..., None] + self.eps)
        )
        loss_denom = self.eps + torch.sum(torsions_mask, dim=(-1, -2))

        # Torsion angle loss
        err_angle = torch.square(
            pred_torsions_normalized - target_torsions
        ).sum(dim=-1)
        loss_angle = torch.sum(
            err_angle * torsions_mask, dim=(-1, -2)
        ) / loss_denom

        # Angle norm loss
        err_norm = torch.abs(pred_torsions_norm - 1.0)

        loss_norm = torch.sum(
            err_norm * torsions_mask, dim=(-1, -2)
        ) / loss_denom

        # Total
        loss = self.angle_weight * loss_angle + self.norm_weight * loss_norm
        loss = torch.mean(loss)

        return loss


class RootFAPE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config.root_fape

        self.clamp_distance = self.config.clamp_distance
        self.length_scale = self.config.length_scale
        self.clamp_prob = self.config.clamp_prob
        self.eps = self.config.eps

    def forward(
        self,
        pred_frames: torch.Tensor,
        target_frames: torch.Tensor,
        frames_mask: torch.Tensor,
    ):
        # Create batch clamping mask
        clamp_mask = torch.full(
            frames_mask.shape[:-1],
            device=target_frames.device,
            dtype=target_frames.dtype,
            fill_value=self.clamp_prob
        )
        clamp_mask = torch.bernoulli(clamp_mask)

        # FAPE loss
        loss = backbone_loss(
            backbone_rigid_tensor=target_frames,
            backbone_rigid_mask=frames_mask,
            traj=pred_frames,
            use_clamped_fape=clamp_mask,
            clamp_distance=self.clamp_distance,
            loss_unit_distance=self.length_scale,
            eps=self.eps,
        )
        loss = torch.mean(loss)

        return loss


class AllFAPE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config.all_fape

        self.clamp_distance = self.config.clamp_distance
        self.length_scale = self.config.length_scale
        self.eps = self.config.eps

    def forward(
        self,
        pred_atom_pos: torch.Tensor,
        pred_frames: torch.Tensor,
        target_atom_pos: torch.Tensor,
        target_frames: torch.Tensor,
        atom_mask: torch.Tensor,
        frames_mask: torch.Tensor,
    ):
        # Adjust frame shapes
        batch_dims = pred_atom_pos.shape[:-3]

        pred_frames = pred_frames.view(*batch_dims, -1, 4, 4)
        pred_frames = Rigid.from_tensor_4x4(pred_frames)
        target_frames = target_frames.view(*batch_dims, -1, 4, 4)
        target_frames = Rigid.from_tensor_4x4(target_frames)
        frames_mask = frames_mask.view(*batch_dims, -1)

        pred_atom_pos = pred_atom_pos.view(*batch_dims, -1, 3)
        target_atom_pos = target_atom_pos.view(*batch_dims, -1, 3)
        atom_mask = atom_mask.view(*batch_dims, -1)

        # FAPE
        loss = compute_fape(
            pred_frames=pred_frames,
            target_frames=target_frames,
            frames_mask=frames_mask,
            pred_positions=pred_atom_pos,
            target_positions=target_atom_pos,
            positions_mask=atom_mask,
            length_scale=self.length_scale,
            l1_clamp_distance=self.clamp_distance,
            eps=self.eps,
        )

        # Get batch mean and apply weight
        loss = torch.mean(loss)

        return loss


class FAPE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config.fape

        self.root_fape = RootFAPE(self.config)
        self.all_fape = AllFAPE(self.config)

        self.root_weight = self.config.root_fape.weight
        self.all_weight = self.config.all_fape.weight

    def forward(
        self,
        pred_atom_pos: torch.Tensor,
        pred_root_frames: torch.Tensor,
        pred_all_frames: torch.Tensor,
        target_atom_pos: torch.Tensor,
        target_root_frames: torch.Tensor,
        target_all_frames: torch.Tensor,
        atom_mask: torch.Tensor,
        root_frames_mask: torch.Tensor,
        all_frames_mask: torch.Tensor,
    ):
        root_loss = self.root_fape(
            pred_frames=pred_root_frames,
            target_frames=target_root_frames,
            frames_mask=root_frames_mask,
        )
        all_loss = self.all_fape(
            pred_atom_pos=pred_atom_pos,
            pred_frames=pred_all_frames,
            target_atom_pos=target_atom_pos,
            target_frames=target_all_frames,
            atom_mask=atom_mask,
            frames_mask=all_frames_mask,
        )

        return self.root_weight * root_loss + self.all_weight * all_loss


class ClashLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config.clash

        self.tolerance = self.config.tolerance
        self.eps = self.config.eps

        # Minimum unbonded distances according to VdW radii
        vdw_radii = np.array(RNA_VDW_RADII, dtype=np.float32)
        min_unbonded_dist = vdw_radii[:, None] + vdw_radii[None, :]
        self.register_buffer(
            "min_unbonded_dist", torch.tensor(min_unbonded_dist),
            persistent=False,
        )

        # Intra-residue bond mask
        # NOTE: This only excludes self-pairs and directly bonded (1-2) atom
        # pairs. Angle neighbours (1-3, two atoms sharing a common bonded
        # neighbour) are NOT excluded -- they sit ~2.4 A apart by bond
        # geometry, which is below the VdW sum (~3.2 A) and would
        # register as a clash. With the current tolerance (~1.5 A) the
        # clash threshold (VdW_sum - tolerance ~= 1.7 A) stays below
        # 2.4 A, so 1-3 pairs are not flagged and this is fine.
        # If the tolerance is lowered, add an explicit 1-3 exclusion
        # here (derive angle pairs from RESIDUE_BONDS), otherwise
        # the loss will start penalising correct geometry.
        no_bond_mask = np.ones(
            (NUM_ATOM_TYPES, NUM_ATOM_TYPES),
            dtype=np.float32
        )
        np.fill_diagonal(no_bond_mask, 0.0)  # Handle "self-pairings"

        for atom1, atom2 in RESIDUE_BONDS:
            no_bond_mask[ATOM_ORDER[atom1], ATOM_ORDER[atom2]] = 0.0
            no_bond_mask[ATOM_ORDER[atom2], ATOM_ORDER[atom1]] = 0.0

        self.register_buffer(
            "no_bond_mask", torch.tensor(no_bond_mask),
            persistent=False,
        )

    def forward(
        self,
        atom_pos: torch.Tensor,
        atom_exists_mask: torch.Tensor,
        res_idx: torch.Tensor,
        chain_asym_id: torch.Tensor,
    ):
        # Inter-residue atom distances
        dists = torch.norm(
            (
                atom_pos[..., None, :, None, :] -
                atom_pos[..., None, :, None, :, :]
            ),
            dim=-1
        )

        # Inter-residue atom pair exists mask
        pair_exists_mask = (
            atom_exists_mask[..., None, :, None, :] *
            atom_exists_mask[..., None, :, None]
        )

        # Mask bonded intra-residue pairings
        seq_len = atom_pos.shape[-3]
        diag_idxs = torch.arange(seq_len, device=pair_exists_mask.device)

        pair_exists_mask[..., diag_idxs, diag_idxs, :, :] = \
            pair_exists_mask[..., diag_idxs, diag_idxs, :, :] * \
            self.no_bond_mask

        # Neighbor residue mask: 1 if residues are consecutive and same chain
        # (i.e., a real O3'-P covalent bond exists), 0 otherwise.
        res_idx_diff = torch.abs(res_idx[..., 1:] - res_idx[..., :-1])
        bonded_res_mask = 2 - res_idx_diff.clamp(min=1, max=2)

        same_chain_mask = 1 - torch.clamp(
            torch.abs(
                chain_asym_id[..., 1:] -
                chain_asym_id[..., :-1]
            ),
            min=0,
            max=1.,
        )

        bonded_res_mask = bonded_res_mask * same_chain_mask

        # Zero out inter-residue bond pairs (O3' - P) from the clash loss
        pair_exists_mask[
            ...,
            diag_idxs[:-1] + 1,
            diag_idxs[1:] - 1,
            ATOM_ORDER["P"], ATOM_ORDER["O3'"]
        ] *= (1.0 - bonded_res_mask)

        pair_exists_mask[
            ...,
            diag_idxs[1:] - 1,
            diag_idxs[:-1] + 1,
            ATOM_ORDER["O3'"], ATOM_ORDER["P"]
        ] *= (1.0 - bonded_res_mask)

        # Clash error
        clash_err = torch.relu(
            (
                self.min_unbonded_dist -
                self.tolerance -
                dists
            )
        )
        del dists

        # Ignore undefined/non-existing atom pairs
        clash_err = clash_err * pair_exists_mask
        del pair_exists_mask

        # Average error (clashes)
        clash_err_sum_per_atom = torch.sum(clash_err, dim=(-1, -3))
        num_clash_per_atom = torch.sum(clash_err > self.eps, dim=(-1, -3))

        clash_err_per_atom = \
            clash_err_sum_per_atom / (self.eps + num_clash_per_atom)

        # NOTE: Atom average was used (instead of batch)
        # Loss = Average error (atoms)
        loss = (
            torch.sum(clash_err_per_atom) /
            (self.eps + torch.sum(atom_exists_mask))
        )

        return loss


class InterResidueLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config.inter_residue

        self.tolerance = self.config.tolerance
        self.eps = self.config.eps

        self.struct_consts = StructureConstants.from_json(STRUCT_STATS_PATH)

    def forward(
        self,
        atom_pos: torch.Tensor,
        atom_exists_mask: torch.Tensor,
        res_idx: torch.Tensor,
        chain_asym_id: torch.Tensor,
    ):
        o3_pos = atom_pos[..., :-1, ATOM_ORDER["O3'"], :]
        p_pos = atom_pos[..., 1:, ATOM_ORDER["P"], :]

        # Neighboring residues mask
        bond_exists_mask = torch.abs(res_idx[..., 1:] - res_idx[..., :-1])
        bond_exists_mask = bond_exists_mask.clamp(min=1, max=2) - 1
        bond_exists_mask = 1 - bond_exists_mask

        same_chain_mask = 1 - torch.clamp(
            torch.abs(
                chain_asym_id[..., 1:] -
                chain_asym_id[..., :-1]
            ),
            min=0,
            max=1.,
        )
        bond_exists_mask = bond_exists_mask * same_chain_mask

        # Bond exists only if both bonded atoms (O3' of i, P of i+1) exist
        o3_exists = atom_exists_mask[..., ATOM_ORDER["O3'"]]
        p_exists = atom_exists_mask[..., ATOM_ORDER["P"]]
        bond_exists_mask = \
            bond_exists_mask * o3_exists[..., :-1] * p_exists[..., 1:]

        # Predicted inter-residue distances (O3'-P bond)
        bond_lens = torch.norm(o3_pos - p_pos, dim=-1)

        # max(|l_pred - l_ideal| - tolerance, 0.0)
        len_err = torch.clamp(
            (
                torch.abs(bond_lens - self.struct_consts.p_o3_distance) -
                self.tolerance
            ),
            min=0.0
        )

        # Average per batch
        loss = (
            torch.sum(len_err * bond_exists_mask, dim=-1) /
            (self.eps + torch.sum(bond_exists_mask, dim=-1))
        )

        # Batch mean
        loss = torch.mean(loss)

        return loss


class ViolationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config.violation

        self.clash_loss = ClashLoss(self.config)
        self.inter_res_loss = InterResidueLoss(self.config)

        self.clash_weight = self.config.clash.weight
        self.inter_res_weight = self.config.inter_residue.weight

    def forward(
        self,
        atom_pos: torch.Tensor,
        atom_exists_mask: torch.Tensor,
        res_idx: torch.Tensor,
        chain_asym_id: torch.Tensor,
    ):
        clash_loss = self.clash_loss(
            atom_pos,
            atom_exists_mask,
            res_idx,
            chain_asym_id,
        )

        inter_res_loss = self.inter_res_loss(
            atom_pos,
            atom_exists_mask,
            res_idx,
            chain_asym_id,
        )

        return (
            self.clash_weight * clash_loss
            + self.inter_res_weight * inter_res_loss
        )
