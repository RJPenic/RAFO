import torch

from rafo.constants import (
    ATOM_ORDER, LDDT_CUTOFF,
    RMSD_REF_ATOM, LDDT_REF_ATOM
)
from rafo.utils.external_tools import run_usalign

from openfold.utils.superimposition import superimpose
from openfold.utils.loss import lddt


def compute_tm_score(
    target_atom_pos: torch.Tensor,
    pred_atom_pos: torch.Tensor,
    atom_mask: torch.Tensor,
    batch_reduce: bool = True,
) -> torch.Tensor:
    batch_size = atom_mask.shape[0]

    tm_score = torch.zeros(
        batch_size,
        dtype=torch.float32,
        device=pred_atom_pos.device
    )

    for i in range(batch_size):
        tm_score[i] = run_usalign(
            query_atom_pos=pred_atom_pos[i],
            query_atom_mask=atom_mask[i],
            target_atom_pos=target_atom_pos[i],
            target_atom_mask=atom_mask[i],
            force_res_order=True,
        )

    if batch_reduce:
        tm_score = tm_score.mean()

    return tm_score


def compute_rmsd(
    target_atom_pos: torch.Tensor,
    pred_atom_pos: torch.Tensor,
    atom_mask: torch.Tensor,
    atom: str = RMSD_REF_ATOM,
    batch_reduce: bool = True,
) -> torch.Tensor:
    _, rmsd = superimpose(
        reference=target_atom_pos[..., ATOM_ORDER[atom], :],
        coords=pred_atom_pos[..., ATOM_ORDER[atom], :],
        mask=atom_mask[..., ATOM_ORDER[atom]],
    )

    if batch_reduce:
        rmsd = rmsd.mean()

    return rmsd


def compute_lddt(
    target_atom_pos: torch.Tensor,
    pred_atom_pos: torch.Tensor,
    atom_mask: torch.Tensor,
    atom: str = LDDT_REF_ATOM,
    batch_reduce: bool = True,
) -> torch.Tensor:
    lddt_val = lddt(
        all_atom_positions=target_atom_pos[..., ATOM_ORDER[atom], :],
        all_atom_pred_pos=pred_atom_pos[..., ATOM_ORDER[atom], :],
        all_atom_mask=atom_mask[..., ATOM_ORDER[atom]].unsqueeze(-1),
        per_residue=False,
        cutoff=LDDT_CUTOFF,
    )

    if batch_reduce:
        lddt_val = lddt_val.mean()

    return lddt_val
