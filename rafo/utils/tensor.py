import torch
from functools import reduce

from rafo.constants import (
    ATOM_ORDER, GLYCOSIDIC_N_ATOM,
    PURINE_N_ATOM, PYRIMIDINE_N_ATOM,
    PURINE_TOKENS, RESIDUE_TOKEN_ORDER
)


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled():
        a = a + b
    else:
        a += b

    return a


def select_atoms(
    atom_positions: torch.Tensor,
    atom_mask: torch.Tensor,
    seq_tokens: torch.Tensor,
    atom_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    sel_atom_idx = [
        ATOM_ORDER[atom]
        if atom != GLYCOSIDIC_N_ATOM else ATOM_ORDER[PYRIMIDINE_N_ATOM]
        for atom in atom_names
    ]
    sel_atom_positions = atom_positions[..., sel_atom_idx, :].clone()
    sel_atom_mask = atom_mask[..., sel_atom_idx].clone()

    if GLYCOSIDIC_N_ATOM in atom_names:
        # Get glycosidic atom indices
        glycosidic_n_idcs = [
            i for i, atom in enumerate(atom_names)
            if atom == GLYCOSIDIC_N_ATOM
        ]

        # Create purine mask
        purine_mask = reduce(
            lambda x, y: x | y,
            [
                seq_tokens == RESIDUE_TOKEN_ORDER[pur_tkn]
                for pur_tkn in PURINE_TOKENS
            ]
        )

        # Extract and replace with purine glycosidic N positions
        purine_n_positions = atom_positions[..., ATOM_ORDER[PURINE_N_ATOM], :]
        purine_n_mask = atom_mask[..., ATOM_ORDER[PURINE_N_ATOM]]

        sel_atom_positions[..., glycosidic_n_idcs, :] = torch.where(
            purine_mask[..., None, None],
            purine_n_positions[..., None, :],
            sel_atom_positions[..., glycosidic_n_idcs, :]
        )
        sel_atom_mask[..., glycosidic_n_idcs] = torch.where(
            purine_mask[..., None],
            purine_n_mask[..., None],
            sel_atom_mask[..., glycosidic_n_idcs]
        )

    return sel_atom_positions, sel_atom_mask
