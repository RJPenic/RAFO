import torch
import torch.nn as nn

from rafo.constants import ATOM_ORDER
from rafo.utils.binning import Binner
from rafo.utils.tensor import add

from openfold.model.primitives import LayerNorm


class Recycler(nn.Module):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        repr_atom: str,
    ):
        super().__init__()

        assert no_bins > 2, \
            f"Number of recycler bins must be greater than 2! (Got {no_bins})"

        self.binner = Binner(
            no_bins=no_bins,
            min_bin=min_bin,
            max_bin=max_bin
        )
        self.dist_embed = nn.Embedding(
            num_embeddings=no_bins,
            embedding_dim=c_z
        )
        self.repr_atom = repr_atom

        self.layer_norm_z = LayerNorm(c_z)
        self.layer_norm_m = LayerNorm(c_m)

    def forward(
        self,
        m_0: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute distogram and bin the values
        ref_pos = x[..., ATOM_ORDER[self.repr_atom], :]

        dist = torch.norm(
            ref_pos[..., None, :] - ref_pos[..., None, :, :],
            dim=-1
        )
        dist = self.dist_embed(
            self.binner.val2bin(dist)
        ).to(dtype=z.dtype)

        # Get recycle embeddings
        z_update = add(
            dist,
            self.layer_norm_z(z)
        )
        m_0_update = self.layer_norm_m(m_0)

        return m_0_update, z_update
