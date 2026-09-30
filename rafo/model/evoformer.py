import torch
from torch import nn

from typing import Optional
from functools import partial

from openfold.model.msa import (
    MSARowAttentionWithPairBias,
    MSAColumnAttention
)
from openfold.model.triangular_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode
)
from openfold.model.triangular_multiplicative_update import (
    FusedTriangleMultiplicationIncoming,
    FusedTriangleMultiplicationOutgoing
)
from openfold.model.dropout import (
    DropoutRowwise,
    DropoutColumnwise
)
from openfold.model.outer_product_mean import OuterProductMean
from openfold.model.primitives import LayerNorm

from rafo.utils.grad_checkpointing import checkpoint_blocks
from rafo.utils.tensor import add


class Evoformer(nn.Module):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_s: int,
        c_hidden_msa_attn: int,
        c_hidden_pair_mul: int,
        c_hidden_pair_attn: int,
        c_hidden_opm: int,
        transition_exp_factor: int,
        no_blocks: int,
        no_heads_msa: int,
        no_heads_pair: int,
        msa_dropout: float,
        pair_dropout: float,
        use_msa_col_attn: bool,
        blocks_per_ckpt: int,
    ):
        super().__init__()

        self.blocks_per_ckpt = blocks_per_ckpt

        self.blocks = nn.ModuleList(
            [
                EvoformerBlock(
                    c_m=c_m,
                    c_z=c_z,
                    c_hidden_msa_attn=c_hidden_msa_attn,
                    c_hidden_pair_mul=c_hidden_pair_mul,
                    c_hidden_pair_attn=c_hidden_pair_attn,
                    c_hidden_opm=c_hidden_opm,
                    transition_exp_factor=transition_exp_factor,
                    no_heads_msa=no_heads_msa,
                    no_heads_pair=no_heads_pair,
                    msa_dropout=msa_dropout,
                    pair_dropout=pair_dropout,
                    use_msa_col_attn=use_msa_col_attn,
                )
                for _ in range(no_blocks)
            ]
        )

        self.linear_single = nn.Linear(c_m, c_s)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
    ):
        m, z = checkpoint_blocks(
            blocks=[
                partial(
                    block,
                    msa_mask=msa_mask,
                    pair_mask=pair_mask,
                    chunk_size=chunk_size,
                    use_lma=use_lma,
                )
                for block in self.blocks
            ],
            args=(m, z),
            blocks_per_ckpt=self.blocks_per_ckpt,
        )

        s = self.linear_single(m[..., 0, :, :])

        return m, z, s


class EvoformerBlock(nn.Module):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_attn: int,
        c_hidden_pair_mul: int,
        c_hidden_pair_attn: int,
        c_hidden_opm: int,
        transition_exp_factor: int,
        no_heads_msa: int,
        no_heads_pair: int,
        msa_dropout: float,
        pair_dropout: float,
        use_msa_col_attn: bool,
    ):
        super().__init__()

        self.msa_stack = MSAStack(
            c_m=c_m,
            c_z=c_z,
            c_hidden_attn=c_hidden_msa_attn,
            transition_exp_factor=transition_exp_factor,
            no_heads=no_heads_msa,
            dropout=msa_dropout,
            use_col_attn=use_msa_col_attn,
        )
        self.pair_stack = PairStack(
            c_z=c_z,
            c_hidden_mul=c_hidden_pair_mul,
            c_hidden_attn=c_hidden_pair_attn,
            transition_exp_factor=transition_exp_factor,
            no_heads=no_heads_pair,
            dropout=pair_dropout,
        )

        self.out_prod_mean = OuterProductMean(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_opm
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
    ):
        m = self.msa_stack(
            m, z, mask=msa_mask,
            chunk_size=chunk_size, use_lma=use_lma,
        )

        z = add(
            z,
            self.out_prod_mean(
                m, mask=msa_mask,
                chunk_size=chunk_size, inplace_safe=not self.training
            )
        )
        z = self.pair_stack(
            z, mask=pair_mask,
            chunk_size=chunk_size, use_lma=use_lma
        )

        return m, z


class Transition(nn.Module):
    def __init__(
        self,
        c: int,
        exp_factor: int
    ):
        super().__init__()

        self.layer_norm = LayerNorm(c)

        self.linear_a = nn.Linear(c, exp_factor * c)
        self.linear_b = nn.Linear(c, exp_factor * c)
        self.swish = nn.SiLU()

        self.linear_out = nn.Linear(exp_factor * c, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.swish(self.linear_a(x)) * self.linear_b(x)
        x = self.linear_out(x)

        return x


class MSAStack(nn.Module):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_attn: int,
        transition_exp_factor: int,
        no_heads: int,
        dropout: float,
        use_col_attn: bool,
    ):
        super().__init__()

        # Axial attention
        self.row_attn = MSARowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_attn,
            no_heads=no_heads,
        )

        self.use_col_attn = use_col_attn

        if self.use_col_attn:
            self.col_attn = MSAColumnAttention(
                c_m=c_m,
                c_hidden=c_hidden_attn,
                no_heads=no_heads,
            )

        # Dropouts
        self.row_dropout = DropoutRowwise(dropout)

        # MLP
        self.transition = Transition(c=c_m, exp_factor=transition_exp_factor)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
    ):
        m = add(
            m,
            self.row_dropout(
                self.row_attn(
                    m=m, z=z, mask=mask,
                    chunk_size=chunk_size, use_lma=use_lma,
                    inplace_safe=not self.training,
                )
            )
        )

        if self.use_col_attn:
            m = add(
                m,
                self.col_attn(
                    m=m, mask=mask,
                    chunk_size=chunk_size, use_lma=use_lma,
                    inplace_safe=not self.training,
                )
            )

        m = add(m, self.transition(m))

        return m


class PairStack(nn.Module):
    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_attn: int,
        transition_exp_factor: int,
        no_heads: int,
        dropout: float
    ):
        super().__init__()

        # Triangular multiplicative update
        self.tri_mul_out = FusedTriangleMultiplicationOutgoing(
            c_z=c_z,
            c_hidden=c_hidden_mul,
        )
        self.tri_mul_in = FusedTriangleMultiplicationIncoming(
            c_z=c_z,
            c_hidden=c_hidden_mul,
        )

        # Triangular self-attention
        self.tri_attn_start = TriangleAttentionStartingNode(
            c_in=c_z,
            c_hidden=c_hidden_attn,
            no_heads=no_heads,
        )
        self.tri_attn_end = TriangleAttentionEndingNode(
            c_in=c_z,
            c_hidden=c_hidden_attn,
            no_heads=no_heads,
        )

        # Dropout
        self.row_dropout = DropoutRowwise(dropout)
        self.col_dropout = DropoutColumnwise(dropout)

        # MLP
        self.transition = Transition(c=c_z, exp_factor=transition_exp_factor)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
    ):
        z = add(
            z,
            self.row_dropout(
                self.tri_mul_out(z, mask=mask)
            )
        )
        z = add(
            z,
            self.col_dropout(
                self.tri_mul_in(z, mask=mask)
            )
        )

        z = add(
            z,
            self.row_dropout(
                self.tri_attn_start(
                    z, mask=mask,
                    chunk_size=chunk_size, use_lma=use_lma,
                )
            )
        )
        z = add(
            z,
            self.col_dropout(
                self.tri_attn_end(
                    z, mask=mask,
                    chunk_size=chunk_size, use_lma=use_lma,
                )
            )
        )

        z = add(z, self.transition(z))

        return z
