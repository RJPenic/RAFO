import torch
from torch import nn
import torch.nn.functional as F

from rafo.utils.binning import Binner
from rafo.utils.seq import RINALMO_EMBED_DIM
from rafo.utils.sec_struct import SS_FEAT_DIM
from rafo.constants import NUM_RESIDUE_TOKENS

from openfold.model.primitives import LayerNorm


def _modality_dropout(
    x: torch.Tensor,
    p: float,
    n_feature_dims: int,
    training: bool,
) -> torch.Tensor:
    if not training or p <= 0.0:
        return x

    batch_shape = x.shape[:-n_feature_dims]

    keep = (torch.rand(batch_shape, device=x.device) >= p).to(x.dtype)
    keep = keep.view(batch_shape + (1,) * n_feature_dims)

    return x * keep


class InputEmbedder(nn.Module):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        rel_res_win_size: int,
        rel_chain_win_size: int,
        lm_dropout: float,
        ss_dropout: float,
    ):
        super().__init__()

        self.seq_embedder = SeqEmbedder(
            c_m=c_m,
            lm_dropout=lm_dropout,
        )

        self.pair_embedder = PairEmbedder(
            c_z=c_z,
            rel_res_win_size=rel_res_win_size,
            rel_chain_win_size=rel_chain_win_size,
            ss_dropout=ss_dropout,
        )

    def forward(
        self,
        seq_tokens: torch.Tensor,
        lm_embed: torch.Tensor,
        res_idx: torch.Tensor,
        asym_id: torch.Tensor,
        sym_id: torch.Tensor,
        entity_id: torch.Tensor,
        ss_feat: torch.Tensor,
    ):
        seq_feat = \
            F.one_hot(seq_tokens, num_classes=NUM_RESIDUE_TOKENS).float()

        m = self.seq_embedder(
            seq_feat=seq_feat,
            lm_embed=lm_embed,
        )
        m = m.unsqueeze(dim=-3)

        z = self.pair_embedder(
            seq_feat=seq_feat,
            res_idx=res_idx,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
            ss_feat=ss_feat,
        )

        return m, z


class SeqEmbedder(nn.Module):
    def __init__(
        self,
        c_m: int,
        lm_dropout: float,
    ):
        super().__init__()

        self.lm_dropout = lm_dropout

        self.mlp_lm = nn.Sequential(
            LayerNorm(RINALMO_EMBED_DIM),
            nn.Linear(RINALMO_EMBED_DIM, c_m),
            nn.GELU(),
            nn.Linear(c_m, c_m),
        )

        self.linear_seq = nn.Linear(NUM_RESIDUE_TOKENS, c_m)

    def forward(
        self,
        seq_feat: torch.Tensor,
        lm_embed: torch.Tensor,
    ):
        m = self.linear_seq(seq_feat)

        lm_contrib = self.mlp_lm(lm_embed)
        lm_contrib = _modality_dropout(
            lm_contrib,
            p=self.lm_dropout,
            n_feature_dims=2,
            training=self.training,
        )
        m = m + lm_contrib

        return m


class PairEmbedder(nn.Module):
    def __init__(
        self,
        c_z: int,
        rel_res_win_size: int,
        rel_chain_win_size: int,
        ss_dropout: float,
    ):
        super().__init__()

        self.ss_dropout = ss_dropout

        self.linear_z_seq_a = nn.Linear(NUM_RESIDUE_TOKENS, c_z)
        self.linear_z_seq_b = nn.Linear(NUM_RESIDUE_TOKENS, c_z)

        self.mlp_ss = nn.Sequential(
            nn.Linear(SS_FEAT_DIM, c_z // 2),
            nn.GELU(),
            nn.Linear(c_z // 2, c_z),
        )

        self.relpos = RelativePositionalEncoding(
            c_z=c_z,
            rel_res_win_size=rel_res_win_size,
            rel_chain_win_size=rel_chain_win_size
        )

    def forward(
        self,
        seq_feat: torch.Tensor,
        res_idx: torch.Tensor,
        asym_id: torch.Tensor,
        sym_id: torch.Tensor,
        entity_id: torch.Tensor,
        ss_feat: torch.Tensor,
    ):
        a = self.linear_z_seq_a(seq_feat)
        b = self.linear_z_seq_b(seq_feat)
        z = a[..., None, :] + b[..., None, :, :]

        z = z + self.relpos(
            res_idx=res_idx,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id
        ).to(z.dtype)

        ss_contrib = self.mlp_ss(ss_feat)
        ss_contrib = _modality_dropout(
            ss_contrib,
            p=self.ss_dropout,
            n_feature_dims=3,
            training=self.training,
        )
        z = z + ss_contrib

        return z


class RelativePositionalEncoding(nn.Module):
    def __init__(
        self,
        c_z: int,
        rel_res_win_size: int,
        rel_chain_win_size: int,
    ):
        super().__init__()

        # Relative residue position
        self.rel_res_binner = Binner(
            min_bin=-rel_res_win_size,
            max_bin=rel_res_win_size,
            no_bins=rel_res_win_size * 2 + 1
        )
        self.rel_res_embed = nn.Embedding(
            num_embeddings=rel_res_win_size * 2 + 2,
            embedding_dim=c_z,
        )

        # Same entity (sequence)
        self.same_entity_embed = nn.Embedding(
            num_embeddings=2,
            embedding_dim=c_z
        )

        # Relative chain ID
        self.rel_chain_binner = Binner(
            min_bin=-rel_chain_win_size,
            max_bin=rel_chain_win_size,
            no_bins=rel_chain_win_size * 2 + 1
        )
        self.rel_chain_embed = nn.Embedding(
            num_embeddings=rel_chain_win_size * 2 + 2,
            embedding_dim=c_z,
        )

    def forward(
        self,
        res_idx: torch.Tensor,
        asym_id: torch.Tensor,
        sym_id: torch.Tensor,
        entity_id: torch.Tensor,
    ):
        # Relative residue position
        same_chain = \
            1 - torch.clip(
                torch.abs(asym_id[..., None] - asym_id[..., None, :]),
                max=1, min=0
            )

        d = res_idx[..., None] - res_idx[..., None, :]
        rel_pos_bin = self.rel_res_binner.val2bin(d)
        rel_pos_bin = rel_pos_bin + 1
        rel_pos_bin = rel_pos_bin * same_chain

        # Residues belong to the same entity?
        same_entity = \
            1 - torch.clip(
                torch.abs(entity_id[..., None] - entity_id[..., None, :]),
                max=1, min=0
            )

        # Relative chain index (same entity)
        d = sym_id[..., None] - sym_id[..., None, :]

        rel_chain_bin = self.rel_chain_binner.val2bin(d)
        rel_chain_bin = rel_chain_bin + 1
        rel_chain_bin = rel_chain_bin * same_entity

        # Positional embedding
        pe = \
            self.rel_res_embed(rel_pos_bin) + \
            self.same_entity_embed(same_entity) + \
            self.rel_chain_embed(rel_chain_bin)

        return pe
