import torch
import torch.nn as nn

from typing import Optional

from rinalmo.pretrained import get_pretrained_model

import random
from rafo.constants import NUM_ATOM_TYPES

from rafo.model.embedders import InputEmbedder
from rafo.model.evoformer import Evoformer
from rafo.model.structure_module import StructureModule
from rafo.model.recycling import Recycler
from rafo.model.heads import AuxiliaryHeads

from rafo.utils.multimer import lm_out_to_multimer_batch


class RAFO(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config.model

        self.lm, _ = get_pretrained_model(model_name="giga-v1")

        self.input_embedder = InputEmbedder(**self.config["input_embedder"])
        self.evoformer = Evoformer(**self.config["evoformer"])
        self.structure_module = StructureModule(
            **self.config["structure_module"]
        )

        self.no_cycles = self.config["no_cycles"]
        self.recycler = None

        if self.no_cycles > 1:
            self.recycler = Recycler(**self.config["recycler"])

        self.aux_heads = AuxiliaryHeads(**self.config["auxiliary_heads"])

    def iteration(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        seq_tokens: torch.Tensor,
        m_0_prev: torch.Tensor,
        z_prev: torch.Tensor,
        x_prev: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        msa_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
    ):
        outputs = {}
        outputs["ef"] = {}

        if self.recycler is not None:
            m_0_update, z_update = self.recycler(m_0_prev, z_prev, x_prev)

            m = m.clone()
            m[..., 0, :, :] = m[..., 0, :, :] + m_0_update
            z = z + z_update

        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        m, z, s = self.evoformer(
            m, z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_lma=use_lma,
        )
        outputs["ef"]["msa"] = m
        outputs["ef"]["pair"] = z
        outputs["ef"]["single"] = s

        outputs["sm"] = self.structure_module(s, z, seq_tokens, seq_mask)

        return outputs

    def forward(
        self,
        seq_tokens: torch.Tensor,
        seq_tokens_lm: torch.Tensor,
        seq_batch_idx: torch.Tensor,
        res_idx: torch.Tensor,
        asym_id: torch.Tensor,
        sym_id: torch.Tensor,
        entity_id: torch.Tensor,
        ss_feat: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        msa_mask: Optional[torch.Tensor] = None,
        no_cycles: Optional[int] = None,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
    ) -> dict:
        # LM frozen
        with torch.no_grad():
            lm_out = self.lm(seq_tokens_lm)["representation"]

        lm_embed = lm_out_to_multimer_batch(
            lm_out=lm_out,
            seq_tokens_lm=seq_tokens_lm,
            seq_batch_idx=seq_batch_idx,
            res_idcs=res_idx,
        )

        m, z = self.input_embedder(
            seq_tokens=seq_tokens,
            lm_embed=lm_embed,
            res_idx=res_idx,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
            ss_feat=ss_feat,
        )

        c_m = m.shape[-1]
        c_z = z.shape[-1]

        # Initialize recycling
        m_0_prev = z.new_zeros(
            z.shape[:-2] + (c_m,)
        )
        z_prev = z.new_zeros(
            z.shape[:-1] + (c_z,)
        )
        x_prev = z.new_zeros(
            z.shape[:-2] + (NUM_ATOM_TYPES, 3)
        )

        # Resolve number of recycling cycles
        if self.training:
            # Randomly sample number of cycles
            no_cycles = random.randint(1, self.no_cycles)
        elif no_cycles is None:
            # Use default number of cycles
            no_cycles = self.no_cycles
        elif self.no_cycles == 1:
            # If model was trained without recycling, disable it
            no_cycles = 1

        is_grad_enabled = torch.is_grad_enabled()

        # Recycle iterations
        for cycle_idx in range(no_cycles):
            is_final_iter = ((cycle_idx + 1) == no_cycles)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                cycle_outs = self.iteration(
                    m, z, seq_tokens,
                    m_0_prev, z_prev, x_prev,
                    seq_mask.to(m.dtype), msa_mask.to(m.dtype),
                    chunk_size, use_lma,
                )

                z_prev = cycle_outs["ef"]["pair"]
                m_0_prev = cycle_outs["ef"]["msa"][..., 0, :, :]
                x_prev = cycle_outs["sm"]["atom_positions"]

        outputs = cycle_outs

        outputs["aux"] = self.aux_heads(
            z=cycle_outs["ef"]["pair"],
            s=cycle_outs["sm"]["single"]
        )

        # Delete output embeddings (not needed, save memory)
        del outputs["ef"]["msa"]
        del outputs["ef"]["pair"]
        del outputs["ef"]["single"]

        del outputs["sm"]["single"]

        return outputs
