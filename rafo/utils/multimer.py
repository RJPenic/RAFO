import torch
from collections import defaultdict

from rafo.utils.seq import LM_TOKENIZER


def get_asym_ids(seqs: list[str]) -> list[int]:
    chain_asym_ids = [(chain_id + 1) for chain_id in range(len(seqs))]
    return chain_asym_ids


def get_entity_ids(seqs: list[str]) -> list[int]:
    unique_seqs = list(dict.fromkeys(seqs))

    seq_to_entity = {seq: (idx + 1) for idx, seq in enumerate(unique_seqs)}
    entity_ids = [seq_to_entity[seq] for seq in seqs]

    return entity_ids


def get_sym_ids(seqs: list[str]) -> list[int]:
    entity_ids = get_entity_ids(seqs)

    occurences = defaultdict(int)
    chain_sym_ids = []
    for entity_id in entity_ids:
        occurences[entity_id] = occurences[entity_id] + 1
        chain_sym_ids.append(occurences[entity_id])

    return chain_sym_ids


def lm_out_to_multimer_batch(
    lm_out: torch.Tensor,
    seq_tokens_lm: torch.Tensor,
    seq_batch_idx: torch.Tensor,
    res_idcs: torch.Tensor,
) -> torch.Tensor:
    lm_bs = lm_out.shape[0]
    mm_bs = res_idcs.shape[0]

    res_embed = lm_out.new_zeros(
        *res_idcs.shape, lm_out.shape[-1],
    )  # Create tensor of shape (B x L x E), zeros

    # Get sequence lengths
    seq_lens = (
        seq_tokens_lm.shape[-1] -
        torch.sum(seq_tokens_lm == LM_TOKENIZER.pad_idx, dim=-1)
    )
    seq_lens = seq_lens - 2  # Remove BOS and EOS

    for mm_idx in range(mm_bs):  # Iterate through batch (multimer) indices
        curr_idx = 0  # Set up batch "fill" pointer
        for lm_idx in range(lm_bs):  # Iterate through LM batch
            if seq_batch_idx[lm_idx] == mm_idx:
                seq_len = seq_lens[lm_idx]

                # Copy values into embedding tensor
                res_embed[
                    seq_batch_idx[lm_idx], curr_idx: curr_idx + seq_len
                ] = lm_out[lm_idx, 1: (seq_len + 1)]
                curr_idx += seq_len  # Update "fill" pointer

    return res_embed
