import numpy as np

from rafo.constants import (
    URACIL_TKN, THYMINE_TKN, ADENINE_TKN,
    CYTOSINE_TKN, GUANINE_TKN, UNDEFINED_TKN,
    RESIDUE_TOKEN_ORDER
)
from rinalmo.pretrained import get_pretrained_model

_, LM_TOKENIZER = get_pretrained_model(model_name="giga-v1")

RINALMO_EMBED_DIM = 1280


def clean_seq(seq: str) -> str:
    seq = seq.upper().replace(THYMINE_TKN, URACIL_TKN)
    non_standard_tkns = set(
        [
            tkn for tkn in seq
            if tkn not in [URACIL_TKN, CYTOSINE_TKN, GUANINE_TKN, ADENINE_TKN]
        ]
    )

    for non_standard_tkn in non_standard_tkns:
        seq = seq.replace(non_standard_tkn, UNDEFINED_TKN)

    return seq


def tokenize(seq: str) -> list[int]:
    seq = clean_seq(seq)
    tokens = [RESIDUE_TOKEN_ORDER[nuc] for nuc in seq]

    return tokens


def get_pur_pyr_masks(seq: str) -> tuple[np.array, np.array]:
    seq = seq.upper().replace("T", "U")
    purine_mask = np.zeros(len(seq), dtype=bool)
    pyrimidine_mask = np.zeros(len(seq), dtype=bool)

    for i, nuc in enumerate(seq):
        if nuc == 'A' or nuc == 'G':
            purine_mask[i] = True
        elif nuc == 'C' or nuc == 'U':
            pyrimidine_mask[i] = True

    return purine_mask, pyrimidine_mask
