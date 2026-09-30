import numpy as np

from rafo.data.preparation.steps import Filter
from rafo.utils.data_structures import MultiRNA
from rafo.constants import (
    ATOM_ORDER,
    MODELED_RES_FILTER_REF_ATOM,
    CHAIN_CLASH_FILTER_REF_ATOM,
    BACKBONE_BREAK_FILTER_REF_ATOM,
    STRUCT_DEPTH_FILTER_REF_ATOM,
    LONG_RANGE_CONTACT_FILTER_REF_ATOM
)


def create_meta_filter(
    max_resolution: float,
    cutoff_date: str,
    is_training: bool
) -> Filter:
    return Filter(
        filter_funcs=[
            # Remove non-RNA entries
            lambda struct: len(struct.chains) > 0,
            # Remove NMR entries
            lambda struct: "nmr solution" not in struct.method.lower(),
            # Remove low-quality entries
            lambda struct: struct.resolution <= max_resolution,
            # Remove entries released after/before the cut-off
            (lambda struct: struct.release_date <= cutoff_date)
            if is_training else
            (lambda struct: struct.release_date > cutoff_date),
        ],
        filter_descs=[
            "RNA",
            "non-NMR",
            f"resolution <= {max_resolution:.2f}",
            f"release date <= '{cutoff_date}'"
            if is_training else
            f"release date > '{cutoff_date}'",
        ]
    )


def create_quality_filter(
    min_seq_len: int,
    max_seq_len: int,
    min_max_chain_len: int,
    max_undef_seq_perc: float,
    min_modeled_res: int,
) -> Filter:
    return Filter(
        filter_funcs=[
            # Remove "short" entries
            lambda struct:
                sum(len(chain.seq) for chain in struct.chains)
                >= min_seq_len,
            # Remove "long" entries
            lambda struct:
                sum(len(chain.seq) for chain in struct.chains)
                <= max_seq_len,
            # Remove IFEs with short longest chains
            lambda struct:
                max(len(chain.seq) for chain in struct.chains)
                >= min_max_chain_len,
            # Remove entries with many non-standard residues
            lambda struct:
                sum(
                    1 for chain in struct.chains
                    for res in chain.seq.upper()
                    if res not in {"A", "C", "T", "U", "G"}
                ) /
                sum(len(chain.seq) for chain in struct.chains)
                <= max_undef_seq_perc,
            # Remove entries with less than N modeled residues
            lambda struct:
                sum(
                    sum(
                        chain.atom_mask[
                            ..., ATOM_ORDER[MODELED_RES_FILTER_REF_ATOM]
                        ]
                    )
                    for chain in struct.chains
                ) >= min_modeled_res
            ],
        filter_descs=[
            f"num. res. >= {min_seq_len}",
            f"num. res. <= {max_seq_len}",
            f"max. chain len. >= {min_max_chain_len}",
            f"undef. res. perc. <= {max_undef_seq_perc:.2f}",
            f"num. modeled res. >= {min_modeled_res}",
        ]
    )


def create_struct_depth_filter(min_depth: float) -> Filter:
    def _get_struct_depth(struct: MultiRNA) -> float:
        # Concatenate chains
        atom_pos = np.concatenate(
            [
                chain.atom_pos[..., ATOM_ORDER["P"], :]
                for chain in struct.chains
            ], axis=0
        )
        atom_mask = np.concatenate(
            [
                chain.atom_mask[..., ATOM_ORDER[STRUCT_DEPTH_FILTER_REF_ATOM]]
                for chain in struct.chains
            ], axis=0
        )

        # If only 3 or fewer residues are defined, depth is 0
        if np.sum(atom_mask) <= 3:
            return 0.0

        # Extract modeled atom positions and center them into origin
        atom_pos = atom_pos[atom_mask > 0.5]
        centered = atom_pos - atom_pos.mean(axis=0)

        # PCA via SVD
        *_, Vt = np.linalg.svd(centered, full_matrices=False)

        # Rotate points into PCA frame
        rotated = centered @ Vt.T

        mins = rotated.min(axis=0)
        maxs = rotated.max(axis=0)

        # Get dimensions of the bounding box
        a, b, c = maxs - mins

        depth = min(a, b, c)
        return depth

    return Filter(
        filter_funcs=[
            # Remove entries with depth less than min_depth
            lambda struct:
                _get_struct_depth(struct) >= min_depth
        ],
        filter_descs=[
            f"struct. depth >= {min_depth}"
        ]
    )


def create_long_range_contact_coverage_filter(
    dist_threshold: float,
    ignore_neighbors: int,
    min_coverage: float,
) -> Filter:
    def _get_long_range_contact_coverage(struct: MultiRNA) -> float:
        # Get P atom positions and masks for all chains
        atom_pos = np.concatenate(
            [
                chain.atom_pos[
                    ..., ATOM_ORDER[LONG_RANGE_CONTACT_FILTER_REF_ATOM], :
                ]
                for chain in struct.chains
            ], axis=0
        )
        atom_mask = np.concatenate(
            [
                chain.atom_mask[
                    ..., ATOM_ORDER[LONG_RANGE_CONTACT_FILTER_REF_ATOM]
                ]
                for chain in struct.chains
            ], axis=0
        )

        n_res = atom_pos.shape[0]

        # Track chain membership for each residue
        chain_ids = np.concatenate(
            [
                np.full(len(chain.seq), i)
                for i, chain in enumerate(struct.chains)
            ], axis=0
        )

        # Compute all-vs-all pairwise distances
        # Shape: (n_res, n_res)
        diff = atom_pos[:, None, :] - atom_pos[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Create mask for valid pairs (both residues must be modeled)
        valid_pair_mask = atom_mask[:, None] * atom_mask[None, :]

        # Create mask to ignore neighboring residues
        # (i to i ± ignore_neighbors)
        # Residues from different chains are never neighbors
        seq_idx = np.arange(n_res)
        seq_diff = np.abs(seq_idx[:, None] - seq_idx[None, :])
        same_chain_mask = (chain_ids[:, None] == chain_ids[None, :])
        non_neighbor_mask = \
            (seq_diff > ignore_neighbors) | ~same_chain_mask

        # Combined mask: valid pairs that are not neighbors
        combined_mask = valid_pair_mask * non_neighbor_mask

        # For each residue, check if it has at least one non-neighbor within
        # dist_threshold
        close_contact = \
            (dist_matrix <= dist_threshold) & (combined_mask > 0.5)
        has_close_contact = np.any(close_contact, axis=1)

        # Only count modeled residues
        valid_residues = (atom_mask > 0.5)

        if np.sum(valid_residues) == 0:
            return 0.0

        # Calculate long-range-contact coverage
        coverage = (
            np.sum(has_close_contact & valid_residues) /
            np.sum(valid_residues)
        )

        return coverage

    return Filter(
        filter_funcs=[
            lambda struct:
                _get_long_range_contact_coverage(struct) >= min_coverage
        ],
        filter_descs=[
            f"long-range-contact coverage >= {min_coverage} " +
            f"(dist <= {dist_threshold}, ignore ±{ignore_neighbors} neighbors)"
        ]
    )


def create_chain_clash_filter() -> Filter:
    DIST_THRESHOLD = 1.0

    def _has_no_chain_clash(struct: MultiRNA) -> bool:
        # Need at least two chains for an inter-chain clash
        if len(struct.chains) < 2:
            return True

        # Get P atom positions and masks for all chains
        atom_pos = np.concatenate(
            [
                chain.atom_pos[..., ATOM_ORDER[CHAIN_CLASH_FILTER_REF_ATOM], :]
                for chain in struct.chains
            ], axis=0
        )
        atom_mask = np.concatenate(
            [
                chain.atom_mask[..., ATOM_ORDER[CHAIN_CLASH_FILTER_REF_ATOM]]
                for chain in struct.chains
            ], axis=0
        )

        # Track chain membership for each residue
        chain_ids = np.concatenate(
            [
                np.full(len(chain.seq), i)
                for i, chain in enumerate(struct.chains)
            ], axis=0
        )

        # Compute all-vs-all pairwise P-P distances (distogram)
        diff = atom_pos[:, None, :] - atom_pos[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Valid inter-chain pairs with both P atoms modeled
        valid_pair_mask = (atom_mask[:, None] * atom_mask[None, :]) > 0.5
        inter_chain_mask = chain_ids[:, None] != chain_ids[None, :]
        combined_mask = valid_pair_mask & inter_chain_mask

        # Clash if any inter-chain P-P distance is below the threshold
        return not bool(np.any((dist_matrix < DIST_THRESHOLD) & combined_mask))

    return Filter(
        filter_funcs=[
            lambda struct: _has_no_chain_clash(struct)
        ],
        filter_descs=[
            f"no inter-chain P-P clash (dist. threshold = {DIST_THRESHOLD})"
        ]
    )


def create_backbone_break_filter(break_dist_threshold: float) -> Filter:
    def _has_no_backbone_break(struct: MultiRNA) -> bool:
        for chain in struct.chains:
            # Need at least two residues to have a sequential gap
            if len(chain.seq) < 2:
                continue

            atom_pos = chain.atom_pos[
                ..., ATOM_ORDER[BACKBONE_BREAK_FILTER_REF_ATOM], :
            ]
            atom_mask = chain.atom_mask[
                ..., ATOM_ORDER[BACKBONE_BREAK_FILTER_REF_ATOM]
            ]

            # Consider consecutive residue pairs where both P atoms are modeled
            both_modeled = (atom_mask[1:] > 0.5) & (atom_mask[:-1] > 0.5)

            # P-P distance between consecutive residues
            diff = atom_pos[1:] - atom_pos[:-1]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))

            # Structural gap: both modeled but too far apart
            if np.any(both_modeled & (dist > break_dist_threshold)):
                return False

        return True

    return Filter(
        filter_funcs=[
            lambda struct: _has_no_backbone_break(struct)
        ],
        filter_descs=[
            f"no backbone break (all consecutive modeled P-P dist. "
            f"<= {break_dist_threshold})"
        ]
    )
