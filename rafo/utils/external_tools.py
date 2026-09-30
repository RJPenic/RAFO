import subprocess
import tempfile
import numpy as np
from numpy.typing import ArrayLike
from collections import defaultdict

from rafo.utils.save import save_to_pdb
from rafo.constants import TM_SCORE_REF_ATOM
from rafo.utils.data_structures import MultiRNA

TM_SCORE_LINE_IDX = 16


def _parse_usalign_output(output: str) -> float:
    lines = output.split("\n")

    tm_line = lines[TM_SCORE_LINE_IDX]

    tm_score = tm_line.split()[1]
    tm_score = float(tm_score)

    return tm_score


def run_usalign_with_multirna(
    query_struct: MultiRNA,
    target_struct: MultiRNA,
    force_res_order: bool = False,
) -> float:
    query_atom_pos = np.concatenate(
        [chain.atom_pos for chain in query_struct.chains],
        axis=0,
    )
    query_atom_mask = np.concatenate(
        [chain.atom_mask for chain in query_struct.chains],
        axis=0,
    )

    target_atom_pos = np.concatenate(
        [chain.atom_pos for chain in target_struct.chains],
        axis=0,
    )
    target_atom_mask = np.concatenate(
        [chain.atom_mask for chain in target_struct.chains],
        axis=0,
    )

    return run_usalign(
        query_atom_pos=query_atom_pos,
        query_atom_mask=query_atom_mask,
        target_atom_pos=target_atom_pos,
        target_atom_mask=target_atom_mask,
        force_res_order=force_res_order,
    )


def run_usalign(
    query_atom_pos: ArrayLike,
    query_atom_mask: ArrayLike,
    target_atom_pos: ArrayLike,
    target_atom_mask: ArrayLike,
    force_res_order: bool = False,
    ref_atom: str = TM_SCORE_REF_ATOM,
) -> float:
    # NOTE: Will fail if the structure has less than 4 modeled residues
    # Do data filtering before calling this method to avoid this
    with (
        tempfile.NamedTemporaryFile() as f_query_struct,
        tempfile.NamedTemporaryFile() as f_target_struct,
    ):
        save_to_pdb(
            atom_pos=query_atom_pos,
            atom_mask=query_atom_mask,
            output_file=f_query_struct.name
        )

        save_to_pdb(
            atom_pos=target_atom_pos,
            atom_mask=target_atom_mask,
            output_file=f_target_struct.name
        )

        usalign_output = subprocess.check_output(
            [
                "USalign",
                "-atom", f" {ref_atom:<3s}",
                "-TMscore", "1" if force_res_order else "0",
                str(f_query_struct.name),
                str(f_target_struct.name),
            ],
        ).decode()

    tm_score = _parse_usalign_output(usalign_output)

    return tm_score


def run_mmseqs_easy_cluster(
    seqs: dict[str, str],
    min_seq_id: float,
    coverage: float,
    threads: int,
) -> dict[str, str]:
    with (
        tempfile.NamedTemporaryFile() as tmp_fasta,
        tempfile.TemporaryDirectory() as tmp_out,
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        with open(tmp_fasta.name, "w") as f:
            for seq_id, seq in seqs.items():
                f.write(f">{seq_id}\n")
                f.write(f"{seq}\n")

        subprocess.check_call(
            [
                "mmseqs", "easy-cluster",
                "--min-seq-id", f"{min_seq_id}",
                "-c", f"{coverage}",
                "--threads", f"{threads}",
                "-s", "7.5",  # Boost sensitivity
                "--max-seqs", "10000",  # Boost sensitivity
                "--cluster-reassign", "1",  # Boost sensitivity
                "--spaced-kmer-mode", "0",  # Why? MMseqs2 issue #794
                f"{tmp_fasta.name}",
                f"{tmp_out}/out",
                f"{tmp_dir}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        clusters = {}
        with open(f"{tmp_out}/out_cluster.tsv") as f:
            for line in f.readlines():
                cluster_id, chain_id = line.rstrip().split()
                clusters[chain_id] = cluster_id

    return clusters


def run_mmseqs_easy_search(
    query_seqs: dict[str, str],
    target_seqs: dict[str, str],
    min_seq_id: float,
    coverage: float,
    threads: int,
) -> dict[str, set[str]]:
    with (
        tempfile.NamedTemporaryFile() as tmp_query_fasta,
        tempfile.NamedTemporaryFile() as tmp_target_fasta,
        tempfile.NamedTemporaryFile() as tmp_out,
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        # Save query sequences to FASTA
        with open(tmp_query_fasta.name, "w") as f:
            for seq_id, seq in query_seqs.items():
                f.write(f">{seq_id}\n")
                f.write(f"{seq}\n")

        # Save target sequences to FASTA
        with open(tmp_target_fasta.name, "w") as f:
            for seq_id, seq in target_seqs.items():
                f.write(f">{seq_id}\n")
                f.write(f"{seq}\n")

        subprocess.check_call(
            [
                "mmseqs", "easy-search",
                "--search-type", "3",
                "--min-seq-id", f"{min_seq_id}",
                "-c", f"{coverage}",
                "--threads", f"{threads}",
                "-s", "7.5",  # Boost sensitivity
                "--max-seqs", "10000",  # Boost sensitivity
                "--spaced-kmer-mode", "0",  # Why? MMseqs2 issue #794
                f"{tmp_query_fasta.name}",
                f"{tmp_target_fasta.name}",
                f"{tmp_out.name}",
                f"{tmp_dir}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        hits = defaultdict(set)
        with open(tmp_out.name) as f:
            for line in f.readlines():
                query_id, target_id = line.rstrip().split()[:2]
                hits[query_id].add(target_id)

        return hits
