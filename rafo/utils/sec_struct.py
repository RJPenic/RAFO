import numpy as np

from pathlib import Path
from typing import Optional
import subprocess
import tempfile

from abc import ABC, abstractmethod

from rafo.utils.seq import clean_seq

RNAFOLD_OUT_DIR = "rnafold"
ETERNAFOLD_OUT_DIR = "eternafold"
RNASTRUCTURE_OUT_DIR = "rnastructure"
IPKNOT_OUT_DIR = "ipknot"

ETERNAFOLD_PROB_CUTOFF = 0.001

SEQ_EXT = "fa"
PROB_EXT = "prob"
SS_EXT = "ss"

SS_FEAT_DIM = 7

def _read_relevant_lines(file_path: Path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    lines = list(
        filter(
            lambda line: not line.lstrip().startswith("#"), lines
        )
    )  # Ignore comment lines

    return lines


def _save_seq(seq: str, seq_id: str, seq_file: Path) -> None:
    seq = clean_seq(seq)

    with open(seq_file, "w") as f:
        f.write(f">{seq_id}\n")
        f.write(f"{seq}\n")


def load_bpseq_file(bpseq_file: Path):
    lines = _read_relevant_lines(bpseq_file)

    seq_len = len(lines)
    seq = ''
    pair_mat = np.zeros((seq_len, seq_len), dtype=np.float32)

    for line in lines:
        res_idx, res_tkn, pair_idx = line.rstrip().split()
        seq += res_tkn

        if pair_idx != '0':
            pair_mat[int(res_idx) - 1, int(pair_idx) - 1] = 1.0

    return pair_mat


def load_dbn_file(dbn_file: Path):
    lines = _read_relevant_lines(dbn_file)
    db_notation = lines[-1]
    db_notation = db_notation.split()[0]  # Remove potential energy notation

    return dot_bracket_to_2d(db_notation)


def dot_bracket_to_2d(db_notation: str):
    seq_len = len(db_notation)
    pair_mat = np.zeros((seq_len, seq_len), dtype=np.float32)

    # Initialize bracket stacks
    stacks = {}
    stacks["("] = stacks[")"] = []
    stacks["["] = stacks["]"] = []
    stacks["{"] = stacks["}"] = []
    stacks["<"] = stacks[">"] = []

    # Iterate through the dot-bracket notation and fill the 2D matrix
    for i in range(seq_len):
        current_tkn = db_notation[i]

        if current_tkn in ("(", "[", "{", "<"):
            stacks[current_tkn].append(i)
        elif current_tkn in (")", "]", "}", ">"):
            j = stacks[current_tkn].pop()
            pair_mat[i, j] = 1.0
        elif db_notation[i] == ".":
            pass
        else:
            raise RuntimeError(
                f"""
                    Encountered unexpected symbol in dot-bracket notation
                    string! (index {i}: '{db_notation[i]}')
                """
            )

    # Symmetrize pairing matrix
    pair_mat = pair_mat + pair_mat.transpose()
    pair_mat = np.minimum(pair_mat, 1.0)

    return pair_mat


class SecStructTool(ABC):
    @staticmethod
    @abstractmethod
    def generate_prediction(
        seq_file: Path,
        ss_file: Path,
        prob_file: Path,
    ) -> None:
        pass

    @staticmethod
    @abstractmethod
    def load_ss_file(seq_len: int, ss_file: Path) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def load_prob_file(seq_len: int, prob_file: Path) -> Optional[np.ndarray]:
        pass

    @classmethod
    def get_ss_feats(
        cls,
        seq: str,
        seq_id: str,
        ss_dir: Path,
        force_overwrite: bool = True,
    ) -> np.ndarray:
        ss_dir.mkdir(parents=True, exist_ok=True)

        seq_file = ss_dir / f"{seq_id}.{SEQ_EXT}"
        prob_file = ss_dir / f"{seq_id}.{PROB_EXT}"
        ss_file = ss_dir / f"{seq_id}.{SS_EXT}"

        if not ss_file.exists() or force_overwrite:
            _save_seq(seq, seq_id, seq_file)  # Create sequence (FASTA) file
            cls.generate_prediction(seq_file, ss_file, prob_file)
            seq_file.unlink()  # Delete sequence file

        seq_len = len(seq)
        ss_pred = cls.load_ss_file(seq_len, ss_file)
        ss_prob = cls.load_prob_file(seq_len, prob_file)

        probs_exist = (ss_prob is not None)

        ss_feat = np.zeros(
            (seq_len, seq_len, 2 if probs_exist else 1),
            dtype=np.float32
        )
        ss_feat[..., 0] = ss_pred

        if probs_exist:
            ss_feat[..., 1] = ss_prob

        return ss_feat


class IPknot(SecStructTool):
    @staticmethod
    def generate_prediction(
        seq_file: Path,
        ss_file: Path,
        prob_file: Path,
    ) -> None:
        subprocess.check_call(
            [
                "ipknot",
                "--input", f"{seq_file}",
                "--bpseq-file", f"{ss_file}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    @staticmethod
    def load_ss_file(seq_len: int, ss_file: Path) -> np.ndarray:
        return load_bpseq_file(ss_file)

    @staticmethod
    def load_prob_file(seq_len: int, prob_file: Path) -> Optional[np.ndarray]:
        return None


class EternaFold(SecStructTool):
    @staticmethod
    def generate_prediction(
        seq_file: Path,
        ss_file: Path,
        prob_file: Path,
    ) -> None:
        subprocess.check_call(
            [
                "eternafold", "predict",
                f"{seq_file}",
                "--bpseq", f"{ss_file}",
                "--posteriors", f"{ETERNAFOLD_PROB_CUTOFF}", f"{prob_file}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    @staticmethod
    def load_ss_file(seq_len: int, ss_file: Path) -> np.ndarray:
        return load_bpseq_file(ss_file)

    @staticmethod
    def load_prob_file(seq_len: int, prob_file: Path) -> Optional[np.ndarray]:
        probs = np.zeros((seq_len, seq_len))

        with open(prob_file, "r") as f:
            for line in f:
                cols = line.split()
                i = int(cols[0])

                for pair in cols[2:]:
                    j, prob = pair.split(":")
                    j = int(j)
                    prob = float(prob)

                    probs[i - 1, j - 1] = prob
                    probs[j - 1, i - 1] = prob

        return probs


class RNAfold(SecStructTool):
    @staticmethod
    def generate_prediction(
        seq_file: Path,
        ss_file: Path,
        prob_file: Path,
    ) -> None:
        subprocess.check_call(
            [
                "RNAfold",
                f"--infile={seq_file.resolve()}",
                f"--outfile={ss_file.name}",
                "--noPS",
                "-p",  # Compute base-pair probabilities
            ],
            # Changing work directory, hence the 'resolve' in infile
            # (because of --outfile limitations)
            cwd=ss_file.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        # RNAfold -p generates a *_dp.ps file with base-pair probabilities
        # Rename it to our expected prob_file location
        dp_file = ss_file.parent / f"{ss_file.stem}_dp.ps"
        dp_file.rename(prob_file)

        # Convert SS file to normalized DBN format
        with open(ss_file, "r") as f:
            lines = f.readlines()

        id_line = lines[0].strip()
        dbn_line = lines[-2].strip()

        with open(ss_file, "w") as f:
            f.write(f"{id_line}\n")
            f.write(f"{dbn_line}\n")

    @staticmethod
    def load_ss_file(seq_len: int, ss_file: Path) -> np.ndarray:
        return load_dbn_file(ss_file)

    @staticmethod
    def load_prob_file(seq_len: int, prob_file: Path) -> Optional[np.ndarray]:
        probs = np.zeros((seq_len, seq_len), dtype=np.float32)

        with open(prob_file, "r") as f:
            for line in f:
                line = line.strip()

                # Look for lines ending with "ubox"
                # and starts with number
                if line.endswith("ubox") and line[0].isdigit():
                    i, j, sqrt_prob, _ = line.split()

                    i = int(i) - 1  # Convert to 0-indexed
                    j = int(j) - 1
                    sqrt_prob = float(sqrt_prob)

                    # Square to get actual probability
                    prob = sqrt_prob ** 2
                    probs[i, j] = prob
                    probs[j, i] = prob  # Symmetric

        return probs


class RNAstructure(SecStructTool):
    @staticmethod
    def generate_prediction(
        seq_file: Path,
        ss_file: Path,
        prob_file: Path,
    ) -> None:
        subprocess.check_call(
            [
                "Fold",
                f"{seq_file}",
                f"{ss_file}",
                "-mfe",
                "--bracket"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    @staticmethod
    def load_ss_file(seq_len: int, ss_file: Path) -> np.ndarray:
        return load_dbn_file(ss_file)

    @staticmethod
    def load_prob_file(seq_len: int, prob_file: Path) -> Optional[np.ndarray]:
        return None


def get_single_seq_ss_feats(
    seq: str,
    seq_id: str,
    ss_dir: Optional[Path] = None,
    force_overwrite: bool = False,
) -> np.ndarray:
    # Create temporary directory (if needed)
    tmp_dir = None
    if ss_dir is None:
        tmp_dir = tempfile.TemporaryDirectory()
        ss_dir = Path(tmp_dir.name)

    # SS calculations
    ss_eternafold_feats = EternaFold.get_ss_feats(
        seq=seq,
        seq_id=seq_id,
        ss_dir=ss_dir / ETERNAFOLD_OUT_DIR,
        force_overwrite=force_overwrite
    )
    ss_ipknot_feats = IPknot.get_ss_feats(
        seq=seq,
        seq_id=seq_id,
        ss_dir=ss_dir / IPKNOT_OUT_DIR,
        force_overwrite=force_overwrite
    )
    ss_rnafold_feats = RNAfold.get_ss_feats(
        seq=seq,
        seq_id=seq_id,
        ss_dir=ss_dir / RNAFOLD_OUT_DIR,
        force_overwrite=force_overwrite
    )
    ss_rnastructure_feats = RNAstructure.get_ss_feats(
        seq=seq,
        seq_id=seq_id,
        ss_dir=ss_dir / RNASTRUCTURE_OUT_DIR,
        force_overwrite=force_overwrite
    )

    if tmp_dir is not None:
        tmp_dir.cleanup()

    # Concatenate features
    ss_feat = np.concatenate(
        (
            ss_eternafold_feats,
            ss_ipknot_feats,
            ss_rnafold_feats,
            ss_rnastructure_feats,
        ),
        axis=-1,
    )

    return ss_feat


def get_multi_seq_ss_feats(
    seqs: list[str],
    seq_ids: list[str],
    ss_dir: Optional[Path] = None,
    force_overwrite: bool = False,
) -> np.ndarray:
    assert len(seqs) == len(seq_ids), "Please provide ID for each sequence!"

    total_size = sum(len(seq) for seq in seqs)
    multi_seq_ss_feats = None

    # SS features for each chain
    offset = 0
    for seq_id, seq in zip(seq_ids, seqs):
        seq_len = len(seq)

        single_seq_ss_feats = get_single_seq_ss_feats(
            seq=seq,
            seq_id=seq_id,
            ss_dir=ss_dir,
            force_overwrite=force_overwrite
        )

        if multi_seq_ss_feats is None:
            multi_seq_ss_feats = np.zeros(
                (
                    total_size, total_size,
                    single_seq_ss_feats.shape[-1]
                ),
                dtype=single_seq_ss_feats.dtype
            )

        multi_seq_ss_feats[
            offset: offset + seq_len,
            offset: offset + seq_len,
        ] = single_seq_ss_feats

        offset += len(seq)

    # Build chain index for each residue
    chain_indices = np.zeros(total_size, dtype=np.int32)
    offset = 0
    for chain_idx, seq in enumerate(seqs):
        seq_len = len(seq)
        chain_indices[offset:offset + seq_len] = chain_idx
        offset += seq_len

    # Create multimeric feature: 1 if residues from different chains, else 0
    # Shape = L x L x 1
    multimeric_feat = (
        chain_indices[:, None] != chain_indices[None, :]
    ).astype(multi_seq_ss_feats.dtype)
    multimeric_feat = multimeric_feat[..., None]

    # Append multimeric feature to SS features
    multi_seq_ss_feats = np.concatenate(
        [multi_seq_ss_feats, multimeric_feat], axis=-1
    )

    return multi_seq_ss_feats
