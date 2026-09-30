import argparse
from pathlib import Path
import pandas as pd
import subprocess
import tempfile
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from Bio.PDB import PDBParser, Superimposer


def calculate_rmsd(target_file: Path, pred_file: Path) -> float:
    # NOTE: Make sure BioPython is installed
    def get_atoms(structure):
        atoms = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != " ":  # skip heteroatoms and waters
                        continue
                    for atom in residue:
                        key = (chain.id, residue.id[1], atom.name)
                        atoms[key] = atom
        return atoms

    # Load structures / atoms
    parser = PDBParser(QUIET=True)
    target_struct = parser.get_structure("ref", target_file)
    pred_struct = parser.get_structure("mob", pred_file)

    target_atoms = get_atoms(target_struct)
    pred_atoms = get_atoms(pred_struct)

    # Only compare atoms present in both structures
    common_keys = sorted(set(target_atoms.keys()) & set(pred_atoms.keys()))
    if not common_keys:
        raise ValueError("No matching atoms found between structures.")

    target_coords = [target_atoms[k] for k in common_keys]
    pred_coords = [pred_atoms[k] for k in common_keys]

    # Calculate RMSD
    sup = Superimposer()
    sup.set_atoms(target_coords, pred_coords)

    return float(sup.rms)


def calculate_tm_score(
    target_file: Path,
    pred_file: Path,
) -> float:
    # NOTE: Make sure USalign is installed
    usalign_output = subprocess.check_output(
        [
            "USalign",
            "-mol", "RNA",
            "-ter", "1",
            "-TMscore", "6",
            str(pred_file),
            str(target_file),
        ],
    ).decode()

    lines = usalign_output.split("\n")

    # Get last line that starts with "TM-score="
    tm_line = next(
        line for line in reversed(lines)
        if line.startswith("TM-score=")
    )

    tm_score = tm_line.split()[1]
    tm_score = float(tm_score)

    return tm_score


def calculate_inf_scores(
    target_file: Path,
    pred_file: Path
) -> dict[str, float]:
    # NOTE: Make sure rna-tools is installed
    with tempfile.NamedTemporaryFile() as f_inf:
        subprocess.check_call(
            [
                "rna_calc_inf.py",
                "-t", str(target_file),
                "-o", str(f_inf.name),
                str(pred_file),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        with open(f_inf.name, "r") as f:
            lines = f.readlines()

        metric_vals = lines[-1].rstrip().split(",")

        inf_all = float(metric_vals[2])
        inf_stack = float(metric_vals[3])
        inf_wc = float(metric_vals[4])
        inf_nwc = float(metric_vals[5])

    return {
        "inf_all": inf_all,
        "inf_stack": inf_stack,
        "inf_wc": inf_wc,
        "inf_nwc": inf_nwc,
    }


def calculate_lddt_c3prime(
    target_file: Path,
    pred_file: Path,
) -> float:
    # NOTE: Make sure OpenStructure is installed
    LDDT_INCLUSION_RADIUS = 30.0

    with tempfile.NamedTemporaryFile() as f_lddt:
        subprocess.check_call(
            [
                "ost", "compare-structures",
                "--bb-lddt",
                "-m", str(pred_file),
                "-r", str(target_file),
                "-o", str(f_lddt.name),
                "--lddt-inclusion-radius", str(LDDT_INCLUSION_RADIUS),
                "--lddt-no-stereochecks",
                "--residue-number-alignment",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        with open(f_lddt.name, "r") as f:
            ost_output = json.load(f)

    lddt_score = ost_output["bb_lddt"]
    return lddt_score


def main(args):
    target_dir = Path(args.target_dir)
    pred_dirs = [Path(pred_dir) for pred_dir in args.pred_dirs]
    output_file = Path(args.output_file)

    dicts = []

    for target_file in tqdm(list(target_dir.rglob("*.pdb"))):
        target_id = target_file.stem

        if args.ignore_missing:
            missing = False
            for pred_dir in pred_dirs:
                pred_subdir = pred_dir / target_id
                if (
                    not pred_subdir.is_dir() or
                    not any(pred_subdir.rglob("*.pdb"))
                ):
                    missing = True
                    break

            if missing:
                continue

        for pred_dir in pred_dirs:
            predictor = pred_dir.name
            pred_subdir = pred_dir / target_id

            for pred_file in sorted(pred_subdir.rglob("*.pdb")):
                rmsd = calculate_rmsd(target_file, pred_file)
                tm_score = calculate_tm_score(target_file, pred_file)
                lddt_c3 = calculate_lddt_c3prime(target_file, pred_file)
                inf_scores = calculate_inf_scores(target_file, pred_file)

                dicts.append(
                    {
                        "predictor": predictor,
                        "struct_id": target_id,
                        "filename": pred_file.name,
                        "rmsd": rmsd,
                        "tm_score": tm_score,
                        "lddt_c3prime": lddt_c3,
                        **inf_scores
                    }
                )

    df = pd.DataFrame(dicts)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics")

    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Directory containing target structures.",
    )
    parser.add_argument(
        "--pred_dirs",
        nargs="+",
        type=str,
        required=True,
        help="Directories containing predicted structures.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output CSV file to store results.",
    )
    parser.add_argument(
        "--ignore_missing",
        action="store_true",
        help="""
            Only evaluate targets with predictions
            available in all prediction directories.
        """,
    )

    args = parser.parse_args()

    main(args)
