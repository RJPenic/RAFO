# Refine structures with QRNAS

# Make sure to install QRNAS before using this script
# https://genesilico.pl/software/stand-alone/qrnas

import argparse
from pathlib import Path
import tempfile
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def refine_with_qrnas(
    input_pdb: Path,
    output_pdb: Path,
    num_steps: int = 20_000,
    qrna_exec: str = "QRNA",
) -> None:
    with tempfile.NamedTemporaryFile() as f_config:
        # Set up config file
        with open(f_config.name, "w") as f:
            f.writelines([
                f"NSTEPS          {num_steps}\n",
                f"WRITEFREQ  {num_steps + 1}\n",
            ])

        # Run QRNAS
        subprocess.check_call(
            [
                qrna_exec,
                "-i", str(input_pdb),
                "-o", str(output_pdb),
                "-c", str(f_config.name),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_pdbs = [
        input_pdb
        for input_pdb in input_dir.rglob("*.pdb")
    ]

    # Mirror each input's subdirectory structure under the output directory
    output_pdbs = []
    for input_pdb in input_pdbs:
        output_pdb = output_dir / input_pdb.relative_to(input_dir)
        output_pdb.parent.mkdir(parents=True, exist_ok=True)
        output_pdbs.append(output_pdb)

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        _ = list(
            tqdm(
                ex.map(
                    refine_with_qrnas,
                    input_pdbs,
                    output_pdbs,
                    len(input_pdbs) * [args.num_refinement_steps]
                ),
                total=len(input_pdbs)
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refine RNA structures with QRNAS"
    )

    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory (e.g. with predicted structures)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory (for refined structures)"
    )

    parser.add_argument(
        "--num_refinement_steps", type=int, default=20_000,
        help="Number of QRNAS structural refinement steps"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of worker processes"
    )

    args = parser.parse_args()
    main(args)
