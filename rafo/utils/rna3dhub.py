from pathlib import Path

from collections import defaultdict
from urllib.request import urlretrieve, Request, urlopen
import csv

RNA3DHUB_NRLIST_URL = "https://rna.bgsu.edu/rna3dhub/nrlist/download/rna"
DEFAULT_RESOLUTION_CUTOFF = "20.0A"


def get_latest_rna3dhub_nrlist_version() -> str:
    url = f"{RNA3DHUB_NRLIST_URL}/current/{DEFAULT_RESOLUTION_CUTOFF}/csv"

    with urlopen(Request(url)) as response:
        # Check filename in headers
        content_disp = response.headers.get('Content-Disposition')

    # Extract version from the filename
    filename = content_disp.split("filename=")[-1]
    version = filename.split("_")[1]

    return version


def download_rna3dhub_nrlist_csv(
    local_csv_file: Path,
    version: str,
    resolution_cutoff: str = DEFAULT_RESOLUTION_CUTOFF
) -> None:
    urlretrieve(
        url=f"{RNA3DHUB_NRLIST_URL}/{version}/{resolution_cutoff}/csv",
        filename=local_csv_file
    )


def _parse_ife(ife: str) -> str:
    pdb_id = ife.split("|")[0].lower()

    chain_ids = [
        struct_id.split("|")[-1]
        for struct_id in ife.split("+")
    ]

    return f"{pdb_id}_{'_'.join(chain_ids)}"


def load_rna3dhub_nrlist_csv(rna3dhub_nrlist_csv: Path) -> dict[str, set[str]]:
    cluster_dict = defaultdict(set)

    with open(rna3dhub_nrlist_csv, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            if row:
                cluster_id = _parse_ife(row[1])
                cluster = row[-1]
                for ife in cluster.split(","):
                    ife = _parse_ife(ife)

                    cluster_dict[cluster_id].add(ife)

    return cluster_dict
