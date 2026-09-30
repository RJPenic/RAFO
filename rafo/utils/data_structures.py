from __future__ import annotations
from typing import Optional

from dataclasses import dataclass
import numpy as np

from typing import Iterable


@dataclass
class SingleRNA:
    id: str
    seq: str
    atom_pos: np.ndarray
    atom_mask: np.ndarray


@dataclass
class MultiRNA:
    id: str
    chains: list[SingleRNA]
    release_date: Optional[str] = None
    method: Optional[str] = None
    resolution: Optional[float] = None

    def split_into_ifes(
        self,
        chain_groups: Iterable[list[str]]
    ) -> list[MultiRNA]:
        ifes = []

        for chain_group in chain_groups:
            # Collect chain IDs and preserve order
            chain_ids = [
                chain.id for chain in self.chains
                if chain.id in chain_group
            ]
            ife_id = self.id
            ife_id += f"_{'_'.join(chain_ids)}"

            ifes.append(
                MultiRNA(
                    id=ife_id,
                    chains=[
                        chain for chain in self.chains
                        if chain.id in chain_group
                    ],
                    release_date=self.release_date,
                    method=self.method,
                    resolution=self.resolution
                )
            )

        return ifes
