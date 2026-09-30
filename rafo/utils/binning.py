import torch
import torch.nn as nn

from typing import Union, Optional


class Binner(nn.Module):
    def __init__(
        self,
        no_bins: int,
        min_bin: Optional[Union[int, float]] = None,
        max_bin: Optional[Union[int, float]] = None,
    ):
        super().__init__()

        assert (min_bin is None) == (max_bin is None), \
            "Please define either both or none of the outermost bins!"

        if min_bin is None and max_bin is None:
            min_bin = 1.0 / (no_bins * 2)
            max_bin = 1.0 - min_bin

        bin_centers = torch.linspace(min_bin, max_bin, steps=no_bins)
        bin_edges = 0.5 * (bin_centers[1:] + bin_centers[:-1])

        self.register_buffer("bin_centers", bin_centers, persistent=False)
        self.register_buffer("bin_edges", bin_edges, persistent=False)

    def bin2val(self, bin_idcs: torch.Tensor) -> torch.Tensor:
        return self.bin_centers[bin_idcs]

    def val2bin(self, vals: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(vals, self.bin_edges)
