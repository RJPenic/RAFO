import torch
import torch.nn as nn

from rafo.utils.binning import Binner

from openfold.model.primitives import LayerNorm


class AuxiliaryHeads(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_lddt_bins: int,
        no_dist_bins: int,
        dist_ref_atoms: list[str],
    ):
        super().__init__()

        self.lddt_predictor = LDDTPredictor(
            c_in=c_s,
            c_hidden=c_hidden,
            no_bins=no_lddt_bins
        )

        self.dist_predictor = DistogramPredictor(
            c_in=c_z,
            no_bins=no_dist_bins,
            no_distograms=len(dist_ref_atoms),
        )

    def forward(self, z, s):
        outputs = {}

        outputs["lddt_logits"], outputs["pLDDT"] = self.lddt_predictor(s)
        outputs["dist_logits"] = self.dist_predictor(z)

        return outputs


class LDDTPredictor(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_bins: int,
    ):
        super().__init__()

        self.binner = Binner(no_bins=no_bins)

        self.net = nn.Sequential(
            LayerNorm(c_in),
            nn.Linear(c_in, c_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(c_hidden, no_bins)
        )

    def forward(self, s):
        logits = self.net(s)
        lddt_pred = torch.sum(
            torch.softmax(logits, dim=-1) * self.binner.bin_centers,
            dim=-1
        )

        return logits, lddt_pred


class DistogramPredictor(nn.Module):
    def __init__(
        self,
        c_in: int,
        no_bins: int,
        no_distograms: int,
    ):
        super().__init__()

        self.no_bins = no_bins
        self.no_distograms = no_distograms

        self.linear = nn.Linear(c_in, no_distograms * no_bins)

    def forward(self, z):
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)

        logits = logits.view(
            *logits.shape[:-1], self.no_distograms, self.no_bins
        )

        return logits
