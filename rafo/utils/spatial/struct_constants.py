from pathlib import Path
import json

from rafo.constants import (
    RESIDUE_TOKEN_ORDER, ATOM_ORDER,
    NUM_RESIDUE_TOKENS, NUM_ATOM_TYPES
)

import numpy as np


class StructureConstants:
    def __init__(self, struct_stats):
        self.struct_stats = struct_stats

    @classmethod
    def from_json(cls, struct_stats_path: Path):
        with open(struct_stats_path, "r") as json_file:
            struct_stats = json.load(json_file)
        return cls(struct_stats)

    @classmethod
    def from_mmcif(cls, mmcif_path: Path):
        # TODO
        pass

    def to_json(self, struct_stats_path: Path):
        with open(struct_stats_path, "w") as f:
            json.dump(self.struct_stats, f, indent=4)

    # - - - Bond lengths - - -
    @property
    def p_o3_distance(self):
        return self.struct_stats["bond_lengths"]["P_O3'"]["median"]

    @property
    def p_o5_distance(self):
        return self.struct_stats["bond_lengths"]["P_O5'"]["median"]

    @property
    def o5_c5_distance(self):
        return self.struct_stats["bond_lengths"]["O5'_C5'"]["median"]

    @property
    def c5_c4_distance(self):
        return self.struct_stats["bond_lengths"]["C5'_C4'"]["median"]

    @property
    def c4_o4_distance(self):
        return self.struct_stats["bond_lengths"]["C4'_O4'"]["median"]

    @property
    def c4_c3_distance(self):
        return self.struct_stats["bond_lengths"]["C4'_C3'"]["median"]

    @property
    def c3_c2_distance(self):
        return self.struct_stats["bond_lengths"]["C3'_C2'"]["median"]

    @property
    def c3_o3_distance(self):
        return self.struct_stats["bond_lengths"]["C3'_O3'"]["median"]

    @property
    def c2_c1_distance(self):
        return self.struct_stats["bond_lengths"]["C2'_C1'"]["median"]

    @property
    def c1_o4_distance(self):
        return self.struct_stats["bond_lengths"]["C1'_O4'"]["median"]

    @property
    def c1_n_distance(self):
        return self.struct_stats["bond_lengths"]["C1'_N"]["median"]

    # - - - Bond angles - - -

    @property
    def p_o5_c5_angle(self):
        return self.struct_stats["bond_angles"]["P_O5'_C5'"]["median"]

    @property
    def o5_c5_c4_angle(self):
        return self.struct_stats["bond_angles"]["O5'_C5'_C4'"]["median"]

    @property
    def c5_c4_c3_angle(self):
        return self.struct_stats["bond_angles"]["C5'_C4'_C3'"]["median"]

    @property
    def c4_c3_c2_angle(self):
        return self.struct_stats["bond_angles"]["C4'_C3'_C2'"]["median"]

    @property
    def c4_c3_o3_angle(self):
        return self.struct_stats["bond_angles"]["C4'_C3'_O3'"]["median"]

    @property
    def o4_c1_n_angle(self):
        return self.struct_stats["bond_angles"]["O4'_C1'_N"]["median"]

    @property
    def c5_c4_o4_angle(self):
        return self.struct_stats["bond_angles"]["C5'_C4'_O4'"]["median"]

    @property
    def c4_o4_c1_angle(self):
        return self.struct_stats["bond_angles"]["C4'_O4'_C1'"]["median"]

    # - - - Local nucleobase coordinates - - -

    @property
    def base_local_positions(self):
        coords = np.zeros((NUM_RESIDUE_TOKENS, NUM_ATOM_TYPES, 3))

        # Adenine
        coords[RESIDUE_TOKEN_ORDER["A"], ATOM_ORDER["N1"]] = \
            self.struct_stats["base_local"]["A"]["N1"]["median"]
        coords[RESIDUE_TOKEN_ORDER["A"], ATOM_ORDER["C2"]] = \
            self.struct_stats["base_local"]["A"]["C2"]["median"]
        coords[RESIDUE_TOKEN_ORDER["A"], ATOM_ORDER["N3"]] = \
            self.struct_stats["base_local"]["A"]["N3"]["median"]
        coords[RESIDUE_TOKEN_ORDER["A"], ATOM_ORDER["C4"]] = \
            self.struct_stats["base_local"]["A"]["C4"]["median"]
        coords[RESIDUE_TOKEN_ORDER["A"], ATOM_ORDER["C5"]] = \
            self.struct_stats["base_local"]["A"]["C5"]["median"]
        coords[RESIDUE_TOKEN_ORDER["A"], ATOM_ORDER["C6"]] = \
            self.struct_stats["base_local"]["A"]["C6"]["median"]
        coords[RESIDUE_TOKEN_ORDER["A"], ATOM_ORDER["N7"]] = \
            self.struct_stats["base_local"]["A"]["N7"]["median"]
        coords[RESIDUE_TOKEN_ORDER["A"], ATOM_ORDER["C8"]] = \
            self.struct_stats["base_local"]["A"]["C8"]["median"]
        coords[RESIDUE_TOKEN_ORDER["A"], ATOM_ORDER["N9"]] = \
            self.struct_stats["base_local"]["A"]["N9"]["median"]
        coords[RESIDUE_TOKEN_ORDER["A"], ATOM_ORDER["N6"]] = \
            self.struct_stats["base_local"]["A"]["N6"]["median"]

        # Guanine
        coords[RESIDUE_TOKEN_ORDER["G"], ATOM_ORDER["N1"]] = \
            self.struct_stats["base_local"]["G"]["N1"]["median"]
        coords[RESIDUE_TOKEN_ORDER["G"], ATOM_ORDER["C2"]] = \
            self.struct_stats["base_local"]["G"]["C2"]["median"]
        coords[RESIDUE_TOKEN_ORDER["G"], ATOM_ORDER["N3"]] = \
            self.struct_stats["base_local"]["G"]["N3"]["median"]
        coords[RESIDUE_TOKEN_ORDER["G"], ATOM_ORDER["C4"]] = \
            self.struct_stats["base_local"]["G"]["C4"]["median"]
        coords[RESIDUE_TOKEN_ORDER["G"], ATOM_ORDER["C5"]] = \
            self.struct_stats["base_local"]["G"]["C5"]["median"]
        coords[RESIDUE_TOKEN_ORDER["G"], ATOM_ORDER["C6"]] = \
            self.struct_stats["base_local"]["G"]["C6"]["median"]
        coords[RESIDUE_TOKEN_ORDER["G"], ATOM_ORDER["N7"]] = \
            self.struct_stats["base_local"]["G"]["N7"]["median"]
        coords[RESIDUE_TOKEN_ORDER["G"], ATOM_ORDER["C8"]] = \
            self.struct_stats["base_local"]["G"]["C8"]["median"]
        coords[RESIDUE_TOKEN_ORDER["G"], ATOM_ORDER["N9"]] = \
            self.struct_stats["base_local"]["G"]["N9"]["median"]
        coords[RESIDUE_TOKEN_ORDER["G"], ATOM_ORDER["N2"]] = \
            self.struct_stats["base_local"]["G"]["N2"]["median"]
        coords[RESIDUE_TOKEN_ORDER["G"], ATOM_ORDER["O6"]] = \
            self.struct_stats["base_local"]["G"]["O6"]["median"]

        # Cytosine
        coords[RESIDUE_TOKEN_ORDER["C"], ATOM_ORDER["N1"]] = \
            self.struct_stats["base_local"]["C"]["N1"]["median"]
        coords[RESIDUE_TOKEN_ORDER["C"], ATOM_ORDER["C2"]] = \
            self.struct_stats["base_local"]["C"]["C2"]["median"]
        coords[RESIDUE_TOKEN_ORDER["C"], ATOM_ORDER["N3"]] = \
            self.struct_stats["base_local"]["C"]["N3"]["median"]
        coords[RESIDUE_TOKEN_ORDER["C"], ATOM_ORDER["C4"]] = \
            self.struct_stats["base_local"]["C"]["C4"]["median"]
        coords[RESIDUE_TOKEN_ORDER["C"], ATOM_ORDER["C5"]] = \
            self.struct_stats["base_local"]["C"]["C5"]["median"]
        coords[RESIDUE_TOKEN_ORDER["C"], ATOM_ORDER["C6"]] = \
            self.struct_stats["base_local"]["C"]["C6"]["median"]
        coords[RESIDUE_TOKEN_ORDER["C"], ATOM_ORDER["O2"]] = \
            self.struct_stats["base_local"]["C"]["O2"]["median"]
        coords[RESIDUE_TOKEN_ORDER["C"], ATOM_ORDER["N4"]] = \
            self.struct_stats["base_local"]["C"]["N4"]["median"]

        # Uracil
        coords[RESIDUE_TOKEN_ORDER["U"], ATOM_ORDER["N1"]] = \
            self.struct_stats["base_local"]["U"]["N1"]["median"]
        coords[RESIDUE_TOKEN_ORDER["U"], ATOM_ORDER["C2"]] = \
            self.struct_stats["base_local"]["U"]["C2"]["median"]
        coords[RESIDUE_TOKEN_ORDER["U"], ATOM_ORDER["N3"]] = \
            self.struct_stats["base_local"]["U"]["N3"]["median"]
        coords[RESIDUE_TOKEN_ORDER["U"], ATOM_ORDER["C4"]] = \
            self.struct_stats["base_local"]["U"]["C4"]["median"]
        coords[RESIDUE_TOKEN_ORDER["U"], ATOM_ORDER["C5"]] = \
            self.struct_stats["base_local"]["U"]["C5"]["median"]
        coords[RESIDUE_TOKEN_ORDER["U"], ATOM_ORDER["C6"]] = \
            self.struct_stats["base_local"]["U"]["C6"]["median"]
        coords[RESIDUE_TOKEN_ORDER["U"], ATOM_ORDER["O2"]] = \
            self.struct_stats["base_local"]["U"]["O2"]["median"]
        coords[RESIDUE_TOKEN_ORDER["U"], ATOM_ORDER["O4"]] = \
            self.struct_stats["base_local"]["U"]["O4"]["median"]

        return coords

    # Phosphorus oxygen side-atoms
    @property
    def op1_local_position(self):
        return self.struct_stats["side_oxygen_local"]["OP1"]["median"]

    @property
    def op2_local_position(self):
        return self.struct_stats["side_oxygen_local"]["OP2"]["median"]

    # Sugar ring oxygen side-atom
    @property
    def o2_local_position(self):
        return self.struct_stats["side_oxygen_local"]["O2'"]["median"]
