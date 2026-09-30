from pathlib import Path

# Structure statistics path
STRUCT_STATS_PATH = Path(__file__).parent / "struct_stats.json"

# Residue tokens
ADENINE_TKN = "A"
THYMINE_TKN = "T"
URACIL_TKN = "U"
GUANINE_TKN = "G"
CYTOSINE_TKN = "C"
UNDEFINED_TKN = "N"

RESIDUE_TOKENS = [
    ADENINE_TKN,
    CYTOSINE_TKN,
    GUANINE_TKN,
    URACIL_TKN,
    UNDEFINED_TKN
]
STANDARD_RESIDUE_TOKENS = [
    ADENINE_TKN,
    CYTOSINE_TKN,
    GUANINE_TKN,
    URACIL_TKN,
    THYMINE_TKN
]
RESIDUE_TOKEN_ORDER = {t: i for i, t in enumerate(RESIDUE_TOKENS)}
NUM_RESIDUE_TOKENS = len(RESIDUE_TOKENS)

PURINE_TOKENS = [ADENINE_TKN, GUANINE_TKN]
PYRIMIDINE_TOKENS = [CYTOSINE_TKN, URACIL_TKN, THYMINE_TKN]

# Atoms
ATOM_TYPES = [
    "P", "OP1", "OP2", "O5'", "C5'",
    "C4'", "O4'", "C3'", "O3'", "C2'",
    "O2'", "C1'", "N9", "C8", "N7",
    "C5", "C6", "N6", "O6", "N1",
    "C2", "N2", "N3", "C4", "N4",
    "O4", "O2"
]
ATOM_ORDER = {atom: i for i, atom in enumerate(ATOM_TYPES)}
NUM_ATOM_TYPES = len(ATOM_TYPES)
BASE_ATOMS = [
        "N9", "C8", "N7", "C5", "C6",
        "N6", "O6", "N1", "C2", "N2",
        "N3", "C4", "N4", "O4", "O2"
]
BACKBONE_ATOMS = list(set(ATOM_TYPES) - set(BASE_ATOMS))

BASE_TO_ATOMS_MAP = {
    ADENINE_TKN: [
        *BACKBONE_ATOMS,
        "N9", "C8", "N7", "C5", "C6",
        "N6", "N1", "C2", "N3", "C4"
    ],
    CYTOSINE_TKN: [
        *BACKBONE_ATOMS,
        "C5", "C6", "N1", "C2", "N3",
        "C4", "N4", "O2"
    ],
    URACIL_TKN: [
        *BACKBONE_ATOMS,
        "C5", "C6", "N1", "C2", "N3",
        "C4", "O4", "O2"
    ],
    GUANINE_TKN: [
        *BACKBONE_ATOMS,
        "N9", "C8", "N7", "C5", "C6",
        "O6", "N1", "C2", "N2", "N3",
        "C4"
    ],
    UNDEFINED_TKN: [
        *BACKBONE_ATOMS
    ]
}
BASE_TO_ATOMS_MAP[THYMINE_TKN] = BASE_TO_ATOMS_MAP[URACIL_TKN]

RESIDUE_BONDS = [
    # Backbone
    ("OP1", "P"),
    ("OP2", "P"),
    ("P", "O5'"),
    ("O5'", "C5'"),
    ("C5'", "C4'"),
    ("C4'", "O4'"),
    ("C4'", "C3'"),
    ("C3'", "O3'"),
    ("C3'", "C2'"),
    ("C2'", "O2'"),
    ("C2'", "C1'"),
    ("C1'", "O4'"),
    # Backbone-to-base
    ("C1'", "N9"),
    ("C1'", "N1"),
    # Base
    ("N1", "C2"),
    ("C2", "O2"),
    ("C2", "N2"),
    ("C2", "N3"),
    ("N3", "C4"),
    ("C4", "N4"),
    ("C4", "O4"),
    ("C4", "C5"),
    ("C5", "C6"),
    ("C6", "N1"),
    ("C6", "N6"),
    ("C6", "O6"),
    ("N9", "C4"),
    ("N9", "C8"),
    ("C8", "N7"),
    ("N7", "C5"),
]

# Van der Waals radii
VDW_RADII = {
    "C": 1.7,
    "O": 1.52,
    "P": 1.8,
    "N": 1.55,
}
RNA_VDW_RADII = [VDW_RADII[atom[0]] for atom in ATOM_TYPES]

# Frames
# [C5', C4', O4']: origin C4', +x along C5' -> C4', O4' in plane.
ROOT_ATOM = "C4'"
ROOT_FRAME = ROOT_ATOM
ROOT_FRAME_ATOMS = ["C5'", "C4'", "O4'"]

FRAME_ATOMS = {
    "C4'": ROOT_FRAME_ATOMS,
    # C4'(root) -> C5'(frame) -> O5' -> P
    "C5'": ["C4'", "C5'", "O4'"],
    "O5'": ["C5'", "O5'", "C4'"],
    "P": ["O5'", "P", "C5'"],
    # C4'(root) -> O4'(frame) -> C1' -> base
    "C1'": ["O4'", "C1'", "C4'"],
    # C4'(root) -> C3' -> C2'/O3'
    "C3'": ["C4'", "C3'", "C5'"],
    "C2'": ["C3'", "C2'", "C4'"],
    "O3'": ["C3'", "O3'", "C4'"],
    # Base attachment
    "N_pre_torsion": {
        "purine": ["C1'", "N9", "O4'"],
        "pyrimidine": ["C1'", "N1", "O4'"],
    },
    "N": {
        "purine": ["C1'", "N9", "C8"],
        "pyrimidine": ["C1'", "N1", "C2"],
    },
}

# Key order preserved (in python 3.7+)
FRAMES = FRAME_ATOMS.keys()
NUM_FRAMES = len(FRAMES)
FRAME_ORDER = {frame: i for i, frame in enumerate(FRAMES)}

# Torsions
TORSION_ATOMS = {
    # P-direction torsions
    "O4'_C4'_C5'_O5'": ["O4'", "C4'", "C5'", "O5'"],
    "C4'_C5'_O5'_P": ["C4'", "C5'", "O5'", "P"],
    "C5'_O5'_P_OP1": ["C5'", "O5'", "P", "OP1"],
    # Sugar torsions
    "C5'_C4'_O4'_C1'": ["C5'", "C4'", "O4'", "C1'"],
    "O4'_C5'_C4'_C3'": ["O4'", "C5'", "C4'", "C3'"],
    "C5'_C4'_C3'_C2'": ["C5'", "C4'", "C3'", "C2'"],
    "C5'_C4'_C3'_O3'": ["C5'", "C4'", "C3'", "O3'"],
    # Base torsions
    "C4'_O4'_C1'_N": {
        "purine": ["C4'", "O4'", "C1'", "N9"],
        "pyrimidine": ["C4'", "O4'", "C1'", "N1"]
    },
    "O4'_C1'_N_C": {
        "purine": ["O4'", "C1'", "N9", "C8"],
        "pyrimidine": ["O4'", "C1'", "N1", "C2"]
    },
}

TORSIONS = TORSION_ATOMS.keys()
NUM_TORSIONS = len(TORSIONS)
TORSION_ORDER = {trsn: i for i, trsn in enumerate(TORSIONS)}

# Other
LDDT_CUTOFF = 30.0

# Metric reference atoms
RMSD_REF_ATOM = "C3'"
TM_SCORE_REF_ATOM = "C3'"
LDDT_REF_ATOM = "P"

# Reference atoms for filters
MODELED_RES_FILTER_REF_ATOM = "P"
CHAIN_CLASH_FILTER_REF_ATOM = "P"
BACKBONE_BREAK_FILTER_REF_ATOM = "P"
STRUCT_DEPTH_FILTER_REF_ATOM = "P"
LONG_RANGE_CONTACT_FILTER_REF_ATOM = "P"

# Reference atoms for model/loss components
GLYCOSIDIC_N_ATOM = "N"
PURINE_N_ATOM = "N9"
PYRIMIDINE_N_ATOM = "N1"

SEQ_DELIMITER = "&"
