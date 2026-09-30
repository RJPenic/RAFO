import ml_collections as mlc
import copy

from rafo.constants import GLYCOSIDIC_N_ATOM


def get_config(name: str, finetune: bool = False) -> mlc.ConfigDict:
    c = copy.deepcopy(default_config)

    if name == "frodo":
        c.globals.c_m = 64
        c.globals.c_z = 64
        c.globals.c_s = 64

        c.model.evoformer.no_blocks = 2
        c.model.structure_module.no_blocks = 2
        c.model.no_cycles = 2
    elif name == "gimli":
        c.globals.c_m = 128
        c.globals.c_z = 128
        c.globals.c_s = 128

        c.model.evoformer.no_blocks = 6
        c.model.structure_module.no_blocks = 4
        c.model.no_cycles = 1
    elif name == "gandalf":
        c.globals.c_m = 192
        c.globals.c_z = 128
        c.globals.c_s = 256

        c.model.evoformer.no_blocks = 24
        c.model.structure_module.no_blocks = 8
        c.model.no_cycles = 3
    elif name == "legolas":
        c.globals.c_m = 128
        c.globals.c_z = 128
        c.globals.c_s = 128

        c.model.evoformer.no_blocks = 14
        c.model.structure_module.no_blocks = 8
        c.model.no_cycles = 3

        c.model.evoformer.msa_dropout = 0.3
        c.model.evoformer.pair_dropout = 0.3
        c.model.structure_module.dropout = 0.2

    elif name == "boromir":
        pass
    elif name == "aragorn":
        pass
    else:
        raise ValueError("Invalid configuration name!")

    if finetune:
        c.loss.violation.weight = 1.0

        c.training.lr_scheduler.cosine.min_lr = 1e-4
        c.training.optimizer.lr = 1e-4

        c.training.max_train_residues = 768

    return c


c_m = mlc.FieldReference(192, field_type=int)
c_z = mlc.FieldReference(128, field_type=int)
c_s = mlc.FieldReference(256, field_type=int)

no_recycle_bins = mlc.FieldReference(32, field_type=int)
min_recycle_bin = mlc.FieldReference(6., field_type=float)
max_recycle_bin = mlc.FieldReference(64., field_type=float)

no_dist_bins = mlc.FieldReference(48, field_type=int)
min_dist_bin = mlc.FieldReference(3., field_type=float)
max_dist_bin = mlc.FieldReference(52., field_type=float)

dist_ref_atoms = mlc.FieldReference(
    ["P", "C4'", GLYCOSIDIC_N_ATOM],
    field_type=list
)

no_lddt_bins = mlc.FieldReference(50, field_type=int)

eps = mlc.FieldReference(1e-6, field_type=float)

default_config = mlc.ConfigDict(
    {
        "globals": {
            "c_m": c_m,
            "c_z": c_z,
            "c_s": c_s,
            "no_recycle_bins": no_recycle_bins,
            "min_recycle_bin": min_recycle_bin,
            "max_recycle_bin": max_recycle_bin,
            "no_dist_bins": no_dist_bins,
            "min_dist_bin": min_dist_bin,
            "max_dist_bin": max_dist_bin,
            "dist_ref_atoms": dist_ref_atoms,
            "no_lddt_bins": no_lddt_bins,
            "eps": eps,
        },
        "model": {
            "input_embedder": {
                "c_m": c_m,
                "c_z": c_z,
                "rel_res_win_size": 64,
                "rel_chain_win_size": 2,
                "lm_dropout": 0.2,
                "ss_dropout": 0.2,
            },
            "evoformer": {
                "c_m": c_m,
                "c_z": c_z,
                "c_s": c_s,
                "c_hidden_msa_attn": 32,
                "c_hidden_pair_mul": 128,
                "c_hidden_pair_attn": 32,
                "c_hidden_opm": 32,
                "transition_exp_factor": 3,
                "no_blocks": 24,
                "no_heads_msa": 8,
                "no_heads_pair": 4,
                "msa_dropout": 0.15,
                "pair_dropout": 0.25,
                "use_msa_col_attn": False,
                "blocks_per_ckpt": 2,
            },
            "structure_module": {
                "c_s": c_s,
                "c_z": c_z,
                "c_ipa": 16,
                "c_angl": 128,
                "no_blocks": 4,
                "no_transition_blocks": 1,
                "no_resnet_blocks": 2,
                "no_angles": 9,
                "no_heads": 12,
                "no_qk_pts": 4,
                "no_v_pts": 8,
                "trans_scale_factor": 30,
                "dropout": 0.1,
            },
            "recycler": {
                "c_m": c_m,
                "c_z": c_z,
                "min_bin": min_recycle_bin,
                "max_bin": max_recycle_bin,
                "no_bins": no_recycle_bins,
                "repr_atom": "C4'",
            },
            "auxiliary_heads": {
                "c_s": c_s,
                "c_z": c_z,
                "c_hidden": 128,
                "no_lddt_bins": no_lddt_bins,
                "no_dist_bins": no_dist_bins,
                "dist_ref_atoms": dist_ref_atoms,
            },
            "no_cycles": 4,
        },
        "loss": {
            "distogram": {
                "weight": 0.3,
                "ref_atoms": dist_ref_atoms,
                "binner": {
                    "min_bin": min_dist_bin,
                    "max_bin": max_dist_bin,
                    "no_bins": no_dist_bins,
                },
                "eps": eps,
            },
            "lddt": {
                "weight": 0.01,
                "binner": {
                    "no_bins": no_lddt_bins,
                },
                "atom": "P",
                "min_resolution": 0.1,
                "max_resolution": 6.0,
                "eps": eps,
            },
            "torsion": {
                "weight": 0.3,
                "angle_weight": 1.0,
                "norm_weight": 0.02,
                "eps": eps,
            },
            "fape": {
                "weight": 1.0,
                "root_fape": {
                    "weight": 1.0,
                    "clamp_distance": 20,
                    "length_scale": 20,
                    "clamp_prob": 0.9,
                    "eps": eps,
                },
                "all_fape": {
                    "weight": 1.0,
                    "clamp_distance": 20,
                    "length_scale": 20,
                    "eps": eps,
                }
            },
            "violation": {
                "weight": 0.00,
                "clash": {
                    "weight": 1.0,
                    "tolerance": 1.5,
                    "eps": eps,
                },
                "inter_residue": {
                    "weight": 1.0,
                    "tolerance": 0.15,
                    "eps": eps,
                }
            }
        },
        "training": {
            "optimizer": {
                "lr": 1e-3,
                "weight_decay": 0.01,
            },
            "lr_scheduler": {
                "warmup": {
                    "iters": 1000,
                },
                "cosine": {
                    "iters": 30_000,
                    "min_lr": 1e-4,
                }
            },
            "gradient_clip_algorithm": "norm",
            "gradient_clip_val": 1.0,
            "max_train_residues": 384,
            "ema": {
                "enabled": True,
                "decay": 0.999,
            },
        }
    }
)
