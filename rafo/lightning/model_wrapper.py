from typing import Optional
from pathlib import Path

import lightning.pytorch as pl
import torch

import ml_collections as mlc

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR

from rafo.model.model import RAFO
from rafo.utils.exponential_moving_average import ExponentialMovingAverage
from rafo.utils.loss import RNAStructureLoss
from rafo.utils.metrics import compute_lddt, compute_rmsd, compute_tm_score
from rafo.utils.save import save_to_pdb


# The RiNALMo language model is frozen and always rebuilt from its pretrained
# weights in `RAFO.__init__`, so its parameters are stripped from every saved
# state dict (they would otherwise dominate the checkpoint size) and filled
# back in when loading.
LM_STATE_DICT_PREFIX = "model.lm."


class RNAStructurePredictionWrapper(pl.LightningModule):
    def __init__(
        self,
        config: mlc.ConfigDict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.chunk_size = None
        self.use_lma = False
        self.inference_dropout = False
        self.inference_seed = None
        self.inference_no_cycles = None

        self.config = config

        self.model = RAFO(self.config)
        self.loss = RNAStructureLoss(self.config)

        # Freeze LM
        for param in self.model.lm.parameters():
            param.requires_grad = False

        self.ema_enabled = self.config.training.ema.enabled
        self.ema = (
            ExponentialMovingAverage(
                model=self.model, decay=self.config.training.ema.decay
            )
            if self.ema_enabled
            else None
        )
        self._ema_cache = None

    def set_chunk_size(self, chunk_size: Optional[int]) -> None:
        self.chunk_size = chunk_size

    def set_use_lma(self, use_lma: bool) -> None:
        self.use_lma = use_lma

    def set_inference_dropout(self, dropout: bool) -> None:
        self.inference_dropout = dropout

    def set_inference_seed(self, seed: int) -> None:
        self.inference_seed = seed

    def set_inference_no_cycles(self, no_cycles: Optional[int]) -> None:
        self.inference_no_cycles = no_cycles

    def enable_dropout(self) -> None:
        for module in self.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

    def forward(
        self,
        seq_tokens: torch.Tensor,
        seq_tokens_lm: torch.Tensor,
        seq_batch_idx: torch.Tensor,
        res_idx: torch.Tensor,
        asym_id: torch.Tensor,
        sym_id: torch.Tensor,
        entity_id: torch.Tensor,
        ss_feat: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        msa_mask: Optional[torch.Tensor] = None,
        no_cycles: Optional[int] = None,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
    ):
        return self.model(
            seq_tokens=seq_tokens,
            seq_tokens_lm=seq_tokens_lm,
            seq_batch_idx=seq_batch_idx,
            res_idx=res_idx,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
            ss_feat=ss_feat,
            seq_mask=seq_mask,
            msa_mask=msa_mask,
            no_cycles=no_cycles,
            chunk_size=chunk_size,
            use_lma=use_lma,
        )

    def _common_step(
        self,
        batch,
        batch_idx,
        log_prefix: str,
        log_rmsd: bool = False,
        log_tm_score: bool = False,
        save_structs: bool = False,
        no_cycles: Optional[int] = None,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
    ):
        model_outs = self(
            seq_tokens=batch["seq_tokens"],
            seq_tokens_lm=batch["seq_tokens_lm"],
            seq_batch_idx=batch["seq_batch_idx"],
            res_idx=batch["residue_index"],
            asym_id=batch["asym_id"],
            sym_id=batch["sym_id"],
            entity_id=batch["entity_id"],
            ss_feat=batch["secondary_structures"],
            seq_mask=batch["seq_mask"],
            msa_mask=batch["msa_mask"],
            no_cycles=no_cycles,
            chunk_size=chunk_size,
            use_lma=use_lma,
        )

        loss = self.loss(model_outs, batch)

        # Loss logging
        log = {
            f"{log_prefix}/loss_total": loss["total"],
            f"{log_prefix}/loss_fape": loss["fape"],
            f"{log_prefix}/loss_torsion": loss["torsion"],
            f"{log_prefix}/loss_distogram": loss["distogram"],
            f"{log_prefix}/loss_lddt": loss["lddt"],
            f"{log_prefix}/loss_violation": loss["violation"],
        }

        # Metric logging (RMSD, LDDT, etc)
        rmsd = None
        if log_rmsd:
            rmsd = compute_rmsd(
                target_atom_pos=batch["atom_positions"],
                pred_atom_pos=model_outs["sm"]["atom_positions"],
                atom_mask=batch["atom_mask"],
                batch_reduce=False,
            )
            log[f"{log_prefix}/rmsd"] = rmsd.mean()

        tm_score = None
        if log_tm_score:
            tm_score = compute_tm_score(
                target_atom_pos=batch["atom_positions"],
                pred_atom_pos=model_outs["sm"]["atom_positions"],
                atom_mask=batch["atom_mask"],
                batch_reduce=False,
            )
            log[f"{log_prefix}/tm_score"] = tm_score.mean()

        lddt_val = compute_lddt(
            target_atom_pos=batch["atom_positions"],
            pred_atom_pos=model_outs["sm"]["atom_positions"],
            atom_mask=batch["atom_mask"],
            batch_reduce=False,
        )
        log[f"{log_prefix}/lddt"] = lddt_val.mean()

        self.log_dict(
            log,
            sync_dist=(log_prefix != "train"),
            batch_size=len(batch["sequence"]),
            add_dataloader_idx=False
        )

        if save_structs:
            batch_size = len(batch["sequence"])
            for i in range(batch_size):
                struct_id = batch['entry_id'][i]

                filename = f"{struct_id}.pdb"
                output_dir = \
                    Path(self.trainer.default_root_dir)
                output_file = output_dir / filename

                output_dir.mkdir(parents=True, exist_ok=True)

                rmsd_str = rmsd[i] if rmsd is not None else 'N/A'
                tm_str = tm_score[i] if tm_score is not None else 'N/A'

                save_to_pdb(
                    atom_pos=model_outs["sm"]["atom_positions"][i],
                    atom_mask=model_outs["sm"]["atom_mask"][i],
                    seqs=batch["sequences"][i],
                    chain_ids=batch["chain_ids"][i],
                    output_file=output_file,
                    res_temp_factors=model_outs["aux"]["pLDDT"][i],
                    remarks=[
                        f"RMSD_P = {rmsd_str}",
                        f"TM_SCORE = {tm_str}",
                        f"LDDT_P = {lddt_val[i]}"
                    ],
                )

        return loss["total"]

    def training_step(self, batch, batch_idx):
        return self._common_step(
            batch,
            batch_idx,
            log_prefix="train",
            log_rmsd=False,
            log_tm_score=False,
        )

    def on_before_zero_grad(self, optimizer):
        if self.ema is not None:
            if self.ema.device != next(self.model.parameters()).device:
                self.ema.to(next(self.model.parameters()).device)
            self.ema.update(self.model)

    def _swap_in_ema(self):
        if self.ema is None or self._ema_cache is not None:
            return
        self.ema.to(next(self.model.parameters()).device)
        self._ema_cache = self.ema.copy_to(self.model)

    def _restore_live_weights(self):
        if self.ema is None or self._ema_cache is None:
            return
        self.ema.restore_from(self.model, self._ema_cache)
        self._ema_cache = None

    def on_validation_start(self):
        self._swap_in_ema()

    def on_validation_end(self):
        self._restore_live_weights()

    def on_test_start(self):
        self._swap_in_ema()

    def on_test_end(self):
        self._restore_live_weights()

    def on_predict_start(self):
        self._swap_in_ema()

    def on_predict_end(self):
        self._restore_live_weights()

    @property
    def _lm_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.model.lm.parameters())

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)

        if not self._lm_frozen:
            # LM is being trained, its weights have to be kept
            return state_dict

        # `state_dict(destination, prefix, keep_vars)` may also be called
        # positionally (deprecated, but still supported by torch)
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        lm_prefix = prefix + LM_STATE_DICT_PREFIX

        # Drop LM-related keys
        for key in [k for k in state_dict if k.startswith(lm_prefix)]:
            del state_dict[key]

        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        # State dicts saved by this wrapper can carry no LM weights,
        # so take them from the pretrained LM that is already in memory
        # (if not in the state dict).
        if not any(k.startswith(LM_STATE_DICT_PREFIX) for k in state_dict):
            state_dict = dict(state_dict)
            state_dict.update({
                LM_STATE_DICT_PREFIX + name: param
                for name, param in self.model.lm.state_dict().items()
            })

        return super().load_state_dict(state_dict, *args, **kwargs)

    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.ema is not None and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._common_step(
            batch,
            batch_idx,
            log_prefix=f"val_{dataloader_idx + 1}",
            log_rmsd=True,
            log_tm_score=True,
            no_cycles=self.inference_no_cycles,
            chunk_size=self.chunk_size,
            use_lma=self.use_lma,
        )

    def test_step(self, batch, batch_idx):
        return self._common_step(
            batch,
            batch_idx,
            log_prefix="test",
            log_rmsd=True,
            log_tm_score=True,
            save_structs=True,
            no_cycles=self.inference_no_cycles,
            chunk_size=self.chunk_size,
            use_lma=self.use_lma,
        )

    def on_predict_model_eval(self) -> None:
        self.eval()

        if self.inference_dropout:
            # Enable Monte Carlo dropout
            self.enable_dropout()

    def predict_step(self, batch, batch_idx):
        if self.inference_seed is not None:
            pl.seed_everything(self.inference_seed)

        model_outs = self(
            seq_tokens=batch["seq_tokens"],
            seq_tokens_lm=batch["seq_tokens_lm"],
            seq_batch_idx=batch["seq_batch_idx"],
            res_idx=batch["residue_index"],
            asym_id=batch["asym_id"],
            sym_id=batch["sym_id"],
            entity_id=batch["entity_id"],
            ss_feat=batch["secondary_structures"],
            seq_mask=batch["seq_mask"],
            msa_mask=batch["msa_mask"],
            no_cycles=self.inference_no_cycles,
            chunk_size=self.chunk_size,
            use_lma=self.use_lma,
        )

        output_dir = \
            Path(self.trainer.default_root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_size = len(batch["sequence"])
        for i in range(batch_size):
            save_to_pdb(
                atom_pos=model_outs["sm"]["atom_positions"][i],
                atom_mask=model_outs["sm"]["atom_mask"][i],
                seqs=batch["sequences"][i],
                chain_ids=batch["chain_ids"][i],
                output_file=output_dir / f"{batch['entry_id'][i]}.pdb",
                res_temp_factors=model_outs["aux"]["pLDDT"][i],
            )

    def configure_optimizers(self):
        optimizer = AdamW(
            params=filter(lambda p: p.requires_grad, self.parameters()),
            **self.config.training.optimizer
        )

        # Scheduler parameters from config
        warmup_iters = self.config.training.lr_scheduler.warmup.iters
        cosine_iters = self.config.training.lr_scheduler.cosine.iters
        start_lr = self.config.training.optimizer.lr
        min_lr = self.config.training.lr_scheduler.cosine.min_lr

        # Warmup scheduler
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda it: min(1.0, (it + 1) / (warmup_iters + 1))
        )

        # Cosine decay scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_iters,
            eta_min=min_lr
        )

        # Stagnate at min value (constant LR)
        stagnate_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda it: min_lr / start_lr  # LR stays at min_lr
        )

        # Sequential scheduler: warmup -> cosine -> stagnate
        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[
                warmup_scheduler, cosine_scheduler, stagnate_scheduler
            ],
            milestones=[warmup_iters, warmup_iters + cosine_iters]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
