from typing import Dict, Optional

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """
    Maintains an exponential moving average of trainable parameters.

    Update rule:
        ema = decay * ema + (1 - decay) * param

    Only parameters with ``requires_grad=True`` are tracked, so the frozen
    RiNALMo language model is skipped automatically.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.device = next(model.parameters()).device
        self.params: Dict[str, torch.Tensor] = {}

        self.reset(model)

    @torch.no_grad()
    def reset(self, model: nn.Module) -> None:
        """Re-seed the averages from the model's current parameters."""
        self.params = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self.device = next(model.parameters()).device

    def to(self, device: torch.device) -> None:
        self.params = {k: v.to(device) for k, v in self.params.items()}
        self.device = device

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            ema_p = self.params[name]
            if ema_p.device != p.device:
                ema_p = ema_p.to(p.device)
                self.params[name] = ema_p
            ema_p.sub_((ema_p - p.detach()) * (1.0 - self.decay))

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Swap EMA params into ``model`` and return a cache of the previous
        live parameters. Pair with :meth:`restore_from` to roll back.
        """
        cache: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if name not in self.params:
                continue
            cache[name] = p.detach().clone()
            p.data.copy_(self.params[name].to(p.device))
        return cache

    @torch.no_grad()
    def restore_from(
        self, model: nn.Module, cache: Dict[str, torch.Tensor]
    ) -> None:
        for name, p in model.named_parameters():
            if name in cache:
                p.data.copy_(cache[name].to(p.device))

    def state_dict(self) -> dict:
        return {"params": self.params, "decay": self.decay}

    def load_state_dict(self, state_dict: dict) -> None:
        loaded: Optional[Dict[str, torch.Tensor]] = state_dict.get("params")
        if loaded is None:
            return
        for k, v in loaded.items():
            if k in self.params:
                self.params[k] = v.clone().to(self.params[k].device)
        if "decay" in state_dict:
            self.decay = state_dict["decay"]
