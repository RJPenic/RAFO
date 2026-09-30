import torch
from torch.utils.checkpoint import checkpoint

from typing import Any, Optional, Callable


def _wrap(x: Any) -> tuple:
    # Non-tuple to tuple conversion
    return (x,) if type(x) is not tuple else x


def _forward_blocks(blocks: list[Callable], args: tuple) -> tuple:
    outs = args

    # Pass through the blocks
    for block in blocks:
        outs = _wrap(block(*outs))

    return outs


def _get_forward_blocks_func(blocks: list[Callable]) -> Callable:
    def _func(args: Any):
        return _forward_blocks(blocks, args)

    return _func


def checkpoint_blocks(
    blocks: list[Callable],
    args: tuple,
    blocks_per_ckpt: Optional[int],
    use_reentrant: bool = False,
) -> tuple:
    if blocks_per_ckpt is None or not torch.is_grad_enabled():
        # Skip gradient checkpointing
        return _forward_blocks(blocks, args)

    outs = args
    # Gradient checkpointing
    for i in range(0, len(blocks), blocks_per_ckpt):
        outs = checkpoint(
            _get_forward_blocks_func(blocks[i: i + blocks_per_ckpt]),
            outs,
            use_reentrant=use_reentrant,
        )

    return outs
