"""Ascend backend for GDN chunked prefill."""

import torch

from tokenspeed_kernel.ops.attention.gdn import (
    GdnCheckpointLayout,
    GdnChunkPrefillResult,
)
from tokenspeed_kernel.ops.attention.gdn._triton.chunk import (
    chunk_gated_delta_rule,
)
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "attention",
    "gdn_chunk_prefill",
    name="ascend_gdn_chunk_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={
        "output_h": frozenset({False, True}),
        "qk_l2norm": frozenset({False, True}),
    },
)
def ascend_gdn_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool = False,
    output_final_state: bool = True,
    output_h: bool = False,
) -> GdnChunkPrefillResult:
    # Shared API uses K-last [N, H, V, K].
    # The existing FLA-style implementation uses [N, H, K, V].
    initial_state_fla = initial_state.transpose(-2, -1).contiguous()

    result = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state_fla,
        output_final_state=output_final_state or output_h,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=qk_l2norm,
        output_h=output_h,
    )

    if output_h:
        out, final_state, h = result
        return GdnChunkPrefillResult(
            out=out,
            final_state=(
                final_state.transpose(-2, -1)
                if final_state is not None else None
            ),
            h=h.transpose(-2, -1),
            h_layout=GdnCheckpointLayout.FLA,
        )

    out, final_state = result
    return GdnChunkPrefillResult(
        out=out,
        final_state=(
            final_state.transpose(-2, -1)
            if final_state is not None else None
        ),
    )
