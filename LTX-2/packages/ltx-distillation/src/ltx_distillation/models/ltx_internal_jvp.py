from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core.model.transformer.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from ltx_core.utils import rms_norm


_TRACE_INTERNAL_JVP = os.environ.get("LTX_INTERNAL_JVP_TRACE", "0") == "1"


def _trace(message: str) -> None:
    if _TRACE_INTERNAL_JVP:
        print(f"[InternalJVP] {message}", flush=True)


@dataclass(frozen=True)
class TransformerArgsWithT:
    primal: TransformerArgs
    t_x: torch.Tensor
    t_timesteps: torch.Tensor
    t_embedded_timestep: torch.Tensor
    t_cross_scale_shift_timestep: torch.Tensor | None = None
    t_cross_gate_timestep: torch.Tensor | None = None


def _unwrap_fsdp(module):
    return getattr(module, "module", module)


def _linear_tangent(linear: torch.nn.Linear, t_x: torch.Tensor) -> torch.Tensor:
    return F.linear(t_x.to(dtype=linear.weight.dtype), linear.weight, None)


def _linear_fp32_no_param_grad(linear: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    linear = _unwrap_fsdp(linear)
    weight = linear.weight.detach().to(torch.float32)
    bias = linear.bias.detach().to(torch.float32) if linear.bias is not None else None
    return F.linear(x.to(torch.float32), weight, bias)


def _timestep_embedding_fp32(timestep_embedder: torch.nn.Module, sample: torch.Tensor) -> torch.Tensor:
    timestep_embedder = _unwrap_fsdp(timestep_embedder)
    sample = _linear_fp32_no_param_grad(timestep_embedder.linear_1, sample)
    if timestep_embedder.act is not None:
        sample = timestep_embedder.act(sample)
    sample = _linear_fp32_no_param_grad(timestep_embedder.linear_2, sample)
    if timestep_embedder.post_act is not None:
        sample = timestep_embedder.post_act(sample)
    return sample


def _adaln_single_fp32(adaln: torch.nn.Module, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """AdaLN time path in fp32, independent of FSDP flat-parameter views."""
    adaln = _unwrap_fsdp(adaln)
    with torch.amp.autocast(device_type=timestep.device.type, enabled=False):
        embedded_timestep = adaln.emb.time_proj(timestep.to(torch.float32))
        embedded_timestep = _timestep_embedding_fp32(
            adaln.emb.timestep_embedder,
            embedded_timestep,
        )
        timestep_values = _linear_fp32_no_param_grad(adaln.linear, adaln.silu(embedded_timestep))
    return timestep_values, embedded_timestep



def _fp32_jvp_impl(fn, *args_tangent_pairs):
    """Promote primal+tangent to FP32 for torch.func.jvp, then cast back."""
    args = tuple(p for p, _ in args_tangent_pairs)
    tangents = tuple(t for _, t in args_tangent_pairs)
    out_fp32, t_out_fp32 = torch.func.jvp(fn, tuple(a.float() for a in args), tuple(t.float() for t in tangents))
    return out_fp32.to(args[0].dtype), t_out_fp32.detach().to(args[0].dtype)


def _rms_norm_with_t(x: torch.Tensor, t_x: torch.Tensor, eps: float):
    # RMS norm has no weights — FP32 safe
    return _fp32_jvp_impl(lambda z: rms_norm(z, eps=eps), (x, t_x))


def _attention_with_t(
    attention: Attention,
    x: torch.Tensor,
    t_x: torch.Tensor,
    *,
    context: torch.Tensor | None,
    t_context: torch.Tensor | None,
    mask: torch.Tensor | None,
    pe: torch.Tensor | None,
    k_pe: torch.Tensor | None,
):
    attention = _unwrap_fsdp(attention)
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        # Promote attention weights to FP32 for exact JVP
        orig_dtype = attention.to_q.weight.dtype
        attention.to(torch.float32)
        try:
            if context is None:
                return _fp32_jvp_impl(
                    lambda xx: attention(xx, context=None, mask=mask, pe=pe, k_pe=k_pe),
                    (x, t_x),
                )
            if t_context is None:
                return _fp32_jvp_impl(
                    lambda xx: attention(xx, context=context, mask=mask, pe=pe, k_pe=k_pe),
                    (x, t_x),
                )
            return _fp32_jvp_impl(
                lambda xx, cc: attention(xx, context=cc, mask=mask, pe=pe, k_pe=k_pe),
                (x, t_x), (context, t_context),
            )
        finally:
            attention.to(orig_dtype)


def _feed_forward_with_t(ff: FeedForward, x: torch.Tensor, t_x: torch.Tensor):
    ff = _unwrap_fsdp(ff)
    orig_dtype = next(ff.parameters()).dtype
    ff.to(torch.float32)
    try:
        return _fp32_jvp_impl(ff, (x, t_x))
    finally:
        ff.to(orig_dtype)


def _layer_norm_with_t(norm: torch.nn.LayerNorm, x: torch.Tensor, t_x: torch.Tensor):
    # LayerNorm has affine weights — but no matmul, so FP32 promotion is safe
    norm = _unwrap_fsdp(norm)
    return _fp32_jvp_impl(norm, (x, t_x))


def _ada_values_tangent(
    num_ada_params: int,
    batch_size: int,
    timestep_tangent: torch.Tensor,
    indices: slice,
) -> tuple[torch.Tensor, ...]:
    return timestep_tangent.reshape(batch_size, timestep_tangent.shape[1], num_ada_params, -1)[:, :, indices, :].unbind(
        dim=2
    )


def _get_ada_values_with_t(
    block: BasicAVTransformerBlock,
    scale_shift_table: torch.Tensor,
    batch_size: int,
    timestep: torch.Tensor,
    timestep_tangent: torch.Tensor,
    indices: slice,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    primal = block.get_ada_values(scale_shift_table, batch_size, timestep, indices)
    tangent = _ada_values_tangent(scale_shift_table.shape[0], batch_size, timestep_tangent, indices)
    return primal, tangent


def _get_av_ca_ada_values_with_t(
    block: BasicAVTransformerBlock,
    scale_shift_table: torch.Tensor,
    batch_size: int,
    scale_shift_timestep: torch.Tensor,
    t_scale_shift_timestep: torch.Tensor,
    gate_timestep: torch.Tensor,
    t_gate_timestep: torch.Tensor,
    num_scale_shift_values: int = 4,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    primal = block.get_av_ca_ada_values(
        scale_shift_table,
        batch_size,
        scale_shift_timestep,
        gate_timestep,
        num_scale_shift_values=num_scale_shift_values,
    )
    scale_shift_tangent = _ada_values_tangent(
        num_scale_shift_values,
        batch_size,
        t_scale_shift_timestep,
        slice(None, None),
    )
    gate_tangent = _ada_values_tangent(
        scale_shift_table.shape[0] - num_scale_shift_values,
        batch_size,
        t_gate_timestep,
        slice(None, None),
    )
    tangent = tuple(t.squeeze(2) for t in scale_shift_tangent) + tuple(t.squeeze(2) for t in gate_tangent)
    return primal, tangent


def prepare_transformer_args_with_t(
    preprocessor: TransformerArgsPreprocessor | MultiModalTransformerArgsPreprocessor,
    modality,
    t_latent: torch.Tensor,
    t_timesteps_input: torch.Tensor,
) -> TransformerArgsWithT:
    primal = preprocessor.prepare(modality)
    timestep_input = modality.timesteps.detach().clone().to(torch.float32)
    t_timesteps_input = t_timesteps_input.detach().clone().to(torch.float32)

    if isinstance(preprocessor, MultiModalTransformerArgsPreprocessor):
        simple_preprocessor = preprocessor.simple_preprocessor
    else:
        simple_preprocessor = preprocessor

    t_x = _linear_tangent(simple_preprocessor.patchify_proj, t_latent)

    def _prepare_timestep_fp32(raw_timestep: torch.Tensor):
        timestep = raw_timestep * simple_preprocessor.timestep_scale_multiplier
        timestep, embedded_timestep = _adaln_single_fp32(simple_preprocessor.adaln, timestep.flatten())
        timestep = timestep.view(modality.latent.shape[0], -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(
            modality.latent.shape[0],
            -1,
            embedded_timestep.shape[-1],
        )
        return timestep, embedded_timestep

    _, (t_timesteps, t_embedded_timestep) = torch.func.jvp(
        _prepare_timestep_fp32,
        (timestep_input,),
        (t_timesteps_input,),
    )

    t_cross_scale_shift_timestep = None
    t_cross_gate_timestep = None
    if isinstance(preprocessor, MultiModalTransformerArgsPreprocessor):
        def _prepare_cross_attention_timestep_fp32(raw_timestep: torch.Tensor):
            timestep = raw_timestep * preprocessor.simple_preprocessor.timestep_scale_multiplier
            av_ca_factor = (
                preprocessor.av_ca_timestep_scale_multiplier
                / preprocessor.simple_preprocessor.timestep_scale_multiplier
            )

            scale_shift_timestep, _ = _adaln_single_fp32(preprocessor.cross_scale_shift_adaln, timestep.flatten())
            scale_shift_timestep = scale_shift_timestep.view(
                modality.latent.shape[0],
                -1,
                scale_shift_timestep.shape[-1],
            )
            gate_noise_timestep, _ = _adaln_single_fp32(
                preprocessor.cross_gate_adaln,
                timestep.flatten() * av_ca_factor,
            )
            gate_noise_timestep = gate_noise_timestep.view(
                modality.latent.shape[0],
                -1,
                gate_noise_timestep.shape[-1],
            )
            return scale_shift_timestep, gate_noise_timestep

        _, (t_cross_scale_shift_timestep, t_cross_gate_timestep) = torch.func.jvp(
            _prepare_cross_attention_timestep_fp32,
            (timestep_input,),
            (t_timesteps_input,),
        )

    return TransformerArgsWithT(
        primal=primal,
        t_x=t_x.detach(),
        t_timesteps=t_timesteps.detach(),
        t_embedded_timestep=t_embedded_timestep.detach(),
        t_cross_scale_shift_timestep=(
            t_cross_scale_shift_timestep.detach() if t_cross_scale_shift_timestep is not None else None
        ),
        t_cross_gate_timestep=t_cross_gate_timestep.detach() if t_cross_gate_timestep is not None else None,
    )


def _replace_x(args_with_t: TransformerArgsWithT, x: torch.Tensor, t_x: torch.Tensor) -> TransformerArgsWithT:
    return TransformerArgsWithT(
        primal=replace(args_with_t.primal, x=x),
        t_x=t_x,
        t_timesteps=args_with_t.t_timesteps,
        t_embedded_timestep=args_with_t.t_embedded_timestep,
        t_cross_scale_shift_timestep=args_with_t.t_cross_scale_shift_timestep,
        t_cross_gate_timestep=args_with_t.t_cross_gate_timestep,
    )


def _block_with_t_impl(
    block: BasicAVTransformerBlock,
    video: TransformerArgsWithT | None,
    audio: TransformerArgsWithT | None,
    perturbations: BatchedPerturbationConfig,
) -> tuple[TransformerArgsWithT | None, TransformerArgsWithT | None]:
    block = _unwrap_fsdp(block)
    _trace(f"block {block.idx} start")
    if video is None and audio is None:
        raise ValueError("At least one modality must be provided")

    batch_size = (video or audio).primal.x.shape[0]

    vx = video.primal.x if video is not None else None
    t_vx = video.t_x if video is not None else None
    ax = audio.primal.x if audio is not None else None
    t_ax = audio.t_x if audio is not None else None

    run_vx = video is not None and video.primal.enabled and vx.numel() > 0
    run_ax = audio is not None and audio.primal.enabled and ax.numel() > 0
    run_a2v = run_vx and (audio is not None and ax.numel() > 0)
    run_v2a = run_ax and (video is not None and vx.numel() > 0)

    if run_vx:
        (vshift_msa, vscale_msa, vgate_msa), (t_vshift_msa, t_vscale_msa, t_vgate_msa) = _get_ada_values_with_t(
            block,
            block.scale_shift_table,
            batch_size,
            video.primal.timesteps,
            video.t_timesteps,
            slice(0, 3),
        )
        if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, block.idx):
            _trace(f"block {block.idx} video_self_attn")
            norm_vx_base, t_norm_vx_base = _rms_norm_with_t(vx, t_vx, block.norm_eps)
            norm_vx = norm_vx_base * (1 + vscale_msa) + vshift_msa
            t_norm_vx = t_norm_vx_base * (1 + vscale_msa) + norm_vx_base * t_vscale_msa + t_vshift_msa
            v_attn, t_v_attn = _attention_with_t(
                block.attn1,
                norm_vx,
                t_norm_vx,
                context=None,
                t_context=None,
                mask=None,
                pe=video.primal.positional_embeddings,
                k_pe=None,
            )
            v_mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, block.idx, vx)
            vx = vx + v_attn * vgate_msa * v_mask
            t_vx = t_vx + (t_v_attn * vgate_msa + v_attn * t_vgate_msa) * v_mask

        _trace(f"block {block.idx} video_ctx_attn")
        norm_vx_ctx, t_norm_vx_ctx = _rms_norm_with_t(vx, t_vx, block.norm_eps)
        v_attn2, t_v_attn2 = _attention_with_t(
            block.attn2,
            norm_vx_ctx,
            t_norm_vx_ctx,
            context=video.primal.context,
            t_context=None,
            mask=video.primal.context_mask,
            pe=None,
            k_pe=None,
        )
        vx = vx + v_attn2
        t_vx = t_vx + t_v_attn2

    if run_ax:
        (ashift_msa, ascale_msa, agate_msa), (t_ashift_msa, t_ascale_msa, t_agate_msa) = _get_ada_values_with_t(
            block,
            block.audio_scale_shift_table,
            batch_size,
            audio.primal.timesteps,
            audio.t_timesteps,
            slice(0, 3),
        )
        if not perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, block.idx):
            _trace(f"block {block.idx} audio_self_attn")
            norm_ax_base, t_norm_ax_base = _rms_norm_with_t(ax, t_ax, block.norm_eps)
            norm_ax = norm_ax_base * (1 + ascale_msa) + ashift_msa
            t_norm_ax = t_norm_ax_base * (1 + ascale_msa) + norm_ax_base * t_ascale_msa + t_ashift_msa
            a_attn, t_a_attn = _attention_with_t(
                block.audio_attn1,
                norm_ax,
                t_norm_ax,
                context=None,
                t_context=None,
                mask=None,
                pe=audio.primal.positional_embeddings,
                k_pe=None,
            )
            a_mask = perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, block.idx, ax)
            ax = ax + a_attn * agate_msa * a_mask
            t_ax = t_ax + (t_a_attn * agate_msa + a_attn * t_agate_msa) * a_mask

        _trace(f"block {block.idx} audio_ctx_attn")
        norm_ax_ctx, t_norm_ax_ctx = _rms_norm_with_t(ax, t_ax, block.norm_eps)
        a_attn2, t_a_attn2 = _attention_with_t(
            block.audio_attn2,
            norm_ax_ctx,
            t_norm_ax_ctx,
            context=audio.primal.context,
            t_context=None,
            mask=audio.primal.context_mask,
            pe=None,
            k_pe=None,
        )
        ax = ax + a_attn2
        t_ax = t_ax + t_a_attn2

    if run_a2v or run_v2a:
        vx_norm3, t_vx_norm3 = _rms_norm_with_t(vx, t_vx, block.norm_eps)
        ax_norm3, t_ax_norm3 = _rms_norm_with_t(ax, t_ax, block.norm_eps)

        (
            scale_ca_audio_hidden_states_a2v,
            shift_ca_audio_hidden_states_a2v,
            scale_ca_audio_hidden_states_v2a,
            shift_ca_audio_hidden_states_v2a,
            gate_out_v2a,
        ), (
            t_scale_ca_audio_hidden_states_a2v,
            t_shift_ca_audio_hidden_states_a2v,
            t_scale_ca_audio_hidden_states_v2a,
            t_shift_ca_audio_hidden_states_v2a,
            t_gate_out_v2a,
        ) = _get_av_ca_ada_values_with_t(
            block,
            block.scale_shift_table_a2v_ca_audio,
            batch_size,
            audio.primal.cross_scale_shift_timestep,
            audio.t_cross_scale_shift_timestep,
            audio.primal.cross_gate_timestep,
            audio.t_cross_gate_timestep,
        )

        (
            scale_ca_video_hidden_states_a2v,
            shift_ca_video_hidden_states_a2v,
            scale_ca_video_hidden_states_v2a,
            shift_ca_video_hidden_states_v2a,
            gate_out_a2v,
        ), (
            t_scale_ca_video_hidden_states_a2v,
            t_shift_ca_video_hidden_states_a2v,
            t_scale_ca_video_hidden_states_v2a,
            t_shift_ca_video_hidden_states_v2a,
            t_gate_out_a2v,
        ) = _get_av_ca_ada_values_with_t(
            block,
            block.scale_shift_table_a2v_ca_video,
            batch_size,
            video.primal.cross_scale_shift_timestep,
            video.t_cross_scale_shift_timestep,
            video.primal.cross_gate_timestep,
            video.t_cross_gate_timestep,
        )

        if run_a2v and not perturbations.all_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, block.idx):
            _trace(f"block {block.idx} a2v_cross_attn")
            vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v) + shift_ca_video_hidden_states_a2v
            t_vx_scaled = (
                t_vx_norm3 * (1 + scale_ca_video_hidden_states_a2v)
                + vx_norm3 * t_scale_ca_video_hidden_states_a2v
                + t_shift_ca_video_hidden_states_a2v
            )
            ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v
            t_ax_scaled = (
                t_ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v)
                + ax_norm3 * t_scale_ca_audio_hidden_states_a2v
                + t_shift_ca_audio_hidden_states_a2v
            )
            a2v_attn, t_a2v_attn = _attention_with_t(
                block.audio_to_video_attn,
                vx_scaled,
                t_vx_scaled,
                context=ax_scaled,
                t_context=t_ax_scaled,
                mask=None,
                pe=video.primal.cross_positional_embeddings,
                k_pe=audio.primal.cross_positional_embeddings,
            )
            a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, block.idx, vx)
            vx = vx + a2v_attn * gate_out_a2v * a2v_mask
            t_vx = t_vx + (t_a2v_attn * gate_out_a2v + a2v_attn * t_gate_out_a2v) * a2v_mask

        if run_v2a and not perturbations.all_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, block.idx):
            _trace(f"block {block.idx} v2a_cross_attn")
            ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
            t_ax_scaled = (
                t_ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a)
                + ax_norm3 * t_scale_ca_audio_hidden_states_v2a
                + t_shift_ca_audio_hidden_states_v2a
            )
            vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a
            t_vx_scaled = (
                t_vx_norm3 * (1 + scale_ca_video_hidden_states_v2a)
                + vx_norm3 * t_scale_ca_video_hidden_states_v2a
                + t_shift_ca_video_hidden_states_v2a
            )
            v2a_attn, t_v2a_attn = _attention_with_t(
                block.video_to_audio_attn,
                ax_scaled,
                t_ax_scaled,
                context=vx_scaled,
                t_context=t_vx_scaled,
                mask=None,
                pe=audio.primal.cross_positional_embeddings,
                k_pe=video.primal.cross_positional_embeddings,
            )
            v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, block.idx, ax)
            ax = ax + v2a_attn * gate_out_v2a * v2a_mask
            t_ax = t_ax + (t_v2a_attn * gate_out_v2a + v2a_attn * t_gate_out_v2a) * v2a_mask

    if run_vx:
        _trace(f"block {block.idx} video_ff")
        (vshift_mlp, vscale_mlp, vgate_mlp), (t_vshift_mlp, t_vscale_mlp, t_vgate_mlp) = _get_ada_values_with_t(
            block,
            block.scale_shift_table,
            batch_size,
            video.primal.timesteps,
            video.t_timesteps,
            slice(3, None),
        )
        vx_scaled_base, t_vx_scaled_base = _rms_norm_with_t(vx, t_vx, block.norm_eps)
        vx_scaled = vx_scaled_base * (1 + vscale_mlp) + vshift_mlp
        t_vx_scaled = t_vx_scaled_base * (1 + vscale_mlp) + vx_scaled_base * t_vscale_mlp + t_vshift_mlp
        v_ff, t_v_ff = _feed_forward_with_t(block.ff, vx_scaled, t_vx_scaled)
        vx = vx + v_ff * vgate_mlp
        t_vx = t_vx + t_v_ff * vgate_mlp + v_ff * t_vgate_mlp

    if run_ax:
        _trace(f"block {block.idx} audio_ff")
        (ashift_mlp, ascale_mlp, agate_mlp), (t_ashift_mlp, t_ascale_mlp, t_agate_mlp) = _get_ada_values_with_t(
            block,
            block.audio_scale_shift_table,
            batch_size,
            audio.primal.timesteps,
            audio.t_timesteps,
            slice(3, None),
        )
        ax_scaled_base, t_ax_scaled_base = _rms_norm_with_t(ax, t_ax, block.norm_eps)
        ax_scaled = ax_scaled_base * (1 + ascale_mlp) + ashift_mlp
        t_ax_scaled = t_ax_scaled_base * (1 + ascale_mlp) + ax_scaled_base * t_ascale_mlp + t_ashift_mlp
        a_ff, t_a_ff = _feed_forward_with_t(block.audio_ff, ax_scaled, t_ax_scaled)
        ax = ax + a_ff * agate_mlp
        t_ax = t_ax + t_a_ff * agate_mlp + a_ff * t_agate_mlp

    _trace(f"block {block.idx} end")
    return (
        _replace_x(video, vx, t_vx) if video is not None else None,
        _replace_x(audio, ax, t_ax) if audio is not None else None,
    )


def _block_with_t(
    block: BasicAVTransformerBlock,
    video: TransformerArgsWithT | None,
    audio: TransformerArgsWithT | None,
    perturbations: BatchedPerturbationConfig,
) -> tuple[TransformerArgsWithT | None, TransformerArgsWithT | None]:
    summon_context = (
        FSDP.summon_full_params(block, recurse=True, writeback=False)
        if isinstance(block, FSDP)
        else nullcontext()
    )
    with summon_context:
        return _block_with_t_impl(
            _unwrap_fsdp(block),
            video,
            audio,
            perturbations,
        )


def _process_output_with_t(
    scale_shift_table: torch.Tensor,
    norm_out: torch.nn.LayerNorm,
    proj_out: torch.nn.Linear,
    x: torch.Tensor,
    t_x: torch.Tensor,
    embedded_timestep: torch.Tensor,
    t_embedded_timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _trace(f"output_head start x={tuple(x.shape)} t_x={tuple(t_x.shape)}")
    scale_shift_values = scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
    t_scale_shift_values = t_embedded_timestep[:, :, None].expand_as(scale_shift_values)
    shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
    t_shift, t_scale = t_scale_shift_values[:, :, 0], t_scale_shift_values[:, :, 1]

    _trace("output_head layer_norm")
    x_norm, t_x_norm = _layer_norm_with_t(norm_out, x, t_x)
    x_mod = x_norm * (1 + scale) + shift
    t_x_mod = t_x_norm * (1 + scale) + x_norm * t_scale + t_shift

    _trace("output_head proj_out")
    x_out = proj_out(x_mod)
    t_x_out = _linear_tangent(proj_out, t_x_mod)
    _trace("output_head end")
    return x_out, t_x_out


def ltx_model_with_t(
    model: LTXModel,
    video: TransformerArgsWithT | None,
    audio: TransformerArgsWithT | None,
    perturbations: BatchedPerturbationConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    model = _unwrap_fsdp(model)

    video_args = video
    audio_args = audio
    for block in model.transformer_blocks:
        video_args, audio_args = _block_with_t(block, video_args, audio_args, perturbations)

    _trace("output_heads start")
    vx = t_vx = ax = t_ax = None
    if video_args is not None:
        vx, t_vx = _process_output_with_t(
            model.scale_shift_table,
            model.norm_out,
            model.proj_out,
            video_args.primal.x,
            video_args.t_x,
            video_args.primal.embedded_timestep,
            video_args.t_embedded_timestep,
        )
    if audio_args is not None:
        ax, t_ax = _process_output_with_t(
            model.audio_scale_shift_table,
            model.audio_norm_out,
            model.audio_proj_out,
            audio_args.primal.x,
            audio_args.t_x,
            audio_args.primal.embedded_timestep,
            audio_args.t_embedded_timestep,
        )

    return vx, ax, t_vx, t_ax
