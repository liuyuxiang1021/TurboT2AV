from __future__ import annotations

import math

import torch
import torch.nn as nn

from ltx_core.model.transformer.attention import AttentionFunction
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_distillation.models.ltx_internal_jvp import prepare_transformer_args_with_t
from ltx_distillation.models.ltx_trig_wrapper import LTX2TrigFlowDiffusionWrapper


def _build_tiny_wrapper(device: torch.device) -> LTX2TrigFlowDiffusionWrapper:
    model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=128,
        out_channels=128,
        num_layers=1,
        cross_attention_dim=8,
        caption_channels=16,
        positional_embedding_max_pos=[4, 4, 4],
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_in_channels=128,
        audio_out_channels=128,
        audio_cross_attention_dim=8,
        audio_positional_embedding_max_pos=[4],
        attention_type=AttentionFunction.PYTORCH,
    ).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        for name, param in model.named_parameters():
            if "scale_shift_table" in name:
                nn.init.zeros_(param)
            elif not torch.isfinite(param).all():
                nn.init.normal_(param, mean=0.0, std=0.02)
    wrapper = LTX2TrigFlowDiffusionWrapper(
        velocity_model=model,
        video_height=32,
        video_width=32,
        vae_spatial_compression=32,
    ).to(device=device, dtype=torch.float32)
    wrapper.eval()
    return wrapper


def _flow_field(noisy_latent: torch.Tensor, pred_x0: torch.Tensor, trig_time: torch.Tensor) -> torch.Tensor:
    if noisy_latent.dim() == 5:
        trig_view = trig_time.view(-1, 1, 1, 1, 1).to(dtype=noisy_latent.dtype, device=noisy_latent.device)
    elif noisy_latent.dim() == 3:
        trig_view = trig_time.view(-1, 1, 1).to(dtype=noisy_latent.dtype, device=noisy_latent.device)
    else:
        raise ValueError(f"Unsupported latent rank: {noisy_latent.dim()}")
    return (torch.cos(trig_view) * noisy_latent - pred_x0) / torch.sin(trig_view)


def test_internal_jvp_matches_central_difference() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = _build_tiny_wrapper(device)

    batch = 1
    video_frames = 2
    audio_frames = 3
    context_tokens = 4

    noisy_video = torch.randn(batch, video_frames, 128, 1, 1, device=device, dtype=torch.float32)
    noisy_audio = torch.randn(batch, audio_frames, 128, device=device, dtype=torch.float32)
    trig_time = torch.full((batch, 1), 0.7, device=device, dtype=torch.float32)

    t_noisy_video = torch.randn_like(noisy_video)
    t_noisy_audio = torch.randn_like(noisy_audio)
    t_trig_time = torch.full_like(trig_time, 0.2)

    conditional_dict = {
        "video_context": torch.randn(batch, context_tokens, 16, device=device, dtype=torch.float32),
        "audio_context": torch.randn(batch, context_tokens, 16, device=device, dtype=torch.float32),
        "attention_mask": None,
    }

    def flow_fn(video_xt: torch.Tensor, audio_xt: torch.Tensor, trig_t: torch.Tensor):
        video_x0, audio_x0 = wrapper(
            noisy_image_or_video=video_xt,
            conditional_dict=conditional_dict,
            timestep=trig_t,
            noisy_audio=audio_xt,
        )
        return (
            _flow_field(video_xt, video_x0, trig_t),
            _flow_field(audio_xt, audio_x0, trig_t),
        )

    internal_video, internal_audio, internal_t_video, internal_t_audio = wrapper(
        noisy_image_or_video=noisy_video,
        conditional_dict=conditional_dict,
        timestep=trig_time,
        noisy_audio=noisy_audio,
        t_noisy_image_or_video=t_noisy_video,
        t_noisy_audio=t_noisy_audio,
        t_timestep=t_trig_time,
        with_t=True,
    )

    reference_video, reference_audio = flow_fn(noisy_video, noisy_audio, trig_time)
    torch.testing.assert_close(internal_video, reference_video, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(internal_audio, reference_audio, atol=1e-4, rtol=1e-4)

    step = 1e-3
    plus_video, plus_audio = flow_fn(
        noisy_video + step * t_noisy_video,
        noisy_audio + step * t_noisy_audio,
        trig_time + step * t_trig_time,
    )
    minus_video, minus_audio = flow_fn(
        noisy_video - step * t_noisy_video,
        noisy_audio - step * t_noisy_audio,
        trig_time - step * t_trig_time,
    )
    finite_diff_video = (plus_video - minus_video) / (2 * step)
    finite_diff_audio = (plus_audio - minus_audio) / (2 * step)

    torch.testing.assert_close(internal_t_video, finite_diff_video, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(internal_t_audio, finite_diff_audio, atol=2e-2, rtol=2e-2)


def test_internal_jvp_time_tangents_are_fp32_for_bf16_model() -> None:
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    device = torch.device("cuda")
    wrapper = _build_tiny_wrapper(device).to(device=device, dtype=torch.bfloat16)
    model = getattr(wrapper.model, "velocity_model", wrapper.model)

    batch = 1
    video_frames = 2
    audio_frames = 3
    context_tokens = 4

    noisy_video = torch.randn(batch, video_frames, 128, 1, 1, device=device, dtype=torch.bfloat16)
    noisy_audio = torch.randn(batch, audio_frames, 128, device=device, dtype=torch.bfloat16)
    t_noisy_video = torch.randn_like(noisy_video)
    t_noisy_audio = torch.randn_like(noisy_audio)
    rf_time = torch.full((batch, 1), 0.42, device=device, dtype=torch.bfloat16)
    t_rf_time = torch.full_like(rf_time, 0.125)

    video_flat = wrapper._flatten_video_latent(noisy_video)
    t_video_flat = wrapper._flatten_video_latent(t_noisy_video)
    video_timesteps = wrapper._compute_timesteps_for_tokens(
        rf_time[:, 0],
        video_flat.shape[1],
        wrapper.video_frame_seqlen,
    )
    t_video_timesteps = wrapper._compute_timesteps_for_tokens(
        t_rf_time[:, 0],
        video_flat.shape[1],
        wrapper.video_frame_seqlen,
    )
    video_modality = Modality(
        latent=video_flat,
        timesteps=video_timesteps,
        positions=wrapper._compute_video_positions(noisy_video),
        context=torch.randn(batch, context_tokens, 16, device=device, dtype=torch.bfloat16),
        context_mask=None,
        enabled=True,
    )

    audio_timesteps = wrapper._compute_timesteps_for_tokens(rf_time[:, 0], audio_frames, 1)
    t_audio_timesteps = wrapper._compute_timesteps_for_tokens(t_rf_time[:, 0], audio_frames, 1)
    audio_modality = Modality(
        latent=noisy_audio,
        timesteps=audio_timesteps,
        positions=wrapper._compute_audio_positions(noisy_audio),
        context=torch.randn(batch, context_tokens, 16, device=device, dtype=torch.bfloat16),
        context_mask=None,
        enabled=True,
    )

    video_args = prepare_transformer_args_with_t(
        model.video_args_preprocessor,
        video_modality,
        t_video_flat,
        t_video_timesteps,
    )
    audio_args = prepare_transformer_args_with_t(
        model.audio_args_preprocessor,
        audio_modality,
        t_noisy_audio,
        t_audio_timesteps,
    )

    assert video_args.primal.timesteps.dtype == torch.bfloat16
    assert video_args.primal.embedded_timestep.dtype == torch.bfloat16
    assert video_args.t_timesteps.dtype == torch.float32
    assert video_args.t_embedded_timestep.dtype == torch.float32
    assert video_args.t_cross_scale_shift_timestep.dtype == torch.float32
    assert video_args.t_cross_gate_timestep.dtype == torch.float32

    assert audio_args.primal.timesteps.dtype == torch.bfloat16
    assert audio_args.primal.embedded_timestep.dtype == torch.bfloat16
    assert audio_args.t_timesteps.dtype == torch.float32
    assert audio_args.t_embedded_timestep.dtype == torch.float32
    assert audio_args.t_cross_scale_shift_timestep.dtype == torch.float32
    assert audio_args.t_cross_gate_timestep.dtype == torch.float32


if __name__ == "__main__":
    test_internal_jvp_matches_central_difference()
    test_internal_jvp_time_tangents_are_fp32_for_bf16_model()
    print("PASS: test_internal_jvp_matches_central_difference")
