"""
LTX-2 Diffusion Model Wrapper with native TrigFlow parameterization.

This wrapper keeps the underlying LTX velocity backbone unchanged, but exposes
the model under the TrigFlow semantics used by rCM:

    x_t = cos(t) * x0 + sin(t) * eps

The wrapped backbone still predicts the RectifiedFlow velocity on the scaled
input/state, while this wrapper handles the TrigFlow <-> RectifiedFlow
preconditioning coefficients directly in the forward pass.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional, Tuple

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer import LTXModel

from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper
from ltx_distillation.models.ltx_internal_jvp import (
    ltx_model_with_t,
    prepare_transformer_args_with_t,
)


_TRACE_INTERNAL_JVP = os.environ.get("LTX_INTERNAL_JVP_TRACE", "0") == "1"


def _trace(message: str) -> None:
    if _TRACE_INTERNAL_JVP:
        print(f"[TrigWrapper] {message}", flush=True)


class LTX2TrigFlowDiffusionWrapper(LTX2DiffusionWrapper):
    """
    LTX wrapper that exposes the backbone under native TrigFlow semantics.

    The wrapped `velocity_model` continues to run in the RectifiedFlow/LTX
    parameterization. This wrapper applies the same TrigFlow -> RectifiedFlow
    preconditioning used in rCM so callers can pass TrigFlow time directly.
    """

    def __init__(
        self,
        velocity_model: LTXModel,
        video_height: int = 512,
        video_width: int = 768,
        vae_spatial_compression: int = 32,
    ):
        super().__init__(
            model=velocity_model,
            video_height=video_height,
            video_width=video_width,
            vae_spatial_compression=vae_spatial_compression,
        )

    @staticmethod
    def _trig_coefficients(trig_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cos_t = torch.cos(trig_t)
        sin_t = torch.sin(trig_t)
        denom = (cos_t + sin_t).clamp_min(1e-8)
        return cos_t, sin_t, denom

    @staticmethod
    def _reshape_time_for_latent(
        trig_t: torch.Tensor,
        latent_dim: int,
    ) -> torch.Tensor:
        if trig_t.dim() == 1:
            return trig_t.reshape(-1, *[1] * (latent_dim - 1))
        if trig_t.dim() == 2:
            return trig_t.reshape(*trig_t.shape, *[1] * (latent_dim - 2))
        raise ValueError(f"Unsupported TrigFlow timestep shape: {trig_t.shape}")

    @staticmethod
    def _rf_time_from_trig(
        trig_t: torch.Tensor,
    ) -> torch.Tensor:
        _, sin_t, denom = LTX2TrigFlowDiffusionWrapper._trig_coefficients(trig_t)
        return sin_t / denom

    @staticmethod
    def _rf_time_and_tangent_from_trig(
        trig_t: torch.Tensor,
        t_trig_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, sin_t, denom = LTX2TrigFlowDiffusionWrapper._trig_coefficients(trig_t)
        rf_time = sin_t / denom
        t_rf_time = t_trig_t / (denom * denom)
        return rf_time, t_rf_time

    def _flow_with_t(
        self,
        noisy_latent: torch.Tensor,
        pred_x0: torch.Tensor,
        trig_time: torch.Tensor,
        t_noisy_latent: torch.Tensor,
        t_pred_x0: torch.Tensor,
        t_trig_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noisy_latent.dim() not in (3, 5):
            raise ValueError(f"Unsupported latent rank for trig flow field: {noisy_latent.dim()}")

        trig_view = self._reshape_time_for_latent(trig_time, noisy_latent.dim()).to(
            dtype=noisy_latent.dtype,
            device=noisy_latent.device,
        )
        t_trig_view = self._reshape_time_for_latent(t_trig_time, noisy_latent.dim()).to(
            dtype=noisy_latent.dtype,
            device=noisy_latent.device,
        )

        sin_t = torch.sin(trig_view).clamp_min(1e-8)
        cos_t = torch.cos(trig_view)
        flow = (cos_t * noisy_latent - pred_x0) / sin_t

        dcos_t = -torch.sin(trig_view) * t_trig_view
        dsin_t = torch.cos(trig_view) * t_trig_view
        numerator = cos_t * noisy_latent - pred_x0
        t_numerator = dcos_t * noisy_latent + cos_t * t_noisy_latent - t_pred_x0
        t_flow = (t_numerator * sin_t - numerator * dsin_t) / (sin_t * sin_t)
        return flow, t_flow

    def forward_flow_with_t(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: Dict[str, Any],
        timestep: torch.Tensor,
        t_noisy_image_or_video: torch.Tensor,
        t_timestep: torch.Tensor,
        noisy_audio: Optional[torch.Tensor] = None,
        t_noisy_audio: Optional[torch.Tensor] = None,
        audio_timestep: Optional[torch.Tensor] = None,
        t_audio_timestep: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        B = noisy_image_or_video.shape[0]
        num_video_frames = noisy_image_or_video.shape[1]

        if t_timestep.shape != timestep.shape:
            if t_timestep.dim() == 2 and t_timestep.shape[1] == 1 and timestep.dim() == 2:
                t_timestep = t_timestep.expand_as(timestep)
            else:
                raise ValueError(
                    f"t_timestep shape {tuple(t_timestep.shape)} must match timestep shape {tuple(timestep.shape)}"
                )

        # Use raw x_t as RF latent (matching forward_rf convention, no scaling).
        video_rf_latent = noisy_image_or_video
        t_video_rf_latent = t_noisy_image_or_video.detach().to(dtype=noisy_image_or_video.dtype)

        video_rf_time_tokens, t_video_rf_time_tokens = self._rf_time_and_tangent_from_trig(
            timestep.to(dtype=noisy_image_or_video.dtype, device=noisy_image_or_video.device),
            t_timestep.to(dtype=noisy_image_or_video.dtype, device=noisy_image_or_video.device),
        )
        if video_rf_time_tokens.dim() == 2 and video_rf_time_tokens.shape[1] == 1:
            video_rf_time_tokens = video_rf_time_tokens[:, 0]
            t_video_rf_time_tokens = t_video_rf_time_tokens[:, 0]

        video_flat = self._flatten_video_latent(video_rf_latent)
        t_video_flat = self._flatten_video_latent(t_video_rf_latent)
        num_video_tokens = video_flat.shape[1]
        video_positions = self._compute_video_positions(noisy_image_or_video)
        video_timesteps = self._compute_timesteps_for_tokens(
            video_rf_time_tokens,
            num_video_tokens,
            self.video_frame_seqlen,
        )
        t_video_timesteps = self._compute_timesteps_for_tokens(
            t_video_rf_time_tokens,
            num_video_tokens,
            self.video_frame_seqlen,
        )

        from ltx_core.model.transformer.modality import Modality

        video_modality = Modality(
            latent=video_flat,
            timesteps=video_timesteps,
            positions=video_positions,
            context=conditional_dict["video_context"],
            context_mask=conditional_dict.get("attention_mask"),
            enabled=True,
        )

        audio_modality = None
        t_audio_timesteps = None
        audio_rf_latent = None
        t_audio_rf_latent = None
        audio_rf_time = None
        t_audio_rf_time = None

        if noisy_audio is not None:
            if audio_timestep is None:
                audio_timestep = timestep if timestep.dim() == 1 else timestep[:, 0]
            if t_audio_timestep is None:
                t_audio_timestep = t_timestep if audio_timestep.dim() == t_timestep.dim() else t_timestep[:, 0]
            if t_audio_timestep.shape != audio_timestep.shape:
                if t_audio_timestep.dim() == 2 and t_audio_timestep.shape[1] == 1 and audio_timestep.dim() == 2:
                    t_audio_timestep = t_audio_timestep.expand_as(audio_timestep)
                else:
                    raise ValueError(
                        "t_audio_timestep shape "
                        f"{tuple(t_audio_timestep.shape)} must match audio_timestep shape {tuple(audio_timestep.shape)}"
                    )
            if t_noisy_audio is None:
                raise ValueError("t_noisy_audio must be provided when noisy_audio is provided")

            # Use raw audio as RF latent (no scaling, matching forward_rf convention).
            audio_rf_latent = noisy_audio
            t_audio_rf_latent = t_noisy_audio.detach().to(dtype=noisy_audio.dtype)

            # Audio uses single scalar timestep, not per-token.
            audio_t_for_rf = audio_timestep[:, 0] if audio_timestep.dim() == 2 else audio_timestep
            t_audio_t_for_rf = t_audio_timestep[:, 0] if t_audio_timestep.dim() == 2 else t_audio_timestep
            audio_rf_time_tokens, t_audio_rf_time_tokens = self._rf_time_and_tangent_from_trig(
                audio_t_for_rf.to(dtype=noisy_audio.dtype, device=noisy_audio.device),
                t_audio_t_for_rf.to(dtype=noisy_audio.dtype, device=noisy_audio.device),
            )
            if audio_rf_time_tokens.dim() == 2 and audio_rf_time_tokens.shape[1] == 1:
                audio_rf_time_tokens = audio_rf_time_tokens[:, 0]
                t_audio_rf_time_tokens = t_audio_rf_time_tokens[:, 0]

            num_audio_tokens = audio_rf_latent.shape[1]
            audio_timesteps = self._compute_timesteps_for_tokens(
                audio_rf_time_tokens,
                num_audio_tokens,
                1,
            )
            t_audio_timesteps = self._compute_timesteps_for_tokens(
                t_audio_rf_time_tokens,
                num_audio_tokens,
                1,
            )
            audio_positions = self._compute_audio_positions(audio_rf_latent)

            audio_modality = Modality(
                latent=audio_rf_latent,
                timesteps=audio_timesteps,
                positions=audio_positions,
                context=conditional_dict["audio_context"],
                context_mask=conditional_dict.get("attention_mask"),
                enabled=True,
            )

            audio_rf_time, t_audio_rf_time = self._rf_time_and_tangent_from_trig(
                audio_timestep.to(dtype=noisy_audio.dtype, device=noisy_audio.device),
                t_audio_timestep.to(dtype=noisy_audio.dtype, device=noisy_audio.device),
            )
            audio_rf_time = audio_rf_time.to(dtype=noisy_audio.dtype)
            t_audio_rf_time = t_audio_rf_time.detach().to(dtype=noisy_audio.dtype)

        model = getattr(self.model, "velocity_model", self.model)
        video_args = prepare_transformer_args_with_t(
            model.video_args_preprocessor,
            video_modality,
            t_video_flat,
            t_video_timesteps,
        )
        audio_args = (
            prepare_transformer_args_with_t(
                model.audio_args_preprocessor,
                audio_modality,
                t_audio_rf_latent,
                t_audio_timesteps,
            )
            if audio_modality is not None
            else None
        )
        perturbations = BatchedPerturbationConfig.empty(batch_size=B)
        _trace("ltx_model_with_t start")
        video_v, audio_v, t_video_v, t_audio_v = ltx_model_with_t(
            model=model,
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )
        _trace("ltx_model_with_t done")

        video_v = self._unflatten_video_latent(video_v, num_video_frames)
        t_video_v = self._unflatten_video_latent(t_video_v, num_video_frames)
        video_rf_time, t_video_rf_time = self._rf_time_and_tangent_from_trig(
            timestep.to(dtype=noisy_image_or_video.dtype, device=noisy_image_or_video.device),
            t_timestep.to(dtype=noisy_image_or_video.dtype, device=noisy_image_or_video.device),
        )
        video_rf_time = self._reshape_time_for_latent(video_rf_time, noisy_image_or_video.dim()).to(dtype=noisy_image_or_video.dtype)
        t_video_rf_time = self._reshape_time_for_latent(t_video_rf_time, noisy_image_or_video.dim()).detach().to(dtype=noisy_image_or_video.dtype)

        video_x0 = (video_rf_latent - video_rf_time * video_v).to(dtype=noisy_image_or_video.dtype)
        t_video_x0 = (t_video_rf_latent - t_video_rf_time * video_v - video_rf_time * t_video_v).to(
            dtype=noisy_image_or_video.dtype
        )
        _trace("video_x0 done")
        video_flow, t_video_flow = self._flow_with_t(
            noisy_image_or_video,
            video_x0,
            timestep,
            t_noisy_image_or_video,
            t_video_x0,
            t_timestep,
        )
        _trace("video_flow done")

        audio_flow = None
        t_audio_flow = None
        if audio_modality is not None and audio_v is not None:
            audio_x0 = (audio_rf_latent - audio_rf_time * audio_v).to(dtype=noisy_audio.dtype)
            t_audio_x0 = (t_audio_rf_latent - t_audio_rf_time * audio_v - audio_rf_time * t_audio_v).to(
                dtype=noisy_audio.dtype
            )
            _trace("audio_x0 done")
            audio_flow, t_audio_flow = self._flow_with_t(
                noisy_audio,
                audio_x0,
                audio_timestep,
                t_noisy_audio,
                t_audio_x0,
                t_audio_timestep,
            )
            _trace("audio_flow done")

        _trace("forward_flow_with_t return")
        return video_flow, audio_flow, t_video_flow.detach(), t_audio_flow.detach() if t_audio_flow is not None else None

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: Dict[str, Any],
        timestep: torch.Tensor,
        noisy_audio: Optional[torch.Tensor] = None,
        audio_timestep: Optional[torch.Tensor] = None,
        use_causal_timestep: bool = False,  # ignored, for API compatibility
        t_noisy_image_or_video: Optional[torch.Tensor] = None,
        t_timestep: Optional[torch.Tensor] = None,
        t_noisy_audio: Optional[torch.Tensor] = None,
        t_audio_timestep: Optional[torch.Tensor] = None,
        with_t: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if with_t:
            if t_noisy_image_or_video is None or t_timestep is None:
                raise ValueError("with_t=True requires t_noisy_image_or_video and t_timestep")
            return self.forward_flow_with_t(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=timestep,
                t_noisy_image_or_video=t_noisy_image_or_video,
                t_timestep=t_timestep,
                noisy_audio=noisy_audio,
                t_noisy_audio=t_noisy_audio,
                audio_timestep=audio_timestep,
                t_audio_timestep=t_audio_timestep,
            )

        # Convert TrigFlow time to native RF sigma and delegate to forward_rf.
        # rf_time = sin/(cos+sin) = sigma. No input scaling needed.
        video_sigma = (torch.sin(timestep.float()) / (torch.cos(timestep.float()) + torch.sin(timestep.float())).clamp_min(1e-8)).to(dtype=noisy_image_or_video.dtype)

        audio_sigma = None
        if noisy_audio is not None:
            if audio_timestep is not None:
                audio_sigma = (torch.sin(audio_timestep.float()) / (torch.cos(audio_timestep.float()) + torch.sin(audio_timestep.float())).clamp_min(1e-8)).to(dtype=noisy_audio.dtype)
            else:
                audio_sigma = video_sigma if video_sigma.dim() == 1 else video_sigma[:, 0]

        return self.forward_rf(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

    def forward_rf(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: Dict[str, Any],
        timestep: torch.Tensor,
        noisy_audio: Optional[torch.Tensor] = None,
        audio_timestep: Optional[torch.Tensor] = None,
        use_causal_timestep: bool = False,  # ignored, for API compatibility
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run the wrapper under native RectifiedFlow/LTX semantics.

        This is mainly used by teacher benchmark generation, which still relies
        on the original LTX sigma/RF scheduler and should not reinterpret its
        timesteps as TrigFlow time.
        """
        B = noisy_image_or_video.shape[0]
        num_video_frames = noisy_image_or_video.shape[1]

        video_flat = self._flatten_video_latent(noisy_image_or_video)
        num_video_tokens = video_flat.shape[1]
        video_positions = self._compute_video_positions(noisy_image_or_video)
        video_timesteps = self._compute_timesteps_for_tokens(
            timestep,
            num_video_tokens,
            self.video_frame_seqlen,
        )

        from ltx_core.model.transformer.modality import Modality  # local import to avoid extra top-level churn

        video_modality = Modality(
            latent=video_flat,
            timesteps=video_timesteps,
            positions=video_positions,
            context=conditional_dict["video_context"],
            context_mask=conditional_dict.get("attention_mask"),
            enabled=True,
        )

        audio_modality = None
        if noisy_audio is not None:
            num_audio_tokens = noisy_audio.shape[1]

            if audio_timestep is None:
                if timestep.dim() == 1:
                    audio_timestep = timestep
                else:
                    audio_timestep = timestep[:, 0]

            audio_timesteps = self._compute_timesteps_for_tokens(
                audio_timestep,
                num_audio_tokens,
                1,
            )
            audio_positions = self._compute_audio_positions(noisy_audio)

            audio_modality = Modality(
                latent=noisy_audio,
                timesteps=audio_timesteps,
                positions=audio_positions,
                context=conditional_dict["audio_context"],
                context_mask=conditional_dict.get("attention_mask"),
                enabled=True,
            )

        perturbations = BatchedPerturbationConfig.empty(batch_size=B)
        video_v, audio_v = self.model(
            video=video_modality,
            audio=audio_modality,
            perturbations=perturbations,
        )

        video_x0 = None
        if video_v is not None:
            video_v = self._unflatten_video_latent(video_v, num_video_frames)
            video_rf_time = self._reshape_time_for_latent(timestep, noisy_image_or_video.dim()).to(
                dtype=noisy_image_or_video.dtype,
                device=noisy_image_or_video.device,
            )
            video_x0 = (noisy_image_or_video - video_rf_time * video_v).to(dtype=noisy_image_or_video.dtype)

        audio_x0 = None
        if audio_v is not None:
            if audio_timestep is None:
                raise ValueError("audio_timestep should be populated before audio_x0 reconstruction.")
            audio_rf_time = self._reshape_time_for_latent(audio_timestep, noisy_audio.dim()).to(
                dtype=noisy_audio.dtype,
                device=noisy_audio.device,
            )
            audio_x0 = (noisy_audio - audio_rf_time * audio_v).to(dtype=noisy_audio.dtype)

        return video_x0, audio_x0


def create_ltx2_trig_wrapper(
    checkpoint_path: str,
    gemma_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    video_height: int = 512,
    video_width: int = 768,
    registry: Registry | None = None,
) -> LTX2TrigFlowDiffusionWrapper:
    from ltx_pipelines.utils.model_ledger import ModelLedger

    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_path,
        registry=registry,
    )

    x0_model = ledger.transformer()
    velocity_model = x0_model.velocity_model.to(device=device, dtype=dtype)

    wrapper = LTX2TrigFlowDiffusionWrapper(
        velocity_model=velocity_model,
        video_height=video_height,
        video_width=video_width,
    )

    return wrapper
