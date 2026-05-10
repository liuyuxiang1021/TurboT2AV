"""
Unit tests for the exact SCM JVP path.

These tests mirror the intent of ``rcm/networks/wan2pt1_jvp_test.py`` but
adapt it to the current LTX SCM implementation:

1. Verify the TrigFlow field helper returns the same value as a manual
   implementation on a tiny toy generator.
2. Verify the exact JVP path matches a finite-difference reference on the same
   toy generator for both video and audio branches.

The tests deliberately avoid loading the real LTX model. They validate the
mathematical wiring of the SCM helper methods in isolation so failures are
easy to localize.
"""

from types import SimpleNamespace

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.modality import Modality
from ltx_core.utils import rms_norm
from torch.nn.attention import SDPBackend, sdpa_kernel
from ltx_distillation.dmd import LTX2DMD
from ltx_distillation.models.ltx_internal_jvp import (
    _attention_with_t,
    _block_with_t_impl,
    _feed_forward_with_t,
    _layer_norm_with_t,
    _process_output_with_t,
    _rms_norm_with_t,
    ltx_model_with_t,
    prepare_transformer_args_with_t,
)
from ltx_distillation.models.ltx_trig_wrapper import LTX2TrigFlowDiffusionWrapper


class _ToyTrigGenerator(torch.nn.Module):
    """
    Small differentiable generator with the same call signature as the real
    LTX wrapper.

    The mapping is intentionally nonlinear in both the latent and time inputs
    so that the JVP is non-trivial and a finite-difference comparison is
    meaningful.
    """

    def forward(
        self,
        noisy_image_or_video,
        conditional_dict,
        timestep,
        noisy_audio=None,
        audio_timestep=None,
        use_causal_timestep=False,
    ):
        del use_causal_timestep

        video_scale = conditional_dict["video_scale"].view(-1, 1, 1, 1, 1)
        video_bias = conditional_dict["video_bias"].view(-1, 1, 1, 1, 1)
        video_time = timestep[:, :, None, None, None]

        video_x0 = (
            0.35 * noisy_image_or_video
            + 0.05 * noisy_image_or_video.square()
            + video_scale * torch.sin(video_time)
            + video_bias * torch.cos(video_time)
        )

        audio_x0 = None
        if noisy_audio is not None:
            audio_scale = conditional_dict["audio_scale"].view(-1, 1, 1)
            audio_bias = conditional_dict["audio_bias"].view(-1, 1, 1)
            audio_time = audio_timestep[:, :, None]
            audio_x0 = (
                0.25 * noisy_audio
                + 0.04 * noisy_audio.square()
                + audio_scale * torch.sin(audio_time)
                + audio_bias * torch.cos(audio_time)
            )

        return video_x0, audio_x0


def _build_stub_dmd(dtype=torch.float64):
    stub = object.__new__(LTX2DMD)
    torch.nn.Module.__init__(stub)
    stub.args = SimpleNamespace()
    stub.device = torch.device("cpu")
    stub.dtype = dtype
    stub.scm_time_eps = 1e-6
    stub.generator = _ToyTrigGenerator().to(dtype=dtype)
    stub._generator_fsdp_jvp_primed = False
    return stub


def _build_tiny_real_wrapper(
    dtype=torch.float32,
    device=None,
    *,
    num_layers=2,
    num_attention_heads=2,
    attention_head_dim=8,
    cross_attention_dim=16,
    caption_channels=32,
    positional_embedding_max_pos=(4, 4, 4),
    audio_positional_embedding_max_pos=(8,),
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    velocity_model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        in_channels=128,
        out_channels=128,
        num_layers=num_layers,
        cross_attention_dim=cross_attention_dim,
        caption_channels=caption_channels,
        positional_embedding_max_pos=list(positional_embedding_max_pos),
        audio_num_attention_heads=num_attention_heads,
        audio_attention_head_dim=attention_head_dim,
        audio_in_channels=128,
        audio_out_channels=128,
        audio_cross_attention_dim=cross_attention_dim,
        audio_positional_embedding_max_pos=list(audio_positional_embedding_max_pos),
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        for name, param in velocity_model.named_parameters():
            if param.ndim >= 2:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            elif "bias" in name:
                param.zero_()
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
    velocity_model.eval()

    wrapper = LTX2TrigFlowDiffusionWrapper(
        velocity_model=velocity_model,
        video_height=32,
        video_width=32,
    ).to(device=device, dtype=dtype)
    wrapper.eval()
    return wrapper


def _metric_dict(lhs: torch.Tensor, rhs: torch.Tensor):
    diff = (lhs - rhs).abs()
    rel = diff / rhs.abs().clamp_min(1e-8)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float(rel.max()),
        "mean_rel": float(rel.mean()),
    }


def _build_tiny_real_case(
    dtype=torch.float32,
    *,
    seed=0,
    wrapper_kwargs=None,
    batch=2,
    video_frames=2,
    audio_frames=3,
    text_len=4,
    text_dim=32,
    trig_time_values=None,
    audio_trig_time_values=None,
    t_trig_time_values=None,
    t_audio_trig_time_values=None,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = _build_tiny_real_wrapper(
        dtype=dtype,
        device=device,
        **(wrapper_kwargs or {}),
    )

    noisy_video = torch.randn(batch, video_frames, 128, 1, 1, device=device, dtype=dtype)
    noisy_audio = torch.randn(batch, audio_frames, 128, device=device, dtype=dtype)
    if trig_time_values is None:
        trig_time_values = [[0.55], [0.95]]
    if audio_trig_time_values is None:
        audio_trig_time_values = [[0.60], [0.90]]
    trig_time = torch.tensor(trig_time_values, device=device, dtype=dtype)
    audio_trig_time = torch.tensor(audio_trig_time_values, device=device, dtype=dtype)

    t_noisy_video = torch.randn_like(noisy_video)
    t_noisy_audio = torch.randn_like(noisy_audio)
    if t_trig_time_values is None:
        t_trig_time_values = [[0.03], [-0.02]]
    if t_audio_trig_time_values is None:
        t_audio_trig_time_values = [[0.02], [0.01]]
    t_trig_time = torch.tensor(t_trig_time_values, device=device, dtype=dtype)
    t_audio_trig_time = torch.tensor(t_audio_trig_time_values, device=device, dtype=dtype)

    conditional_dict = {
        "video_context": torch.randn(batch, text_len, text_dim, device=device, dtype=dtype),
        "audio_context": torch.randn(batch, text_len, text_dim, device=device, dtype=dtype),
        "attention_mask": torch.ones(batch, text_len, device=device, dtype=torch.long),
    }

    return {
        "wrapper": wrapper,
        "device": device,
        "dtype": dtype,
        "noisy_video": noisy_video,
        "noisy_audio": noisy_audio,
        "trig_time": trig_time,
        "audio_trig_time": audio_trig_time,
        "t_noisy_video": t_noisy_video,
        "t_noisy_audio": t_noisy_audio,
        "t_trig_time": t_trig_time,
        "t_audio_trig_time": t_audio_trig_time,
        "conditional_dict": conditional_dict,
    }


def _prepare_modalities_for_case(case):
    wrapper = case["wrapper"]
    noisy_video = case["noisy_video"]
    noisy_audio = case["noisy_audio"]
    trig_time = case["trig_time"]
    audio_trig_time = case["audio_trig_time"]
    t_noisy_video = case["t_noisy_video"]
    t_noisy_audio = case["t_noisy_audio"]
    t_trig_time = case["t_trig_time"]
    t_audio_trig_time = case["t_audio_trig_time"]
    conditional_dict = case["conditional_dict"]

    num_video_frames = noisy_video.shape[1]
    video_trig_bcast = wrapper._reshape_time_for_latent(trig_time, noisy_video.dim()).to(
        dtype=noisy_video.dtype,
        device=noisy_video.device,
    )
    _, _, video_denom = wrapper._trig_coefficients(video_trig_bcast)
    video_rf_latent = (noisy_video / video_denom).to(dtype=noisy_video.dtype)
    video_denom_tangent = (torch.cos(video_trig_bcast) - torch.sin(video_trig_bcast)) * wrapper._reshape_time_for_latent(
        t_trig_time, noisy_video.dim()
    ).to(dtype=noisy_video.dtype, device=noisy_video.device)
    t_video_rf_latent = (
        t_noisy_video / video_denom - noisy_video * video_denom_tangent / (video_denom * video_denom)
    ).to(dtype=noisy_video.dtype)
    video_rf_time_tokens, t_video_rf_time_tokens = wrapper._rf_time_and_tangent_from_trig(trig_time, t_trig_time)
    if video_rf_time_tokens.dim() == 2 and video_rf_time_tokens.shape[1] == 1:
        video_rf_time_tokens = video_rf_time_tokens[:, 0]
        t_video_rf_time_tokens = t_video_rf_time_tokens[:, 0]
    video_flat = wrapper._flatten_video_latent(video_rf_latent)
    t_video_flat = wrapper._flatten_video_latent(t_video_rf_latent)
    num_video_tokens = video_flat.shape[1]
    video_positions = wrapper._compute_video_positions(noisy_video)
    video_timesteps = wrapper._compute_timesteps_for_tokens(video_rf_time_tokens, num_video_tokens, wrapper.video_frame_seqlen)
    t_video_timesteps = wrapper._compute_timesteps_for_tokens(
        t_video_rf_time_tokens, num_video_tokens, wrapper.video_frame_seqlen
    )
    video_modality = Modality(
        latent=video_flat,
        timesteps=video_timesteps,
        positions=video_positions,
        context=conditional_dict["video_context"],
        context_mask=conditional_dict.get("attention_mask"),
        enabled=True,
    )

    audio_trig_bcast = wrapper._reshape_time_for_latent(audio_trig_time, noisy_audio.dim()).to(
        dtype=noisy_audio.dtype,
        device=noisy_audio.device,
    )
    _, _, audio_denom = wrapper._trig_coefficients(audio_trig_bcast)
    audio_rf_latent = (noisy_audio / audio_denom).to(dtype=noisy_audio.dtype)
    audio_denom_tangent = (torch.cos(audio_trig_bcast) - torch.sin(audio_trig_bcast)) * wrapper._reshape_time_for_latent(
        t_audio_trig_time, noisy_audio.dim()
    ).to(dtype=noisy_audio.dtype, device=noisy_audio.device)
    t_audio_rf_latent = (
        t_noisy_audio / audio_denom - noisy_audio * audio_denom_tangent / (audio_denom * audio_denom)
    ).to(dtype=noisy_audio.dtype)
    audio_rf_time_tokens, t_audio_rf_time_tokens = wrapper._rf_time_and_tangent_from_trig(
        audio_trig_time, t_audio_trig_time
    )
    if audio_rf_time_tokens.dim() == 2 and audio_rf_time_tokens.shape[1] == 1:
        audio_rf_time_tokens = audio_rf_time_tokens[:, 0]
        t_audio_rf_time_tokens = t_audio_rf_time_tokens[:, 0]
    num_audio_tokens = audio_rf_latent.shape[1]
    audio_timesteps = wrapper._compute_timesteps_for_tokens(audio_rf_time_tokens, num_audio_tokens, 1)
    t_audio_timesteps = wrapper._compute_timesteps_for_tokens(t_audio_rf_time_tokens, num_audio_tokens, 1)
    audio_positions = wrapper._compute_audio_positions(audio_rf_latent)
    audio_modality = Modality(
        latent=audio_rf_latent,
        timesteps=audio_timesteps,
        positions=audio_positions,
        context=conditional_dict["audio_context"],
        context_mask=conditional_dict.get("attention_mask"),
        enabled=True,
    )

    return {
        "video_modality": video_modality,
        "audio_modality": audio_modality,
        "t_video_flat": t_video_flat,
        "t_audio_rf_latent": t_audio_rf_latent,
        "t_video_timesteps": t_video_timesteps,
        "t_audio_timesteps": t_audio_timesteps,
        "num_video_frames": num_video_frames,
        "video_rf_latent": video_rf_latent,
        "audio_rf_latent": audio_rf_latent,
        "video_rf_time_tokens": video_rf_time_tokens,
        "audio_rf_time_tokens": audio_rf_time_tokens,
    }


def _manual_flow_field(noisy_latent, pred_x0, trig_time):
    if noisy_latent.dim() == 5:
        trig_view = trig_time.view(-1, 1, 1, 1, 1)
    elif noisy_latent.dim() == 3:
        trig_view = trig_time.view(-1, 1, 1)
    else:
        raise ValueError(f"unsupported latent rank: {noisy_latent.dim()}")

    return (torch.cos(trig_view) * noisy_latent - pred_x0) / torch.sin(trig_view)


def _finite_difference_jvp(flow_fn, primals, tangents, h=1e-5):
    plus_inputs = tuple(x + h * t for x, t in zip(primals, tangents))
    minus_inputs = tuple(x - h * t for x, t in zip(primals, tangents))
    plus = flow_fn(*plus_inputs)
    minus = flow_fn(*minus_inputs)
    if isinstance(plus, tuple):
        return tuple((p - m) / (2 * h) for p, m in zip(plus, minus))
    return (plus - minus) / (2 * h)


def test_student_trig_flow_field_matches_manual_formula():
    dmd = _build_stub_dmd()

    batch = 2
    noisy_video = torch.randn(batch, 3, 2, 2, 2, dtype=dmd.dtype)
    noisy_audio = torch.randn(batch, 5, 2, dtype=dmd.dtype)
    trig_time = torch.tensor([[0.7], [1.1]], dtype=dmd.dtype)
    conditional_dict = {
        "video_scale": torch.tensor([0.3, -0.2], dtype=dmd.dtype),
        "video_bias": torch.tensor([0.1, 0.05], dtype=dmd.dtype),
        "audio_scale": torch.tensor([-0.4, 0.25], dtype=dmd.dtype),
        "audio_bias": torch.tensor([0.15, -0.1], dtype=dmd.dtype),
    }

    video_flow, audio_flow = dmd._student_trig_flow_field(
        noisy_video=noisy_video,
        noisy_audio=noisy_audio,
        conditional_dict=conditional_dict,
        trig_time=trig_time,
    )

    video_timestep = trig_time.expand(batch, noisy_video.shape[1])
    audio_timestep = trig_time.expand(batch, noisy_audio.shape[1])
    pred_video_x0, pred_audio_x0 = dmd.generator(
        noisy_image_or_video=noisy_video,
        conditional_dict=conditional_dict,
        timestep=video_timestep,
        noisy_audio=noisy_audio,
        audio_timestep=audio_timestep,
    )

    expected_video = _manual_flow_field(noisy_video, pred_video_x0, trig_time)
    expected_audio = _manual_flow_field(noisy_audio, pred_audio_x0, trig_time)

    torch.testing.assert_close(video_flow, expected_video, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(audio_flow, expected_audio, rtol=1e-7, atol=1e-7)


def test_student_trig_flow_jvp_matches_central_difference():
    dmd = _build_stub_dmd()

    batch = 2
    noisy_video = torch.randn(batch, 3, 2, 2, 2, dtype=dmd.dtype)
    noisy_audio = torch.randn(batch, 5, 2, dtype=dmd.dtype)
    trig_time = torch.tensor([[0.65], [1.05]], dtype=dmd.dtype)
    conditional_dict = {
        "video_scale": torch.tensor([0.2, -0.15], dtype=dmd.dtype),
        "video_bias": torch.tensor([0.05, 0.12], dtype=dmd.dtype),
        "audio_scale": torch.tensor([0.1, -0.3], dtype=dmd.dtype),
        "audio_bias": torch.tensor([0.08, -0.06], dtype=dmd.dtype),
    }

    t_noisy_video = torch.randn_like(noisy_video)
    t_noisy_audio = torch.randn_like(noisy_audio)
    t_trig_time = torch.tensor([[0.07], [-0.04]], dtype=dmd.dtype)

    flow_video, flow_audio, t_flow_video, t_flow_audio = dmd._student_trig_flow_jvp(
        noisy_video=noisy_video,
        noisy_audio=noisy_audio,
        conditional_dict=conditional_dict,
        trig_time=trig_time,
        t_noisy_video=t_noisy_video,
        t_noisy_audio=t_noisy_audio,
        t_trig_time=t_trig_time,
    )

    def flow_fn(video_xt, audio_xt, trig_t):
        return dmd._student_trig_flow_field(
            noisy_video=video_xt,
            noisy_audio=audio_xt,
            conditional_dict=conditional_dict,
            trig_time=trig_t,
        )

    ref_t_flow_video, ref_t_flow_audio = _finite_difference_jvp(
        flow_fn,
        (noisy_video, noisy_audio, trig_time),
        (t_noisy_video, t_noisy_audio, t_trig_time),
        h=1e-5,
    )

    ref_flow_video, ref_flow_audio = flow_fn(noisy_video, noisy_audio, trig_time)

    torch.testing.assert_close(flow_video, ref_flow_video, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(flow_audio, ref_flow_audio, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(t_flow_video, ref_t_flow_video, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(t_flow_audio, ref_t_flow_audio, rtol=1e-4, atol=1e-5)


def test_student_trig_flow_fd1_matches_semi_continuous_formula():
    dmd = _build_stub_dmd()

    batch = 2
    noisy_video = torch.randn(batch, 3, 2, 2, 2, dtype=dmd.dtype)
    noisy_audio = torch.randn(batch, 5, 2, dtype=dmd.dtype)
    trig_time = torch.tensor([[0.65], [1.05]], dtype=dmd.dtype)
    conditional_dict = {
        "video_scale": torch.tensor([0.2, -0.15], dtype=dmd.dtype),
        "video_bias": torch.tensor([0.05, 0.12], dtype=dmd.dtype),
        "audio_scale": torch.tensor([0.1, -0.3], dtype=dmd.dtype),
        "audio_bias": torch.tensor([0.08, -0.06], dtype=dmd.dtype),
    }

    t_noisy_video = torch.randn_like(noisy_video)
    t_noisy_audio = torch.randn_like(noisy_audio)
    t_trig_time = torch.tensor([[0.07], [-0.04]], dtype=dmd.dtype)
    zero_t_trig_time = torch.zeros_like(t_trig_time)
    h = 1.0e-4
    cos_h = torch.tensor(torch.cos(torch.tensor(h, dtype=dmd.dtype)).item(), dtype=dmd.dtype)
    sin_h = torch.tensor(torch.sin(torch.tensor(h, dtype=dmd.dtype)).item(), dtype=dmd.dtype)
    cs_video = torch.cos(trig_time).view(-1, 1, 1, 1, 1) * torch.sin(trig_time).view(-1, 1, 1, 1, 1)
    cs_audio = torch.cos(trig_time).view(-1, 1, 1) * torch.sin(trig_time).view(-1, 1, 1)

    flow_video_now, flow_audio_now, jvp_video_zero_t, jvp_audio_zero_t = dmd._student_trig_flow_jvp(
        noisy_video=noisy_video,
        noisy_audio=noisy_audio,
        conditional_dict=conditional_dict,
        trig_time=trig_time,
        t_noisy_video=t_noisy_video,
        t_noisy_audio=t_noisy_audio,
        t_trig_time=zero_t_trig_time,
    )

    flow_video_prev, flow_audio_prev = dmd._student_trig_flow_field(
        noisy_video=noisy_video,
        noisy_audio=noisy_audio,
        conditional_dict=conditional_dict,
        trig_time=trig_time - h,
    )

    expected_t_flow_video = jvp_video_zero_t + cs_video * ((cos_h * flow_video_now - flow_video_prev) / sin_h)
    expected_t_flow_audio = jvp_audio_zero_t + cs_audio * ((cos_h * flow_audio_now - flow_audio_prev) / sin_h)

    def fd1_formula(video_xt, audio_xt, trig_t):
        cur_v, cur_a = dmd._student_trig_flow_field(
            noisy_video=video_xt,
            noisy_audio=audio_xt,
            conditional_dict=conditional_dict,
            trig_time=trig_t,
        )
        prev_v, prev_a = dmd._student_trig_flow_field(
            noisy_video=video_xt,
            noisy_audio=audio_xt,
            conditional_dict=conditional_dict,
            trig_time=trig_t - h,
        )
        return cur_v, cur_a, prev_v, prev_a

    plus_cur_v, plus_cur_a, plus_prev_v, plus_prev_a = fd1_formula(
        noisy_video + 1.0e-5 * t_noisy_video,
        noisy_audio + 1.0e-5 * t_noisy_audio,
        trig_time,
    )
    minus_cur_v, minus_cur_a, minus_prev_v, minus_prev_a = fd1_formula(
        noisy_video - 1.0e-5 * t_noisy_video,
        noisy_audio - 1.0e-5 * t_noisy_audio,
        trig_time,
    )
    ref_jvp_video_zero_t = (plus_cur_v - minus_cur_v) / (2.0e-5)
    ref_jvp_audio_zero_t = (plus_cur_a - minus_cur_a) / (2.0e-5)
    ref_prev_video = (plus_prev_v + minus_prev_v) / 2
    ref_prev_audio = (plus_prev_a + minus_prev_a) / 2
    ref_cur_video = (plus_cur_v + minus_cur_v) / 2
    ref_cur_audio = (plus_cur_a + minus_cur_a) / 2
    ref_t_flow_video = ref_jvp_video_zero_t + cs_video * ((cos_h * ref_cur_video - ref_prev_video) / sin_h)
    ref_t_flow_audio = ref_jvp_audio_zero_t + cs_audio * ((cos_h * ref_cur_audio - ref_prev_audio) / sin_h)

    torch.testing.assert_close(expected_t_flow_video, ref_t_flow_video, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(expected_t_flow_audio, ref_t_flow_audio, rtol=1e-4, atol=1e-5)


def run_internal_jvp_vs_external_reference_diagnostic():
    torch.manual_seed(0)
    case = _build_tiny_real_case(dtype=torch.float32)
    wrapper = case["wrapper"]
    noisy_video = case["noisy_video"]
    noisy_audio = case["noisy_audio"]
    trig_time = case["trig_time"]
    audio_trig_time = case["audio_trig_time"]
    t_noisy_video = case["t_noisy_video"]
    t_noisy_audio = case["t_noisy_audio"]
    t_trig_time = case["t_trig_time"]
    t_audio_trig_time = case["t_audio_trig_time"]
    conditional_dict = case["conditional_dict"]

    video_flow_internal, audio_flow_internal, t_video_flow_internal, t_audio_flow_internal = wrapper(
        noisy_image_or_video=noisy_video,
        conditional_dict=conditional_dict,
        timestep=trig_time,
        noisy_audio=noisy_audio,
        audio_timestep=audio_trig_time,
        t_noisy_image_or_video=t_noisy_video,
        t_timestep=t_trig_time,
        t_noisy_audio=t_noisy_audio,
        t_audio_timestep=t_audio_trig_time,
        with_t=True,
    )

    def flow_fn(video_xt, audio_xt, trig_t, audio_trig_t):
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            pred_video_x0, pred_audio_x0 = wrapper(
                noisy_image_or_video=video_xt,
                conditional_dict=conditional_dict,
                timestep=trig_t,
                noisy_audio=audio_xt,
                audio_timestep=audio_trig_t,
            )
        return (
            _manual_flow_field(video_xt, pred_video_x0, trig_t),
            _manual_flow_field(audio_xt, pred_audio_x0, audio_trig_t),
        )

    (video_flow_external, audio_flow_external), (t_video_flow_external, t_audio_flow_external) = torch.func.jvp(
        flow_fn,
        (noisy_video, noisy_audio, trig_time, audio_trig_time),
        (t_noisy_video, t_noisy_audio, t_trig_time, t_audio_trig_time),
    )

    metrics = {}
    for name, lhs, rhs in [
        ("video_primal", video_flow_internal, video_flow_external),
        ("audio_primal", audio_flow_internal, audio_flow_external),
        ("video_tangent", t_video_flow_internal, t_video_flow_external),
        ("audio_tangent", t_audio_flow_internal, t_audio_flow_external),
    ]:
        diff = (lhs - rhs).abs()
        rel = diff / rhs.abs().clamp_min(1e-8)
        metrics[name] = {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "max_rel": float(rel.max()),
            "mean_rel": float(rel.mean()),
        }

    return metrics


def run_internal_stage_diagnostic():
    torch.manual_seed(0)
    case = _build_tiny_real_case(dtype=torch.float32)
    wrapper = case["wrapper"]
    model = getattr(wrapper.model, "velocity_model", wrapper.model)
    prepared = _prepare_modalities_for_case(case)
    video_modality = prepared["video_modality"]
    audio_modality = prepared["audio_modality"]
    t_video_flat = prepared["t_video_flat"]
    t_audio_rf_latent = prepared["t_audio_rf_latent"]
    t_video_timesteps = prepared["t_video_timesteps"]
    t_audio_timesteps = prepared["t_audio_timesteps"]

    diagnostics = {}

    video_args_ref = model.video_args_preprocessor.prepare(video_modality)
    audio_args_ref = model.audio_args_preprocessor.prepare(audio_modality)
    video_args_with_t = prepare_transformer_args_with_t(
        model.video_args_preprocessor,
        video_modality,
        t_video_flat,
        t_video_timesteps,
    )
    audio_args_with_t = prepare_transformer_args_with_t(
        model.audio_args_preprocessor,
        audio_modality,
        t_audio_rf_latent,
        t_audio_timesteps,
    )

    diagnostics["prepare/video_x"] = _metric_dict(video_args_with_t.primal.x, video_args_ref.x)
    diagnostics["prepare/video_timesteps"] = _metric_dict(video_args_with_t.primal.timesteps, video_args_ref.timesteps)
    diagnostics["prepare/video_embedded_timestep"] = _metric_dict(
        video_args_with_t.primal.embedded_timestep, video_args_ref.embedded_timestep
    )
    diagnostics["prepare/audio_x"] = _metric_dict(audio_args_with_t.primal.x, audio_args_ref.x)
    diagnostics["prepare/audio_timesteps"] = _metric_dict(audio_args_with_t.primal.timesteps, audio_args_ref.timesteps)
    diagnostics["prepare/audio_embedded_timestep"] = _metric_dict(
        audio_args_with_t.primal.embedded_timestep, audio_args_ref.embedded_timestep
    )

    block = model.transformer_blocks[0]
    zero_video = video_args_with_t.t_x.new_zeros(video_args_with_t.t_x.shape)
    zero_audio = audio_args_with_t.t_x.new_zeros(audio_args_with_t.t_x.shape)
    video_args_zero_t = type(video_args_with_t)(
        primal=video_args_with_t.primal,
        t_x=zero_video,
        t_timesteps=video_args_with_t.t_timesteps.new_zeros(video_args_with_t.t_timesteps.shape),
        t_embedded_timestep=video_args_with_t.t_embedded_timestep.new_zeros(video_args_with_t.t_embedded_timestep.shape),
        t_cross_scale_shift_timestep=video_args_with_t.t_cross_scale_shift_timestep.new_zeros(
            video_args_with_t.t_cross_scale_shift_timestep.shape
        ),
        t_cross_gate_timestep=video_args_with_t.t_cross_gate_timestep.new_zeros(video_args_with_t.t_cross_gate_timestep.shape),
    )
    audio_args_zero_t = type(audio_args_with_t)(
        primal=audio_args_with_t.primal,
        t_x=zero_audio,
        t_timesteps=audio_args_with_t.t_timesteps.new_zeros(audio_args_with_t.t_timesteps.shape),
        t_embedded_timestep=audio_args_with_t.t_embedded_timestep.new_zeros(audio_args_with_t.t_embedded_timestep.shape),
        t_cross_scale_shift_timestep=audio_args_with_t.t_cross_scale_shift_timestep.new_zeros(
            audio_args_with_t.t_cross_scale_shift_timestep.shape
        ),
        t_cross_gate_timestep=audio_args_with_t.t_cross_gate_timestep.new_zeros(audio_args_with_t.t_cross_gate_timestep.shape),
    )
    perturbations = BatchedPerturbationConfig.empty(batch_size=video_modality.latent.shape[0])

    with sdpa_kernel(backends=[SDPBackend.MATH]):
        rms_ref = rms_norm(video_args_ref.x, eps=block.norm_eps)
        rms_internal, _ = _rms_norm_with_t(video_args_ref.x, torch.zeros_like(video_args_ref.x), block.norm_eps)
        diagnostics["primitive/rms_norm"] = _metric_dict(rms_internal, rms_ref)

        ff_ref = block.ff(video_args_ref.x)
        ff_internal, _ = _feed_forward_with_t(block.ff, video_args_ref.x, torch.zeros_like(video_args_ref.x))
        diagnostics["primitive/feed_forward"] = _metric_dict(ff_internal, ff_ref)

        ln_ref = model.norm_out(video_args_ref.x)
        ln_internal, _ = _layer_norm_with_t(model.norm_out, video_args_ref.x, torch.zeros_like(video_args_ref.x))
        diagnostics["primitive/layer_norm"] = _metric_dict(ln_internal, ln_ref)

        attn_ref = block.attn1(video_args_ref.x, context=None, mask=None, pe=video_args_ref.positional_embeddings, k_pe=None)
        attn_internal, _ = _attention_with_t(
            block.attn1,
            video_args_ref.x,
            torch.zeros_like(video_args_ref.x),
            context=None,
            t_context=None,
            mask=None,
            pe=video_args_ref.positional_embeddings,
            k_pe=None,
        )
        diagnostics["primitive/attention_self"] = _metric_dict(attn_internal, attn_ref)

        ctx_ref = block.attn2(
            video_args_ref.x,
            context=video_args_ref.context,
            mask=video_args_ref.context_mask,
            pe=None,
            k_pe=None,
        )
        ctx_internal, _ = _attention_with_t(
            block.attn2,
            video_args_ref.x,
            torch.zeros_like(video_args_ref.x),
            context=video_args_ref.context,
            t_context=None,
            mask=video_args_ref.context_mask,
            pe=None,
            k_pe=None,
        )
        diagnostics["primitive/attention_ctx"] = _metric_dict(ctx_internal, ctx_ref)

        block_ref_video, block_ref_audio = block(video_args_ref, audio_args_ref, perturbations)
        block_internal_video, block_internal_audio = _block_with_t_impl(
            block,
            video_args_zero_t,
            audio_args_zero_t,
            perturbations,
        )
        diagnostics["block/video_x"] = _metric_dict(block_internal_video.primal.x, block_ref_video.x)
        diagnostics["block/audio_x"] = _metric_dict(block_internal_audio.primal.x, block_ref_audio.x)

        full_ref_video, full_ref_audio = model(video_modality, audio_modality, perturbations)
        full_internal_video, full_internal_audio, _, _ = ltx_model_with_t(
            model=model,
            video=video_args_zero_t,
            audio=audio_args_zero_t,
            perturbations=perturbations,
        )
        diagnostics["full_model/video"] = _metric_dict(full_internal_video, full_ref_video)
        diagnostics["full_model/audio"] = _metric_dict(full_internal_audio, full_ref_audio)

        full_internal_video_nz, full_internal_audio_nz, _, _ = ltx_model_with_t(
            model=model,
            video=video_args_with_t,
            audio=audio_args_with_t,
            perturbations=perturbations,
        )
        diagnostics["full_model_nonzero_tangent/video"] = _metric_dict(full_internal_video_nz, full_ref_video)
        diagnostics["full_model_nonzero_tangent/audio"] = _metric_dict(full_internal_audio_nz, full_ref_audio)

        ordinary_video_x0, ordinary_audio_x0 = wrapper(
            noisy_image_or_video=case["noisy_video"],
            conditional_dict=case["conditional_dict"],
            timestep=case["trig_time"],
            noisy_audio=case["noisy_audio"],
            audio_timestep=case["audio_trig_time"],
        )

        video_trig_bcast = wrapper._reshape_time_for_latent(case["trig_time"], case["noisy_video"].dim()).to(
            dtype=case["noisy_video"].dtype,
            device=case["noisy_video"].device,
        )
        audio_trig_bcast = wrapper._reshape_time_for_latent(case["audio_trig_time"], case["noisy_audio"].dim()).to(
            dtype=case["noisy_audio"].dtype,
            device=case["noisy_audio"].device,
        )
        video_rf_time, _ = wrapper._rf_time_and_tangent_from_trig(
            video_trig_bcast,
            wrapper._reshape_time_for_latent(case["t_trig_time"], case["noisy_video"].dim()).to(
                dtype=case["noisy_video"].dtype,
                device=case["noisy_video"].device,
            ),
        )
        audio_rf_time, _ = wrapper._rf_time_and_tangent_from_trig(
            audio_trig_bcast,
            wrapper._reshape_time_for_latent(case["t_audio_trig_time"], case["noisy_audio"].dim()).to(
                dtype=case["noisy_audio"].dtype,
                device=case["noisy_audio"].device,
            ),
        )
        internal_video_v_unflat = wrapper._unflatten_video_latent(
            full_internal_video_nz,
            prepared["num_video_frames"],
        )
        internal_video_x0 = (prepared["video_rf_latent"] - video_rf_time * internal_video_v_unflat).to(
            dtype=case["noisy_video"].dtype
        )
        internal_audio_x0 = (prepared["audio_rf_latent"] - audio_rf_time * full_internal_audio_nz).to(
            dtype=case["noisy_audio"].dtype
        )
        diagnostics["wrapper_x0/video"] = _metric_dict(internal_video_x0, ordinary_video_x0)
        diagnostics["wrapper_x0/audio"] = _metric_dict(internal_audio_x0, ordinary_audio_x0)

        zero_t_video = torch.zeros_like(case["noisy_video"])
        zero_t_audio = torch.zeros_like(case["noisy_audio"])
        zero_t_time = torch.zeros_like(case["trig_time"])
        zero_t_audio_time = torch.zeros_like(case["audio_trig_time"])
        internal_video_flow_from_helper, internal_t_video_flow = wrapper._flow_with_t(
            case["noisy_video"],
            internal_video_x0,
            case["trig_time"],
            zero_t_video,
            torch.zeros_like(internal_video_x0),
            zero_t_time,
        )
        internal_audio_flow_from_helper, internal_t_audio_flow = wrapper._flow_with_t(
            case["noisy_audio"],
            internal_audio_x0,
            case["audio_trig_time"],
            zero_t_audio,
            torch.zeros_like(internal_audio_x0),
            zero_t_audio_time,
        )
        del internal_t_video_flow, internal_t_audio_flow
        diagnostics["flow_helper/video"] = _metric_dict(
            internal_video_flow_from_helper,
            _manual_flow_field(case["noisy_video"], internal_video_x0, case["trig_time"]),
        )
        diagnostics["flow_helper/audio"] = _metric_dict(
            internal_audio_flow_from_helper,
            _manual_flow_field(case["noisy_audio"], internal_audio_x0, case["audio_trig_time"]),
        )

        out_ref_video = model._process_output(
            model.scale_shift_table,
            model.norm_out,
            model.proj_out,
            block_ref_video.x,
            block_ref_video.embedded_timestep,
        )
        out_internal_video, _ = _process_output_with_t(
            model.scale_shift_table,
            model.norm_out,
            model.proj_out,
            block_ref_video.x,
            torch.zeros_like(block_ref_video.x),
            block_ref_video.embedded_timestep,
            torch.zeros_like(block_ref_video.embedded_timestep),
        )
        diagnostics["output_head/video"] = _metric_dict(out_internal_video, out_ref_video)

        out_ref_audio = model._process_output(
            model.audio_scale_shift_table,
            model.audio_norm_out,
            model.audio_proj_out,
            block_ref_audio.x,
            block_ref_audio.embedded_timestep,
        )
        out_internal_audio, _ = _process_output_with_t(
            model.audio_scale_shift_table,
            model.audio_norm_out,
            model.audio_proj_out,
            block_ref_audio.x,
            torch.zeros_like(block_ref_audio.x),
            block_ref_audio.embedded_timestep,
            torch.zeros_like(block_ref_audio.embedded_timestep),
        )
        diagnostics["output_head/audio"] = _metric_dict(out_internal_audio, out_ref_audio)

    return diagnostics


def run_internal_jvp_sweep_diagnostic():
    cases = [
        dict(
            seed=0,
            wrapper_kwargs=dict(
                num_layers=2,
                num_attention_heads=2,
                attention_head_dim=8,
                cross_attention_dim=16,
                caption_channels=32,
                positional_embedding_max_pos=(4, 4, 4),
                audio_positional_embedding_max_pos=(8,),
            ),
            batch=2,
            video_frames=2,
            audio_frames=3,
            text_len=4,
            text_dim=32,
            trig_time_values=[[0.12], [1.46]],
            audio_trig_time_values=[[0.18], [1.42]],
            t_trig_time_values=[[0.015], [-0.018]],
            t_audio_trig_time_values=[[0.011], [0.007]],
        ),
        dict(
            seed=1,
            wrapper_kwargs=dict(
                num_layers=3,
                num_attention_heads=2,
                attention_head_dim=8,
                cross_attention_dim=16,
                caption_channels=48,
                positional_embedding_max_pos=(6, 4, 4),
                audio_positional_embedding_max_pos=(10,),
            ),
            batch=2,
            video_frames=3,
            audio_frames=5,
            text_len=5,
            text_dim=48,
            trig_time_values=[[0.08], [1.50]],
            audio_trig_time_values=[[0.10], [1.48]],
            t_trig_time_values=[[0.01], [-0.012]],
            t_audio_trig_time_values=[[0.009], [0.006]],
        ),
        dict(
            seed=2,
            wrapper_kwargs=dict(
                num_layers=2,
                num_attention_heads=4,
                attention_head_dim=8,
                cross_attention_dim=32,
                caption_channels=64,
                positional_embedding_max_pos=(5, 5, 5),
                audio_positional_embedding_max_pos=(12,),
            ),
            batch=2,
            video_frames=4,
            audio_frames=6,
            text_len=6,
            text_dim=64,
            trig_time_values=[[0.20], [1.20]],
            audio_trig_time_values=[[0.24], [1.16]],
            t_trig_time_values=[[0.02], [-0.017]],
            t_audio_trig_time_values=[[0.014], [0.008]],
        ),
    ]

    all_metrics = []
    for idx, case_kwargs in enumerate(cases):
        case = _build_tiny_real_case(dtype=torch.float32, **case_kwargs)
        wrapper = case["wrapper"]
        conditional_dict = case["conditional_dict"]

        video_flow_internal, audio_flow_internal, t_video_flow_internal, t_audio_flow_internal = wrapper(
            noisy_image_or_video=case["noisy_video"],
            conditional_dict=conditional_dict,
            timestep=case["trig_time"],
            noisy_audio=case["noisy_audio"],
            audio_timestep=case["audio_trig_time"],
            t_noisy_image_or_video=case["t_noisy_video"],
            t_timestep=case["t_trig_time"],
            t_noisy_audio=case["t_noisy_audio"],
            t_audio_timestep=case["t_audio_trig_time"],
            with_t=True,
        )

        def flow_fn(video_xt, audio_xt, trig_t, audio_trig_t):
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                pred_video_x0, pred_audio_x0 = wrapper(
                    noisy_image_or_video=video_xt,
                    conditional_dict=conditional_dict,
                    timestep=trig_t,
                    noisy_audio=audio_xt,
                    audio_timestep=audio_trig_t,
                )
            return (
                _manual_flow_field(video_xt, pred_video_x0, trig_t),
                _manual_flow_field(audio_xt, pred_audio_x0, audio_trig_t),
            )

        (video_flow_external, audio_flow_external), (t_video_flow_external, t_audio_flow_external) = torch.func.jvp(
            flow_fn,
            (case["noisy_video"], case["noisy_audio"], case["trig_time"], case["audio_trig_time"]),
            (case["t_noisy_video"], case["t_noisy_audio"], case["t_trig_time"], case["t_audio_trig_time"]),
        )

        metrics = {"case": idx}
        for name, lhs, rhs in [
            ("video_primal", video_flow_internal, video_flow_external),
            ("audio_primal", audio_flow_internal, audio_flow_external),
            ("video_tangent", t_video_flow_internal, t_video_flow_external),
            ("audio_tangent", t_audio_flow_internal, t_audio_flow_external),
        ]:
            case_metrics = _metric_dict(lhs, rhs)
            metrics[name] = case_metrics
        all_metrics.append(metrics)

    return all_metrics


def test_internal_jvp_matches_external_reference_on_tiny_ltx_wrapper():
    metrics = run_internal_jvp_vs_external_reference_diagnostic()
    for name, values in metrics.items():
        assert values["max_abs"] < 1e-5, f"{name} max_abs too large: {values['max_abs']}"
        assert values["mean_abs"] < 1e-6, f"{name} mean_abs too large: {values['mean_abs']}"
        assert values["mean_rel"] < 1e-4, f"{name} mean_rel too large: {values['mean_rel']}"


def test_internal_stage_primal_equivalence_on_tiny_ltx_wrapper():
    diagnostics = run_internal_stage_diagnostic()
    for name, values in diagnostics.items():
        assert values["max_abs"] < 1e-6, f"{name} max_abs too large: {values['max_abs']}"
        assert values["mean_abs"] < 1e-7, f"{name} mean_abs too large: {values['mean_abs']}"


def test_internal_jvp_matches_external_reference_across_randomized_ltx_cases():
    metrics_list = run_internal_jvp_sweep_diagnostic()
    for metrics in metrics_list:
        case_id = metrics["case"]
        for name in ("video_primal", "audio_primal", "video_tangent", "audio_tangent"):
            values = metrics[name]
            assert values["max_abs"] < 2e-5, (
                f"case {case_id} {name} max_abs too large: {values['max_abs']}"
            )
            assert values["mean_abs"] < 2e-6, (
                f"case {case_id} {name} mean_abs too large: {values['mean_abs']}"
            )
            assert values["mean_rel"] < 2e-4, (
                f"case {case_id} {name} mean_rel too large: {values['mean_rel']}"
            )


def _run_all_tests():
    test_student_trig_flow_field_matches_manual_formula()
    print("PASS: test_student_trig_flow_field_matches_manual_formula")
    test_student_trig_flow_jvp_matches_central_difference()
    print("PASS: test_student_trig_flow_jvp_matches_central_difference")
    test_student_trig_flow_fd1_matches_semi_continuous_formula()
    print("PASS: test_student_trig_flow_fd1_matches_semi_continuous_formula")
    test_internal_jvp_matches_external_reference_on_tiny_ltx_wrapper()
    print("PASS: test_internal_jvp_matches_external_reference_on_tiny_ltx_wrapper")
    test_internal_stage_primal_equivalence_on_tiny_ltx_wrapper()
    print("PASS: test_internal_stage_primal_equivalence_on_tiny_ltx_wrapper")
    test_internal_jvp_matches_external_reference_across_randomized_ltx_cases()
    print("PASS: test_internal_jvp_matches_external_reference_across_randomized_ltx_cases")
    metrics = run_internal_jvp_vs_external_reference_diagnostic()
    print("DIAG: internal JVP vs external torch.func.jvp on tiny LTX wrapper")
    for name, values in metrics.items():
        print(
            f"  {name}: max_abs={values['max_abs']:.6f} mean_abs={values['mean_abs']:.6f} "
            f"max_rel={values['max_rel']:.6f} mean_rel={values['mean_rel']:.6f}"
        )
    stage_metrics = run_internal_stage_diagnostic()
    print("DIAG: stage-by-stage primal equivalence on tiny LTX wrapper")
    for name, values in stage_metrics.items():
        print(
            f"  {name}: max_abs={values['max_abs']:.6f} mean_abs={values['mean_abs']:.6f} "
            f"max_rel={values['max_rel']:.6f} mean_rel={values['mean_rel']:.6f}"
        )
    sweep_metrics = run_internal_jvp_sweep_diagnostic()
    print("DIAG: randomized internal-vs-external JVP sweep on LTX wrappers")
    for metrics in sweep_metrics:
        print(f"  case={metrics['case']}")
        for name in ("video_primal", "audio_primal", "video_tangent", "audio_tangent"):
            values = metrics[name]
            print(
                f"    {name}: max_abs={values['max_abs']:.6f} mean_abs={values['mean_abs']:.6f} "
                f"max_rel={values['max_rel']:.6f} mean_rel={values['mean_rel']:.6f}"
            )


if __name__ == "__main__":
    _run_all_tests()
