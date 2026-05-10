"""
Run standalone teacher/base-model inference for RF vs TrigFlow evaluation.

This script is intentionally single-process and loads only the components
required for teacher-style generation:
    - text encoder
    - teacher wrapper
    - VAEs

It can be used to:
1. Re-run teacher benchmark outputs for an existing training run without
   restarting training.
2. Compare native RF vs migrated rCM/TrigFlow teacher inference on a larger
   prompt set using full inference steps.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Sequence, Tuple

import torch
from omegaconf import OmegaConf

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader.registry import StateDictRegistry
from ltx_distillation.models.ltx_trig_wrapper import create_ltx2_trig_wrapper
from ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper
from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper
from ltx_distillation.models.vae_wrapper import create_vae_wrappers
from ltx_distillation.time_utils import rf_to_trig_time
from ltx_distillation.train_distillation import compute_latent_shapes


def _load_prompts(prompts_file: str, num_prompts: int) -> list[str]:
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts[:num_prompts]


def _save_prompt_file(prompts: Sequence[str], output_dir: str) -> str:
    path = os.path.join(output_dir, "prompts.txt")
    with open(path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")
    return path


def _decode_and_save_sample(
    video_vae,
    audio_vae,
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    prompt_idx: int,
    output_dir: str,
    video_fps: int,
    audio_sample_rate: int,
) -> None:
    video_pixel = video_vae.decode_to_pixel(video_latent)

    audio_waveform = None
    try:
        audio_waveform = audio_vae.decode_to_waveform(audio_latent)
    except Exception as e:  # pragma: no cover - best-effort logging path
        print(f"[Decode] Audio decode failed for prompt {prompt_idx}: {e}", flush=True)

    vid = video_pixel[0]
    if vid.shape[0] == 3:
        vid = vid.permute(1, 0, 2, 3)
    vid = vid.permute(0, 2, 3, 1)
    vid = (vid.clamp(0, 1) * 255).cpu().to(torch.uint8)

    sample_path = os.path.join(output_dir, f"sample_{prompt_idx}.mp4")
    written_with_audio = False

    if audio_waveform is not None:
        try:
            from torchvision.io import write_video

            wav_float = audio_waveform[0].cpu().float()
            write_video(
                sample_path,
                vid,
                fps=video_fps,
                audio_array=wav_float,
                audio_fps=audio_sample_rate,
                audio_codec="aac",
            )
            written_with_audio = True
        except Exception as e:  # pragma: no cover - best-effort logging path
            print(f"[Decode] write_video with audio failed for prompt {prompt_idx}: {e}", flush=True)

    if not written_with_audio:
        from torchvision.io import write_video

        write_video(sample_path, vid, fps=video_fps)

        if audio_waveform is not None:
            try:
                import torchaudio

                wav = audio_waveform[0].cpu().float()
                wav_path = os.path.join(output_dir, f"sample_{prompt_idx}.wav")
                torchaudio.save(wav_path, wav, audio_sample_rate)
            except Exception as e:  # pragma: no cover - best-effort logging path
                print(f"[Decode] Saving wav failed for prompt {prompt_idx}: {e}", flush=True)

    del video_pixel, audio_waveform
    torch.cuda.empty_cache()


@torch.no_grad()
def _generate_teacher_sample(
    teacher,
    video_shape: Tuple[int, ...],
    audio_shape: Tuple[int, ...],
    sigmas: torch.Tensor,
    conditional_dict,
    unconditional_dict,
    device: torch.device,
    dtype: torch.dtype,
    video_cfg: float,
    audio_cfg: float,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz = video_shape[0]
    video_frames = video_shape[1]
    audio_frames = audio_shape[1]

    video = torch.randn(video_shape, device=device, dtype=dtype)
    audio = torch.randn(audio_shape, device=device, dtype=dtype)

    if mode == "native_rf":
        teacher_forward = teacher.forward_rf if hasattr(teacher, "forward_rf") else teacher
        schedule = sigmas
    elif mode == "rcm_trig":
        teacher_forward = teacher
        schedule = rf_to_trig_time(sigmas.double()).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    for i in range(len(schedule) - 1):
        sigma = schedule[i]
        video_sigma = sigma * torch.ones([bsz, video_frames], device=device, dtype=dtype)
        audio_sigma = sigma * torch.ones([bsz, audio_frames], device=device, dtype=dtype)

        video_x0_cond, audio_x0_cond = teacher_forward(
            noisy_image_or_video=video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=audio,
            audio_timestep=audio_sigma,
        )
        video_x0_uncond, audio_x0_uncond = teacher_forward(
            noisy_image_or_video=video,
            conditional_dict=unconditional_dict,
            timestep=video_sigma,
            noisy_audio=audio,
            audio_timestep=audio_sigma,
        )

        video_x0 = video_x0_uncond + video_cfg * (video_x0_cond - video_x0_uncond)
        audio_x0 = audio_x0_uncond + audio_cfg * (audio_x0_cond - audio_x0_uncond)

        sigma_next = schedule[i + 1]
        if sigma_next > 0 and sigma > 0:
            if mode == "native_rf":
                video_velocity = (video.float() - video_x0.float()) / sigma.float()
                audio_velocity = (audio.float() - audio_x0.float()) / sigma.float()
                dt = (sigma_next - sigma).float()
                video = (video.float() + video_velocity * dt).to(dtype)
                audio = (audio.float() + audio_velocity * dt).to(dtype)
            else:
                next_t_video = sigma_next.view(1, 1, 1, 1, 1).to(device=device, dtype=dtype)
                next_t_audio = sigma_next.view(1, 1, 1).to(device=device, dtype=dtype)
                video = (
                    torch.cos(next_t_video) * video_x0
                    + torch.sin(next_t_video) * torch.randn_like(video)
                ).to(dtype)
                audio = (
                    torch.cos(next_t_audio) * audio_x0
                    + torch.sin(next_t_audio) * torch.randn_like(audio)
                ).to(dtype)
        else:
            video = video_x0
            audio = audio_x0

    return video, audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone LTX teacher inference runner")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--num_prompts", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["native_rf", "rcm_trig"], required=True)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--start_index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if bool(getattr(cfg, "mixed_precision", True)) else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = _load_prompts(args.prompts_file, args.start_index + args.num_prompts)
    prompts = prompts[args.start_index : args.start_index + args.num_prompts]
    prompt_path = _save_prompt_file(prompts, args.output_dir)

    steps = int(
        args.num_inference_steps
        if args.num_inference_steps is not None
        else getattr(cfg, "teacher_benchmark_num_inference_steps", 40)
    )

    print(
        f"[TeacherEval] mode={args.mode} prompts={len(prompts)} steps={steps} "
        f"output={args.output_dir} prompts_file={prompt_path}",
        flush=True,
    )

    registry = StateDictRegistry()
    if args.mode == "native_rf":
        teacher_factory = create_ltx2_wrapper
    else:
        teacher_factory = (
            create_ltx2_trig_wrapper
            if str(getattr(cfg, "dmd_style", "legacy")) == "rcm_trig"
            else create_ltx2_wrapper
        )
    teacher = teacher_factory(
        checkpoint_path=cfg.checkpoint_path,
        gemma_path=cfg.gemma_path,
        device=device,
        dtype=dtype,
        video_height=cfg.video_height,
        video_width=cfg.video_width,
        registry=registry,
    ).eval()
    text_encoder = create_text_encoder_wrapper(
        checkpoint_path=cfg.checkpoint_path,
        gemma_path=cfg.gemma_path,
        device=device,
        dtype=dtype,
        registry=registry,
    ).eval()
    video_vae, audio_vae = create_vae_wrappers(
        checkpoint_path=cfg.checkpoint_path,
        device=device,
        dtype=dtype,
        registry=registry,
    )
    video_vae.eval()
    audio_vae.eval()

    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(steps=steps).to(device=device, dtype=dtype)
    video_shape, audio_shape = compute_latent_shapes(
        num_frames=int(cfg.num_frames),
        video_height=int(cfg.video_height),
        video_width=int(cfg.video_width),
        batch_size=1,
    )

    start_time = time.perf_counter()
    for idx, prompt in enumerate(prompts):
        global_idx = args.start_index + idx
        conditional_dict = text_encoder(text_prompts=[prompt])
        unconditional_dict = text_encoder(text_prompts=[cfg.negative_prompt])

        prompt_seed = int(args.seed) + global_idx
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(prompt_seed)
            torch.cuda.manual_seed(prompt_seed)
            gen_start = time.perf_counter()
            video_latent, audio_latent = _generate_teacher_sample(
                teacher=teacher,
                video_shape=tuple(video_shape),
                audio_shape=tuple(audio_shape),
                sigmas=sigmas,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                device=device,
                dtype=dtype,
                video_cfg=float(getattr(cfg, "teacher_benchmark_video_guidance_scale", 3.0)),
                audio_cfg=float(getattr(cfg, "teacher_benchmark_audio_guidance_scale", 5.0)),
                mode=args.mode,
            )
            gen_elapsed = time.perf_counter() - gen_start

        _decode_and_save_sample(
            video_vae=video_vae,
            audio_vae=audio_vae,
            video_latent=video_latent,
            audio_latent=audio_latent,
            prompt_idx=idx,
            output_dir=args.output_dir,
            video_fps=int(getattr(cfg, "benchmark_video_fps", 24)),
            audio_sample_rate=int(getattr(cfg, "benchmark_audio_sample_rate", 24000)),
        )

        print(
            f"[TeacherEval] mode={args.mode} prompt={idx + 1}/{len(prompts)} "
            f"seed={prompt_seed} gen={gen_elapsed:.2f}s",
            flush=True,
        )
        del conditional_dict, unconditional_dict, video_latent, audio_latent
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - start_time
    print(
        f"[TeacherEval] done mode={args.mode} prompts={len(prompts)} "
        f"wall={elapsed:.2f}s ({elapsed / max(1, len(prompts)):.2f}s/video) "
        f"saved_to={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
