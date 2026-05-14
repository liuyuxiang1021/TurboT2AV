"""
DMD Distillation Training Script for LTX-2.

Usage:
    torchrun --nproc_per_node=8 -m ltx_distillation.train_distillation \
        --config_path configs/ltx2_bidirectional_dmd.yaml
"""

import argparse
import math
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_distillation.rcm import LTX2RCM
from ltx_distillation.data import TextDataset, ODERegressionLMDBDataset
from ltx_distillation.time_utils import rf_to_trig_time
from ltx_distillation.util import (
    launch_distributed_job,
    set_seed,
    init_logging_folder,
    fsdp_wrap,
    fsdp_state_dict,
    barrier,
    cycle,
)


def compute_latent_shapes(
    num_frames: int,
    video_height: int,
    video_width: int,
    batch_size: int = 1,
    latent_channels: int = 128,
    vae_temporal_compression: int = 8,
    vae_spatial_compression: int = 32,
    video_fps: float = 24.0,
    audio_sample_rate: int = 16000,
    audio_hop_length: int = 160,
    audio_latent_downsample: int = 4,
) -> Tuple[list, list]:
    """
    Compute latent shapes from video frames and resolution.

    Calculation logic matches LTX-2 native implementation (see ltx_core/types.py):
    - Video: frames = (num_frames - 1) // 8 + 1
    - Audio: frames = round(video_duration * audio_latent_fps)
             where audio_latent_fps = sample_rate / hop_length / downsample = 25

    Args:
        num_frames: Number of raw video frames (must satisfy 1 + 8*k constraint)
        video_height: Video height in pixels
        video_width: Video width in pixels
        batch_size: Batch size
        latent_channels: Number of latent channels
        vae_temporal_compression: VAE temporal compression ratio (default 8)
        vae_spatial_compression: VAE spatial compression ratio (default 32)
        video_fps: Video frame rate (default 24.0)
        audio_sample_rate: Audio sample rate (default 16000)
        audio_hop_length: Audio hop length (default 160)
        audio_latent_downsample: Audio latent downsampling factor (default 4)

    Returns:
        (video_shape, audio_shape)
        - video_shape: [B, latent_frames, C, H, W]
        - audio_shape: [B, audio_frames, C]
    """
    # Check frame count constraint
    if (num_frames - 1) % vae_temporal_compression != 0:
        raise ValueError(
            f"num_frames must be 1 + 8*k, got {num_frames}. "
            f"Valid values: 1, 9, 17, 25, ..., 121, ..., 241, ..."
        )

    # Compute video latent frames (matches LTX types.py:73)
    latent_frames = 1 + (num_frames - 1) // vae_temporal_compression

    # Compute latent spatial dimensions
    latent_h = video_height // vae_spatial_compression
    latent_w = video_width // vae_spatial_compression

    # Compute audio frames (matches LTX types.py:140-156)
    # video_duration = num_frames / video_fps
    # audio_latent_fps = sample_rate / hop_length / downsample = 16000/160/4 = 25
    # audio_frames = round(video_duration * audio_latent_fps)
    video_duration = float(num_frames) / float(video_fps)
    audio_latent_fps = float(audio_sample_rate) / float(audio_hop_length) / float(audio_latent_downsample)
    audio_frames = round(video_duration * audio_latent_fps)

    video_shape = [batch_size, latent_frames, latent_channels, latent_h, latent_w]
    audio_shape = [batch_size, audio_frames, latent_channels]

    return video_shape, audio_shape


class Trainer:
    """
    DMD Distillation Trainer for LTX-2.

    Handles:
    - Distributed training with FSDP
    - Alternating generator and critic training
    - Checkpointing and logging
    """

    def __init__(self, config):
        self.config = config
        self.wandb_enabled = True

        # Initialize distributed environment
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        rank, world_size, local_rank = launch_distributed_job()
        self.global_rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = self.global_rank == 0

        # Set seed
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            if world_size > 1:
                dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + self.global_rank)

        # Initialize logging (main process only) then broadcast output_path
        # to all ranks so every rank can save benchmark files to shared FS.
        # Avoid NCCL object broadcast here: on this cluster it can fail during
        # early initialization with socket connection errors. Use the shared
        # filesystem plus a barrier instead.
        sync_token = f"{os.environ.get('MASTER_ADDR', 'localhost')}_{os.environ.get('MASTER_PORT', '29500')}"
        sync_token = sync_token.replace("/", "_").replace(":", "_")
        shared_run_path_file = os.path.join(config.output_path, f".run_path_{sync_token}.txt")
        if self.is_main_process:
            self.output_path, self.wandb_folder = init_logging_folder(config)
            os.makedirs(config.output_path, exist_ok=True)
            with open(shared_run_path_file, "w", encoding="utf-8") as f:
                f.write(self.output_path)
        else:
            self.output_path = None
            self.wandb_folder = None

        barrier()

        if not self.is_main_process:
            with open(shared_run_path_file, "r", encoding="utf-8") as f:
                self.output_path = f.read().strip()

        self.wandb_folder = os.path.join(self.output_path, "wandb")

        barrier()
        if self.is_main_process:
            try:
                os.remove(shared_run_path_file)
            except FileNotFoundError:
                pass

        # Initialize unified rCM module.
        # Keep self.dmd as a compatibility alias because the rest of the
        # trainer/benchmark code still uses the historical attribute name.
        self.rcm = LTX2RCM(config, device=self.device)
        self.dmd = self.rcm

        # Initialize models from checkpoints BEFORE FSDP wrapping
        # Models must exist before we can wrap them with FSDP
        self.rcm.init_models()
        self._validate_preinstalled_bidirectional_delegate()

        # FSDP wrapping
        self._wrap_with_fsdp()

        # Optimizers
        weight_decay = getattr(config, "weight_decay", 0.0)
        generator_lr = getattr(config, "generator_lr", config.lr)
        critic_lr = getattr(config, "critic_lr", config.lr)

        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.dmd.generator.parameters() if p.requires_grad],
            lr=generator_lr,
            betas=(config.beta1, config.beta2),
            weight_decay=weight_decay,
        )

        if self.dmd.fake_score is not None:
            self.critic_optimizer = torch.optim.AdamW(
                [p for p in self.dmd.fake_score.parameters() if p.requires_grad],
                lr=critic_lr,
                betas=(config.beta1, config.beta2),
                weight_decay=weight_decay,
            )
        else:
            self.critic_optimizer = None

        # Learning rate schedulers
        self.generator_scheduler = self._create_lr_scheduler(self.generator_optimizer)
        self.critic_scheduler = self._create_lr_scheduler(self.critic_optimizer)

        # Dataloader
        self._init_dataloader()

        # Benchmark prompts (for periodic inference visualization)
        self._init_benchmark_prompts()

        self.step = 0
        self.max_grad_norm = getattr(config, "max_grad_norm", 10.0)
        self.log_iters = int(getattr(config, "log_iters", 0))
        self.train_log_all_scalars = bool(getattr(config, "train_log_all_scalars", False))
        self.checkpoint_iters = int(getattr(config, "checkpoint_iters", self.log_iters))
        self.layerwise_grad_log_interval = max(
            1, int(getattr(config, "layerwise_grad_log_interval", config.log_iters))
        )
        self.previous_time = None

        # Resume from a DMD checkpoint (generator + critic + step counter).
        # Checkpoints are saved after finishing the current step's update, so
        # resume should continue from the *next* step, not rerun the recorded one.
        resume_ckpt = getattr(config, "resume_checkpoint", None)
        if resume_ckpt:
            if self.is_main_process:
                print(f"[Resume] Loading causal DMD checkpoint from {resume_ckpt}")
            ckpt = torch.load(resume_ckpt, map_location="cpu")
            self.dmd.generator.load_state_dict(ckpt["generator"])
            if self.dmd.fake_score is not None and "critic" in ckpt:
                self.dmd.fake_score.load_state_dict(ckpt["critic"])
            if self.dmd.ema_enabled and "generator_ema" in ckpt and ckpt["generator_ema"] is not None:
                self.dmd._ema_state_dict = {k: v.cpu() for k, v in ckpt["generator_ema"].items()}

            # Load optimizer states for seamless resume
            if "generator_optimizer" in ckpt and ckpt["generator_optimizer"] is not None:
                try:
                    osd = ckpt["generator_optimizer"]
                    flattened_osd = self.generator_optimizer.state_dict()
                    for key in flattened_osd["state"].keys():
                        if key in osd["state"]:
                            flattened_osd["state"][key] = osd["state"][key]
                    self.generator_optimizer.load_state_dict(flattened_osd)
                    if self.is_main_process:
                        print("[Resume] Generator optimizer state loaded")
                except Exception as e:
                    print(f"[Resume] Failed to load generator optimizer state: {e}")
            if self.critic_optimizer is not None and "critic_optimizer" in ckpt and ckpt["critic_optimizer"] is not None:
                try:
                    osd = ckpt["critic_optimizer"]
                    flattened_osd = self.critic_optimizer.state_dict()
                    for key in flattened_osd["state"].keys():
                        if key in osd["state"]:
                            flattened_osd["state"][key] = osd["state"][key]
                    self.critic_optimizer.load_state_dict(flattened_osd)
                    if self.is_main_process:
                        print("[Resume] Critic optimizer state loaded")
                except Exception as e:
                    print(f"[Resume] Failed to load critic optimizer state: {e}")
            if "generator_scheduler" in ckpt and ckpt["generator_scheduler"] is not None and self.generator_scheduler is not None:
                self.generator_scheduler.load_state_dict(ckpt["generator_scheduler"])
            if "critic_scheduler" in ckpt and ckpt["critic_scheduler"] is not None and self.critic_scheduler is not None:
                self.critic_scheduler.load_state_dict(ckpt["critic_scheduler"])

            completed_step = int(ckpt.get("completed_step", ckpt.get("step", -1)))
            self.step = int(ckpt.get("next_step", completed_step + 1))
            if self.is_main_process:
                if completed_step >= 0:
                    print(
                        f"[Resume] Loaded checkpoint after completed step {completed_step}; "
                        f"resuming from step {self.step}"
                    )
                else:
                    print(f"[Resume] Resumed at step {self.step}")

    def _disable_wandb(self, reason: str, finish: bool = False):
        if not self.is_main_process or not self.wandb_enabled:
            return
        self.wandb_enabled = False
        os.environ["WANDB_MODE"] = "disabled"
        if finish:
            try:
                wandb.finish()
            except Exception:
                pass
        print(f"[WandB] disabled for the rest of training: {reason}", flush=True)

    def _run_wandb_call_with_timeout(self, label: str, fn) -> bool:
        timeout_s = float(getattr(self.config, "wandb_log_timeout_seconds", 20.0))
        result: Dict[str, Any] = {}

        def _target():
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 - keep training alive.
                result["exc"] = exc

        thread = threading.Thread(target=_target, name=f"wandb-{label}", daemon=True)
        thread.start()
        thread.join(timeout_s)

        if thread.is_alive():
            self._disable_wandb(
                f"{label} timed out after {timeout_s:.1f}s; continuing without WandB"
            )
            return False

        exc = result.get("exc")
        if exc is not None:
            self._disable_wandb(
                f"{label} failed with {type(exc).__name__}: {exc}"
            )
            return False

        return True

    def _safe_wandb_log(self, payload: Dict[str, Any], step: Optional[int] = None):
        if not self.is_main_process or not self.wandb_enabled:
            return
        self._run_wandb_call_with_timeout(
            label="log",
            fn=lambda: wandb.log(payload, step=step),
        )

    def _safe_wandb_finish(self):
        if not self.is_main_process or not self.wandb_enabled:
            return
        ok = self._run_wandb_call_with_timeout(
            label="finish",
            fn=wandb.finish,
        )
        if ok:
            self.wandb_enabled = False

    def _create_lr_scheduler(self, optimizer):
        """Create learning rate scheduler based on config.

        IMPORTANT: The scheduler is NOT stepped per-optimizer-call. Instead,
        both generator and critic schedulers are stepped once per global
        training step (in the training loop), so they stay synchronized
        even though the generator only trains every dfake_gen_update_ratio steps.

        Supported scheduler_type values:
        - None / "constant": No scheduling (constant LR)
        - "cosine_warmup": Linear warmup then cosine decay to min_lr
        """
        scheduler_type = getattr(self.config, "scheduler_type", None)
        if scheduler_type is None or scheduler_type == "constant":
            return None

        warmup_steps = getattr(self.config, "warmup_steps", 1000)
        max_steps = getattr(self.config, "max_steps", 20000)
        min_lr = getattr(self.config, "min_lr", 1e-7)
        base_lr = optimizer.param_groups[0]["lr"]

        if scheduler_type == "cosine_warmup":
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                else:
                    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
                    progress = min(progress, 1.0)
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return max(min_lr / base_lr, cosine_decay)

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    def _validate_preinstalled_bidirectional_delegate(self) -> None:
        """Fail early if causal benchmark fallback would need lazy delegate construction."""
        if not getattr(self.dmd, "generator_use_causal_wrapper", False):
            return

        has_delegate = getattr(self.dmd.generator, "has_bidirectional_delegate", None)
        if callable(has_delegate) and has_delegate():
            return

        raise RuntimeError(
            "Causal Stage-3 generator is missing a pre-installed bidirectional delegate before FSDP "
            "wrapping. Install it during model init (for example from "
            "bootstrap_bidirectional_ckpt_path / generator_ckpt) instead of relying on lazy "
            "delegate construction at benchmark time."
        )

    def _wrap_with_fsdp(self):
        """Wrap models with FSDP for distributed training."""
        config = self.config
        use_orig_params = bool(getattr(config, "fsdp_use_orig_params", True))
        try:
            from ltx_core.model.transformer.transformer import BasicAVTransformerBlock

            transformer_module = (BasicAVTransformerBlock,)
        except Exception:
            transformer_module = None

        if bool(getattr(config, "generator_fsdp_enabled", True)):
            self.dmd.generator = fsdp_wrap(
                self.dmd.generator,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.generator_fsdp_wrap_strategy,
                transformer_module=transformer_module,
                cpu_offload=bool(getattr(config, "generator_cpu_offload", False)),
                use_orig_params=use_orig_params,
            )

        # EMA stored as lightweight state_dict on CPU — no FSDP wrapping needed

        if bool(getattr(config, "real_score_fsdp_enabled", True)):
            self.dmd.real_score = fsdp_wrap(
                self.dmd.real_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.real_score_fsdp_wrap_strategy,
                transformer_module=transformer_module,
                cpu_offload=bool(getattr(config, "real_score_cpu_offload", False)),
                use_orig_params=use_orig_params,
            )

        if self.dmd.fake_score is not None and bool(getattr(config, "fake_score_fsdp_enabled", True)):
            self.dmd.fake_score = fsdp_wrap(
                self.dmd.fake_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.fake_score_fsdp_wrap_strategy,
                transformer_module=transformer_module,
                cpu_offload=bool(getattr(config, "fake_score_cpu_offload", False)),
                use_orig_params=use_orig_params,
            )

        if bool(getattr(config, "text_encoder_fsdp_enabled", True)):
            self.dmd.text_encoder = fsdp_wrap(
                self.dmd.text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
                cpu_offload=bool(getattr(config, "text_encoder_cpu_offload", False)),
                use_orig_params=use_orig_params,
            )

        # Keep VAEs on CPU to save GPU memory during training.
        # They are only needed for periodic visualization and benchmark decoding.
        # Use _vae_to_device() / _vae_to_cpu() to move them on-demand.
        if self.dmd.video_vae is not None:
            self.dmd.video_vae = self.dmd.video_vae.to(dtype=self.dtype)
        if self.dmd.audio_vae is not None:
            self.dmd.audio_vae = self.dmd.audio_vae.to(dtype=self.dtype)

    def _init_dataloader(self):
        """Initialize data loader."""
        from ltx_distillation.data import collate_text_prompts, collate_ode_data

        config = self.config

        self.backward_simulation = getattr(config, "backward_simulation", True)
        self.scm_enabled = bool(getattr(config, "scm_enabled", False))
        self.dcm_enabled = bool(getattr(config, "dcm_enabled", False))

        if self.backward_simulation:
            dataset = TextDataset(config.data_path)
            collate_fn = collate_text_prompts
        else:
            dataset = ODERegressionLMDBDataset(
                config.data_path,
                max_pair=int(1e8),
            )
            collate_fn = collate_ode_data

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
        )

        self.dataloader = cycle(dataloader)

        self.scm_dataloader = None
        if self.scm_enabled or self.dcm_enabled:
            scm_data_path = getattr(config, "scm_data_path", None)
            if not scm_data_path:
                raise ValueError("scm_enabled=true or dcm_enabled=true requires scm_data_path to be set")

            scm_dataset = ODERegressionLMDBDataset(
                scm_data_path,
                max_pair=int(getattr(config, "scm_max_pair", 1e8)),
            )
            scm_sampler = torch.utils.data.distributed.DistributedSampler(
                scm_dataset,
                shuffle=True,
                drop_last=True,
            )
            scm_dataloader = torch.utils.data.DataLoader(
                scm_dataset,
                batch_size=int(getattr(config, "scm_batch_size", config.batch_size)),
                sampler=scm_sampler,
                collate_fn=collate_ode_data,
            )
            self.scm_dataloader = cycle(scm_dataloader)

            # Diagnostic mode: pre-fetch fixed samples for variance analysis
            self.scm_diagnostic_mode = bool(getattr(config, "scm_diagnostic_mode", False))
            self.scm_diagnostic_num_repeats = int(getattr(config, "scm_diagnostic_num_repeats", 20))
            self.scm_diagnostic_samples = []
            if self.scm_diagnostic_mode:
                num_samples = int(getattr(config, "scm_diagnostic_num_samples", 4))
                raw_dataset = scm_dataset  # un-shuffled, direct index access
                for i in range(min(num_samples, len(raw_dataset))):
                    sample = raw_dataset[i]
                    prompts = sample["prompts"]
                    # raw dataset returns [T, C, H, W] without batch dim.
                    # Add batch dim to match the dataloader-collated shape [B=1, C, H, W].
                    video = sample["ode_latent"][:, -1].unsqueeze(0).clone()
                    audio = None
                    if "ode_audio_latent" in sample and sample["ode_audio_latent"] is not None:
                        audio = sample["ode_audio_latent"][:, -1].unsqueeze(0).clone()
                    self.scm_diagnostic_samples.append((prompts, video, audio))
                if self.is_main_process:
                    print(f"[SCM Diagnostic] Pinned {len(self.scm_diagnostic_samples)} samples, "
                          f"each repeated {self.scm_diagnostic_num_repeats}x "
                          f"(total {len(self.scm_diagnostic_samples) * self.scm_diagnostic_num_repeats} diagnostic steps)")

    def _get_unconditional_dict(self, batch_size: int) -> Dict[str, Any]:
        """Cache unconditional text embeddings by batch size."""
        if not hasattr(self, "_unconditional_dict_cache"):
            self._unconditional_dict_cache = {}

        if batch_size not in self._unconditional_dict_cache:
            unconditional_dict = self.dmd.text_encoder(
                text_prompts=[self.config.negative_prompt] * batch_size
            )
            unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
            self._unconditional_dict_cache[batch_size] = unconditional_dict

        return self._unconditional_dict_cache[batch_size]

    def _prepare_scm_batch(self) -> Tuple[list, torch.Tensor, Optional[torch.Tensor]]:
        """Fetch one faithful SCM batch containing real clean latents."""
        if self.scm_dataloader is None:
            raise RuntimeError("CM batch requested but scm_dataloader is not initialized")

        # Diagnostic mode: repeat fixed samples to measure within/between-sample variance
        if self.scm_diagnostic_mode and self.scm_diagnostic_samples:
            num_samples = len(self.scm_diagnostic_samples)
            num_repeats = self.scm_diagnostic_num_repeats
            # Map step to (sample_idx, repeat_idx)
            diag_step = self.step % (num_samples * num_repeats)
            sample_idx = diag_step % num_samples
            repeat_idx = diag_step // num_samples
            prompts, video, audio = self.scm_diagnostic_samples[sample_idx]
            # Clone to avoid mutating the cached copy
            clean_video = video.clone().to(device=self.device, dtype=self.dtype)
            clean_audio = audio.clone().to(device=self.device, dtype=self.dtype) if audio is not None else None
            return [prompts], clean_video, clean_audio

        batch = next(self.scm_dataloader)
        text_prompts = batch["prompts"]
        clean_video = batch["ode_latent"][:, -1].to(
            device=self.device,
            dtype=self.dtype,
        )
        if "ode_audio_latent" in batch and batch["ode_audio_latent"] is not None:
            clean_audio = batch["ode_audio_latent"][:, -1].to(
                device=self.device,
                dtype=self.dtype,
            )
        else:
            clean_audio = None

        return text_prompts, clean_video, clean_audio

    def _init_benchmark_prompts(self):
        """
        Load fixed benchmark prompts from the training prompt file.

        Reads the first ``benchmark_num_prompts`` lines from ``config.data_path``
        so that every benchmark run uses exactly the same prompts for comparison.

        **All ranks** load the prompts because FSDP-wrapped models require all
        ranks to participate in forward passes during benchmark inference.
        """
        config = self.config
        self.benchmark_enabled = getattr(config, "benchmark_enabled", True)
        self.benchmark_iters = int(getattr(config, "benchmark_iters", config.log_iters))
        self.benchmark_seed = getattr(config, "benchmark_seed", 12345)
        self.benchmark_num_prompts = getattr(config, "benchmark_num_prompts", 2)
        self.benchmark_video_fps = getattr(config, "benchmark_video_fps", 24)
        self.benchmark_audio_sample_rate = getattr(config, "benchmark_audio_sample_rate", 24000)
        self.benchmark_mode = str(getattr(config, "benchmark_mode", "bidirectional")).lower()
        if self.benchmark_mode not in {"bidirectional", "causal"}:
            if self.is_main_process:
                print(f"[Benchmark] Invalid benchmark_mode={self.benchmark_mode}, falling back to bidirectional.")
            self.benchmark_mode = "bidirectional"
        self.benchmark_num_frame_per_block = int(getattr(config, "benchmark_num_frame_per_block", getattr(config, "num_frame_per_block", 3)))
        self.benchmark_use_kv_cache = bool(getattr(config, "benchmark_use_kv_cache", False))
        self.benchmark_clear_cuda_cache_per_round = bool(getattr(config, "benchmark_clear_cuda_cache_per_round", True))
        self.teacher_benchmark_enabled = bool(
            getattr(config, "teacher_benchmark_enabled", self.benchmark_enabled)
        )
        self.teacher_benchmark_num_inference_steps = int(
            getattr(config, "teacher_benchmark_num_inference_steps", 40)
        )
        self.teacher_benchmark_video_guidance_scale = float(
            getattr(
                config,
                "teacher_benchmark_video_guidance_scale",
                getattr(config, "real_video_guidance_scale", 3.0),
            )
        )
        self.teacher_benchmark_audio_guidance_scale = float(
            getattr(
                config,
                "teacher_benchmark_audio_guidance_scale",
                getattr(config, "real_audio_guidance_scale", 7.0),
            )
        )
        default_teacher_benchmark_mode = "rcm_trig" if self.dmd.use_rcm_style_dmd else "native_rf"
        self.teacher_benchmark_mode = str(
            getattr(config, "teacher_benchmark_mode", default_teacher_benchmark_mode)
        ).lower()
        self.teacher_benchmark_include_native_rf_reference = bool(
            getattr(
                config,
                "teacher_benchmark_include_native_rf_reference",
                self.dmd.use_rcm_style_dmd,
            )
        )
        if self.teacher_benchmark_mode not in {"rcm_trig", "native_rf"}:
            if self.is_main_process:
                print(
                    f"[Benchmark] Invalid teacher_benchmark_mode={self.teacher_benchmark_mode}, "
                    f"falling back to {default_teacher_benchmark_mode}."
                )
            self.teacher_benchmark_mode = default_teacher_benchmark_mode
        if self.teacher_benchmark_mode == "rcm_trig" and not self.dmd.use_rcm_style_dmd:
            if self.is_main_process:
                print(
                    "[Benchmark] teacher_benchmark_mode=rcm_trig requested, but current DMD "
                    "style is not trig-based. Falling back to native_rf."
                )
            self.teacher_benchmark_mode = "native_rf"
        self.benchmark_prompts = []

        if self.benchmark_iters <= 0:
            self.benchmark_enabled = False
            if self.is_main_process:
                print("[Benchmark] Disabled because benchmark_iters <= 0.")

        if self.benchmark_mode == "causal" and self.benchmark_use_kv_cache:
            if self.is_main_process:
                print(
                    "[Benchmark] benchmark_use_kv_cache=true requested, but the current "
                    "causal wrapper does not expose a stable KV-cache runtime API. "
                    "Falling back to prefix-rerun autoregressive benchmark mode."
                )
            self.benchmark_use_kv_cache = False

        if not self.benchmark_enabled:
            return

        try:
            # When backward_simulation=false, data_path is an LMDB directory.
            # Use benchmark_prompt_file if specified, otherwise fall back to data_path.
            data_path = getattr(config, "benchmark_prompt_file", None) or config.data_path
            with open(data_path, "r", encoding="utf-8") as f:
                all_prompts = [line.strip() for line in f if line.strip()]
            self.benchmark_prompts = all_prompts[: self.benchmark_num_prompts]
            if self.is_main_process:
                print(f"[Benchmark] Loaded {len(self.benchmark_prompts)} prompts from {data_path}")
                print(f"[Benchmark] mode={self.benchmark_mode}, kv_cache={self.benchmark_use_kv_cache}, frames_per_block={self.benchmark_num_frame_per_block}")
                if self.teacher_benchmark_enabled:
                    print(
                        "[Benchmark] teacher reference enabled: "
                        f"{self.teacher_benchmark_num_inference_steps} steps, "
                        f"mode={self.teacher_benchmark_mode}, "
                        f"video_cfg={self.teacher_benchmark_video_guidance_scale}, "
                        f"audio_cfg={self.teacher_benchmark_audio_guidance_scale}"
                    )
                    if self.teacher_benchmark_include_native_rf_reference and self.teacher_benchmark_mode != "native_rf":
                        print("[Benchmark] teacher native RF comparison reference enabled.")
                for i, p in enumerate(self.benchmark_prompts):
                    print(f"  [{i}] {p[:80]}{'...' if len(p) > 80 else ''}")
        except Exception as e:
            if self.is_main_process:
                print(f"[Benchmark] Failed to load prompts: {e}")
            self.benchmark_enabled = False

    def _vae_to_device(self):
        """Move VAEs to GPU for decoding (visualization / benchmark)."""
        if self.dmd.video_vae is not None:
            self.dmd.video_vae = self.dmd.video_vae.to(device=self.device)
        if self.dmd.audio_vae is not None:
            self.dmd.audio_vae = self.dmd.audio_vae.to(device=self.device)

    def _save_prompt_file(self, output_dir: str) -> str:
        """Save benchmark prompts in sample index order."""
        prompt_path = os.path.join(output_dir, "prompts.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            for prompt in self.benchmark_prompts:
                f.write(f"{prompt}\n")
        return prompt_path

    def _vae_to_cpu(self):
        """Offload VAEs back to CPU to free GPU memory."""
        if self.dmd.video_vae is not None:
            self.dmd.video_vae = self.dmd.video_vae.to(device="cpu")
        if self.dmd.audio_vae is not None:
            self.dmd.audio_vae = self.dmd.audio_vae.to(device="cpu")
        torch.cuda.empty_cache()

    def save(self):
        """Save checkpoint with optimizer states for seamless resume."""
        print("Gathering distributed model states...")

        generator_state_dict = fsdp_state_dict(self.dmd.generator)
        critic_state_dict = fsdp_state_dict(self.dmd.fake_score) if self.dmd.fake_score is not None else None
        generator_ema_state_dict = self.dmd.ema_state_dict()

        state_dict = {
            "generator": generator_state_dict,
            "critic": critic_state_dict,
            "generator_ema": generator_ema_state_dict,
            "step": self.step,
            "completed_step": self.step,
            "next_step": self.step + 1,
        }

        if self.is_main_process:
            checkpoints_dir = os.path.join(self.output_path, "checkpoints")
            checkpoint_dir = os.path.join(
                checkpoints_dir,
                f"checkpoint_{self.step:06d}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            save_path = os.path.join(checkpoint_dir, "model.pth")
            torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)
            print(f"Checkpoint saved to {save_path}")

    @staticmethod
    def _to_scalar(value):
        """Convert tensor-like values to Python scalars for WandB logging."""
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.item()
            return value.detach().float().mean().item()
        return value

    def _compute_layerwise_grad_norms(self, module, prefix):
        """
        Compute per-layer gradient L2 norm for monitoring.

        Aggregation strategy:
        - For transformer blocks, log at block granularity: blocks.{idx}
        - For others, log at up-to-2-level module granularity.
        """
        layer_sq_norm = {}
        fsdp_prefix = "_fsdp_wrapped_module."

        for name, param in module.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue

            normalized_name = name[len(fsdp_prefix):] if name.startswith(fsdp_prefix) else name
            parts = normalized_name.split(".")
            if len(parts) >= 3 and parts[1] == "blocks" and parts[2].isdigit():
                layer_key = f"blocks.{parts[2]}"
            elif len(parts) >= 2:
                layer_key = f"{parts[0]}.{parts[1]}"
            else:
                layer_key = parts[0]

            grad_sq = param.grad.detach().float().pow(2).sum().item()
            layer_sq_norm[layer_key] = layer_sq_norm.get(layer_key, 0.0) + grad_sq

        return {
            f"train/{prefix}_grad_norm/{k}": math.sqrt(v) for k, v in layer_sq_norm.items()
        }

    def train_one_step(self):
        """Execute one training step."""
        # Set all models to eval mode first (disables dropout/batchnorm),
        # then re-enable train mode for generator and fake_score so that
        # gradient checkpointing remains active during their gradient-enabled
        # forward passes. This is critical for the 19B model's memory footprint.
        # The real_score (teacher) stays in eval mode since it's frozen.
        #
        # For backward simulation's @torch.no_grad() forward passes, the
        # generator is temporarily switched to eval() inside
        # _consistency_backward_simulation() to avoid FSDP+checkpoint conflicts.
        self.dmd.eval()
        self.dmd.generator.train()
        if self.dmd.critic_enabled:
            self.dmd.fake_score.train()

        # Pass current step to DMD for step-dependent loss weighting
        self.dmd.current_step = self.step

        config = self.config
        TRAIN_GENERATOR = (
            True if not self.dmd.critic_enabled else self.step % config.dfake_gen_update_ratio == 0
        )
        LOG_LAYERWISE_GRAD = self.step % self.layerwise_grad_log_interval == 0

        # Periodic cache clearing
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Get batch
        need_dmd_inputs = self.dmd.dmd_enabled or self.dmd.critic_enabled
        text_prompts = None
        clean_video = None
        clean_audio = None
        conditional_dict = None
        unconditional_dict = None

        if need_dmd_inputs:
            if not self.backward_simulation:
                batch = next(self.dataloader)
                text_prompts = batch["prompts"]
                # ODE latent format: [B, T, F, C, H, W], take last timestep (clean)
                clean_video = batch["ode_latent"][:, -1].to(
                    device=self.device,
                    dtype=self.dtype,
                )
                # Audio ODE latent format: [B, T, F_a, C], take last timestep (clean)
                if "ode_audio_latent" in batch and batch["ode_audio_latent"] is not None:
                    clean_audio = batch["ode_audio_latent"][:, -1].to(
                        device=self.device,
                        dtype=self.dtype,
                    )
                else:
                    clean_audio = None
            else:
                text_prompts = next(self.dataloader)

        scm_clean_video = None
        scm_clean_audio = None
        scm_conditional_dict = None
        scm_unconditional_dict = None

        batch_size = len(text_prompts) if text_prompts is not None else None
        with torch.no_grad():
            if need_dmd_inputs:
                conditional_dict = self.dmd.text_encoder(text_prompts=text_prompts)
                unconditional_dict = self._get_unconditional_dict(batch_size)

            if self.scm_enabled or self.dcm_enabled:
                scm_text_prompts, scm_clean_video, scm_clean_audio = self._prepare_scm_batch()
                scm_conditional_dict = self.dmd.text_encoder(text_prompts=scm_text_prompts)
                scm_unconditional_dict = self._get_unconditional_dict(len(scm_text_prompts))
                if batch_size is None:
                    batch_size = len(scm_text_prompts)

        if batch_size is None:
            raise ValueError("Could not determine batch size for the current training step")

        # Compute latent shapes
        video_shape, audio_shape = compute_latent_shapes(
            num_frames=config.num_frames,
            video_height=config.video_height,
            video_width=config.video_width,
            batch_size=batch_size,
        )

        # Train generator
        if TRAIN_GENERATOR:
            generator_loss, generator_log_dict = self.dmd.generator_loss(
                video_shape=video_shape,
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_video=clean_video,
                clean_audio=clean_audio,
                scm_clean_video=scm_clean_video,
                scm_clean_audio=scm_clean_audio,
                scm_conditional_dict=scm_conditional_dict,
                scm_unconditional_dict=scm_unconditional_dict,
            )

            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_layerwise_grad_dict = (
                self._compute_layerwise_grad_norms(self.dmd.generator, "generator")
                if LOG_LAYERWISE_GRAD else {}
            )
            # Use FSDP's clip_grad_norm_ if available, otherwise fall back to torch utility
            if hasattr(self.dmd.generator, 'clip_grad_norm_'):
                generator_grad_norm = self.dmd.generator.clip_grad_norm_(self.max_grad_norm)
            else:
                generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.dmd.generator.parameters(), self.max_grad_norm
                )
            self.generator_optimizer.step()

            # Update EMA (exponential moving average) of generator weights
            ema_update_iters = int(getattr(self.config, "ema_update_iters", 50))
            if self.step > 0 and self.step % ema_update_iters == 0:
                self.dmd.update_ema(self.step)

            # ---- Memory cleanup between generator and critic training ----
            # Save scalar metrics before freeing the computation graph.
            # This is critical because step 0 first allocates Adam optimizer
            # states (momentum + variance ≈ 2× param size), and the remaining
            # graph/activation memory must be released before critic training.
            generator_loss_val = generator_loss.item()
            generator_grad_norm_val = generator_grad_norm.item()
            gen_grad_norm_video = generator_log_dict.get("dmdtrain_gradient_norm_video", 0)
            gen_grad_norm_audio = generator_log_dict.get("dmdtrain_gradient_norm_audio", 0)

            del generator_loss, generator_grad_norm
            torch.cuda.empty_cache()
        else:
            generator_log_dict = {}
            generator_loss_val = None
            generator_grad_norm_val = None
            gen_grad_norm_video = 0
            gen_grad_norm_audio = 0
            generator_layerwise_grad_dict = {}

        # Train critic
        if self.dmd.critic_enabled:
            critic_loss, critic_log_dict = self.dmd.critic_loss(
                video_shape=video_shape,
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_video=clean_video,
                clean_audio=clean_audio,
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_layerwise_grad_dict = (
                self._compute_layerwise_grad_norms(self.dmd.fake_score, "critic")
                if LOG_LAYERWISE_GRAD else {}
            )
            # Use FSDP's clip_grad_norm_ if available, otherwise fall back to torch utility
            if hasattr(self.dmd.fake_score, 'clip_grad_norm_'):
                critic_grad_norm = self.dmd.fake_score.clip_grad_norm_(self.max_grad_norm)
            else:
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.dmd.fake_score.parameters(), self.max_grad_norm
                )
            self.critic_optimizer.step()
        else:
            critic_loss = None
            critic_log_dict = {}
            critic_layerwise_grad_dict = {}
            critic_grad_norm = None

        # Benchmark: periodic 4-step inference visualization
        # ALL ranks must participate because FSDP forward passes require
        # collective communication across all ranks.
        benchmark_min_step = int(getattr(config, "benchmark_min_step", 1))
        BENCHMARK = (
            self.benchmark_enabled
            and len(self.benchmark_prompts) > 0
            and self.step >= benchmark_min_step
            and self.step % self.benchmark_iters == 0
            and not getattr(config, "no_visualize", False)
        )

        if BENCHMARK:
            self._run_benchmark_and_log()

        # Logging (all scalars, no GPU tensors)
        if self.is_main_process:
            wandb_dict = {
                "train/critic_loss": self._to_scalar(critic_loss) if critic_loss is not None else 0.0,
                "train/critic_grad_norm": self._to_scalar(critic_grad_norm) if critic_grad_norm is not None else 0.0,
            }

            # Add per-component critic losses from log_dict
            wandb_dict.update({
                f"train/{k}": self._to_scalar(v) for k, v in critic_log_dict.items()
            })
            wandb_dict.update(critic_layerwise_grad_dict)

            if TRAIN_GENERATOR:
                wandb_dict.update({
                    "train/generator_loss": generator_loss_val,
                    "train/generator_grad_norm": generator_grad_norm_val,
                    "train/dmdtrain_gradient_norm_video": gen_grad_norm_video,
                    "train/dmdtrain_gradient_norm_audio": gen_grad_norm_audio,
                })
                wandb_dict.update(generator_layerwise_grad_dict)
                for gk, gv in generator_log_dict.items():
                    wandb_dict[f"train/{gk}"] = self._to_scalar(gv)

            wandb_dict["train/lr_generator"] = self.generator_optimizer.param_groups[0]["lr"]
            wandb_dict["train/lr_critic"] = (
                self.critic_optimizer.param_groups[0]["lr"] if self.critic_optimizer is not None else 0.0
            )

            # Diagnostic mode: tag each step with sample/repeat index
            if self.scm_diagnostic_mode and self.scm_diagnostic_samples:
                num_samples = len(self.scm_diagnostic_samples)
                num_repeats = self.scm_diagnostic_num_repeats
                diag_step = self.step % (num_samples * num_repeats)
                wandb_dict["train/diag_sample_idx"] = diag_step % num_samples
                wandb_dict["train/diag_repeat_idx"] = diag_step // num_samples

            self._safe_wandb_log(wandb_dict, step=self.step)

            if self.log_iters > 0 and self.step % self.log_iters == 0:
                summary_parts = [
                    f"step={self.step}",
                ]
                if self.scm_diagnostic_mode and self.scm_diagnostic_samples:
                    diag_s = self.step % (len(self.scm_diagnostic_samples) * self.scm_diagnostic_num_repeats)
                    summary_parts.append(
                        f"diag_s={diag_s % len(self.scm_diagnostic_samples)}/"
                        f"r={diag_s // len(self.scm_diagnostic_samples)}"
                    )
                summary_parts += [
                    f"gen_loss={wandb_dict.get('train/generator_loss', 0.0):.6f}",
                    f"critic_loss={wandb_dict.get('train/critic_loss', 0.0):.6f}",
                    f"lr_g={wandb_dict.get('train/lr_generator', 0.0):.2e}",
                    f"grad_g={wandb_dict.get('train/generator_grad_norm', 0.0):.4f}",
                ]
                if "train/alignment/scm_rf_time_mean" in wandb_dict:
                    summary_parts.append(
                        f"rf_t={wandb_dict['train/alignment/scm_rf_time_mean']:.4f}"
                    )
                if "train/alignment/scm_trig_time_mean" in wandb_dict:
                    summary_parts.append(
                        f"trig_t={wandb_dict['train/alignment/scm_trig_time_mean']:.4f}"
                    )
                if "train/scm_video_loss" in wandb_dict:
                    summary_parts.append(
                        f"scm_video_loss={wandb_dict['train/scm_video_loss']:.6f}"
                    )
                if "train/scm_audio_loss" in wandb_dict:
                    summary_parts.append(
                        f"scm_audio_loss={wandb_dict['train/scm_audio_loss']:.6f}"
                    )
                if "train/dcm_video_loss" in wandb_dict:
                    summary_parts.append(
                        f"dcm_video_loss={wandb_dict['train/dcm_video_loss']:.6f}"
                    )
                if "train/dcm_audio_loss" in wandb_dict:
                    summary_parts.append(
                        f"dcm_audio_loss={wandb_dict['train/dcm_audio_loss']:.6f}"
                    )
                if "train/scm_warmup_ratio" in wandb_dict:
                    summary_parts.append(
                        f"scm_warmup={wandb_dict['train/scm_warmup_ratio']:.4f}"
                    )
                if "train/scm_g_normalization_per_modality" in wandb_dict:
                    mode = (
                        "per_modality"
                        if wandb_dict["train/scm_g_normalization_per_modality"] > 0.5
                        else "joint"
                    )
                    summary_parts.append(f"scm_g_norm={mode}")
                if "train/video_loss_weight" in wandb_dict:
                    summary_parts.append(
                        f"video_w={wandb_dict['train/video_loss_weight']:.3f}"
                    )
                if "train/audio_loss_weight" in wandb_dict:
                    summary_parts.append(
                        f"audio_w={wandb_dict['train/audio_loss_weight']:.3f}"
                    )
                if "train/alignment/dcm_trig_t0_mean" in wandb_dict:
                    summary_parts.append(
                        f"dcm_t0={wandb_dict['train/alignment/dcm_trig_t0_mean']:.4f}"
                    )
                if "train/alignment/dcm_trig_tN_mean" in wandb_dict:
                    summary_parts.append(
                        f"dcm_tN={wandb_dict['train/alignment/dcm_trig_tN_mean']:.4f}"
                    )
                if "train/alignment/dcm_video_x0_gap" in wandb_dict:
                    summary_parts.append(
                        f"dcm_video_gap={wandb_dict['train/alignment/dcm_video_x0_gap']:.6f}"
                    )
                if "train/alignment/dcm_audio_x0_gap" in wandb_dict:
                    summary_parts.append(
                        f"dcm_audio_gap={wandb_dict['train/alignment/dcm_audio_x0_gap']:.6f}"
                    )
                if "train/alignment/scm_video_direction_gap" in wandb_dict:
                    summary_parts.append(
                        f"video_gap={wandb_dict['train/alignment/scm_video_direction_gap']:.6f}"
                    )
                if "train/alignment/scm_video_tangent_norm" in wandb_dict:
                    summary_parts.append(
                        f"video_tangent={wandb_dict['train/alignment/scm_video_tangent_norm']:.6f}"
                    )
                if "train/alignment/scm_video_tangent_p99" in wandb_dict:
                    summary_parts.append(
                        f"video_tp99={wandb_dict['train/alignment/scm_video_tangent_p99']:.4f}"
                    )
                if "train/alignment/scm_video_teacher_norm" in wandb_dict:
                    summary_parts.append(
                        f"video_teacher={wandb_dict['train/alignment/scm_video_teacher_norm']:.6f}"
                    )
                if "train/alignment/scm_video_student_norm" in wandb_dict:
                    summary_parts.append(
                        f"video_student={wandb_dict['train/alignment/scm_video_student_norm']:.6f}"
                    )
                if "train/alignment/scm_video_g_norm" in wandb_dict:
                    summary_parts.append(
                        f"video_g={wandb_dict['train/alignment/scm_video_g_norm']:.6f}"
                    )
                if "train/alignment/scm_video_g_post_norm" in wandb_dict:
                    summary_parts.append(
                        f"video_g_post={wandb_dict['train/alignment/scm_video_g_post_norm']:.6f}"
                    )
                if "train/alignment/scm_video_loss_share" in wandb_dict:
                    summary_parts.append(
                        f"video_share={wandb_dict['train/alignment/scm_video_loss_share']:.4f}"
                    )
                if "train/alignment/scm_video_nan_ratio" in wandb_dict:
                    summary_parts.append(
                        f"video_nan={wandb_dict['train/alignment/scm_video_nan_ratio']:.4f}"
                    )
                if "train/alignment/scm_audio_direction_gap" in wandb_dict:
                    summary_parts.append(
                        f"audio_gap={wandb_dict['train/alignment/scm_audio_direction_gap']:.6f}"
                    )
                if "train/alignment/scm_audio_tangent_norm" in wandb_dict:
                    summary_parts.append(
                        f"audio_tangent={wandb_dict['train/alignment/scm_audio_tangent_norm']:.6f}"
                    )
                if "train/alignment/scm_audio_tangent_p99" in wandb_dict:
                    summary_parts.append(
                        f"audio_tp99={wandb_dict['train/alignment/scm_audio_tangent_p99']:.4f}"
                    )
                if "train/alignment/scm_audio_teacher_norm" in wandb_dict:
                    summary_parts.append(
                        f"audio_teacher={wandb_dict['train/alignment/scm_audio_teacher_norm']:.6f}"
                    )
                if "train/alignment/scm_audio_student_norm" in wandb_dict:
                    summary_parts.append(
                        f"audio_student={wandb_dict['train/alignment/scm_audio_student_norm']:.6f}"
                    )
                if "train/alignment/scm_audio_g_norm" in wandb_dict:
                    summary_parts.append(
                        f"audio_g={wandb_dict['train/alignment/scm_audio_g_norm']:.6f}"
                    )
                if "train/alignment/scm_audio_g_post_norm" in wandb_dict:
                    summary_parts.append(
                        f"audio_g_post={wandb_dict['train/alignment/scm_audio_g_post_norm']:.6f}"
                    )
                if "train/alignment/scm_audio_loss_share" in wandb_dict:
                    summary_parts.append(
                        f"audio_share={wandb_dict['train/alignment/scm_audio_loss_share']:.4f}"
                    )
                if "train/alignment/scm_audio_nan_ratio" in wandb_dict:
                    summary_parts.append(
                        f"audio_nan={wandb_dict['train/alignment/scm_audio_nan_ratio']:.4f}"
                    )
                print("[Train] " + " | ".join(summary_parts), flush=True)
                now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                print(
                    f"[TrainTimestamp] step={self.step} | utc={now_utc}",
                    flush=True,
                )
                if self.train_log_all_scalars:
                    scalar_parts = []
                    for key in sorted(wandb_dict):
                        value = wandb_dict[key]
                        if isinstance(value, (int, float)):
                            scalar_parts.append(f"{key}={value:.8g}")
                    if scalar_parts:
                        print("[TrainFull] " + " | ".join(scalar_parts), flush=True)

        del critic_loss, critic_grad_norm
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _run_benchmark_and_log(self):
        """
        Run 4-step inference on fixed benchmark prompts, distributing work
        across all ranks for maximum parallelism.

        **All ranks** must call this method because the generator and text
        encoder are FSDP-wrapped and require collective communication.

        Flow (per round, one prompt per rank):
        1. ALL ranks: encode 1 prompt each (FSDP collective, batch_size=1)
        2. ALL ranks: run inference pipeline (FSDP collective, batch_size=1)
        3. ALL ranks: decode video/audio with local VAE, save mp4 to shared FS
        4. Rank 0: collect all saved files, log to WandB

        This distributes N prompts across W ranks in ceil(N/W) rounds,
        reducing per-rank memory vs the old single-rank-decodes-all approach.

        RNG is forked per prompt for reproducibility without affecting training.
        """
        from ltx_distillation.inference.bidirectional_pipeline import (
            BidirectionalAVInferencePipeline,
        )
        from ltx_distillation.inference.causal_pipeline import (
            CausalAVInferencePipeline,
        )

        config = self.config

        # Free training intermediate memory before benchmark
        torch.cuda.empty_cache()

        if self.teacher_benchmark_enabled and self.step == 0:
            self._run_teacher_reference_and_log()

        num_prompts = len(self.benchmark_prompts)
        num_rounds = math.ceil(num_prompts / self.world_size)

        if self.is_main_process:
            print(
                f"[Benchmark] Step {self.step}: generating {num_prompts} samples "
                f"({self.benchmark_mode} mode) across {self.world_size} ranks "
                f"({num_rounds} round(s))..."
            )

        step_dir = os.path.join(
            self.output_path, "benchmark", "student", f"step_{self.step:07d}"
        )
        os.makedirs(step_dir, exist_ok=True)

        video_shape_single, audio_shape_single = compute_latent_shapes(
            num_frames=config.num_frames,
            video_height=config.video_height,
            video_width=config.video_width,
            batch_size=1,
        )

        # Keep Stage 3 benchmark aligned with the Stage-2 ODE benchmark:
        # temporarily switch the FSDP-wrapped generator to eval() under no_grad,
        # then restore the previous mode afterwards.
        was_training = self.dmd.generator.training
        self.dmd.generator.eval()

        # Switch to EMA weights for inference (critical for quality)
        ema_state = None
        ema_sd = self.dmd.ema_state_dict()
        if self.dmd.ema_enabled and ema_sd is not None:
            ema_state = {k: v.cpu() for k, v in self.dmd.generator.state_dict().items()}
            self.dmd.generator.load_state_dict(ema_sd)

        try:
            if self.benchmark_mode == "causal":
                pipeline = CausalAVInferencePipeline(
                    generator=self.dmd.generator,
                    add_noise_fn=self.dmd.add_noise,
                    denoising_sigmas=self.dmd.denoising_sigmas,
                    num_frame_per_block=self.benchmark_num_frame_per_block,
                    use_kv_cache=self.benchmark_use_kv_cache,
                    clear_cuda_cache_per_round=self.benchmark_clear_cuda_cache_per_round,
                )
            else:
                pipeline = BidirectionalAVInferencePipeline(
                    generator=self.dmd.generator,
                    add_noise_fn=self.dmd.add_noise,
                    denoising_sigmas=self.dmd.denoising_sigmas,
                    use_trigflow=self.dmd.use_rcm_style_dmd,
                )

            self._vae_to_device()

            # Timing: wall-clock for full benchmark, and per-video generation time
            benchmark_wall_start = time.perf_counter()
            my_total_generate_seconds = 0.0

            for round_idx in range(num_rounds):
                prompt_idx = round_idx * self.world_size + self.global_rank
                has_real_prompt = prompt_idx < num_prompts

                if has_real_prompt:
                    my_prompt = [self.benchmark_prompts[prompt_idx]]
                else:
                    my_prompt = [self.benchmark_prompts[0]]

                with torch.no_grad():
                    conditional_dict = self.dmd.text_encoder(text_prompts=my_prompt)

                prompt_seed = self.benchmark_seed + prompt_idx
                with torch.random.fork_rng(devices=[self.device]):
                    torch.manual_seed(prompt_seed)
                    torch.cuda.manual_seed(prompt_seed)

                    gen_start = time.perf_counter()
                    video_latent, audio_latent = pipeline.generate(
                        video_shape=tuple(video_shape_single),
                        audio_shape=tuple(audio_shape_single),
                        conditional_dict=conditional_dict,
                    )
                    gen_elapsed = time.perf_counter() - gen_start
                    my_total_generate_seconds += gen_elapsed

                if has_real_prompt:
                    self._decode_and_save_sample(
                        video_latent=video_latent,
                        audio_latent=audio_latent,
                        prompt_idx=prompt_idx,
                        step_dir=step_dir,
                    )

                del video_latent, audio_latent, conditional_dict
                if self.benchmark_clear_cuda_cache_per_round:
                    torch.cuda.empty_cache()

                barrier()
        finally:
            # Restore training weights if EMA was used
            if ema_state is not None:
                self.dmd.generator.load_state_dict(ema_state)
                del ema_state
            if was_training:
                self.dmd.generator.train()

        benchmark_wall_elapsed = time.perf_counter() - benchmark_wall_start

        # Gather total generation time from all ranks (each rank sums its own generate times)
        total_generate_tensor = torch.tensor(
            [my_total_generate_seconds], device=self.device, dtype=torch.float64
        )
        dist.all_reduce(total_generate_tensor, op=dist.ReduceOp.SUM)
        total_generate_seconds = total_generate_tensor.item()

        self._vae_to_cpu()

        barrier()

        # ---- Rank 0: log all samples to WandB and print benchmark timing ----
        if self.is_main_process:
            time_per_video_wall = benchmark_wall_elapsed / max(1, num_prompts)
            time_per_video_generate = total_generate_seconds / max(1, num_prompts)

            benchmark_wandb_dict = {}
            prompt_rows = []

            for idx in range(num_prompts):
                sample_path = os.path.join(step_dir, f"sample_{idx}.mp4")
                if os.path.exists(sample_path):
                    benchmark_wandb_dict[f"benchmark/sample_{idx}"] = wandb.Video(
                        sample_path, fps=self.benchmark_video_fps, format="mp4"
                    )
                    prompt_rows.append(
                        [idx, self.benchmark_prompts[idx], sample_path]
                    )

            if prompt_rows:
                table = wandb.Table(
                    columns=["index", "prompt", "local_path"],
                    data=prompt_rows,
                )
                benchmark_wandb_dict["benchmark/prompt_table"] = table

            if benchmark_wandb_dict:
                self._safe_wandb_log(benchmark_wandb_dict, step=self.step)

            # One line: timing + save path (flush so it always appears in logs)
            print(
                f"[Benchmark] Step {self.step}: {num_prompts} video(s) | "
                f"wall {benchmark_wall_elapsed:.2f}s ({time_per_video_wall:.2f}s/video) | "
                f"generate {total_generate_seconds:.2f}s ({time_per_video_generate:.2f}s/video) | "
                f"saved to {step_dir}",
                flush=True,
            )

        barrier()

    @torch.no_grad()
    def _run_teacher_reference_and_log(self):
        teacher_runs = [
            (
                self.teacher_benchmark_mode,
                os.path.join(self.output_path, "benchmark", "teacher"),
                "benchmark_teacher",
                "Teacher",
            )
        ]
        if self.teacher_benchmark_include_native_rf_reference and self.teacher_benchmark_mode != "native_rf":
            teacher_runs.append(
                (
                    "native_rf",
                    os.path.join(self.output_path, "benchmark", "teacher_native_rf"),
                    "benchmark_teacher_native_rf",
                    "Teacher-NativeRF",
                )
            )

        for mode, ref_dir, wandb_prefix, label in teacher_runs:
            self._run_teacher_reference_and_log_single(
                mode=mode,
                ref_dir=ref_dir,
                wandb_prefix=wandb_prefix,
                label=label,
            )

    @torch.no_grad()
    def _run_teacher_reference_and_log_single(
        self,
        mode: str,
        ref_dir: str,
        wandb_prefix: str,
        label: str,
    ):
        """
        Run one teacher-reference benchmark at step 0.

        The teacher is frozen, so we only generate this reference once and
        compare later student checkpoints against the fixed outputs.
        """
        config = self.config
        num_prompts = len(self.benchmark_prompts)
        num_rounds = math.ceil(num_prompts / self.world_size)
        os.makedirs(ref_dir, exist_ok=True)
        prompt_path = None

        if self.is_main_process:
            prompt_path = self._save_prompt_file(ref_dir)
            print(
                f"[Benchmark][{label}] Step 0: generating "
                f"{num_prompts} reference sample(s) with "
                f"{self.teacher_benchmark_num_inference_steps}-step teacher "
                f"({mode}) "
                f"across {self.world_size} ranks ({num_rounds} round(s))... "
                f"prompts saved to {prompt_path}"
            )

        video_shape_single, audio_shape_single = compute_latent_shapes(
            num_frames=config.num_frames,
            video_height=config.video_height,
            video_width=config.video_width,
            batch_size=1,
        )

        was_teacher_training = self.dmd.real_score.training
        was_text_encoder_training = self.dmd.text_encoder.training
        self.dmd.real_score.eval()
        self.dmd.text_encoder.eval()

        self._vae_to_device()

        teacher_scheduler = LTX2Scheduler()
        sigmas = teacher_scheduler.execute(
            steps=self.teacher_benchmark_num_inference_steps
        ).to(device=self.device, dtype=self.dtype)

        teacher_wall_start = time.perf_counter()
        my_total_generate_seconds = 0.0

        try:
            for round_idx in range(num_rounds):
                prompt_idx = round_idx * self.world_size + self.global_rank
                has_real_prompt = prompt_idx < num_prompts

                if has_real_prompt:
                    my_prompt = self.benchmark_prompts[prompt_idx]
                else:
                    my_prompt = self.benchmark_prompts[0]

                conditional_dict = self.dmd.text_encoder(text_prompts=[my_prompt])
                unconditional_dict = self.dmd.text_encoder(
                    text_prompts=[config.negative_prompt]
                )

                prompt_seed = self.benchmark_seed + prompt_idx
                with torch.random.fork_rng(devices=[self.device]):
                    torch.manual_seed(prompt_seed)
                    torch.cuda.manual_seed(prompt_seed)

                    gen_start = time.perf_counter()
                    video_latent, audio_latent = self._generate_teacher_reference_sample(
                        video_shape=tuple(video_shape_single),
                        audio_shape=tuple(audio_shape_single),
                        sigmas=sigmas,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        mode=mode,
                    )
                    gen_elapsed = time.perf_counter() - gen_start
                    my_total_generate_seconds += gen_elapsed

                if has_real_prompt:
                    self._decode_and_save_sample(
                        video_latent=video_latent,
                        audio_latent=audio_latent,
                        prompt_idx=prompt_idx,
                        step_dir=ref_dir,
                    )

                del video_latent, audio_latent, conditional_dict, unconditional_dict
                if self.benchmark_clear_cuda_cache_per_round:
                    torch.cuda.empty_cache()

                barrier()
        finally:
            if was_teacher_training:
                self.dmd.real_score.train()
            if was_text_encoder_training:
                self.dmd.text_encoder.train()

        teacher_wall_elapsed = time.perf_counter() - teacher_wall_start

        total_generate_tensor = torch.tensor(
            [my_total_generate_seconds], device=self.device, dtype=torch.float64
        )
        dist.all_reduce(total_generate_tensor, op=dist.ReduceOp.SUM)
        total_generate_seconds = total_generate_tensor.item()

        self._vae_to_cpu()
        barrier()

        if self.is_main_process:
            time_per_video_wall = teacher_wall_elapsed / max(1, num_prompts)
            time_per_video_generate = total_generate_seconds / max(1, num_prompts)

            teacher_wandb_dict = {}
            prompt_rows = []

            for idx in range(num_prompts):
                sample_path = os.path.join(ref_dir, f"sample_{idx}.mp4")
                if os.path.exists(sample_path):
                    teacher_wandb_dict[f"{wandb_prefix}/sample_{idx}"] = wandb.Video(
                        sample_path, fps=self.benchmark_video_fps, format="mp4"
                    )
                    prompt_rows.append(
                        [idx, self.benchmark_prompts[idx], sample_path]
                    )

            if prompt_rows:
                teacher_wandb_dict[f"{wandb_prefix}/prompt_table"] = wandb.Table(
                    columns=["index", "prompt", "local_path"],
                    data=prompt_rows,
                )

            if teacher_wandb_dict:
                self._safe_wandb_log(teacher_wandb_dict, step=self.step)

            print(
                f"[Benchmark][{label}] Step 0: "
                f"{num_prompts} video(s) | "
                f"wall {teacher_wall_elapsed:.2f}s ({time_per_video_wall:.2f}s/video) | "
                f"generate {total_generate_seconds:.2f}s ({time_per_video_generate:.2f}s/video) | "
                f"saved to {ref_dir}"
                + (f" | prompts {prompt_path}" if prompt_path is not None else ""),
                flush=True,
            )

        barrier()

    @torch.no_grad()
    def _generate_teacher_reference_sample(
        self,
        video_shape: Tuple[int, ...],
        audio_shape: Tuple[int, ...],
        sigmas: torch.Tensor,
        conditional_dict,
        unconditional_dict,
        mode: str = "native_rf",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate one sample with the frozen teacher under the requested benchmark mode."""
        B = video_shape[0]
        F_v = video_shape[1]
        F_a = audio_shape[1]

        video = torch.randn(video_shape, device=self.device, dtype=self.dtype)
        audio = torch.randn(audio_shape, device=self.device, dtype=self.dtype)
        if mode == "native_rf":
            teacher_forward = (
                self.dmd.real_score.forward_rf
                if self.dmd.use_rcm_style_dmd and hasattr(self.dmd.real_score, "forward_rf")
                else self.dmd.real_score
            )
            schedule = sigmas
        elif mode == "rcm_trig":
            teacher_forward = self.dmd.real_score
            schedule = rf_to_trig_time(sigmas.double()).to(device=self.device, dtype=self.dtype)
        else:
            raise ValueError(f"Unsupported teacher benchmark mode: {mode}")

        for i in range(len(schedule) - 1):
            sigma = schedule[i]
            video_sigma = sigma * torch.ones([B, F_v], device=self.device, dtype=self.dtype)
            audio_sigma = sigma * torch.ones([B, F_a], device=self.device, dtype=self.dtype)

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

            video_x0 = video_x0_uncond + self.teacher_benchmark_video_guidance_scale * (
                video_x0_cond - video_x0_uncond
            )
            audio_x0 = audio_x0_uncond + self.teacher_benchmark_audio_guidance_scale * (
                audio_x0_cond - audio_x0_uncond
            )

            sigma_next = schedule[i + 1]
            if sigma_next > 0 and sigma > 0:
                if mode == "native_rf":
                    video_velocity = (video.float() - video_x0.float()) / sigma.float()
                    audio_velocity = (audio.float() - audio_x0.float()) / sigma.float()
                    dt = (sigma_next - sigma).float()
                    video = (video.float() + video_velocity * dt).to(self.dtype)
                    audio = (audio.float() + audio_velocity * dt).to(self.dtype)
                else:
                    next_t_video = sigma_next.view(1, 1, 1, 1, 1).to(device=self.device, dtype=self.dtype)
                    next_t_audio = sigma_next.view(1, 1, 1).to(device=self.device, dtype=self.dtype)
                    video = (
                        torch.cos(next_t_video) * video_x0
                        + torch.sin(next_t_video) * torch.randn_like(video)
                    ).to(self.dtype)
                    audio = (
                        torch.cos(next_t_audio) * audio_x0
                        + torch.sin(next_t_audio) * torch.randn_like(audio)
                    ).to(self.dtype)
            else:
                video = video_x0
                audio = audio_x0

        return video, audio

    def _decode_and_save_sample(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        prompt_idx: int,
        step_dir: str,
    ):
        """
        Decode one (video, audio) latent pair and save as mp4 with audio.

        Called by every rank that owns a real benchmark prompt.  VAEs must
        already be on GPU (via ``_vae_to_device``) before calling this.
        """
        # Decode video → pixel  [1, C, F, H, W]  →  [0, 1]
        video_pixel = self.dmd.video_vae.decode_to_pixel(video_latent)

        # Decode audio → waveform  [1, 1, samples]
        audio_waveform = None
        try:
            audio_waveform = self.dmd.audio_vae.decode_to_waveform(audio_latent)
        except Exception as e:
            print(
                f"[Benchmark][Rank {self.global_rank}] Audio decode failed "
                f"for prompt {prompt_idx}: {e}"
            )

        # Prepare video tensor: -> uint8 [F, H, W, C]
        vid = video_pixel[0]  # [C, F, H, W]
        if vid.shape[0] == 3:
            vid = vid.permute(1, 0, 2, 3)  # -> [F, C, H, W]
        vid = vid.permute(0, 2, 3, 1)  # -> [F, H, W, C]
        vid = (vid.clamp(0, 1) * 255).cpu().to(torch.uint8)

        sample_path = os.path.join(step_dir, f"sample_{prompt_idx}.mp4")

        # Try writing mp4 with embedded audio track
        written_with_audio = False
        if audio_waveform is not None:
            try:
                wav_float = audio_waveform[0].cpu().float()  # [1, samples]
                from torchvision.io import write_video

                write_video(
                    sample_path,
                    vid,
                    fps=self.benchmark_video_fps,
                    audio_array=wav_float,
                    audio_fps=self.benchmark_audio_sample_rate,
                    audio_codec="aac",
                )
                written_with_audio = True
            except Exception as e:
                print(
                    f"[Benchmark][Rank {self.global_rank}] write_video with "
                    f"audio failed for prompt {prompt_idx}: {e}"
                )

        # Fallback: silent video + separate wav
        if not written_with_audio:
            try:
                from torchvision.io import write_video

                write_video(sample_path, vid, fps=self.benchmark_video_fps)
            except Exception as e:
                print(
                    f"[Benchmark][Rank {self.global_rank}] write_video (silent) "
                    f"failed for prompt {prompt_idx}: {e}"
                )
                return

            if audio_waveform is not None:
                try:
                    import torchaudio

                    wav = audio_waveform[0].cpu().float()
                    wav_path = os.path.join(
                        step_dir, f"sample_{prompt_idx}.wav"
                    )
                    torchaudio.save(
                        wav_path, wav, self.benchmark_audio_sample_rate
                    )
                except Exception as e:
                    print(
                        f"[Benchmark][Rank {self.global_rank}] torchaudio.save "
                        f"failed for prompt {prompt_idx}: {e}"
                    )

        # Free decoded tensors
        del video_pixel, audio_waveform
        torch.cuda.empty_cache()

    def train(self):
        """Main training loop."""
        while True:
            self.train_one_step()

            # Save checkpoint
            if (
                not getattr(self.config, "no_save", False)
                and self.checkpoint_iters > 0
                and self.step % self.checkpoint_iters == 0
                and not (
                    self.step == 0
                    and getattr(self.config, "skip_initial_checkpoint", False)
                )
            ):
                self.save()
                torch.cuda.empty_cache()

            barrier()

            # Timing
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is not None:
                    self._safe_wandb_log(
                        {"per_iteration_time": current_time - self.previous_time},
                        step=self.step,
                    )
                self.previous_time = current_time

            self.step += 1

            # Step LR schedulers based on global step (both stay synchronized)
            if self.generator_scheduler is not None:
                self.generator_scheduler.step(self.step)
            if self.critic_scheduler is not None:
                self.critic_scheduler.step(self.step)

            # Optional: max steps limit
            max_steps = getattr(self.config, "max_steps", None)
            if max_steps and self.step >= max_steps:
                break

        if self.is_main_process:
            if not getattr(self.config, "no_save", False):
                self.save()
            self._safe_wandb_finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    trainer = Trainer(config)
    try:
        trainer.train()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
