"""
Generate teacher-created pseudo-SCM latent LMDB data.

This path is prompt-only and teacher-generated, so it is useful for large-scale
pseudo-SCM experiments but should not be confused with faithful real-data SCM.

The writer supports:
- resumable generation
- shard-per-rank output for distributed generation
- incremental LMDB metadata updates so partially written shards remain readable
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import lmdb
import torch
from omegaconf import OmegaConf

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader.registry import StateDictRegistry
from ltx_distillation.models.ltx_trig_wrapper import create_ltx2_trig_wrapper
from ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper
from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper
from ltx_distillation.models.vae_wrapper import create_vae_wrappers
from ltx_distillation.tools.run_teacher_inference_eval import (
    _decode_and_save_sample,
    _generate_teacher_sample,
)
from ltx_distillation.train_distillation import compute_latent_shapes


SHARD_STRATEGIES = {"block", "modulo"}


@dataclass
class PromptAssignment:
    prompts: list[str]
    global_indices: list[int]
    local_count: int
    shard_strategy: str

    @property
    def global_start_index(self) -> int | None:
        return self.global_indices[0] if self.global_indices else None

    @property
    def global_end_index(self) -> int | None:
        return (self.global_indices[-1] + 1) if self.global_indices else None


def _validate_shard_args(num_shards: int, shard_id: int) -> None:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")


@dataclass
class OutputPaths:
    lmdb_path: Path
    preview_dir: Path | None
    meta_path: Path


def _compute_shard_partition(total: int, num_shards: int, shard_id: int) -> tuple[int, int]:
    _validate_shard_args(num_shards, shard_id)
    base = total // num_shards
    remainder = total % num_shards
    count = base + (1 if shard_id < remainder else 0)
    start = shard_id * base + min(shard_id, remainder)
    return start, count


def _load_prompt_assignment(
    prompts_file: str,
    num_prompts: int,
    start_index: int,
    shard_id: int,
    num_shards: int,
    shard_strategy: str,
) -> PromptAssignment:
    _validate_shard_args(num_shards, shard_id)
    if shard_strategy not in SHARD_STRATEGIES:
        raise ValueError(f"Unsupported shard_strategy={shard_strategy}. Expected one of {sorted(SHARD_STRATEGIES)}")

    end_index = start_index + num_prompts
    prompts: list[str] = []
    global_indices: list[int] = []
    logical_index = 0

    if shard_strategy == "block":
        shard_offset, shard_count = _compute_shard_partition(num_prompts, num_shards, shard_id)
        slice_start = start_index + shard_offset
        slice_end = slice_start + shard_count
    else:
        slice_start = start_index
        slice_end = end_index
        shard_count = None

    with open(prompts_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            prompt = raw_line.strip()
            if not prompt:
                continue
            if logical_index < slice_start:
                logical_index += 1
                continue
            if logical_index >= slice_end:
                break
            if shard_strategy == "block":
                prompts.append(prompt)
                global_indices.append(logical_index)
            else:
                relative_index = logical_index - start_index
                if relative_index % num_shards == shard_id:
                    prompts.append(prompt)
                    global_indices.append(logical_index)
            logical_index += 1

    if shard_strategy == "block" and len(prompts) != shard_count:
        raise ValueError(
            f"Requested prompts [{slice_start}, {slice_end}) from {prompts_file}, "
            f"but only loaded {len(prompts)} prompt(s)."
        )

    return PromptAssignment(
        prompts=prompts,
        global_indices=global_indices,
        local_count=len(prompts),
        shard_strategy=shard_strategy,
    )


def _resolve_output_paths(
    output_lmdb: str,
    preview_dir: str,
    preview_count: int,
    shard_id: int,
    num_shards: int,
) -> OutputPaths:
    lmdb_root = Path(output_lmdb).expanduser().resolve()
    preview_root = Path(preview_dir).expanduser().resolve()

    if num_shards > 1:
        lmdb_path = lmdb_root / f"shard_{shard_id:05d}"
        resolved_preview_dir = preview_root / f"shard_{shard_id:05d}" if preview_count > 0 else None
    else:
        lmdb_path = lmdb_root
        resolved_preview_dir = preview_root if preview_count > 0 else None

    return OutputPaths(
        lmdb_path=lmdb_path,
        preview_dir=resolved_preview_dir,
        meta_path=lmdb_path / "teacher_scm_meta.json",
    )


def _prepare_output_path(path: Path, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        elif not resume:
            raise FileExistsError(
                f"Output path already exists: {path}. Pass --resume to continue or --overwrite to replace it."
            )

    path.mkdir(parents=True, exist_ok=True)


def _prepare_preview_dir(path: Path | None, overwrite: bool, resume: bool) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        elif not resume:
            raise FileExistsError(
                f"Preview path already exists: {path}. Pass --resume to continue or --overwrite to replace it."
            )
    path.mkdir(parents=True, exist_ok=True)


def _read_written_count(env: lmdb.Environment) -> int:
    with env.begin(write=False) as txn:
        count_bytes = txn.get("num_written".encode())
        if count_bytes is not None:
            return int(count_bytes.decode())

        video_shape_bytes = txn.get("video_latents_shape".encode())
        if video_shape_bytes is not None:
            return int(video_shape_bytes.decode().split()[0])

    count = 0
    with env.begin(write=False) as txn:
        while True:
            video_key = f"video_latents_{count}_data".encode()
            prompt_key = f"prompts_{count}_data".encode()
            if txn.get(video_key) is None or txn.get(prompt_key) is None:
                break
            count += 1
    return count


def _write_progress_metadata(
    txn: lmdb.Transaction,
    count: int,
    video_shape: tuple[int, ...],
    audio_shape: tuple[int, ...] | None,
) -> None:
    txn.put("num_written".encode(), str(count).encode())
    txn.put("video_latents_shape".encode(), " ".join(map(str, [count, *video_shape])).encode())
    txn.put("prompts_shape".encode(), str(count).encode())
    if audio_shape is not None:
        txn.put("audio_latents_shape".encode(), " ".join(map(str, [count, *audio_shape])).encode())


def _store_entry(
    env: lmdb.Environment,
    index: int,
    prompt: str,
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor | None,
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    video_entry = video_latent.squeeze(0).unsqueeze(0).to(torch.float16).cpu().numpy()
    audio_entry = None
    if audio_latent is not None:
        audio_entry = audio_latent.squeeze(0).unsqueeze(0).to(torch.float16).cpu().numpy()

    with env.begin(write=True) as txn:
        txn.put(f"video_latents_{index}_data".encode(), video_entry.tobytes())
        txn.put(f"prompts_{index}_data".encode(), prompt.encode("utf-8"))
        if audio_entry is not None:
            txn.put(f"audio_latents_{index}_data".encode(), audio_entry.tobytes())
        _write_progress_metadata(
            txn=txn,
            count=index + 1,
            video_shape=tuple(video_entry.shape),
            audio_shape=tuple(audio_entry.shape) if audio_entry is not None else None,
        )

    return tuple(video_entry.shape), tuple(audio_entry.shape) if audio_entry is not None else None


def _write_generation_meta(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_existing_meta(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_effective_shard_strategy(
    meta_path: Path,
    requested_strategy: str,
    resume: bool,
    prompts_file: str,
    start_index: int,
    num_prompts: int,
    num_shards: int,
) -> str:
    existing_meta = _load_existing_meta(meta_path)
    if existing_meta is None:
        return requested_strategy

    existing_strategy = str(existing_meta.get("shard_strategy", "block"))
    existing_num_shards = int(existing_meta.get("num_shards", num_shards))
    existing_start_index = int(existing_meta.get("start_index", start_index))
    existing_prompts_file = str(existing_meta.get("prompts_file", prompts_file))
    existing_num_prompts = int(existing_meta.get("num_prompts_requested", num_prompts))

    if existing_num_shards != num_shards:
        raise ValueError(
            f"Existing shard metadata uses num_shards={existing_num_shards}, "
            f"but requested num_shards={num_shards}."
        )
    if existing_start_index != start_index:
        raise ValueError(
            f"Existing shard metadata uses start_index={existing_start_index}, "
            f"but requested start_index={start_index}."
        )
    if existing_prompts_file != prompts_file:
        raise ValueError(
            f"Existing shard metadata uses prompts_file={existing_prompts_file}, "
            f"but requested prompts_file={prompts_file}."
        )

    if not resume:
        return requested_strategy

    if existing_strategy == "block" and num_prompts != existing_num_prompts:
        raise ValueError(
            "This shard was generated with legacy block sharding, which only supports "
            "resume with the same num_prompts. Start a fresh output directory to switch "
            "to expandable modulo sharding."
        )
    if existing_strategy == "modulo" and num_prompts < existing_num_prompts:
        raise ValueError(
            f"Cannot resume modulo-sharded generation with a smaller num_prompts "
            f"({num_prompts} < {existing_num_prompts})."
        )

    if existing_strategy != requested_strategy:
        print(
            f"[TeacherLatent] Existing shard metadata uses shard_strategy={existing_strategy}; "
            f"overriding requested {requested_strategy}.",
            flush=True,
        )
    return existing_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create teacher-generated pseudo-SCM latent LMDB.")
    parser.add_argument("--config_path", required=True)
    parser.add_argument(
        "--prompts_file",
        default="/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/release_prompts.txt",
    )
    parser.add_argument("--num_prompts", type=int, default=1000)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument(
        "--output_lmdb",
        default="/data/datasets/turbodiff_datasets_and_ckpt/my_omniforcing/scm_latent_teacher_1000",
    )
    parser.add_argument(
        "--preview_dir",
        default="/data/datasets/turbodiff_datasets_and_ckpt/my_omniforcing/scm_latent_teacher_1000_preview",
    )
    parser.add_argument("--preview_count", type=int, default=16)
    parser.add_argument("--mode", choices=["native_rf", "rcm_trig"], default="native_rf")
    parser.add_argument("--shard_strategy", choices=sorted(SHARD_STRATEGIES), default="modulo")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--map_size", type=int, default=500_000_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = str(Path(args.config_path).expanduser().resolve())
    prompts_file = str(Path(args.prompts_file).expanduser().resolve())
    cfg = OmegaConf.load(config_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for teacher latent generation.")

    dtype = torch.bfloat16 if bool(getattr(cfg, "mixed_precision", True)) else torch.float32
    device = torch.device("cuda")

    output_paths = _resolve_output_paths(
        output_lmdb=args.output_lmdb,
        preview_dir=args.preview_dir,
        preview_count=args.preview_count,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    _prepare_output_path(output_paths.lmdb_path, overwrite=args.overwrite, resume=args.resume)
    _prepare_preview_dir(output_paths.preview_dir, overwrite=args.overwrite, resume=args.resume)

    shard_strategy = _resolve_effective_shard_strategy(
        meta_path=output_paths.meta_path,
        requested_strategy=args.shard_strategy,
        resume=args.resume,
        prompts_file=prompts_file,
        start_index=args.start_index,
        num_prompts=args.num_prompts,
        num_shards=args.num_shards,
    )

    prompt_slice = _load_prompt_assignment(
        prompts_file=prompts_file,
        num_prompts=args.num_prompts,
        start_index=args.start_index,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        shard_strategy=shard_strategy,
    )
    if not prompt_slice.prompts:
        raise ValueError("No prompts loaded for teacher latent generation.")

    env = lmdb.open(str(output_paths.lmdb_path), map_size=args.map_size, subdir=True)
    resume_count = _read_written_count(env) if args.resume else 0
    if resume_count > prompt_slice.local_count:
        raise ValueError(
            f"Resume state reports {resume_count} completed prompt(s), "
            f"but shard only contains {prompt_slice.local_count} prompt(s)."
        )

    steps = int(
        args.num_inference_steps
        if args.num_inference_steps is not None
        else getattr(cfg, "teacher_benchmark_num_inference_steps", 40)
    )

    meta = {
        "config_path": config_path,
        "prompts_file": prompts_file,
        "num_prompts_requested": args.num_prompts,
        "start_index": args.start_index,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "shard_strategy": shard_strategy,
        "shard_prompt_count": prompt_slice.local_count,
        "shard_global_start_index": prompt_slice.global_start_index,
        "shard_global_end_index": prompt_slice.global_end_index,
        "mode": args.mode,
        "num_inference_steps": steps,
        "seed": args.seed,
        "preview_count": args.preview_count,
        "resume": args.resume,
        "output_lmdb": str(output_paths.lmdb_path),
        "preview_dir": str(output_paths.preview_dir) if output_paths.preview_dir is not None else None,
    }
    _write_generation_meta(output_paths.meta_path, meta)

    if resume_count >= prompt_slice.local_count:
        print(
            f"[TeacherLatent] shard={args.shard_id + 1}/{args.num_shards} already complete: "
            f"{resume_count}/{prompt_slice.local_count} prompts at {output_paths.lmdb_path}",
            flush=True,
        )
        env.close()
        return

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

    video_vae = None
    audio_vae = None
    if args.preview_count > 0:
        video_vae, audio_vae = create_vae_wrappers(
            checkpoint_path=cfg.checkpoint_path,
            device=device,
            dtype=dtype,
            registry=registry,
        )
        video_vae.eval()
        audio_vae.eval()

        prompts_path = output_paths.preview_dir / "prompts.txt"
        if args.overwrite or not prompts_path.exists():
            with prompts_path.open("w", encoding="utf-8") as f:
                for prompt in prompt_slice.prompts:
                    f.write(f"{prompt}\n")
        (output_paths.preview_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(steps=steps).to(device=device, dtype=dtype)
    video_shape, audio_shape = compute_latent_shapes(
        num_frames=int(cfg.num_frames),
        video_height=int(cfg.video_height),
        video_width=int(cfg.video_width),
        batch_size=1,
    )

    start_time = time.perf_counter()
    print(
        f"[TeacherLatent] shard={args.shard_id + 1}/{args.num_shards} "
        f"resume={resume_count}/{prompt_slice.local_count} "
        f"strategy={shard_strategy} "
        f"global_range=[{prompt_slice.global_start_index}, {prompt_slice.global_end_index}) "
        f"mode={args.mode} steps={steps} output={output_paths.lmdb_path}",
        flush=True,
    )

    try:
        for local_idx in range(resume_count, prompt_slice.local_count):
            prompt = prompt_slice.prompts[local_idx]
            global_idx = prompt_slice.global_indices[local_idx]

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

            _store_entry(
                env=env,
                index=local_idx,
                prompt=prompt,
                video_latent=video_latent,
                audio_latent=audio_latent,
            )

            if args.preview_count > 0 and local_idx < args.preview_count:
                _decode_and_save_sample(
                    video_vae=video_vae,
                    audio_vae=audio_vae,
                    video_latent=video_latent,
                    audio_latent=audio_latent,
                    prompt_idx=local_idx,
                    output_dir=str(output_paths.preview_dir),
                    video_fps=int(getattr(cfg, "benchmark_video_fps", 24)),
                    audio_sample_rate=int(getattr(cfg, "benchmark_audio_sample_rate", 24000)),
                )

            print(
                f"[TeacherLatent] shard={args.shard_id + 1}/{args.num_shards} "
                f"prompt={local_idx + 1}/{prompt_slice.local_count} "
                f"global={global_idx} seed={prompt_seed} gen={gen_elapsed:.2f}s",
                flush=True,
            )
            del conditional_dict, unconditional_dict, video_latent, audio_latent
            torch.cuda.empty_cache()
    finally:
        env.close()

    elapsed = time.perf_counter() - start_time
    print(
        f"[TeacherLatent] done shard={args.shard_id + 1}/{args.num_shards} "
        f"prompts={prompt_slice.local_count} mode={args.mode} "
        f"wall={elapsed:.2f}s ({elapsed / max(1, prompt_slice.local_count - resume_count):.2f}s/generated) "
        f"lmdb={output_paths.lmdb_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
