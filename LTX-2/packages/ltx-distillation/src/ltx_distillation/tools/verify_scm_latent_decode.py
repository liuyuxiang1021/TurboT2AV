from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import lmdb
import numpy as np
import torch
from ltx_core.loader.registry import StateDictRegistry
from ltx_distillation.models.vae_wrapper import create_vae_wrappers
from ltx_pipelines.utils.media_io import encode_video
from ltx_distillation.tools.create_scm_latent_lmdb import (
    _build_audio_encoder,
    _prepare_audio_waveform,
    _prepare_video_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly decode SCM latent samples and save side-by-side folders with the original videos."
    )
    parser.add_argument("--lmdb_root", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--samples_per_shard", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--video_height", type=int, default=512)
    parser.add_argument("--video_width", type=int, default=768)
    parser.add_argument("--video_fps", type=int, default=24)
    parser.add_argument("--audio_sample_rate", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def select_samples(
    available: list[tuple[int, int]],
    num_shards: int,
    samples_per_shard: int | None,
    num_samples: int,
    seed: int,
) -> list[tuple[int, int]]:
    sampler = random.Random(seed)

    if samples_per_shard is None:
        sample_count = min(num_samples, len(available))
        return sampler.sample(available, sample_count)

    selected: list[tuple[int, int]] = []
    for shard_idx in range(num_shards):
        shard_items = [item for item in available if item[0] == shard_idx]
        if not shard_items:
            continue
        shard_sample_count = min(samples_per_shard, len(shard_items))
        selected.extend(sampler.sample(shard_items, shard_sample_count))
    return selected


def shard_paths(lmdb_root: str) -> list[Path]:
    root = Path(lmdb_root)
    shards = sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.startswith("shard_") and (path / "data.mdb").exists()
    )
    if shards:
        return shards
    if (root / "data.mdb").exists():
        return [root]
    raise FileNotFoundError(f"No LMDB shard found under {lmdb_root}")


def open_env(path: Path) -> lmdb.Environment:
    return lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)


def read_num_written(env: lmdb.Environment) -> int:
    with env.begin(write=False) as txn:
        value = txn.get(b"num_written")
        if value is not None:
            return int(value.decode())
        video_shape = txn.get(b"video_latents_shape")
        if video_shape is None:
            raise ValueError("Missing video_latents_shape in LMDB")
        return int(video_shape.decode().split()[0])


def read_entry_shapes(env: lmdb.Environment) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    with env.begin(write=False) as txn:
        video_shape = tuple(map(int, txn.get(b"video_latents_shape").decode().split()[1:]))
        audio_shape_bytes = txn.get(b"audio_latents_shape")
        audio_shape = tuple(map(int, audio_shape_bytes.decode().split()[1:])) if audio_shape_bytes is not None else None
    return video_shape, audio_shape


def read_latents(
    env: lmdb.Environment,
    local_idx: int,
    video_shape: tuple[int, ...],
    audio_shape: tuple[int, ...] | None,
) -> tuple[torch.Tensor, torch.Tensor | None, str, str | None, str | None, int | None]:
    with env.begin(write=False) as txn:
        video_bytes = txn.get(f"video_latents_{local_idx}_data".encode())
        if video_bytes is None:
            raise KeyError(f"Missing video latent for local_idx={local_idx}")
        prompt_bytes = txn.get(f"prompts_{local_idx}_data".encode())
        audio_bytes = txn.get(f"audio_latents_{local_idx}_data".encode()) if audio_shape is not None else None
        video_path_bytes = txn.get(f"video_paths_{local_idx}_data".encode())
        video_name_bytes = txn.get(f"video_names_{local_idx}_data".encode())
        source_id_bytes = txn.get(f"source_ids_{local_idx}_data".encode())

    video = torch.from_numpy(np.frombuffer(video_bytes, dtype=np.float16).reshape(video_shape).copy())
    audio = None
    if audio_bytes is not None and audio_shape is not None:
        audio = torch.from_numpy(np.frombuffer(audio_bytes, dtype=np.float16).reshape(audio_shape).copy())
    prompt = prompt_bytes.decode("utf-8") if prompt_bytes is not None else ""
    video_path = video_path_bytes.decode("utf-8") if video_path_bytes is not None else None
    video_name = video_name_bytes.decode("utf-8") if video_name_bytes is not None else None
    source_id = int(source_id_bytes.decode()) if source_id_bytes is not None else None
    return video, audio, prompt, video_path, video_name, source_id


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def sanitize_stem(name: str) -> str:
    stem = Path(name).stem
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return sanitized or "unknown_video"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)

    shards = shard_paths(args.lmdb_root)
    registry = StateDictRegistry()
    video_vae, audio_vae = create_vae_wrappers(
        checkpoint_path=args.checkpoint_path,
        device=device,
        dtype=dtype,
        registry=registry,
    )
    _, audio_processor = _build_audio_encoder(
        checkpoint_path=args.checkpoint_path,
        registry=registry,
        device=torch.device("cpu"),
        dtype=dtype,
    )

    available: list[tuple[int, int]] = []
    shard_video_shapes: list[tuple[int, ...]] = []
    shard_audio_shapes: list[tuple[int, ...] | None] = []

    for shard_idx, shard_path in enumerate(shards):
        env = open_env(shard_path)
        try:
            num_written = read_num_written(env)
            video_shape, audio_shape = read_entry_shapes(env)
            shard_video_shapes.append(video_shape)
            shard_audio_shapes.append(audio_shape)

            for local_idx in range(num_written):
                with env.begin(write=False) as txn:
                    if txn.get(f"video_paths_{local_idx}_data".encode()) is None:
                        continue
                available.append((shard_idx, local_idx))
                if args.max_samples is not None and len(available) >= args.max_samples:
                    break
        finally:
            env.close()
        if args.max_samples is not None and len(available) >= args.max_samples:
            break

    if not available:
        raise RuntimeError("No written latent samples available for verification.")

    selected = select_samples(
        available=available,
        num_shards=len(shards),
        samples_per_shard=args.samples_per_shard,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for sample_rank, (shard_idx, local_idx) in enumerate(selected):
        env = open_env(shards[shard_idx])
        try:
            video_shape = shard_video_shapes[shard_idx]
            audio_shape = shard_audio_shapes[shard_idx]
            video_latent, audio_latent, prompt, video_path, video_name, source_id = read_latents(
                env, local_idx, video_shape, audio_shape
            )
            if video_path is None:
                print(
                    f"[verify] skipping shard={shard_idx} local_idx={local_idx} because LMDB lacks video path metadata",
                    flush=True,
                )
                continue

            shard_dir = output_root / f"shard_{shard_idx:02d}"
            shard_dir.mkdir(parents=True, exist_ok=True)

            display_name = video_name or Path(video_path).name
            sample_dir = shard_dir / sanitize_stem(display_name)
            sample_dir.mkdir(parents=True, exist_ok=True)

            original_path = Path(video_path).expanduser().resolve()
            original_copy_path = sample_dir / original_path.name
            shutil.copy2(original_path, original_copy_path)

            save_text(sample_dir / "prompt.txt", prompt)
            save_text(sample_dir / "metadata.json", json.dumps(
                {
                    "shard_idx": shard_idx,
                    "local_idx": local_idx,
                    "source_index": source_id,
                    "video_name": video_name,
                    "original_video_path": str(original_path),
                    "prompt": prompt,
                },
                ensure_ascii=False,
                indent=2,
            ))

            preprocessed_video = _prepare_video_tensor(
                entry=type("Entry", (), {"video_path": str(original_path)})(),
                num_frames=args.num_frames,
                video_height=args.video_height,
                video_width=args.video_width,
                video_fps=float(args.video_fps),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            preprocessed_frames = (
                ((preprocessed_video.squeeze(0).permute(1, 2, 3, 0).contiguous() + 1.0) * 127.5)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
            )
            preprocessed_audio = _prepare_audio_waveform(
                entry=type("Entry", (), {"video_path": str(original_path), "audio_path": None})(),
                num_frames=args.num_frames,
                video_fps=float(args.video_fps),
                audio_processor=audio_processor,
                allow_missing_audio=False,
                device=torch.device("cpu"),
            )
            encode_video(
                video=preprocessed_frames,
                fps=args.video_fps,
                audio=preprocessed_audio.transpose(0, 1).contiguous(),
                audio_sample_rate=audio_processor.sample_rate,
                output_path=str(sample_dir / "original_preprocessed.mp4"),
                video_chunks_number=1,
            )

            video_latent = video_latent.to(device=device, dtype=dtype)
            audio_latent = audio_latent.to(device=device, dtype=dtype) if audio_latent is not None else None

            with torch.no_grad():
                decoded_video = video_vae.decode_to_pixel(video_latent).mul(255).round().to(torch.uint8)
                decoded_frames = decoded_video.squeeze(0).permute(1, 2, 3, 0).contiguous()
                decoded_audio = audio_vae.decode_to_waveform(audio_latent) if audio_latent is not None else None

            reconstructed_path = sample_dir / "reconstructed.mp4"
            encode_video(
                video=decoded_frames,
                fps=args.video_fps,
                audio=decoded_audio[0].transpose(0, 1).cpu() if decoded_audio is not None else None,
                audio_sample_rate=args.audio_sample_rate if decoded_audio is not None else None,
                output_path=str(reconstructed_path),
                video_chunks_number=1,
            )

            print(
                f"[verify] saved sample_{sample_rank:02d} "
                f"(shard={shard_idx}, local_idx={local_idx}) to {sample_dir}",
                flush=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
