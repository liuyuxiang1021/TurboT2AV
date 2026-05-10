"""
Create faithful SCM latent LMDB data from real video/audio samples.

The output format is intentionally aligned with ``ODERegressionLMDBDataset`` so
the SCM branch can reuse the existing dataloader without additional code
changes. Each entry is stored as a trajectory of length ``T=1``:

- ``video_latents_{idx}_data``: ``[1, F, C, H, W]``
- ``audio_latents_{idx}_data``: ``[1, F_a, C]`` (optional)
- ``prompts_{idx}_data``: UTF-8 prompt string
- ``video_latents_shape``: ``"N 1 F C H W"``
- ``audio_latents_shape``: ``"N 1 F_a C"`` if audio exists

Metadata input formats:
    1. Manifest file (.jsonl/.json/.csv), e.g.
       {"prompt": "A dog barking in a park", "video_path": "/abs/path/sample.mp4"}
       {"prompt": "A piano solo", "video_path": "/abs/path/sample.mp4", "audio_path": "/abs/path/sample.wav"}

    2. Caption text file + video directory, one sample per line:
       clip_name.mp4 A dog barking in a park
       another_clip.mp4 A piano solo
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import av
import lmdb
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from ltx_core.loader.registry import StateDictRegistry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    AudioEncoderConfigurator,
    AudioProcessor,
)
from ltx_pipelines.utils.media_io import normalize_latent, resize_and_center_crop
from ltx_pipelines.utils.model_ledger import ModelLedger


@dataclass
class ManifestEntry:
    prompt: str
    video_path: str
    audio_path: str | None = None
    source_index: int | None = None
    video_name: str | None = None


def _parse_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _target_audio_samples(num_frames: int, video_fps: float, sample_rate: int) -> int:
    return int(round(float(num_frames) / float(video_fps) * float(sample_rate)))


def _is_valid_frame_count(num_frames: int) -> bool:
    return num_frames >= 1 and (num_frames - 1) % 8 == 0


def _resolve_path(manifest_dir: Path, raw_path: str | None) -> str | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_dir / path
    return str(path.resolve())


def _entry_from_payload(payload: dict, manifest_dir: Path, item_desc: str) -> ManifestEntry:
    prompt = str(payload.get("prompt") or payload.get("caption") or "").strip()
    video_path = (
        payload.get("video_path")
        or payload.get("media_path")
        or payload.get("video")
        or payload.get("path")
    )
    if not prompt:
        raise ValueError(f"Missing prompt/caption in {item_desc}")
    if not video_path:
        raise ValueError(f"Missing video_path/media_path in {item_desc}")

    return ManifestEntry(
        prompt=prompt,
        video_path=_resolve_path(manifest_dir, str(video_path)),
        audio_path=_resolve_path(manifest_dir, payload.get("audio_path")),
        source_index=int(payload["source_index"]) if payload.get("source_index") is not None else None,
        video_name=str(payload.get("video_name") or Path(str(video_path)).name),
    )


def _normalize_entry_indices(entries: list[ManifestEntry]) -> list[ManifestEntry]:
    normalized: list[ManifestEntry] = []
    for idx, entry in enumerate(entries):
        source_index = idx if entry.source_index is None else int(entry.source_index)
        video_name = entry.video_name or Path(entry.video_path).name
        normalized.append(
            ManifestEntry(
                prompt=entry.prompt,
                video_path=entry.video_path,
                audio_path=entry.audio_path,
                source_index=source_index,
                video_name=video_name,
            )
        )
    return normalized


def _load_manifest(manifest_path: str, max_samples: int | None) -> list[ManifestEntry]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest_dir = manifest_file.parent
    suffix = manifest_file.suffix.lower()

    entries: list[ManifestEntry] = []

    if suffix == ".jsonl":
        with manifest_file.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                entries.append(_entry_from_payload(payload, manifest_dir, f"manifest line {line_no}"))
                if max_samples is not None and len(entries) >= max_samples:
                    break
    elif suffix == ".json":
        with manifest_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {manifest_path}")
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"Expected dict items in {manifest_path}, got {type(item)} at index {idx}")
            entries.append(_entry_from_payload(item, manifest_dir, f"manifest item {idx}"))
            if max_samples is not None and len(entries) >= max_samples:
                break
    elif suffix == ".csv":
        with manifest_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=1):
                entries.append(_entry_from_payload(dict(row), manifest_dir, f"manifest csv row {row_idx}"))
                if max_samples is not None and len(entries) >= max_samples:
                    break
    else:
        raise ValueError(
            f"Unsupported metadata format: {manifest_path}. "
            "Use .jsonl, .json, or .csv."
        )

    if not entries:
        raise ValueError(f"No valid entries found in manifest: {manifest_path}")
    return _normalize_entry_indices(entries)


def _load_caption_lookup(
    captions_path: str,
    ) -> dict[str, str]:
    captions_file = Path(captions_path).expanduser().resolve()

    if not captions_file.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_file}")

    caption_lookup: dict[str, str] = {}
    skipped_bad_lines = 0
    pending_filename: str | None = None

    with captions_file.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if pending_filename is not None:
                filename = pending_filename
                prompt = line
                pending_filename = None
            else:
                parts = line.split(maxsplit=1)
                if len(parts) == 1:
                    token = parts[0]
                    if token.endswith(".mp4"):
                        pending_filename = token
                        continue
                    skipped_bad_lines += 1
                    continue
                filename, prompt = parts

            prompt = prompt.strip()
            if not prompt:
                skipped_bad_lines += 1
                continue

            caption_lookup[filename] = prompt

    if pending_filename is not None:
        skipped_bad_lines += 1

    if not caption_lookup:
        raise ValueError(f"No valid captions found in {captions_file}")
    if skipped_bad_lines > 0:
        print(
            f"[SCM LMDB] Skipped {skipped_bad_lines} malformed caption line(s) in {captions_file}",
            flush=True,
        )
    return caption_lookup


def _load_entries_from_video_dir(
    captions_path: str,
    video_dir: str,
    max_samples: int | None,
) -> list[ManifestEntry]:
    resolved_video_dir = Path(video_dir).expanduser().resolve()
    if not resolved_video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {resolved_video_dir}")

    caption_lookup = _load_caption_lookup(captions_path)
    video_paths = sorted(resolved_video_dir.glob("*.mp4"))

    entries: list[ManifestEntry] = []
    missing_captions = 0

    for video_path in video_paths:
        prompt = caption_lookup.get(video_path.name)
        if not prompt:
            missing_captions += 1
            continue
        entries.append(
            ManifestEntry(
                prompt=prompt,
                video_path=str(video_path.resolve()),
                source_index=len(entries),
                video_name=video_path.name,
            )
        )
        if max_samples is not None and len(entries) >= max_samples:
            break

    if not entries:
        raise ValueError(
            f"No valid video-caption matches found between {resolved_video_dir} and {captions_path}"
        )
    if missing_captions > 0:
        print(
            f"[SCM LMDB] Skipped {missing_captions} video file(s) without matching captions "
            f"in {captions_path}",
            flush=True,
        )
    return _normalize_entry_indices(entries)


def _validate_shard_args(num_shards: int, shard_id: int) -> None:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")


def _select_shard_entries(
    entries: list[ManifestEntry],
    num_shards: int,
    shard_id: int,
) -> list[ManifestEntry]:
    _validate_shard_args(num_shards, shard_id)
    if num_shards == 1:
        return entries
    return [entry for entry in entries if int(entry.source_index) % num_shards == shard_id]


def _filter_entries_by_source_index(
    entries: list[ManifestEntry],
    source_index_start: int | None,
    source_index_end: int | None,
) -> list[ManifestEntry]:
    filtered = entries
    if source_index_start is not None:
        filtered = [entry for entry in filtered if int(entry.source_index) >= source_index_start]
    if source_index_end is not None:
        filtered = [entry for entry in filtered if int(entry.source_index) < source_index_end]
    return filtered


def _resolve_output_lmdb_path(output_lmdb: str, num_shards: int, shard_id: int) -> str:
    _validate_shard_args(num_shards, shard_id)
    root = Path(output_lmdb).expanduser().resolve()
    if num_shards == 1:
        return str(root)
    return str((root / f"shard_{shard_id:05d}").resolve())


def _ensure_video_frames(video: torch.Tensor, target_frames: int) -> torch.Tensor:
    # video: [1, C, F, H, W]
    frames = video.shape[2]
    if frames == target_frames:
        return video
    if frames > target_frames:
        return video[:, :, :target_frames].contiguous()
    if frames <= 0:
        raise ValueError("Decoded video has no frames")
    pad_count = target_frames - frames
    last_frame = video[:, :, -1:].expand(-1, -1, pad_count, -1, -1)
    return torch.cat([video, last_frame], dim=2).contiguous()


def _frame_timestamp_seconds(frame: av.VideoFrame, stream: av.video.stream.VideoStream, fallback_index: int) -> float:
    if frame.pts is not None and frame.time_base is not None:
        return float(frame.pts * frame.time_base)
    if frame.time is not None:
        return float(frame.time)
    average_rate = stream.average_rate
    if average_rate is not None and float(average_rate) > 0:
        return float(fallback_index) / float(average_rate)
    base_rate = stream.base_rate
    if base_rate is not None and float(base_rate) > 0:
        return float(fallback_index) / float(base_rate)
    guessed_rate = stream.guessed_rate
    if guessed_rate is not None and float(guessed_rate) > 0:
        return float(fallback_index) / float(guessed_rate)
    return float(fallback_index)


def _load_video_conditioning_aligned_fps(
    video_path: str,
    height: int,
    width: int,
    num_frames: int,
    target_fps: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    target_times = [float(frame_idx) / float(target_fps) for frame_idx in range(num_frames)]
    max_target_time = target_times[-1] if target_times else 0.0

    container = av.open(video_path)
    try:
        video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if video_stream is None:
            raise ValueError(f"No video stream found in {video_path}")

        decoded_frames: list[torch.Tensor] = []
        decoded_times: list[float] = []

        for fallback_index, frame in enumerate(container.decode(video_stream)):
            decoded_frames.append(torch.from_numpy(frame.to_rgb().to_ndarray()).to(torch.uint8))
            decoded_times.append(_frame_timestamp_seconds(frame, video_stream, fallback_index))
            if decoded_times[-1] >= max_target_time:
                break

        if not decoded_frames:
            raise ValueError(f"Decoded video has no frames: {video_path}")

        sampled_frames: list[torch.Tensor] = []
        for target_time in target_times:
            frame_pos = bisect.bisect_right(decoded_times, target_time) - 1
            if frame_pos < 0:
                frame_pos = 0
            sampled_frames.append(decoded_frames[frame_pos])

        video = None
        for frame in sampled_frames:
            processed = resize_and_center_crop(frame.to(torch.float32), height, width)
            processed = normalize_latent(processed, device, dtype)
            video = processed if video is None else torch.cat([video, processed], dim=2)
        assert video is not None
        return video.contiguous()
    finally:
        container.close()


def _pad_or_trim_waveform(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    # waveform: [channels, samples]
    samples = waveform.shape[-1]
    if samples == target_samples:
        return waveform
    if samples > target_samples:
        return waveform[..., :target_samples].contiguous()
    pad = target_samples - samples
    return torch.nn.functional.pad(waveform, (0, pad))


def _ensure_stereo(waveform: torch.Tensor) -> torch.Tensor:
    # waveform: [channels, samples]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform.contiguous()


def _decode_audio_with_av(path: str) -> tuple[torch.Tensor | None, int | None]:
    container = av.open(path)
    try:
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            return None, None

        sample_rate = int(audio_stream.rate or audio_stream.sample_rate or 16000)
        chunks: list[torch.Tensor] = []
        for frame in container.decode(audio_stream):
            array = frame.to_ndarray()
            tensor = torch.from_numpy(array).float()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 2:
                if tensor.shape[0] > tensor.shape[1] and tensor.shape[1] <= 8:
                    tensor = tensor.transpose(0, 1)
                if tensor.shape[0] > 8:
                    tensor = tensor.reshape(1, -1)
            else:
                tensor = tensor.reshape(1, -1)
            chunks.append(tensor)

        if not chunks:
            return None, sample_rate

        waveform = torch.cat(chunks, dim=-1)
        return waveform, sample_rate
    finally:
        container.close()


def _load_audio_waveform(path: str) -> tuple[torch.Tensor | None, int | None]:
    try:
        waveform, sample_rate = torchaudio.load(path)
        return waveform.float(), int(sample_rate)
    except Exception as exc:
        print(
            f"[SCM LMDB] torchaudio.load failed for {path}; "
            f"falling back to av decode ({type(exc).__name__}: {exc})",
            flush=True,
        )
        return _decode_audio_with_av(path)


def _load_sample_audio(
    entry: ManifestEntry,
    target_samples: int,
    processor: AudioProcessor,
    allow_missing_audio: bool,
    device: torch.device,
) -> torch.Tensor:
    audio_path = entry.audio_path or entry.video_path
    waveform, sample_rate = _load_audio_waveform(audio_path)

    if waveform is None or sample_rate is None:
        if not allow_missing_audio:
            raise ValueError(f"No decodable audio found for sample: {entry.video_path}")
        waveform = torch.zeros(2, target_samples, dtype=torch.float32)
        sample_rate = processor.sample_rate

    waveform = _ensure_stereo(waveform.float())
    waveform = processor.resample_waveform(waveform, sample_rate, processor.sample_rate)
    waveform = _pad_or_trim_waveform(waveform, target_samples)
    return waveform.to(device=device, dtype=torch.float32)


def _build_audio_encoder(
    checkpoint_path: str,
    registry: StateDictRegistry,
    device: torch.device,
    dtype: torch.dtype,
):
    builder = Builder(
        model_path=checkpoint_path,
        model_class_configurator=AudioEncoderConfigurator,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        registry=registry,
    )
    encoder = builder.build(device=torch.device("cpu"), dtype=dtype).to(device=device, dtype=dtype).eval()
    processor = AudioProcessor(
        sample_rate=encoder.sample_rate,
        mel_bins=encoder.mel_bins,
        mel_hop_length=encoder.mel_hop_length,
        n_fft=encoder.n_fft,
    ).to(device=device)
    return encoder, processor


def _encode_video_latent(
    video_encoder,
    entry: ManifestEntry,
    num_frames: int,
    video_height: int,
    video_width: int,
    video_fps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    video = _load_video_conditioning_aligned_fps(
        video_path=entry.video_path,
        height=video_height,
        width=video_width,
        num_frames=num_frames,
        target_fps=video_fps,
        dtype=dtype,
        device=device,
    )
    video = _ensure_video_frames(video, num_frames)
    with torch.no_grad():
        latent = video_encoder(video).permute(0, 2, 1, 3, 4).contiguous()
    return latent


def _prepare_video_tensor(
    entry: ManifestEntry,
    num_frames: int,
    video_height: int,
    video_width: int,
    video_fps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    video = _load_video_conditioning_aligned_fps(
        video_path=entry.video_path,
        height=video_height,
        width=video_width,
        num_frames=num_frames,
        target_fps=video_fps,
        dtype=dtype,
        device=device,
    )
    return _ensure_video_frames(video, num_frames)


def _encode_audio_latent(
    audio_encoder,
    audio_processor: AudioProcessor,
    entry: ManifestEntry,
    num_frames: int,
    video_fps: float,
    allow_missing_audio: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    target_samples = _target_audio_samples(num_frames, video_fps, audio_processor.sample_rate)
    waveform = _load_sample_audio(
        entry=entry,
        target_samples=target_samples,
        processor=audio_processor,
        allow_missing_audio=allow_missing_audio,
        device=device,
    )
    with torch.no_grad():
        mel = audio_processor.waveform_to_mel(waveform.unsqueeze(0), waveform_sample_rate=audio_processor.sample_rate)
        mel = mel.to(device=device, dtype=dtype)
        latent_4d = audio_encoder(mel)
        latent = latent_4d.permute(0, 2, 1, 3).reshape(latent_4d.shape[0], latent_4d.shape[2], -1).contiguous()
    return latent


def _prepare_audio_waveform(
    entry: ManifestEntry,
    num_frames: int,
    video_fps: float,
    audio_processor: AudioProcessor,
    allow_missing_audio: bool,
    device: torch.device,
) -> torch.Tensor:
    target_samples = _target_audio_samples(num_frames, video_fps, audio_processor.sample_rate)
    return _load_sample_audio(
        entry=entry,
        target_samples=target_samples,
        processor=audio_processor,
        allow_missing_audio=allow_missing_audio,
        device=device,
    )


def _iter_batches(items: list[ManifestEntry], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _prepare_batch_on_cpu(
    entries_batch: list[ManifestEntry],
    audio_processor: AudioProcessor,
    num_frames: int,
    video_height: int,
    video_width: int,
    video_fps: float,
    allow_missing_audio: bool,
    video_dtype: torch.dtype,
    pin_memory: bool,
) -> tuple[list[ManifestEntry], torch.Tensor, torch.Tensor, int]:
    valid_entries: list[ManifestEntry] = []
    video_tensors: list[torch.Tensor] = []
    audio_waveforms: list[torch.Tensor] = []
    skipped_decode_errors = 0

    for entry in entries_batch:
        try:
            video_tensor = _prepare_video_tensor(
                entry=entry,
                num_frames=num_frames,
                video_height=video_height,
                video_width=video_width,
                video_fps=video_fps,
                device=torch.device("cpu"),
                dtype=video_dtype,
            )
            audio_waveform = _prepare_audio_waveform(
                entry=entry,
                num_frames=num_frames,
                video_fps=video_fps,
                audio_processor=audio_processor,
                allow_missing_audio=allow_missing_audio,
                device=torch.device("cpu"),
            )
        except Exception as exc:
            skipped_decode_errors += 1
            print(
                f"[SCM LMDB] Skipping sample due to decode/encode error: "
                f"{entry.video_path} | {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        valid_entries.append(entry)
        video_tensors.append(video_tensor)
        audio_waveforms.append(audio_waveform)

    if not valid_entries:
        return valid_entries, torch.empty(0), torch.empty(0), skipped_decode_errors

    video_batch = torch.cat(video_tensors, dim=0)
    audio_batch = torch.stack(audio_waveforms, dim=0)

    if pin_memory:
        video_batch = video_batch.pin_memory()
        audio_batch = audio_batch.pin_memory()

    return valid_entries, video_batch, audio_batch, skipped_decode_errors


def _encode_prepared_batch(
    video_batch: torch.Tensor,
    audio_batch: torch.Tensor,
    video_encoder,
    audio_encoder,
    audio_processor: AudioProcessor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if video_batch.numel() == 0 or audio_batch.numel() == 0:
        return torch.empty(0), torch.empty(0)

    move_kwargs = {}
    if device.type == "cuda":
        move_kwargs["non_blocking"] = True

    video_batch = video_batch.to(device=device, dtype=dtype, **move_kwargs)
    audio_batch = audio_batch.to(device=device, dtype=torch.float32, **move_kwargs)

    with torch.no_grad():
        video_latents = video_encoder(video_batch).permute(0, 2, 1, 3, 4).contiguous()
        mel = audio_processor.waveform_to_mel(
            audio_batch,
            waveform_sample_rate=audio_processor.sample_rate,
        ).to(device=device, dtype=dtype)
        audio_latents_4d = audio_encoder(mel)
        audio_latents = audio_latents_4d.permute(0, 2, 1, 3).reshape(
            audio_latents_4d.shape[0], audio_latents_4d.shape[2], -1
        ).contiguous()

    return video_latents, audio_latents


def _iter_prepared_batches(
    entries: list[ManifestEntry],
    batch_size: int,
    num_workers: int,
    prefetch_batches: int,
    audio_processor: AudioProcessor,
    num_frames: int,
    video_height: int,
    video_width: int,
    video_fps: float,
    allow_missing_audio: bool,
    video_dtype: torch.dtype,
    pin_memory: bool,
):
    batches = list(_iter_batches(entries, batch_size))

    if num_workers <= 0:
        for _, entries_batch in batches:
            yield _prepare_batch_on_cpu(
                entries_batch=entries_batch,
                audio_processor=audio_processor,
                num_frames=num_frames,
                video_height=video_height,
                video_width=video_width,
                video_fps=video_fps,
                allow_missing_audio=allow_missing_audio,
                video_dtype=video_dtype,
                pin_memory=pin_memory,
            )
        return

    max_pending = max(prefetch_batches, 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        next_to_submit = 0
        next_to_yield = 0
        pending: dict[int, concurrent.futures.Future] = {}

        while next_to_yield < len(batches):
            while next_to_submit < len(batches) and len(pending) < max_pending:
                _, entries_batch = batches[next_to_submit]
                pending[next_to_submit] = executor.submit(
                    _prepare_batch_on_cpu,
                    entries_batch=entries_batch,
                    audio_processor=audio_processor,
                    num_frames=num_frames,
                    video_height=video_height,
                    video_width=video_width,
                    video_fps=video_fps,
                    allow_missing_audio=allow_missing_audio,
                    video_dtype=video_dtype,
                    pin_memory=pin_memory,
                )
                next_to_submit += 1

            future = pending.pop(next_to_yield)
            yield future.result()
            next_to_yield += 1


def _prepare_lmdb_path(path: str, overwrite: bool, resume: bool) -> None:
    output_path = Path(path).expanduser().resolve()
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if output_path.exists():
        if overwrite:
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        elif not resume:
            raise FileExistsError(
                f"Output LMDB already exists: {output_path}. "
                "Pass --overwrite (or OVERWRITE=1) to replace it, or --resume to continue."
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)


def _completed_records_path(output_lmdb_path: str) -> Path:
    return Path(output_lmdb_path).expanduser().resolve() / "completed_records.jsonl"


def _load_completed_source_ids(records_path: Path) -> set[int]:
    if not records_path.exists():
        return set()

    completed: set[int] = set()
    with records_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            completed.add(int(payload["source_index"]))
    return completed


def _read_resume_count(env: lmdb.Environment) -> int:
    with env.begin(write=False) as txn:
        value = txn.get("num_written".encode())
        if value is not None:
            return int(value.decode())

    count = 0
    with env.begin(write=False) as txn:
        while txn.get(f"prompts_{count}_data".encode()) is not None:
            count += 1
    return count


def _read_completed_source_ids_from_lmdb(env: lmdb.Environment, count: int) -> set[int]:
    completed: set[int] = set()
    with env.begin(write=False) as txn:
        for index in range(count):
            source_id_bytes = txn.get(f"source_ids_{index}_data".encode())
            if source_id_bytes is not None:
                completed.add(int(source_id_bytes.decode()))
    return completed


def _write_progress_metadata(
    txn: lmdb.Transaction,
    count: int,
    video_shape: Iterable[int],
    audio_shape: Iterable[int] | None,
) -> None:
    txn.put("num_written".encode(), str(count).encode())
    txn.put("video_latents_shape".encode(), " ".join(map(str, [count, *video_shape])).encode())
    txn.put("prompts_shape".encode(), f"{count}".encode())
    if audio_shape is not None:
        txn.put("audio_latents_shape".encode(), " ".join(map(str, [count, *audio_shape])).encode())


def _store_sample(
    env: lmdb.Environment,
    index: int,
    source_index: int,
    video_name: str,
    video_path: str,
    prompt: str,
    video_entry: np.ndarray,
    audio_entry: np.ndarray | None,
) -> None:
    with env.begin(write=True) as txn:
        txn.put(f"video_latents_{index}_data".encode(), video_entry.tobytes())
        if audio_entry is not None:
            txn.put(f"audio_latents_{index}_data".encode(), audio_entry.tobytes())
        txn.put(f"prompts_{index}_data".encode(), prompt.encode("utf-8"))
        txn.put(f"source_ids_{index}_data".encode(), str(source_index).encode())
        txn.put(f"video_names_{index}_data".encode(), video_name.encode("utf-8"))
        txn.put(f"video_paths_{index}_data".encode(), video_path.encode("utf-8"))
        _write_progress_metadata(
            txn=txn,
            count=index + 1,
            video_shape=video_entry.shape,
            audio_shape=audio_entry.shape if audio_entry is not None else None,
        )


def _append_completed_record(
    records_path: Path,
    index: int,
    entry: ManifestEntry,
) -> None:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "lmdb_index": index,
                    "source_index": int(entry.source_index),
                    "video_name": entry.video_name or Path(entry.video_path).name,
                    "video_path": entry.video_path,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _finalize_metadata(
    env: lmdb.Environment,
    count: int,
    video_shape: Iterable[int],
    audio_shape: Iterable[int] | None,
) -> None:
    with env.begin(write=True) as txn:
        _write_progress_metadata(
            txn=txn,
            count=count,
            video_shape=video_shape,
            audio_shape=audio_shape,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create faithful SCM latent LMDB from real AV samples.")
    parser.add_argument(
        "--manifest_path",
        default=None,
        help=(
            "Metadata file describing real samples. Supports .jsonl/.json/.csv and "
            "accepts either prompt/video_path or caption/media_path keys."
        ),
    )
    parser.add_argument(
        "--captions_path",
        default=None,
        help=(
            "Plain-text captions file where each line is "
            "'<video_filename> <caption>'. Use together with --video_dir."
        ),
    )
    parser.add_argument(
        "--video_dir",
        default=None,
        help="Directory containing the video files referenced by --captions_path.",
    )
    parser.add_argument(
        "--output_lmdb",
        default="/data/datasets/turbodiff_datasets_and_ckpt/my_omniforcing/scm_latent",
        help="Output LMDB path.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/data/datasets/turbodiff_datasets_and_ckpt/omniforcing/ltx-2-19b-dev.safetensors",
        help="LTX checkpoint containing video/audio VAE weights.",
    )
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--video_height", type=int, default=512)
    parser.add_argument("--video_width", type=int, default=768)
    parser.add_argument("--video_fps", type=float, default=24.0)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_batches", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--map_size", type=int, default=500_000_000_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow_missing_audio", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--source_index_start", type=int, default=None)
    parser.add_argument("--source_index_end", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.manifest_path:
        if args.captions_path or args.video_dir:
            raise ValueError(
                "Use either --manifest_path OR (--captions_path + --video_dir), not both."
            )
    else:
        if not args.captions_path or not args.video_dir:
            raise ValueError(
                "Provide either --manifest_path, or both --captions_path and --video_dir."
            )

    if not _is_valid_frame_count(args.num_frames):
        raise ValueError(
            f"num_frames must satisfy 1 + 8*k for the video VAE encoder, got {args.num_frames}."
        )
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {args.num_workers}")
    if args.prefetch_batches < 1:
        raise ValueError(f"prefetch_batches must be >= 1, got {args.prefetch_batches}")
    _validate_shard_args(args.num_shards, args.shard_id)
    if args.source_index_start is not None and args.source_index_start < 0:
        raise ValueError(f"source_index_start must be >= 0, got {args.source_index_start}")
    if args.source_index_end is not None and args.source_index_end < 0:
        raise ValueError(f"source_index_end must be >= 0, got {args.source_index_end}")
    if (
        args.source_index_start is not None
        and args.source_index_end is not None
        and args.source_index_end <= args.source_index_start
    ):
        raise ValueError(
            "source_index_end must be greater than source_index_start when both are provided."
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    dtype = _parse_dtype(args.dtype)
    if args.manifest_path:
        entries = _load_manifest(args.manifest_path, args.max_samples)
    else:
        entries = _load_entries_from_video_dir(args.captions_path, args.video_dir, args.max_samples)
    entries = _filter_entries_by_source_index(entries, args.source_index_start, args.source_index_end)
    entries = _select_shard_entries(entries, args.num_shards, args.shard_id)
    output_lmdb_path = _resolve_output_lmdb_path(args.output_lmdb, args.num_shards, args.shard_id)
    _prepare_lmdb_path(output_lmdb_path, args.overwrite, args.resume)
    completed_records = _completed_records_path(output_lmdb_path)

    if not entries:
        raise RuntimeError(
            f"No entries assigned to shard {args.shard_id}/{args.num_shards}. "
            "Check max_samples and shard settings."
        )

    registry = StateDictRegistry()
    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=args.checkpoint_path,
        registry=registry,
    )

    print(f"[SCM LMDB] Loading VAE encoders from {args.checkpoint_path}", flush=True)
    video_encoder = ledger.video_encoder().to(device=device, dtype=dtype).eval()
    audio_encoder, audio_processor = _build_audio_encoder(
        checkpoint_path=args.checkpoint_path,
        registry=registry,
        device=device,
        dtype=dtype,
    )

    env = lmdb.open(output_lmdb_path, map_size=args.map_size)
    pin_memory = args.pin_memory

    stored = 0
    completed_source_ids: set[int] = set()
    skipped_decode_errors = 0
    expected_video_shape: tuple[int, ...] | None = None
    expected_audio_shape: tuple[int, ...] | None = None

    try:
        if args.resume:
            stored = _read_resume_count(env)
            completed_source_ids.update(_load_completed_source_ids(completed_records))
            lmdb_source_ids = _read_completed_source_ids_from_lmdb(env, stored)
            completed_source_ids.update(lmdb_source_ids)

            if stored > 0:
                print(
                    f"[SCM LMDB] Resuming from sample index {stored} for shard "
                    f"{args.shard_id}/{args.num_shards}",
                    flush=True,
                )
                print(
                    f"[SCM LMDB] Resume state: {len(completed_source_ids)} completed source id(s) "
                    f"for shard {args.shard_id}/{args.num_shards}",
                    flush=True,
                )
            elif completed_source_ids:
                print(
                    f"[SCM LMDB] Found {len(completed_source_ids)} completed source id(s) "
                    f"with empty LMDB count on shard {args.shard_id}/{args.num_shards}",
                    flush=True,
                )

            if stored > 0 and not lmdb_source_ids:
                legacy_bootstrap = entries[: min(stored, len(entries))]
                if legacy_bootstrap and not completed_source_ids:
                    completed_source_ids.update(int(entry.source_index) for entry in legacy_bootstrap)
                    for bootstrap_idx, entry in enumerate(legacy_bootstrap):
                        _append_completed_record(completed_records, bootstrap_idx, entry)
                    print(
                        f"[SCM LMDB] Bootstrapped {len(legacy_bootstrap)} legacy entry record(s) "
                        f"for shard {args.shard_id}/{args.num_shards} from existing LMDB count.",
                        flush=True,
                    )
                else:
                    print(
                        f"[SCM LMDB] Warning: shard {args.shard_id}/{args.num_shards} contains legacy LMDB "
                        "entries without source-id metadata. Future resume is stable only for newly written entries.",
                        flush=True,
                    )

            if len(completed_source_ids) >= len(entries):
                print(
                    f"[SCM LMDB] Shard {args.shard_id}/{args.num_shards} already complete "
                    f"({len(completed_source_ids)}/{len(entries)} samples).",
                    flush=True,
                )
                return

        pending_entries = [entry for entry in entries if int(entry.source_index) not in completed_source_ids]
        progress = tqdm(
            total=len(entries),
            desc="Encoding SCM latents",
            initial=len(entries) - len(pending_entries),
        )
        try:
            for valid_entries, video_batch, audio_batch, batch_skipped in _iter_prepared_batches(
                entries=pending_entries,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch_batches=args.prefetch_batches,
                audio_processor=audio_processor,
                num_frames=args.num_frames,
                video_height=args.video_height,
                video_width=args.video_width,
                video_fps=args.video_fps,
                allow_missing_audio=args.allow_missing_audio,
                video_dtype=dtype,
                pin_memory=pin_memory,
            ):
                skipped_decode_errors += batch_skipped
                entries_in_batch = len(valid_entries) + batch_skipped

                if valid_entries:
                    video_latents, audio_latents = _encode_prepared_batch(
                        video_batch=video_batch,
                        audio_batch=audio_batch,
                        video_encoder=video_encoder,
                        audio_encoder=audio_encoder,
                        audio_processor=audio_processor,
                        device=device,
                        dtype=dtype,
                    )

                    for batch_idx, entry in enumerate(valid_entries):
                        video_entry = video_latents[batch_idx : batch_idx + 1].to(torch.float16).cpu().numpy()
                        audio_entry = audio_latents[batch_idx : batch_idx + 1].to(torch.float16).cpu().numpy()

                        if expected_video_shape is None:
                            expected_video_shape = tuple(video_entry.shape)
                        elif tuple(video_entry.shape) != expected_video_shape:
                            raise ValueError(
                                f"Video latent shape mismatch for {entry.video_path}: "
                                f"expected {expected_video_shape}, got {tuple(video_entry.shape)}"
                            )

                        if expected_audio_shape is None:
                            expected_audio_shape = tuple(audio_entry.shape)
                        elif tuple(audio_entry.shape) != expected_audio_shape:
                            raise ValueError(
                                f"Audio latent shape mismatch for {entry.video_path}: "
                                f"expected {expected_audio_shape}, got {tuple(audio_entry.shape)}"
                            )

                        _store_sample(
                            env=env,
                            index=stored,
                            source_index=int(entry.source_index),
                            video_name=entry.video_name or Path(entry.video_path).name,
                            video_path=entry.video_path,
                            prompt=entry.prompt,
                            video_entry=video_entry,
                            audio_entry=audio_entry,
                        )
                        _append_completed_record(completed_records, stored, entry)
                        completed_source_ids.add(int(entry.source_index))
                        stored += 1
                        progress.update(1)

                if batch_skipped > 0:
                    progress.update(batch_skipped)
        finally:
            progress.close()

        if stored == 0 or expected_video_shape is None:
            if args.resume:
                print(
                    f"[SCM LMDB] No new SCM latents were written for {output_lmdb_path} "
                    f"(shard {args.shard_id}/{args.num_shards}); treating as successful resume.",
                    flush=True,
                )
                return
            raise RuntimeError("No SCM latents were written.")

        _finalize_metadata(
            env=env,
            count=stored,
            video_shape=expected_video_shape,
            audio_shape=expected_audio_shape,
        )
    finally:
        env.close()

    print(
        f"[SCM LMDB] Wrote {stored} samples to {output_lmdb_path} "
        f"(shard {args.shard_id}/{args.num_shards}) "
        f"(video_shape={expected_video_shape}, audio_shape={expected_audio_shape}, "
        f"skipped_decode_errors={skipped_decode_errors})",
        flush=True,
    )


if __name__ == "__main__":
    main()
