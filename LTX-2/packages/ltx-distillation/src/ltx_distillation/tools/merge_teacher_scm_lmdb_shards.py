"""
Merge teacher-generated pseudo-SCM LMDB shards into a single LMDB.

This operates at the raw-bytes level for speed: entries are copied directly
from shard LMDBs into a merged LMDB without decoding tensors.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import lmdb


@dataclass
class ShardInfo:
    path: Path
    count: int
    video_shape: list[int]
    audio_shape: list[int] | None
    has_audio: bool
    global_start_index: int | None
    shard_strategy: str
    shard_id: int | None
    num_shards: int | None
    start_index: int


@dataclass(order=True)
class MergeEntry:
    sort_key: int
    shard_name: str
    local_index: int


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
            if txn.get(f"video_latents_{count}_data".encode()) is None:
                break
            count += 1
    return count


def _read_shard_info(path: Path) -> ShardInfo:
    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
    )
    try:
        with env.begin(write=False) as txn:
            video_shape_bytes = txn.get("video_latents_shape".encode())
            if video_shape_bytes is None:
                raise ValueError(f"Missing video_latents_shape in shard {path}")
            video_shape = list(map(int, video_shape_bytes.decode().split()))
            count = video_shape[0]
            video_entry_shape = video_shape[1:]

            audio_shape_bytes = txn.get("audio_latents_shape".encode())
            if audio_shape_bytes is not None:
                audio_shape = list(map(int, audio_shape_bytes.decode().split()))
                audio_entry_shape = audio_shape[1:]
                has_audio = True
            else:
                audio_entry_shape = None
                has_audio = False
    finally:
        env.close()

    meta_path = path / "teacher_scm_meta.json"
    global_start_index = None
    shard_strategy = "block"
    shard_id = None
    num_shards = None
    start_index = 0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "shard_global_start_index" in meta:
            global_start_index = int(meta["shard_global_start_index"])
        shard_strategy = str(meta.get("shard_strategy", "block"))
        if "shard_id" in meta:
            shard_id = int(meta["shard_id"])
        if "num_shards" in meta:
            num_shards = int(meta["num_shards"])
        if "start_index" in meta:
            start_index = int(meta["start_index"])

    return ShardInfo(
        path=path,
        count=count,
        video_shape=video_entry_shape,
        audio_shape=audio_entry_shape,
        has_audio=has_audio,
        global_start_index=global_start_index,
        shard_strategy=shard_strategy,
        shard_id=shard_id,
        num_shards=num_shards,
        start_index=start_index,
    )


def _discover_shards(shards_root: Path) -> list[ShardInfo]:
    shard_paths = sorted(
        (
            path
            for path in shards_root.iterdir()
            if path.is_dir() and path.name.startswith("shard_") and (path / "data.mdb").exists()
        ),
        key=lambda path: path.name,
    )
    if not shard_paths:
        raise FileNotFoundError(
            f"No shard_* LMDB directories found under {shards_root}"
        )

    shards = [_read_shard_info(path) for path in shard_paths]
    shards.sort(
        key=lambda shard: (
            shard.global_start_index is None,
            shard.global_start_index if shard.global_start_index is not None else shard.path.name,
        )
    )

    reference = shards[0]
    for shard in shards[1:]:
        if shard.video_shape != reference.video_shape:
            raise ValueError(
                f"Inconsistent video shape: {shard.path} has {shard.video_shape}, "
                f"expected {reference.video_shape}"
            )
        if shard.audio_shape != reference.audio_shape:
            raise ValueError(
                f"Inconsistent audio shape: {shard.path} has {shard.audio_shape}, "
                f"expected {reference.audio_shape}"
            )
        if shard.has_audio != reference.has_audio:
            raise ValueError(
                f"Inconsistent audio availability: {shard.path} has {shard.has_audio}, "
                f"expected {reference.has_audio}"
            )
        if shard.shard_strategy != reference.shard_strategy:
            raise ValueError(
                f"Inconsistent shard strategy: {shard.path} uses {shard.shard_strategy}, "
                f"expected {reference.shard_strategy}"
            )
        if shard.start_index != reference.start_index:
            raise ValueError(
                f"Inconsistent start_index: {shard.path} uses {shard.start_index}, "
                f"expected {reference.start_index}"
            )
        if shard.shard_strategy == "modulo":
            if shard.num_shards != reference.num_shards:
                raise ValueError(
                    f"Inconsistent num_shards: {shard.path} uses {shard.num_shards}, "
                    f"expected {reference.num_shards}"
                )
            if shard.shard_id is None:
                raise ValueError(f"Missing shard_id metadata for modulo shard {shard.path}")

    return shards


def _build_ordered_entries(shards: list[ShardInfo]) -> list[MergeEntry]:
    entries: list[MergeEntry] = []
    fallback_offset = 0
    seen_keys: set[int] = set()

    for shard in shards:
        for local_index in range(shard.count):
            if shard.shard_strategy == "modulo":
                if shard.shard_id is None or shard.num_shards is None:
                    raise ValueError(f"Modulo shard {shard.path} is missing shard metadata")
                sort_key = shard.start_index + shard.shard_id + local_index * shard.num_shards
            elif shard.global_start_index is not None:
                sort_key = shard.global_start_index + local_index
            else:
                sort_key = fallback_offset + local_index

            if sort_key in seen_keys:
                raise ValueError(f"Duplicate global ordering key {sort_key} detected while merging shards")
            seen_keys.add(sort_key)
            entries.append(
                MergeEntry(
                    sort_key=sort_key,
                    shard_name=shard.path.name,
                    local_index=local_index,
                )
            )

        if shard.shard_strategy == "block" and shard.global_start_index is None:
            fallback_offset += shard.count

    entries.sort()
    return entries


def _prepare_output(path: Path, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        elif not resume:
            raise FileExistsError(
                f"Output path already exists: {path}. Pass --resume to continue or --overwrite to replace it."
            )
    path.mkdir(parents=True, exist_ok=True)


def _write_metadata(
    txn: lmdb.Transaction,
    count: int,
    video_shape: list[int],
    audio_shape: list[int] | None,
) -> None:
    txn.put("num_written".encode(), str(count).encode())
    txn.put("video_latents_shape".encode(), " ".join(map(str, [count, *video_shape])).encode())
    txn.put("prompts_shape".encode(), str(count).encode())
    if audio_shape is not None:
        txn.put("audio_latents_shape".encode(), " ".join(map(str, [count, *audio_shape])).encode())


def _copy_entry(
    src_txn: lmdb.Transaction,
    dst_txn: lmdb.Transaction,
    src_index: int,
    dst_index: int,
    has_audio: bool,
) -> None:
    video_bytes = src_txn.get(f"video_latents_{src_index}_data".encode())
    prompt_bytes = src_txn.get(f"prompts_{src_index}_data".encode())
    if video_bytes is None or prompt_bytes is None:
        raise KeyError(
            f"Missing required entry in shard at index {src_index}: "
            f"video={video_bytes is not None}, prompt={prompt_bytes is not None}"
        )

    dst_txn.put(f"video_latents_{dst_index}_data".encode(), video_bytes)
    dst_txn.put(f"prompts_{dst_index}_data".encode(), prompt_bytes)

    if has_audio:
        audio_bytes = src_txn.get(f"audio_latents_{src_index}_data".encode())
        if audio_bytes is not None:
            dst_txn.put(f"audio_latents_{dst_index}_data".encode(), audio_bytes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge teacher pseudo-SCM LMDB shards.")
    parser.add_argument("--shards_root", required=True)
    parser.add_argument("--output_lmdb", required=True)
    parser.add_argument("--map_size", type=int, default=2_000_000_000_000)
    parser.add_argument("--commit_interval", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards_root = Path(args.shards_root).expanduser().resolve()
    output_lmdb = Path(args.output_lmdb).expanduser().resolve()

    shards = _discover_shards(shards_root)
    total_count = sum(shard.count for shard in shards)
    ordered_entries = _build_ordered_entries(shards)
    _prepare_output(output_lmdb, overwrite=args.overwrite, resume=args.resume)

    env = lmdb.open(str(output_lmdb), map_size=args.map_size, subdir=True)
    resume_count = _read_written_count(env) if args.resume else 0
    if resume_count > len(ordered_entries):
        raise ValueError(
            f"Resume state reports {resume_count} entries, but shard total is only {len(ordered_entries)}."
        )

    start_time = time.perf_counter()
    print(
        f"[MergeTeacherSCM] shards={len(shards)} total={len(ordered_entries)} resume={resume_count} "
        f"input={shards_root} output={output_lmdb}",
        flush=True,
    )

    shard_map = {shard.path.name: shard for shard in shards}
    shard_envs = {
        shard.path.name: lmdb.open(
            str(shard.path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=True,
        )
        for shard in shards
    }
    global_index = resume_count
    try:
        dst_txn = env.begin(write=True)
        pending = 0
        active_shard_name = None
        active_txn = None
        for merge_index, entry in enumerate(ordered_entries):
            if merge_index < resume_count:
                continue

            shard = shard_map[entry.shard_name]
            if active_shard_name != entry.shard_name:
                if active_txn is not None:
                    active_txn.abort()
                active_shard_name = entry.shard_name
                active_txn = shard_envs[entry.shard_name].begin(write=False)

            _copy_entry(
                src_txn=active_txn,
                dst_txn=dst_txn,
                src_index=entry.local_index,
                dst_index=global_index,
                has_audio=shard.has_audio,
            )
            global_index += 1
            pending += 1

            if pending >= args.commit_interval:
                _write_metadata(
                    txn=dst_txn,
                    count=global_index,
                    video_shape=shards[0].video_shape,
                    audio_shape=shards[0].audio_shape,
                )
                dst_txn.commit()
                dst_txn = env.begin(write=True)
                pending = 0

        if active_txn is not None:
            active_txn.abort()

        _write_metadata(
            txn=dst_txn,
            count=global_index,
            video_shape=shards[0].video_shape,
            audio_shape=shards[0].audio_shape,
        )
        dst_txn.commit()
    finally:
        for shard_env in shard_envs.values():
            shard_env.close()
        env.close()

    meta = {
        "input_shards_root": str(shards_root),
        "output_lmdb": str(output_lmdb),
        "num_shards": len(shards),
        "total_entries": total_count,
        "merged_entries": global_index,
        "shard_strategy": shards[0].shard_strategy,
        "has_audio": shards[0].has_audio,
        "video_shape": shards[0].video_shape,
        "audio_shape": shards[0].audio_shape,
    }
    (output_lmdb / "merge_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - start_time
    print(
        f"[MergeTeacherSCM] done entries={global_index} shards={len(shards)} wall={elapsed:.2f}s "
        f"output={output_lmdb}",
        flush=True,
    )


if __name__ == "__main__":
    main()
