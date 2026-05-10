"""
Build or incrementally update a persistent video-caption registry.

The registry is append-only and treats the video directory as the source of
truth. Each discovered video gets a stable ``source_index`` exactly once, so
shard assignment via ``source_index % num_shards`` remains stable across
restarts even when new videos are added later.

Output format:
    {
      "source_index": 123,
      "video_name": "clip.mp4",
      "prompt": "...",
      "video_path": "/abs/path/to/clip.mp4"
    }
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RegistryEntry:
    source_index: int
    video_name: str
    prompt: str
    video_path: str


def load_caption_lookup(captions_file: Path) -> tuple[dict[str, str], int]:
    lookup: dict[str, str] = {}
    malformed_lines = 0
    pending_filename: str | None = None

    with captions_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if pending_filename is not None:
                if line:
                    lookup[pending_filename] = line
                else:
                    malformed_lines += 1
                pending_filename = None
                continue

            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                token = parts[0]
                if token.endswith(".mp4"):
                    pending_filename = token
                else:
                    malformed_lines += 1
                continue

            video_name, prompt = parts
            prompt = prompt.strip()
            if prompt:
                lookup[video_name] = prompt
            else:
                malformed_lines += 1

    if pending_filename is not None:
        malformed_lines += 1

    return lookup, malformed_lines


def load_existing_registry(output_file: Path) -> list[RegistryEntry]:
    if not output_file.exists():
        return []

    entries: list[RegistryEntry] = []
    with output_file.open("r", encoding="utf-8") as f:
        next_index = 0
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            source_index = payload.get("source_index")
            if source_index is None:
                source_index = next_index
            entries.append(
                RegistryEntry(
                    source_index=int(source_index),
                    video_name=str(payload.get("video_name") or Path(str(payload["video_path"])).name),
                    prompt=str(payload["prompt"]),
                    video_path=str(payload["video_path"]),
                )
            )
            next_index = max(next_index, int(source_index) + 1)
    return entries


def acquire_lock(lock_file: Path, poll_interval: float = 0.2) -> int:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            return os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            time.sleep(poll_interval)


def release_lock(lock_fd: int, lock_file: Path) -> None:
    try:
        os.close(lock_fd)
    finally:
        try:
            lock_file.unlink()
        except FileNotFoundError:
            pass


def sync_registry(
    captions_file: Path,
    video_dir: Path,
    output_file: Path,
) -> tuple[int, int, int, int]:
    lock_file = output_file.with_suffix(output_file.suffix + ".lock")
    lock_fd = acquire_lock(lock_file)
    try:
        caption_lookup, malformed_lines = load_caption_lookup(captions_file)
        existing_entries = load_existing_registry(output_file)

        existing_by_name = {entry.video_name: entry for entry in existing_entries}
        next_index = 0 if not existing_entries else max(entry.source_index for entry in existing_entries) + 1

        updated_entries = list(existing_entries)
        added = 0
        missing_caption = 0
        existing_found = 0

        for video_path in sorted(video_dir.glob("*.mp4")):
            video_name = video_path.name
            prompt = caption_lookup.get(video_name)
            if not prompt:
                missing_caption += 1
                continue

            resolved_video_path = str(video_path.resolve())
            existing_entry = existing_by_name.get(video_name)
            if existing_entry is not None:
                existing_found += 1
                # Keep source_index stable, but refresh prompt/path if upstream files changed.
                if existing_entry.prompt != prompt or existing_entry.video_path != resolved_video_path:
                    existing_entry.prompt = prompt
                    existing_entry.video_path = resolved_video_path
                continue

            entry = RegistryEntry(
                source_index=next_index,
                video_name=video_name,
                prompt=prompt,
                video_path=resolved_video_path,
            )
            updated_entries.append(entry)
            existing_by_name[video_name] = entry
            next_index += 1
            added += 1

        output_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_output = output_file.with_suffix(
            output_file.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        with tmp_output.open("w", encoding="utf-8") as out:
            for entry in sorted(updated_entries, key=lambda item: item.source_index):
                out.write(
                    json.dumps(
                        {
                            "source_index": entry.source_index,
                            "video_name": entry.video_name,
                            "prompt": entry.prompt,
                            "video_path": entry.video_path,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        tmp_output.replace(output_file)

        return len(updated_entries), added, missing_caption, malformed_lines
    finally:
        release_lock(lock_fd, lock_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or update a persistent video-caption registry.")
    parser.add_argument("--captions_file", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    captions_file = Path(args.captions_file).expanduser().resolve()
    video_dir = Path(args.video_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()

    if not captions_file.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_file}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    total, added, missing_caption, malformed_lines = sync_registry(
        captions_file=captions_file,
        video_dir=video_dir,
        output_file=output_file,
    )

    print(f"[manifest] output: {output_file}")
    print(f"[manifest] total registry entries: {total}")
    print(f"[manifest] newly appended entries: {added}")
    print(f"[manifest] video files missing captions: {missing_caption}")
    print(f"[manifest] malformed caption lines skipped: {malformed_lines}")


if __name__ == "__main__":
    main()
