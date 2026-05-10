"""
Reconstruct TAVGBench audio-video clips from the released caption file.

Each input line is expected to look like:

    P2RPnCJZMBo_290000_300000.mp4 A caption describing the segment.

The filename encodes:
- YouTube video id: ``P2RPnCJZMBo``
- clip start time in milliseconds: ``290000``
- clip end time in milliseconds: ``300000``

This tool:
1. Parses the caption file.
2. Downloads the requested YouTube section directly to the final clip file by default.
3. Optionally caches the full source video before trimming when requested.
4. Writes a manifest that is compatible with ``create_scm_latent_lmdb.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import re
import shutil
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path


FILENAME_RE = re.compile(
    r"^(?P<video_id>[A-Za-z0-9_-]{11})_(?P<start_ms>\d+)_(?P<end_ms>\d+)\.mp4$"
)


@dataclass(frozen=True)
class TavgEntry:
    index: int
    filename: str
    caption: str
    video_id: str
    start_ms: int
    end_ms: int

    @property
    def start_seconds(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_seconds(self) -> float:
        return self.end_ms / 1000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and reconstruct TAVGBench clips from release_captions.txt."
    )
    parser.add_argument(
        "--captions_file",
        required=True,
        help="Path to release_captions.txt.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root output directory for clips, cache, and manifest.",
    )
    parser.add_argument(
        "--raw_cache_dir",
        default=None,
        help="Optional cache directory for full downloaded YouTube videos. Defaults to <output_dir>/raw_videos when caching is enabled.",
    )
    parser.add_argument(
        "--clips_dir",
        default=None,
        help="Optional clip output directory. Defaults to <output_dir>/clips.",
    )
    parser.add_argument(
        "--manifest_path",
        default=None,
        help="Optional manifest output path. Defaults to <output_dir>/manifest.jsonl.",
    )
    parser.add_argument(
        "--failures_path",
        default=None,
        help="Optional failure log path. Defaults to <output_dir>/failures.jsonl.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start processing from this line index in the caption file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Optional maximum number of samples to reconstruct.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split selected entries across this many shards.",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Current shard id in [0, num_shards).",
    )
    parser.add_argument(
        "--yt_dlp_bin",
        default="yt-dlp",
        help="Path or command name for yt-dlp.",
    )
    parser.add_argument(
        "--ffmpeg_bin",
        default="ffmpeg",
        help="Path or command name for ffmpeg.",
    )
    parser.add_argument(
        "--yt_dlp_format",
        default="bv*+ba/b",
        help="yt-dlp format selector used for downloads.",
    )
    parser.add_argument(
        "--yt_dlp_extractor_args",
        default=None,
        help="Optional yt-dlp extractor args string, e.g. 'youtube:player_client=tv'.",
    )
    parser.add_argument(
        "--yt_dlp_verbose",
        action="store_true",
        help="Enable verbose yt-dlp logging for debugging downloader issues.",
    )
    parser.add_argument(
        "--yt_dlp_no_js_runtimes",
        action="store_true",
        help="Pass --no-js-runtimes to yt-dlp before enabling explicit runtimes.",
    )
    parser.add_argument(
        "--yt_dlp_js_runtimes",
        action="append",
        default=None,
        help="Explicit yt-dlp JS runtime entry, e.g. 'node:/path/to/node'. May be repeated.",
    )
    parser.add_argument(
        "--download_retries",
        type=int,
        default=3,
        help="Number of retries for yt-dlp download failures.",
    )
    parser.add_argument(
        "--sleep_between_downloads",
        type=float,
        default=0.0,
        help="Optional sleep in seconds after each download.",
    )
    parser.add_argument(
        "--cookies_file",
        default=None,
        help="Optional Netscape-format cookies.txt to pass to yt-dlp.",
    )
    parser.add_argument(
        "--cookies_from_browser",
        default=None,
        help="Optional browser spec for yt-dlp, e.g. chromium, chrome, firefox.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing clip files instead of skipping them.",
    )
    parser.add_argument(
        "--copy_codecs",
        action="store_true",
        help="Use ffmpeg stream copy for clipping instead of re-encoding. Faster but less frame-accurate.",
    )
    parser.add_argument(
        "--cache_source_videos",
        action="store_true",
        help="Download and keep full source YouTube videos in raw_cache_dir before trimming clips.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Parse and report planned work without downloading or clipping.",
    )
    parser.add_argument(
        "--completed_index_path",
        default=None,
        help="Optional newline-delimited file of already completed clip filenames to skip before sharding.",
    )
    return parser.parse_args()


def _ensure_binary(name_or_path: str) -> str:
    resolved = shutil.which(name_or_path) if not Path(name_or_path).is_absolute() else name_or_path
    if not resolved or not Path(resolved).exists():
        raise FileNotFoundError(
            f"Required executable not found: {name_or_path}. "
            "Install it or pass an explicit path."
        )
    return resolved


def _iter_entries(captions_file: Path, start_index: int = 0):
    pending_entry: TavgEntry | None = None
    with captions_file.open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            if idx < start_index:
                continue

            parts = line.split(maxsplit=1)
            first_token = parts[0]
            rest = parts[1].strip() if len(parts) > 1 else ""
            match = FILENAME_RE.match(first_token)

            if match:
                if pending_entry is not None:
                    yield pending_entry
                    pending_entry = None
                entry = TavgEntry(
                    index=idx,
                    filename=first_token,
                    caption=rest,
                    video_id=match.group("video_id"),
                    start_ms=int(match.group("start_ms")),
                    end_ms=int(match.group("end_ms")),
                )
                if rest:
                    yield entry
                else:
                    pending_entry = entry
            elif pending_entry is not None:
                pending_entry = TavgEntry(
                    index=pending_entry.index,
                    filename=pending_entry.filename,
                    caption=line,
                    video_id=pending_entry.video_id,
                    start_ms=pending_entry.start_ms,
                    end_ms=pending_entry.end_ms,
                )
                yield pending_entry
                pending_entry = None
            else:
                continue
    if pending_entry is not None:
        yield pending_entry


def _iter_selected_entries(
    captions_file: Path,
    start_index: int,
    num_samples: int | None,
    num_shards: int,
    shard_id: int,
    completed_filenames: set[str] | None = None,
):
    selected_seen = 0
    shard_seen = 0
    for entry in _iter_entries(captions_file, start_index=start_index):
        if completed_filenames is not None and entry.filename in completed_filenames:
            continue
        if num_samples is not None and selected_seen >= num_samples:
            break
        if num_shards > 1 and selected_seen % num_shards != shard_id:
            selected_seen += 1
            continue
        selected_seen += 1
        shard_seen += 1
        yield shard_seen, entry


def _find_cached_source(raw_cache_dir: Path, video_id: str) -> Path | None:
    candidates = []
    for path in raw_cache_dir.glob(f"{video_id}.*"):
        if path.suffix in {".part", ".ytdl", ".temp"}:
            continue
        if path.is_file():
            candidates.append(path)
    if not candidates:
        return None

    suffix_priority = {".mp4": 0, ".mkv": 1, ".webm": 2, ".mov": 3}
    candidates.sort(key=lambda p: (suffix_priority.get(p.suffix.lower(), 99), p.name))
    return candidates[0]


def _download_source_video(
    video_id: str,
    raw_cache_dir: Path,
    yt_dlp_bin: str,
    yt_dlp_format: str,
    yt_dlp_extractor_args: str | None,
    yt_dlp_verbose: bool,
    yt_dlp_no_js_runtimes: bool,
    yt_dlp_js_runtimes: list[str] | None,
    retries: int,
    sleep_seconds: float,
    cookies_file: str | None,
    cookies_from_browser: str | None,
) -> Path:
    cached = _find_cached_source(raw_cache_dir, video_id)
    if cached is not None:
        return cached

    output_template = str(raw_cache_dir / f"{video_id}.%(ext)s")
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, retries + 1):
        cmd = [
            yt_dlp_bin,
            "--newline",
            "--no-warnings",
            "--no-progress",
            "-f",
            yt_dlp_format,
            "--merge-output-format",
            "mp4",
            "-o",
            output_template,
            video_url,
        ]
        if yt_dlp_no_js_runtimes:
            cmd.append("--no-js-runtimes")
        for runtime in yt_dlp_js_runtimes or []:
            cmd += ["--js-runtimes", runtime]
        if yt_dlp_extractor_args:
            cmd += ["--extractor-args", yt_dlp_extractor_args]
        if yt_dlp_verbose:
            cmd.append("--verbose")
        if cookies_file:
            cmd += ["--cookies", cookies_file]
        if cookies_from_browser:
            cmd += ["--cookies-from-browser", cookies_from_browser]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            cached = _find_cached_source(raw_cache_dir, video_id)
            if cached is None:
                raise FileNotFoundError(
                    f"yt-dlp reported success for {video_id}, but no cached file was found."
                )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return cached
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(5.0 * attempt, 20.0))

    stderr = "" if last_error is None else (last_error.stderr or "").strip()
    raise RuntimeError(f"Failed to download {video_id} after {retries} attempts. {stderr}")


def _download_clip_direct(
    entry: TavgEntry,
    clip_path: Path,
    yt_dlp_bin: str,
    ffmpeg_bin: str,
    yt_dlp_format: str,
    yt_dlp_extractor_args: str | None,
    yt_dlp_verbose: bool,
    yt_dlp_no_js_runtimes: bool,
    yt_dlp_js_runtimes: list[str] | None,
    retries: int,
    sleep_seconds: float,
    cookies_file: str | None,
    cookies_from_browser: str | None,
    copy_codecs: bool,
) -> None:
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    video_url = f"https://www.youtube.com/watch?v={entry.video_id}"
    section = f"*{entry.start_seconds:.3f}-{entry.end_seconds:.3f}"

    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, retries + 1):
        cmd = [
            yt_dlp_bin,
            "--newline",
            "--no-warnings",
            "--no-progress",
            "--force-overwrites",
            "--ffmpeg-location",
            str(Path(ffmpeg_bin).resolve().parent),
            "-f",
            yt_dlp_format,
            "--merge-output-format",
            "mp4",
            "--download-sections",
            section,
            "-o",
            str(clip_path),
            video_url,
        ]
        if yt_dlp_no_js_runtimes:
            cmd.append("--no-js-runtimes")
        for runtime in yt_dlp_js_runtimes or []:
            cmd += ["--js-runtimes", runtime]
        if yt_dlp_extractor_args:
            cmd += ["--extractor-args", yt_dlp_extractor_args]
        if yt_dlp_verbose:
            cmd.append("--verbose")
        if not copy_codecs:
            # Make section boundaries more accurate when we want the final clip
            # rather than a fast container-level trim.
            cmd.append("--force-keyframes-at-cuts")
        if cookies_file:
            cmd += ["--cookies", cookies_file]
        if cookies_from_browser:
            cmd += ["--cookies-from-browser", cookies_from_browser]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if not clip_path.exists():
                raise FileNotFoundError(
                    f"yt-dlp reported success for {entry.filename}, but the clip file was not created."
                )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(5.0 * attempt, 20.0))

    stderr = "" if last_error is None else (last_error.stderr or "").strip()
    raise RuntimeError(f"Failed to download clip {entry.filename} after {retries} attempts. {stderr}")


def _clip_video(
    source_video: Path,
    clip_path: Path,
    start_seconds: float,
    end_seconds: float,
    ffmpeg_bin: str,
    copy_codecs: bool,
) -> None:
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end_seconds - start_seconds)
    if duration <= 0:
        raise ValueError(f"Invalid clip duration: {start_seconds} -> {end_seconds}")

    tmp_path = clip_path.with_suffix(clip_path.suffix + ".tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()

    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_video),
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]

    if copy_codecs:
        cmd += [
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
        ]
    else:
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
        ]

    cmd.append(str(tmp_path))

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        tmp_path.replace(clip_path)
    except subprocess.CalledProcessError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"ffmpeg failed for {source_video.name} [{start_seconds:.3f}, {end_seconds:.3f}]. "
            f"{(exc.stderr or '').strip()}"
        ) from exc


def _manifest_payload(entry: TavgEntry, clip_path: Path, manifest_path: Path) -> dict:
    return {
        "caption": entry.caption,
        "media_path": os.path.relpath(clip_path, manifest_path.parent),
        "youtube_id": entry.video_id,
        "segment_filename": entry.filename,
        "start_seconds": entry.start_seconds,
        "end_seconds": entry.end_seconds,
        "source_line_index": entry.index,
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_completed_filenames(completed_index_path: Path | None) -> set[str]:
    if completed_index_path is None or not completed_index_path.exists():
        return set()
    out: set[str] = set()
    with completed_index_path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                out.add(name)
    return out


def main() -> None:
    args = _parse_args()

    captions_file = Path(args.captions_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    raw_cache_dir = Path(args.raw_cache_dir).expanduser().resolve() if args.raw_cache_dir else output_dir / "raw_videos"
    clips_dir = Path(args.clips_dir).expanduser().resolve() if args.clips_dir else output_dir / "clips"
    manifest_path = Path(args.manifest_path).expanduser().resolve() if args.manifest_path else output_dir / "manifest.jsonl"
    failures_path = Path(args.failures_path).expanduser().resolve() if args.failures_path else output_dir / "failures.jsonl"
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must be in [0, num_shards)")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    preview_entries = []
    for _, entry in _iter_selected_entries(
        captions_file=captions_file,
        start_index=args.start_index,
        num_samples=args.num_samples,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ):
        preview_entries.append(entry)
        if len(preview_entries) >= 3:
            break
    if not preview_entries:
        raise ValueError("No entries selected from the caption file.")

    if args.dry_run:
        count = sum(
            1
            for _ in _iter_selected_entries(
                captions_file=captions_file,
                start_index=args.start_index,
                num_samples=args.num_samples,
                num_shards=args.num_shards,
                shard_id=args.shard_id,
            )
        )
        print(f"Loaded {count} entries from {captions_file}")
        print(f"Output dir:    {output_dir}")
        print(f"Raw cache dir: {raw_cache_dir}")
        print(f"Clips dir:     {clips_dir}")
        print(f"Manifest path: {manifest_path}")
        print(f"Cache sources: {args.cache_source_videos}")
        print(f"Shard:         {args.shard_id}/{args.num_shards}")
        print(f"First sample:  {preview_entries[0].filename} -> {preview_entries[0].caption[:120]}")
        return

    yt_dlp_bin = _ensure_binary(args.yt_dlp_bin)
    ffmpeg_bin = _ensure_binary(args.ffmpeg_bin)

    manifest_count = 0
    failures_count = 0
    requested_count = 0
    completed_index_path = (
        Path(args.completed_index_path).expanduser().resolve()
        if args.completed_index_path
        else None
    )
    completed_filenames = _load_completed_filenames(completed_index_path)
    with manifest_path.open("a", encoding="utf-8") as manifest_f, failures_path.open(
        "a", encoding="utf-8"
    ) as failures_f:
        for item_idx, entry in _iter_selected_entries(
            captions_file=captions_file,
            start_index=args.start_index,
            num_samples=args.num_samples,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            completed_filenames=completed_filenames,
        ):
            requested_count = item_idx
            clip_path = clips_dir / entry.filename
            started_at = _utc_now_iso()
            started_ts = time.time()
            if clip_path.exists() and not args.overwrite:
                payload = _manifest_payload(entry, clip_path, manifest_path)
                payload.update(
                    {
                        "status": "skipped_existing",
                        "started_at_utc": started_at,
                        "finished_at_utc": _utc_now_iso(),
                        "elapsed_seconds": round(time.time() - started_ts, 3),
                    }
                )
                manifest_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                manifest_f.flush()
                manifest_count += 1
                continue

            try:
                if args.cache_source_videos:
                    source_video = _download_source_video(
                        video_id=entry.video_id,
                        raw_cache_dir=raw_cache_dir,
                        yt_dlp_bin=yt_dlp_bin,
                        yt_dlp_format=args.yt_dlp_format,
                        yt_dlp_extractor_args=args.yt_dlp_extractor_args,
                        yt_dlp_verbose=args.yt_dlp_verbose,
                        yt_dlp_no_js_runtimes=args.yt_dlp_no_js_runtimes,
                        yt_dlp_js_runtimes=args.yt_dlp_js_runtimes,
                        retries=args.download_retries,
                        sleep_seconds=args.sleep_between_downloads,
                        cookies_file=args.cookies_file,
                        cookies_from_browser=args.cookies_from_browser,
                    )
                    _clip_video(
                        source_video=source_video,
                        clip_path=clip_path,
                        start_seconds=entry.start_seconds,
                        end_seconds=entry.end_seconds,
                        ffmpeg_bin=ffmpeg_bin,
                        copy_codecs=args.copy_codecs,
                    )
                else:
                    _download_clip_direct(
                        entry=entry,
                        clip_path=clip_path,
                        yt_dlp_bin=yt_dlp_bin,
                        ffmpeg_bin=ffmpeg_bin,
                        yt_dlp_format=args.yt_dlp_format,
                        yt_dlp_extractor_args=args.yt_dlp_extractor_args,
                        yt_dlp_verbose=args.yt_dlp_verbose,
                        yt_dlp_no_js_runtimes=args.yt_dlp_no_js_runtimes,
                        yt_dlp_js_runtimes=args.yt_dlp_js_runtimes,
                        retries=args.download_retries,
                        sleep_seconds=args.sleep_between_downloads,
                        cookies_file=args.cookies_file,
                        cookies_from_browser=args.cookies_from_browser,
                        copy_codecs=args.copy_codecs,
                    )
                payload = _manifest_payload(entry, clip_path, manifest_path)
                payload.update(
                    {
                        "status": "downloaded",
                        "started_at_utc": started_at,
                        "finished_at_utc": _utc_now_iso(),
                        "elapsed_seconds": round(time.time() - started_ts, 3),
                    }
                )
                manifest_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                manifest_f.flush()
                manifest_count += 1
            except Exception as exc:  # noqa: BLE001
                failures_f.write(
                    json.dumps(
                        {
                            "filename": entry.filename,
                            "caption": entry.caption,
                            "youtube_id": entry.video_id,
                            "start_seconds": entry.start_seconds,
                            "end_seconds": entry.end_seconds,
                            "started_at_utc": started_at,
                            "finished_at_utc": _utc_now_iso(),
                            "elapsed_seconds": round(time.time() - started_ts, 3),
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                failures_f.flush()
                failures_count += 1
            if item_idx % 50 == 0:
                print(
                    f"Processed {item_idx} entries in shard {args.shard_id}/{args.num_shards} "
                    f"(manifest={manifest_count}, failures={failures_count})"
                )

    print(f"Requested entries: {requested_count}")
    print(f"Manifest clips:    {manifest_count}")
    print(f"Failures:          {failures_count}")
    print(f"Manifest path:     {manifest_path}")
    print(f"Failures path:     {failures_path}")


if __name__ == "__main__":
    main()
