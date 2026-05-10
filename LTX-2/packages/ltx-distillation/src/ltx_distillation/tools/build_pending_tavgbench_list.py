from __future__ import annotations

import argparse
import re
from pathlib import Path


PATTERN = re.compile(r"^([A-Za-z0-9_-]{11})_(\d+)_(\d+)\.mp4$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a pending TAVGBench caption list by excluding already-downloaded clip filenames."
    )
    parser.add_argument("--captions_file", required=True)
    parser.add_argument("--completed_files", required=True)
    parser.add_argument("--output_file", required=True)
    return parser.parse_args()


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                completed.add(name)
    return completed


def main() -> None:
    args = parse_args()
    captions_file = Path(args.captions_file).expanduser().resolve()
    completed_files = Path(args.completed_files).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed(completed_files)

    kept = 0
    skipped = 0
    ignored = 0

    with captions_file.open("r", encoding="utf-8") as src, output_file.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw in src:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue

            clip_name = stripped.split(" ", 1)[0]
            if not PATTERN.match(clip_name):
                ignored += 1
                continue

            if clip_name in completed:
                skipped += 1
                continue

            dst.write(line + "\n")
            kept += 1

    print(f"completed_loaded={len(completed)}")
    print(f"pending_written={kept}")
    print(f"completed_skipped={skipped}")
    print(f"ignored_invalid={ignored}")
    print(f"output_file={output_file}")


if __name__ == "__main__":
    main()
