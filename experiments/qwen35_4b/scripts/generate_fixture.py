#!/usr/bin/env python3
"""Generate the byte-pinned image fixture for the Qwen3.5-4B BCG DeepStack
investigation.

Regenerating this file MUST be bit-identical every time. That is the only
way the byte-pinned SHA-256 in ``fixtures/manifest.json`` stays valid
across runs.

CPU-only, no CUDA, no model download, no network. Safe to run in any
environment with Pillow installed.

Usage
-----

    python3 scripts/generate_fixture.py --check           # regenerate, verify SHA
    python3 scripts/generate_fixture.py --write           # regenerate, overwrite
    python3 scripts/generate_fixture.py --check --strict  # nonzero exit on mismatch
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FIXTURE_NAME = "image_bands.png"
WIDTH = 1280
HEIGHT = 720
# Muted RGB bands, split at exact thirds.
BAND_COLORS = [
    (200, 96, 96),   # muted red
    (96, 160, 96),   # muted green
    (96, 96, 200),   # muted blue
]


def build_image() -> Image.Image:
    """Deterministic image build.

    Three vertical bands plus a small 32x32 all-white sync square in the
    bottom-right corner so a downstream visual-token quantiser cannot
    collapse the whole image into a single 'flat colour' embedding — that
    would defeat the whole point of exercising DeepStack.
    """
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    band_width = WIDTH // 3
    for i, colour in enumerate(BAND_COLORS):
        x0 = i * band_width
        x1 = (i + 1) * band_width if i < 2 else WIDTH
        draw.rectangle([x0, 0, x1, HEIGHT], fill=colour)
    # sync square (bottom-right)
    draw.rectangle([WIDTH - 32, HEIGHT - 32, WIDTH, HEIGHT], fill=(255, 255, 255))
    return img


def canonical_png_bytes(img: Image.Image) -> bytes:
    """Encode to PNG with fixed, PIL-version-stable options.

    Pillow's PNG writer is deterministic given (compress_level, optimize,
    pnginfo, mode) — we pin all four.
    """
    buf = io.BytesIO()
    img.save(
        buf,
        format="PNG",
        optimize=False,
        compress_level=6,
        pnginfo=None,
    )
    return buf.getvalue()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def write_manifest(path: Path, entry: dict) -> None:
    payload = json.dumps(entry, indent=2, sort_keys=True) + "\n"
    path.write_text(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures-dir",
        default=None,
        help="Path to fixtures/ directory (default: sibling of scripts/).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate and overwrite the fixture on disk.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate in memory, compare SHA-256 against manifest.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="With --check, exit non-zero on any mismatch.",
    )
    args = parser.parse_args(argv)

    if not args.write and not args.check:
        parser.error("pass one of --write or --check")

    here = Path(__file__).resolve().parent
    default_fixtures = here.parent / "fixtures"
    fixtures_dir = Path(args.fixtures_dir) if args.fixtures_dir else default_fixtures
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    fixture_path = fixtures_dir / FIXTURE_NAME
    manifest_path = fixtures_dir / "manifest.json"

    img = build_image()
    data = canonical_png_bytes(img)
    sha = sha256_hex(data)

    manifest = load_manifest(manifest_path)
    entry = {
        "path": FIXTURE_NAME,
        "sha256": sha,
        "bytes": len(data),
        "width": WIDTH,
        "height": HEIGHT,
        "mime": "image/png",
        "bands": [
            {"start_x": 0, "end_x": WIDTH // 3, "rgb": list(BAND_COLORS[0])},
            {"start_x": WIDTH // 3, "end_x": 2 * WIDTH // 3, "rgb": list(BAND_COLORS[1])},
            {"start_x": 2 * WIDTH // 3, "end_x": WIDTH, "rgb": list(BAND_COLORS[2])},
        ],
        "generator": "experiments/qwen35_4b/scripts/generate_fixture.py",
        "pillow_version": Image.__version__,
    }

    if args.write:
        fixture_path.write_bytes(data)
        manifest[FIXTURE_NAME] = entry
        write_manifest(manifest_path, manifest)
        print(f"wrote {fixture_path} ({len(data)} bytes, sha256={sha})")
        return 0

    # --check
    pinned = manifest.get(FIXTURE_NAME) or {}
    pinned_sha = pinned.get("sha256")
    on_disk_ok = False
    if fixture_path.is_file():
        on_disk_bytes = fixture_path.read_bytes()
        on_disk_sha = sha256_hex(on_disk_bytes)
        on_disk_ok = on_disk_sha == sha
        print(f"on-disk: sha256={on_disk_sha} ({len(on_disk_bytes)} bytes)")
    else:
        print("on-disk: MISSING")

    print(f"regenerated: sha256={sha} ({len(data)} bytes)")
    print(f"pinned:      sha256={pinned_sha}")

    all_match = (
        pinned_sha is not None
        and pinned_sha == sha
        and (not fixture_path.is_file() or on_disk_ok)
    )
    if all_match:
        print("MATCH")
        return 0

    print("MISMATCH")
    if args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
