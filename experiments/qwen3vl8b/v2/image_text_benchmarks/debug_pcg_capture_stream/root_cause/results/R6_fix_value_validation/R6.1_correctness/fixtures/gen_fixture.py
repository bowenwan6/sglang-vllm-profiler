#!/usr/bin/env python3
"""Deterministic R6.1 correctness-fixture image generator.

CPU-only. No CUDA, no model, no server. Produces a 1280x720 RGB PNG with
three solid vertical color bars (muted red / green / blue) on a white
background. The output is intended to give Qwen3-VL a visually
unambiguous target so cross-variant output comparison is a meaningful
correctness signal.

Determinism:
- Fixed pixel values (no random source).
- PIL PNG encoder with explicit `compress_level=6`, `optimize=False`
  produces byte-identical output across runs on the same Pillow
  version. If the R6 environment is upgraded, R6.0 provenance must be
  re-committed and the fixture SHA-256 re-recorded.

Usage:
    python3 gen_fixture.py [OUT_PATH]
        default OUT_PATH: ./R6.1_fixture.png
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from PIL import Image

WIDTH = 1280
HEIGHT = 720
BG = (255, 255, 255)
BAND_COLORS = (
    (220, 40, 40),
    (40, 180, 60),
    (40, 80, 200),
)


def make_fixture() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BG)
    px = img.load()
    third = WIDTH // 3
    band_slices = (
        (0, third),
        (third, 2 * third),
        (2 * third, WIDTH),
    )
    for (x_lo, x_hi), color in zip(band_slices, BAND_COLORS):
        for x in range(x_lo, x_hi):
            for y in range(HEIGHT):
                px[x, y] = color
    return img


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "R6.1_fixture.png")
    img = make_fixture()
    img.save(out, format="PNG", compress_level=6, optimize=False)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"WROTE {out.resolve()}")
    print(f"SIZE  {out.stat().st_size} bytes")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
