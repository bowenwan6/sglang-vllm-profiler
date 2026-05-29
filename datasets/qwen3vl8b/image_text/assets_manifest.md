# Image asset manifest — qwen3vl8b/image_text (issue #4)

Template for hash-recorded image assets. **Empty by design:** the #4 plan uses the harness-native
**synthetic** image dataset (no checked-in assets) — see [`README.md`](README.md). This manifest exists so
that *if* real images are ever substituted, every asset is checked in and hashed, never downloaded at
runtime.

## Policy

- Every image under `assets/` MUST have a row below with its sha256.
- No runtime external download — assets are read from `assets/` only.
- Keep assets small and intentional; do not bulk-add large binaries.
- Record the source/license of any real image used.

## Assets

| file (under `assets/`) | sha256 | resolution | source / license | notes |
|---|---|---|---|---|
| _(none — synthetic dataset in use)_ | | | | |

## Generated payload dumps (optional provenance)

| file (under `image_text/`) | sha256 | recipe (seed / image params / lengths) | harness commit |
|---|---|---|---|
| _(none yet)_ | | | |
