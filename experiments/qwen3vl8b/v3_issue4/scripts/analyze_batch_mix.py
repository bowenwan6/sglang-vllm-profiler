#!/usr/bin/env python3
"""
Q2 supporting analysis: how often did a prefill batch actually mix classes?

The plan deferred instrumenting this until an effect appeared, but it can be
recovered from the existing logs without touching the source. The two classes
have distinct token counts — a text request is `text_tokens + template` and an
image request is `visual + text + template` — so for a batch reporting
`#new-seq: k` and `#new-token: n`, the composition (t text, i image, t + i = k)
is recoverable whenever one combination fits n within tolerance.

This is what separates "text requests were slower when images were present" from
"text requests were slower *because they shared batches with images*". Without
it, queueing and co-batching are indistinguishable.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

BATCH = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+)")


def classify(seqs, toks, w_text, w_img, tol):
    """Return (n_text, n_image) if exactly one split explains the token count."""
    fits = [(t, seqs - t) for t in range(seqs + 1)
            if abs(t * w_text + (seqs - t) * w_img - toks) <= tol * max(seqs, 1)]
    return fits[0] if len(fits) == 1 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, nargs="+", type=Path)
    ap.add_argument("--text-tokens", type=int, default=520,
                    help="server-side tokens for one text request")
    ap.add_argument("--image-tokens", type=int, default=1024,
                    help="server-side tokens for one image request")
    ap.add_argument("--tol", type=int, default=24,
                    help="per-request token tolerance")
    a = ap.parse_args()

    total = mixed = pure_text = pure_img = single = ambiguous = 0
    seq_hist = Counter()
    mixed_examples = []
    for lp in a.log:
        try:
            txt = lp.read_text(errors="replace")
        except Exception:
            continue
        for m in BATCH.finditer(txt):
            k, n = int(m.group(1)), int(m.group(2))
            if n < 8:                      # server readiness probes
                continue
            total += 1
            seq_hist[k] += 1
            if k == 1:
                single += 1
                continue
            c = classify(k, n, a.text_tokens, a.image_tokens, a.tol)
            if c is None:
                ambiguous += 1
            elif c[0] and c[1]:
                mixed += 1
                if len(mixed_examples) < 5:
                    mixed_examples.append((k, n, c))
            elif c[1]:
                pure_img += 1
            else:
                pure_text += 1

    if not total:
        print("no prefill batches found")
        return
    multi = total - single
    print(f"prefill batches            {total}")
    print(f"  single-request           {single:>6}  ({100*single/total:5.1f}%)")
    print(f"  multi-request            {multi:>6}  ({100*multi/total:5.1f}%)")
    if multi:
        print(f"    of which mixed classes {mixed:>4}  "
              f"({100*mixed/multi:5.1f}% of multi, {100*mixed/total:5.1f}% of all)")
        print(f"    pure text              {pure_text:>4}")
        print(f"    pure image             {pure_img:>4}")
        print(f"    unresolvable           {ambiguous:>4}")
    print(f"#new-seq distribution      "
          f"{dict(sorted(seq_hist.items())[:8])}")
    for k, n, c in mixed_examples:
        print(f"  e.g. #new-seq={k} #new-token={n} -> {c[0]} text + {c[1]} image")


if __name__ == "__main__":
    main()
