# V2 — tiny `<|video_pad|>` serving repro

> Run: 2026-05-31T20:34:53.117859+00:00  GPU=7  clean (no KAPI / no profiler)
> SGLang server, single image-only `/v1/chat/completions` probes (NOT a benchmark).

## Verdict: PASS

| probe | text | status | expectation | met |
|---|---|---|---|---|
| A (failing) | `<|video_pad|>describe the image` | 400 | 400 + `No data iterator found for token: <|video_pad|>` | True |
| B (control) | `describe the image` | 200 | 200 + non-empty | True |

## Probe A error body (head)
```
{"object":"error","message":"No data iterator found for token: <|video_pad|>","type":"BadRequestError","param":null,"code":400}
```

## Probe B output (head)
```
This image is a classic example of **
```

## Interpretation

V2 confirms the serving symptom is real: an image-only request whose text contains `<|video_pad|>` returns HTTP 400, while safe text succeeds. This does **not** make SGLang serving the primary bug — the server is correctly rejecting a video placeholder with no video payload. The fix target is the benchmark **generator** (`gen_mm_prompt`), which must not emit such tokens in synthetic random text (validated separately in V1).