## Debug ladder (one slot, a long request holding it)

| run | build | checks passed | failed |
|---|---|---|---|
| ladder_dummy | patched | 17 of 17 | — |
| ladder_real | patched | 17 of 17 | — |
| ladder_global2 | patched | 5 of 5 | — |
| ladder_control | base | 2 of 2 | — |
| v2_ladder_real | patched | 17 of 17 | — |
| v2_ladder_global2 | patched | 5 of 5 | — |
| ladder_dummy | patched | 17 of 17 | — |
| ladder_real | patched | 17 of 17 | — |
| ladder_global2 | patched | 5 of 5 | — |
| ladder_control | base | 2 of 2 | — |
| tp2_ladder_real | patched | 17 of 17 | — |
| tp2_ladder_global2 | patched | 5 of 5 | — |

**ladder_dummy**

| check | result | observed |
|---|---|---|
| R0 idle server, no field | pass | http 200, 64 tokens in 0.20 s |
| R2 /generate, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.00 s |
| R3 /generate, streaming | pass | http 200, abort event True, token False, after 1.00 s |
| R4 chat, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R5 chat, streaming | pass | http 200, abort event True, token False, after 1.00 s |
| R6 completions, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.00 s |
| R6b completions, streaming | pass | http 200, abort event True, token False, after 1.00 s |
| R9 /generate rejects waiting_timeout=0 | pass | http 400 after 0.00 s: {'error': {'message': 'waiting_timeout should be a positive, finite number of seconds.'}} |
| R9 /generate rejects waiting_timeout=-1 | pass | http 400 after 0.00 s: {'error': {'message': 'waiting_timeout should be a positive, finite number of seconds.'}} |
| R9 /generate rejects waiting_timeout=abc | pass | http 400 after 0.00 s: 1 validation error:
  {'type': 'float_parsing', 'loc': ('body', 'waiting_timeout'), 'msg': |
| R9 /generate rejects waiting_timeout=10000000 | pass | http 400 after 0.00 s: 1 validation error:
  {'type': 'json_invalid', 'loc': ('body', 138), 'msg': 'JSON decode e |
| R9 chat rejects waiting_timeout=0 | pass | http 400 after 0.00 s |
| R7/R8 still queued behind the holder after 2 s | pass | loose done False, unbounded done False, holder done False |
| R1 the holder is undisturbed | pass | http 200, 9416 of 9416 tokens, finish length, 16.0 s |
| R7 loose bound (3600 s) is served once the slot frees | pass | http 200, 16 tokens after 8.5 s |
| R8 no bound is served once the slot frees | pass | http 200, 16 tokens after 8.5 s |
| server healthy at the end | pass | http 200 |

**ladder_real**

| check | result | observed |
|---|---|---|
| R0 idle server, no field | pass | http 200, 64 tokens in 0.34 s |
| R2 /generate, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R3 /generate, streaming | pass | http 200, abort event True, token False, after 1.02 s |
| R4 chat, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R5 chat, streaming | pass | http 200, abort event True, token False, after 1.01 s |
| R6 completions, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R6b completions, streaming | pass | http 200, abort event True, token False, after 1.01 s |
| R9 /generate rejects waiting_timeout=0 | pass | http 400 after 0.00 s: {'error': {'message': 'waiting_timeout should be a positive, finite number of seconds.'}} |
| R9 /generate rejects waiting_timeout=-1 | pass | http 400 after 0.00 s: {'error': {'message': 'waiting_timeout should be a positive, finite number of seconds.'}} |
| R9 /generate rejects waiting_timeout=abc | pass | http 400 after 0.00 s: 1 validation error:
  {'type': 'float_parsing', 'loc': ('body', 'waiting_timeout'), 'msg': |
| R9 /generate rejects waiting_timeout=10000000 | pass | http 400 after 0.00 s: 1 validation error:
  {'type': 'json_invalid', 'loc': ('body', 138), 'msg': 'JSON decode e |
| R9 chat rejects waiting_timeout=0 | pass | http 400 after 0.00 s |
| R7/R8 still queued behind the holder after 2 s | pass | loose done False, unbounded done False, holder done False |
| R1 the holder is undisturbed | pass | http 200, 5608 of 5608 tokens, finish length, 28.9 s |
| R7 loose bound (3600 s) is served once the slot frees | pass | http 200, 16 tokens after 21.4 s |
| R8 no bound is served once the slot frees | pass | http 200, 16 tokens after 21.5 s |
| server healthy at the end | pass | http 200 |

**ladder_global2**

| check | result | observed |
|---|---|---|
| G1 request 30 s, global 2 s | pass | http 503, message 'Request waiting timeout reached.', after 2.01 s |
| G2 request 0.5 s, global 2 s | pass | http 503, message 'Request waiting timeout reached.', after 0.51 s |
| G3 no field, global 2 s | pass | http 503, message 'Request waiting timeout reached.', after 2.01 s |
| G4 request 0.5 s, streaming chat | pass | http 200, abort event True, token False, after 0.51 s |
| server healthy at the end | pass | http 200 |

**ladder_control**

| check | result | observed |
|---|---|---|
| C1 unpatched build ignores the field | pass | http 200, 16 tokens after 8.8 s (bound 1.0 s) |
| server healthy at the end | pass | http 200 |

**v2_ladder_real**

| check | result | observed |
|---|---|---|
| R0 idle server, no field | pass | http 200, 64 tokens in 0.34 s |
| R2 /generate, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R3 /generate, streaming | pass | http 200, abort event True, token False, after 1.02 s |
| R4 chat, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R5 chat, streaming | pass | http 200, abort event True, token False, after 1.01 s |
| R6 completions, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R6b completions, streaming | pass | http 200, abort event True, token False, after 1.01 s |
| R9 /generate rejects waiting_timeout=0 | pass | http 400 after 0.00 s: {'error': {'message': 'waiting_timeout should be a positive, finite number of seconds.'}} |
| R9 /generate rejects waiting_timeout=-1 | pass | http 400 after 0.00 s: {'error': {'message': 'waiting_timeout should be a positive, finite number of seconds.'}} |
| R9 /generate rejects waiting_timeout=abc | pass | http 400 after 0.00 s: 1 validation error:
  {'type': 'float_parsing', 'loc': ('body', 'waiting_timeout'), 'msg': |
| R9 /generate rejects waiting_timeout=10000000 | pass | http 400 after 0.00 s: 1 validation error:
  {'type': 'json_invalid', 'loc': ('body', 138), 'msg': 'JSON decode e |
| R9 chat rejects waiting_timeout=0 | pass | http 400 after 0.00 s |
| R7/R8 still queued behind the holder after 2 s | pass | loose done False, unbounded done False, holder done False |
| R1 the holder is undisturbed | pass | http 200, 5605 of 5605 tokens, finish length, 28.9 s |
| R7 loose bound (3600 s) is served once the slot frees | pass | http 200, 16 tokens after 21.4 s |
| R8 no bound is served once the slot frees | pass | http 200, 16 tokens after 21.5 s |
| server healthy at the end | pass | http 200 |

**v2_ladder_global2**

| check | result | observed |
|---|---|---|
| G1 request 30 s, global 2 s | pass | http 503, message 'Request waiting timeout reached.', after 2.01 s |
| G2 request 0.5 s, global 2 s | pass | http 503, message 'Request waiting timeout reached.', after 0.51 s |
| G3 no field, global 2 s | pass | http 503, message 'Request waiting timeout reached.', after 2.01 s |
| G4 request 0.5 s, streaming chat | pass | http 200, abort event True, token False, after 0.51 s |
| server healthy at the end | pass | http 200 |

**ladder_dummy**

| check | result | observed |
|---|---|---|
| R0 idle server, no field | pass | http 200, 64 tokens in 0.20 s |
| R2 /generate, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R3 /generate, streaming | pass | http 200, abort event True, token False, after 1.00 s |
| R4 chat, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R5 chat, streaming | pass | http 200, abort event True, token False, after 1.00 s |
| R6 completions, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.00 s |
| R6b completions, streaming | pass | http 200, abort event True, token False, after 1.00 s |
| R9 /generate rejects waiting_timeout=0 | pass | http 400 after 0.00 s: {'error': {'message': 'waiting_timeout should be a positive, finite number of seconds.'}} |
| R9 /generate rejects waiting_timeout=-1 | pass | http 400 after 0.00 s: {'error': {'message': 'waiting_timeout should be a positive, finite number of seconds.'}} |
| R9 /generate rejects waiting_timeout=abc | pass | http 400 after 0.00 s: 1 validation error:
  {'type': 'float_parsing', 'loc': ('body', 'waiting_timeout'), 'msg': |
| R9 /generate rejects waiting_timeout=10000000 | pass | http 400 after 0.00 s: 1 validation error:
  {'type': 'json_invalid', 'loc': ('body', 138), 'msg': 'JSON decode e |
| R9 chat rejects waiting_timeout=0 | pass | http 400 after 0.00 s |
| R7/R8 still queued behind the holder after 2 s | pass | loose done False, unbounded done False, holder done False |
| R1 the holder is undisturbed | pass | http 200, 9401 of 9401 tokens, finish length, 16.2 s |
| R7 loose bound (3600 s) is served once the slot frees | pass | http 200, 16 tokens after 8.7 s |
| R8 no bound is served once the slot frees | pass | http 200, 16 tokens after 8.7 s |
| server healthy at the end | pass | http 200 |

**ladder_real**

| check | result | observed |
|---|---|---|
| R0 idle server, no field | pass | http 200, 64 tokens in 0.34 s |
| R2 /generate, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R3 /generate, streaming | pass | http 200, abort event True, token False, after 1.02 s |
| R4 chat, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R5 chat, streaming | pass | http 200, abort event True, token False, after 1.01 s |
| R6 completions, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R6b completions, streaming | pass | http 200, abort event True, token False, after 1.01 s |
| R9 /generate rejects waiting_timeout=0 | pass | http 400 after 0.00 s: {'error': {'message': 'waiting_timeout should be a positive, finite number of seconds.'}} |
| R9 /generate rejects waiting_timeout=-1 | pass | http 400 after 0.00 s: {'error': {'message': 'waiting_timeout should be a positive, finite number of seconds.'}} |
| R9 /generate rejects waiting_timeout=abc | pass | http 400 after 0.00 s: 1 validation error:
  {'type': 'float_parsing', 'loc': ('body', 'waiting_timeout'), 'msg': |
| R9 /generate rejects waiting_timeout=10000000 | pass | http 400 after 0.00 s: 1 validation error:
  {'type': 'json_invalid', 'loc': ('body', 138), 'msg': 'JSON decode e |
| R9 chat rejects waiting_timeout=0 | pass | http 400 after 0.00 s |
| R7/R8 still queued behind the holder after 2 s | pass | loose done False, unbounded done False, holder done False |
| R1 the holder is undisturbed | pass | http 200, 5652 of 5652 tokens, finish length, 29.0 s |
| R7 loose bound (3600 s) is served once the slot frees | pass | http 200, 16 tokens after 21.6 s |
| R8 no bound is served once the slot frees | pass | http 200, 16 tokens after 21.7 s |
| server healthy at the end | pass | http 200 |

**ladder_global2**

| check | result | observed |
|---|---|---|
| G1 request 30 s, global 2 s | pass | http 503, message 'Request waiting timeout reached.', after 2.01 s |
| G2 request 0.5 s, global 2 s | pass | http 503, message 'Request waiting timeout reached.', after 0.51 s |
| G3 no field, global 2 s | pass | http 503, message 'Request waiting timeout reached.', after 2.01 s |
| G4 request 0.5 s, streaming chat | pass | http 200, abort event True, token False, after 0.51 s |
| server healthy at the end | pass | http 200 |

**ladder_control**

| check | result | observed |
|---|---|---|
| C1 unpatched build ignores the field | pass | http 200, 16 tokens after 8.8 s (bound 1.0 s) |
| server healthy at the end | pass | http 200 |

**tp2_ladder_real**

| check | result | observed |
|---|---|---|
| R0 idle server, no field | pass | http 200, 64 tokens in 0.25 s |
| R2 /generate, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R3 /generate, streaming | pass | http 200, abort event True, token False, after 1.01 s |
| R4 chat, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R5 chat, streaming | pass | http 200, abort event True, token False, after 1.00 s |
| R6 completions, non-streaming | pass | http 503, message 'Request waiting timeout reached.', after 1.01 s |
| R6b completions, streaming | pass | http 200, abort event True, token False, after 1.01 s |
| R9 /generate rejects waiting_timeout=0 | pass | http 400 after 0.00 s: {'error': {'message': 'waiting_timeout should be a positive, finite number of seconds.'}} |
| R9 /generate rejects waiting_timeout=-1 | pass | http 400 after 0.00 s: {'error': {'message': 'waiting_timeout should be a positive, finite number of seconds.'}} |
| R9 /generate rejects waiting_timeout=abc | pass | http 400 after 0.00 s: 1 validation error:
  {'type': 'float_parsing', 'loc': ('body', 'waiting_timeout'), 'msg': |
| R9 /generate rejects waiting_timeout=10000000 | pass | http 400 after 0.00 s: 1 validation error:
  {'type': 'json_invalid', 'loc': ('body', 138), 'msg': 'JSON decode e |
| R9 chat rejects waiting_timeout=0 | pass | http 400 after 0.00 s |
| R7/R8 still queued behind the holder after 2 s | pass | loose done False, unbounded done False, holder done False |
| R1 the holder is undisturbed | pass | http 200, 7716 of 7716 tokens, finish length, 27.2 s |
| R7 loose bound (3600 s) is served once the slot frees | pass | http 200, 16 tokens after 19.7 s |
| R8 no bound is served once the slot frees | pass | http 200, 16 tokens after 19.8 s |
| server healthy at the end | pass | http 200 |

**tp2_ladder_global2**

| check | result | observed |
|---|---|---|
| G1 request 30 s, global 2 s | pass | http 503, message 'Request waiting timeout reached.', after 2.01 s |
| G2 request 0.5 s, global 2 s | pass | http 503, message 'Request waiting timeout reached.', after 0.51 s |
| G3 no field, global 2 s | pass | http 503, message 'Request waiting timeout reached.', after 2.01 s |
| G4 request 0.5 s, streaming chat | pass | http 200, abort event True, token False, after 0.51 s |
| server healthy at the end | pass | http 200 |

## Capacity of each class alone (closed loop, concurrency 128, `--max-running-requests 128`)

| class | requests | req/s | output tok/s | mean TTFT ms | mean TPOT ms |
|---|---|---|---|---|---|
| chat | 1280 | 66.59 | 8524 | 446 | 11.2 |
| batch | 512 | 19.49 | 4991 | 1646 | 17.9 |

## The client against `bench_serving --goodput` (32.851 req/s, 256-token prompts, 128 output tokens)

| tool | requests | attainment % | mean TTFT ms | mean TPOT ms | output tok/s |
|---|---|---|---|---|---|
| slo_client.py | 1323 | 100.0 | 20 | 7.9 | 4163 |
| bench_serving | 1314 | 100.0 | 25 | 8.0 | 3870 |

## U1 pilot (one seed)

| arm | runs | chat % | batch % | all % | refused % | wasted tokens % | chat mean TTFT ms | batch p99 E2E s | out tok/s |
|---|---|---|---|---|---|---|---|---|---|
| per_request | 1 | 79.0 | 100.0 | 82.5 | 17.5 | 0 | 939 | 26.0 | 6621 |

## U1 — chat bursts above capacity, steady batch, priority to chat

| arm | runs | chat % | batch % | all % | refused % | wasted tokens % | chat mean TTFT ms | batch p99 E2E s | out tok/s |
|---|---|---|---|---|---|---|---|---|---|
| none | 2 | 31.5 (0.8) | 50.3 (5.3) | 34.7 (0.1) | 0.0 | 63 | 4424 (148) | 69.8 (2.8) | 6913 (11) |
| global_1.5 | 3 | 78.9 (0.5) | 67.5 (1.0) | 77.0 (0.3) | 23.0 | 0 | 900 (2) | 4.9 (0.3) | 6234 (69) |
| per_request | 3 | 79.0 (0.4) | 100.0 (0.0) | 82.6 (0.4) | 17.4 | 0 | 956 (21) | 28.1 (2.7) | 6664 (43) |
| per_request_chat_only | 2 | 78.9 (0.1) | 100.0 (0.0) | 82.5 (0.0) | 17.5 | 0 | 950 (17) | 26.4 (0.8) | 6647 (40) |
| global_30 | 2 | 32.3 (0.8) | 81.0 (0.8) | 40.6 (0.8) | 3.2 | 51 | 4346 (203) | 35.3 (0.4) | 7062 (12) |
| global_5 | 2 | 37.4 (0.6) | 66.2 (0.4) | 42.3 (0.4) | 16.5 | 44 | 2605 (19) | 9.8 (0.4) | 6526 (65) |
| global_20 | 2 | 31.7 (1.3) | 75.3 (1.5) | 39.1 (1.0) | 4.2 | 52 | 4420 (207) | 24.7 (0.7) | 7104 (14) |
| hang_up | 2 | 69.9 (0.8) | 100.0 (0.0) | 75.0 (0.6) | 0.0 | 0 | 839 (9) | 26.3 (0.3) | 6142 (7) |

**A3.1′**: batch +32.5 pp against `global_1.5` (needs ≥ max(10, 3σ = 2.1)); chat +0.1 pp (needs ≥ −max(3, 3σ = 1.3)); total +5.6 pp against the best global arm, `global_1.5` (needs ≥ max(3, 3σ = 1.0)) → PASS.

## Two GPUs, tensor parallel (`--tp-size 2`): chat alone at 1.3 × the one-GPU c_chat, 1.5 s bound

| form | sent | attainment % | refused | statuses | mean TTFT ms | out tok/s |
|---|---|---|---|---|---|---|
| field | 5127 | 100.0 | 0 | {'ok': 5127} | 39 | 10798 |
| global | 5127 | 100.0 | 0 | {'ok': 5127} | 37 | 10785 |

## U3 — no-op control at 0.8 of capacity, a fresh server process per arm

| arm | build | attainment % | chat mean ttft ms | chat p99 ttft ms | chat mean tpot ms | batch mean ttft ms | batch p99 ttft ms | batch mean tpot ms | out tok/s |
|---|---|---|---|---|---|---|---|---|---|
| base_a | base | 100.0 | 27.7 | 82.9 | 12.1 | 43.7 | 118.2 | 11.9 | 5236 |
| patched_absent | patched | 100.0 | 27.5 | 74.8 | 12.2 | 43.1 | 90.4 | 12.1 | 5236 |
| patched_loose | patched | 100.0 | 27.5 | 73.0 | 12.1 | 43.0 | 92.7 | 12.0 | 5236 |
| base_b | base | 100.0 | 27.6 | 70.8 | 12.1 | 43.0 | 90.1 | 12.0 | 5236 |

**A3.3**: the largest excess over the allowance (the A/A spread of the unpatched build, or 3 %) is patched_loose batch mean_ttft_ms: 0.8 % from the unpatched mean, A/A spread 1.6 % → PASS.

## U4 — the same 1.5 s bound as a request field and as the global knob (chat alone, 1.3 × c_chat)

| form | seed | sent | attainment % | refused | mean TTFT ms | p99 TTFT ms |
|---|---|---|---|---|---|---|
| field | 1 | 5127 | 82.4 | 903 | 1319 | 1588 |
| field | 2 | 5172 | 80.5 | 1006 | 1372 | 1628 |
| global | 1 | 5127 | 82.2 | 913 | 1339 | 1590 |
| global | 2 | 5172 | 80.3 | 1019 | 1361 | 1637 |

**A3.4′**: field − global = +0.2 pp (allowance max(2, 3σ = 4.0)) → PASS.

## T4 with the fixed benchmark, three seeds per setting (ShareGPT, 1.5 × c0, cap 128)

| waiting timeout (s) | runs | failed requests | out tok/s | goodput (req/s) | attainment % | printed = recomputed |
|---|---|---|---|---|---|---|
| off | 3 | 0 (0) | 7743 (346) | 7.33 (0.80) | 18.7 (2.7) | yes |
| 2 | 3 | 732 (16) | 7493 (398) | 24.75 (1.65) | 48.7 (1.5) | yes |
| 10 | 3 | 376 (29) | 7608 (386) | 8.36 (0.84) | 19.0 (2.4) | yes |

## T1 load sweep with repeats (session P1's runs plus two more seeds at three points)

| × c0 | runs | out tok/s | goodput (req/s) | attainment % |
|---|---|---|---|---|
| 0.5 | 1 | 3586 | 18.30 | 100.0 |
| 0.8 | 1 | 5412 | 28.13 | 99.8 |
| 1 | 3 | 6984 (174) | 35.76 (0.83) | 99.9 (0.1) |
| 1.25 | 3 | 7727 (290) | 37.56 (2.09) | 95.2 (1.6) |
| 1.6 | 3 | 8808 (255) | 26.73 (8.89) | 59.4 (19.5) |
| 2 | 3 | 9668 (381) | 9.02 (2.03) | 18.4 (4.1) |
