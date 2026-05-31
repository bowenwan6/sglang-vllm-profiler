# D0 — Payload audit results (`<|video_pad|>` blocker)

> Run: 2026-05-31T20:21:13.636368+00:00  (NO GPU, NO server)
> Sample: 30 seeds × 430 prompts = 12900 prompts, input_len=128

## Verdict

✅ **ROOT CAUSE CONFIRMED.** The buggy generator emits forbidden multimodal special tokens; the sanitized generator does not.

## Token identities

- `image_pad_id` = 151655 (`<|image_pad|>`) — excluded by current generator
- `video_pad_id` = 151656 (`<|video_pad|>`) — **NOT excluded** (the bug)
- total special ids = 14 (range 151643–151656)

## Buggy generator (current `gen_mm_prompt`)

- prompts containing `<|video_pad|>`: 8 / 12900 = 0.062%
- prompts containing ANY forbidden mm token: 42 / 12900 = 0.3256%
- expected failures per 430-request rep: 1.4

## Sanitized generator (proposed: exclude all special ids)

- prompts containing ANY forbidden mm token: 0 / 12900
- clean: ✅ yes

## Example hits (sanitized snippets, hashes only — no full prompts)

| seed | index | tokens | prompt_sha12 | snippet |
|---|---|---|---|---|
| 1 | 42 | `<|vision_pad|>` | `e8b33e051cd9` | `…n uttered Schul LINE<\|vision_pad\|>Ә那样的añ Higgins"math率领кер/p…` |
| 1 | 78 | `<|video_pad|>` | `6f18ecf41b92` | `…<\|video_pad\|>ಲ Goblin mysterious Seconda…` |
| 1 | 139 | `<|vision_pad|>` | `4e49bb71de8e` | `…hority convertersינה<\|vision_pad\|>.Global enlight:j pav题材 fp…` |
| 1 | 385 | `<|vision_pad|>` | `2529ea8fe44a` | `…li prisonנם.Paths.Ex<\|vision_pad\|>） 还不如乐园交融 amongรักษาmaze.P…` |
| 2 | 31 | `<|video_pad|>` | `8fa46da3b040` | `…ticalangles.event及以上<\|video_pad\|>ulturalGl combust-rounded q…` |
| 2 | 417 | `<|vision_pad|>` | `a7be68b67660` | `…s 로 gums𣷭.animation노<\|vision_pad\|>生命的_eventsfre Louisfal뻑䲟/d…` |
| 3 | 253 | `<|vision_start|>` | `9ef0916c64db` | `…eANY institutesitag蠼<\|vision_start\|>lxdings面貌/memoryINTEGERO…` |
| 4 | 85 | `<|vision_pad|>` | `3d538f942e18` | `…ir battleConditional<\|vision_pad\|>(sooguemarine Maurit đang�…` |
| 5 | 83 | `<|vision_pad|>` | `3ea3213f711d` | `…iazza𨱔 pasa.seｶ?>"> <\|vision_pad\|> yat Vir中华人民 מק]\=[]  roo…` |
| 6 | 182 | `<|vision_end|>` | `f13ad366e3a2` | `…影音.Paths_NOTEнемrive<\|vision_end\|>)?.Migration>): 항-tech_Com…` |

## Conclusion

The failure is a **benchmark-generator special-token bug**, not a server cache/state or CUDA-IPC performance effect. Sanitizing the random text (exclude all special ids) eliminates the forbidden tokens. The runner can proceed with a sanitized prompt path; an upstream `gen_mm_prompt` fix is a follow-up.