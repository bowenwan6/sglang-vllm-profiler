"""
Phase 0 — Tier B: greedy output collection from vLLM + comparison against SGLang.

Usage:
    python tier_b_vllm_compare.py \
        --port 30001 \
        --sglang-output sglang_outputs.json \
        --output vllm_outputs.json

Sends the same 3 prompts to vLLM with temperature=0, top_p=1, max_tokens=128.
Saves vLLM outputs, then compares first-token and full-output against the
SGLang reference file produced by tier_b_sglang.py.

Exit code 0 = all Tier-B checks pass.
Exit code 1 = first-token divergence detected (blocker per plan §Phase 0).
"""
import argparse, json, sys, urllib.request

PROMPTS = [
    "What is 2+2? Answer in one word.",
    "Explain gradient descent in exactly one sentence.",
    "Write a Python function that reverses a string. Just the code.",
]

def query(port, prompt, model_name, max_tokens=128):
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["message"]["content"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--sglang-output", default="sglang_outputs.json")
    parser.add_argument("--output", default="vllm_outputs.json")
    args = parser.parse_args()

    models = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models").read())
    model_id = models["data"][0]["id"]
    print(f"vLLM model_id: {model_id!r}")

    vllm_outputs = []
    for p in PROMPTS:
        out = query(args.port, p, model_id)
        vllm_outputs.append(out)
        print(f"\nPrompt : {p!r}")
        print(f"Output : {out!r}")

    result = {"framework": "vllm", "port": args.port,
              "model_id": model_id, "prompts": PROMPTS, "outputs": vllm_outputs}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {args.output}")

    # Comparison
    with open(args.sglang_output) as f:
        sg = json.load(f)

    print("\n" + "=" * 60)
    print("Tier B comparison — SGLang vs vLLM")
    print("=" * 60)
    first_token_fail = False
    for i, (p, so, vo) in enumerate(zip(PROMPTS, sg["outputs"], vllm_outputs)):
        exact = so.strip() == vo.strip()
        first_tok = so.strip().split()[0].lower() == vo.strip().split()[0].lower()
        if not first_tok:
            first_token_fail = True
        label = "EXACT" if exact else ("FIRST-TOK-MATCH" if first_tok else "DIVERGE")
        print(f"\n  [{label}] Prompt {i}: {p!r}")
        if not exact:
            print(f"    SGLang : {so[:120]!r}")
            print(f"    vLLM   : {vo[:120]!r}")

    if first_token_fail:
        print("\nTier-B FAIL — first-token divergence (blocker, see plan §Phase 0)")
        sys.exit(1)
    else:
        print("\nTier-B PASS — first tokens match on all prompts")
        sys.exit(0)

if __name__ == "__main__":
    main()
