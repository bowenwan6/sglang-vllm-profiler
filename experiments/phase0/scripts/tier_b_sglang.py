"""
Phase 0 — Tier B: greedy output collection from SGLang.

Usage:
    python tier_b_sglang.py --port 30000 --output sglang_outputs.json

Sends 3 fixed prompts with temperature=0, top_p=1, max_tokens=128.
Saves prompts + outputs to --output for later comparison.
"""
import argparse, json, urllib.request

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
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--output", default="sglang_outputs.json")
    args = parser.parse_args()

    models = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models").read())
    model_id = models["data"][0]["id"]
    print(f"model_id: {model_id!r}")

    outputs = []
    for p in PROMPTS:
        out = query(args.port, p, model_id)
        outputs.append(out)
        print(f"\nPrompt : {p!r}")
        print(f"Output : {out!r}")

    result = {"framework": "sglang", "port": args.port,
              "model_id": model_id, "prompts": PROMPTS, "outputs": outputs}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {args.output}")

if __name__ == "__main__":
    main()
