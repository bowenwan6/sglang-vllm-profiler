"""Send one deterministic image prompt and one text prompt, greedily.

The image is generated in-process (fixed RGB vertical stripes) so the arms are
byte-identical inputs with no download or disk dependency.
"""

import base64
import io
import json
import os
import urllib.request

from PIL import Image

PORT = os.environ["PORT"]
ARM = os.environ["ARM"]


def stripe_image_b64(w=336, h=336, stripe=42):
    img = Image.new("RGB", (w, h))
    px = img.load()
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for x in range(w):
        c = palette[(x // stripe) % len(palette)]
        for y in range(h):
            px[x, y] = c
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def post(payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


b64 = stripe_image_b64()
results = {}

image_msg = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
            {"type": "text", "text": "Describe the colors in this image in order."},
        ],
    }
]
text_msg = [{"role": "user", "content": "Name the first four prime numbers."}]

for label, msgs in [("image", image_msg), ("text", text_msg)]:
    out = post(
        {
            "model": "default",
            "messages": msgs,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 48,
            "seed": 0,
        }
    )
    results[label] = out["choices"][0]["message"]["content"]

print(json.dumps({"arm": ARM, **results}, ensure_ascii=False, indent=2))
