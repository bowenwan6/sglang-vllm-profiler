#!/usr/bin/env python3
"""R6 host-libcuda preflight.

Called by R6 GPU runners after they set
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05
    CUDA_VISIBLE_DEVICES=<physical GPU>
Verifies that libcuda.so.1 resolves to that exact host path (NOT to
any /usr/local/cuda-*/compat/libcuda.so.*), prints torch / driver /
GPU identity, and runs a minimal CUDA tensor smoke on the given
*physical* GPU (which torch sees as its logical index 0 because
CUDA_VISIBLE_DEVICES restricts visibility to a single device).

Rationale: this container ships cuda-compat-13-0 which places
libcuda.so.1 → libcuda.so.580.82.07 in /usr/local/cuda-13.0/compat/.
That copy is older than the host driver (595.71.05) and torch 2.11.0
+cu130 fails cudaGetDeviceCount() against it with Error 803.
LD_PRELOAD'ing the host library fixes the resolution order.

Exit codes:
    0  preflight PASS
    3  libcuda.so.1 could not be loaded
    4  resolved libcuda path is not the required host path (or is
       under /usr/local/cuda-*/compat/)
    5  CUDA smoke returned an unexpected value
    64 bad usage
"""
from __future__ import annotations

import argparse
import ctypes
import os
import re
import subprocess
import sys

REQUIRED_LIBCUDA = "/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05"


def resolve_libcuda() -> tuple[str | None, str | None]:
    """Load libcuda.so.1 via ctypes, then find its file-backed mapping
    in /proc/self/maps and return that path.
    """
    try:
        _ = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        return None, f"ctypes.CDLL('libcuda.so.1') failed: {e!r}"
    try:
        with open("/proc/self/maps", "r") as f:
            for line in f:
                m = re.search(r"(\S*/libcuda\.so[\w.\-]*)", line)
                if m:
                    return m.group(1), None
    except OSError as e:
        return None, f"could not read /proc/self/maps: {e!r}"
    return None, "libcuda not present in /proc/self/maps after CDLL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True,
                    help="GPU index the runner is about to use")
    args = ap.parse_args()

    path, err = resolve_libcuda()
    if err or path is None:
        print(f"FAIL: {err}", file=sys.stderr)
        return 3
    if "/compat/" in path:
        print(f"FAIL: compat libcuda loaded: {path}\n"
              f"       (LD_PRELOAD is not effective; check the "
              f"       runner env)", file=sys.stderr)
        return 4
    if path != REQUIRED_LIBCUDA:
        print(f"FAIL: loaded libcuda is not the pinned host lib\n"
              f"       loaded:   {path}\n"
              f"       required: {REQUIRED_LIBCUDA}", file=sys.stderr)
        return 4
    print(f"OK: libcuda.so.1 -> {path}")

    import torch
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")

    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version",
         "--format=csv,noheader", "-i", str(args.gpu)],
        capture_output=True, text=True, timeout=10)
    print(f"nvidia driver (physical GPU {args.gpu}): {r.stdout.strip()}")
    print(f"target physical GPU: {args.gpu}")

    # The runner has already set CUDA_VISIBLE_DEVICES=<physical GPU>
    # before spawning us. Under that env, torch sees only one device
    # and numbers it cuda:0 regardless of the physical index. Verify
    # that mapping before running the smoke so we fail loudly if the
    # env was not set as expected.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES: {cvd!r}")
    if not cvd:
        print("FAIL: CUDA_VISIBLE_DEVICES not set; runner did not "
              "restrict visibility before preflight", file=sys.stderr)
        return 4
    cvd_list = [x.strip() for x in cvd.split(",") if x.strip()]
    if cvd_list != [str(args.gpu)]:
        print(f"FAIL: CUDA_VISIBLE_DEVICES={cvd!r} does not restrict "
              f"visibility to exactly the approved GPU {args.gpu}",
              file=sys.stderr)
        return 4
    device_count = torch.cuda.device_count()
    if device_count != 1:
        print(f"FAIL: torch.cuda.device_count()={device_count} "
              f"(expected 1 under CVD={cvd!r})", file=sys.stderr)
        return 4
    logical_name = torch.cuda.get_device_name(0)
    print(f"torch sees {device_count} device -> cuda:0 ({logical_name})")

    # cuda:0 here is the physical GPU restricted by CVD.
    dev = torch.device("cuda:0")
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
    b = a * 2
    got = b.sum().item()
    expected = 20.0
    if abs(got - expected) > 1e-6:
        print(f"FAIL: smoke sum {got} != {expected}", file=sys.stderr)
        return 5
    print(f"CUDA_SMOKE_OK: sum([2,4,6,8]) = {got}")

    # Free the tensor before returning so the runner sees a low
    # memory reading immediately after (though the CUDA context
    # itself will keep a few hundred MiB reserved until process exit).
    del a, b
    torch.cuda.synchronize(dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
