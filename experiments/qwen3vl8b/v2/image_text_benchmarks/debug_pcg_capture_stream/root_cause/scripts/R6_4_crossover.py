#!/usr/bin/env python3
"""R6.4 analytical crossover for mixed text+image workloads.

Reads:
  * R6.2 verdict.json -> per-variant rep-level mean_ttft (text-only Case A)
  * R6.3 verdict.json -> R6.3a rep-level mean_ttft (image IMG-A)

Computes:
  * G = mean(stock_default text) - mean(fork_pcg text)   # retained text gain
  * C = mean(fork_pcg image) - mean(stock_default image) # image path cost
  * p* = C / (G + C)  where p is the fraction of text requests
    such that:
      mean_off(p) = p * mean_stock_default_text + (1-p) * mean_stock_default_image
      mean_on(p)  = p * mean_fork_pcg_text     + (1-p) * mean_fork_pcg_image
      mean_on(p*) == mean_off(p*)

Bootstrap CI on p* from the rep-level samples.
Table at p ∈ {0.5, 0.7, 0.8, p*, 0.9, 0.95, 1.0}.
"""
from __future__ import annotations
import argparse, json, math, random, statistics, sys
from pathlib import Path

random.seed(42)


def mean(xs): return sum(xs)/len(xs) if xs else float("nan")


def bootstrap_pstar(sd_text, fp_text, sd_img, fp_img, n_boot=2000):
    ps = []
    for _ in range(n_boot):
        sdt = [random.choice(sd_text) for _ in sd_text]
        fpt = [random.choice(fp_text) for _ in fp_text]
        sdi = [random.choice(sd_img) for _ in sd_img]
        fpi = [random.choice(fp_img) for _ in fp_img]
        G = mean(sdt) - mean(fpt)
        C = mean(fpi) - mean(sdi)
        if (G + C) == 0:
            continue
        ps.append(C / (G + C))
    if not ps: return None
    ps.sort()
    lo = ps[int(0.025 * len(ps))]
    hi = ps[int(0.975 * len(ps))]
    return {"n_boot": len(ps), "mean": mean(ps), "median": statistics.median(ps),
            "ci95_lo": lo, "ci95_hi": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r62-json", required=True, type=Path)
    ap.add_argument("--r63-json", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    r62 = json.loads(args.r62_json.read_text())
    r63 = json.loads(args.r63_json.read_text())

    def rep_ttfts(v_dict, key="mean_ttft_ms"):
        reps = v_dict.get("reps", [])
        return [r.get(key) for r in reps if isinstance(r.get(key), (int, float))]

    # R6.2: text-only means
    sd_text = rep_ttfts(r62["per_variant"]["stock_default"])
    fp_text = rep_ttfts(r62["per_variant"]["fork_pcg"])

    # R6.3a: image means
    sd_img = rep_ttfts(r63["rebaseline"]["stock_default"])
    fp_img = rep_ttfts(r63["rebaseline"]["fork_pcg"])

    if not (sd_text and fp_text and sd_img and fp_img):
        print("MISSING INPUT DATA", file=sys.stderr)
        # Still emit a partial artifact
        args.out_md.write_text("# R6.4 crossover — INSUFFICIENT DATA\n")
        args.out_json.write_text(json.dumps({"verdict": "INSUFFICIENT_DATA"}, indent=2))
        return 3

    mean_sd_text, mean_fp_text = mean(sd_text), mean(fp_text)
    mean_sd_img, mean_fp_img = mean(sd_img), mean(fp_img)
    G = mean_sd_text - mean_fp_text
    C = mean_fp_img - mean_sd_img
    if (G + C) == 0:
        p_star = None
    else:
        p_star = C / (G + C)

    boot = bootstrap_pstar(sd_text, fp_text, sd_img, fp_img)

    # Sensitivity table
    ps_list = [0.5, 0.7, 0.8]
    if p_star is not None and 0 <= p_star <= 1:
        ps_list.append(round(p_star, 4))
    ps_list.extend([0.9, 0.95, 1.0])
    ps_list = sorted(set(ps_list))
    table = []
    for p in ps_list:
        off = p * mean_sd_text + (1 - p) * mean_sd_img
        on  = p * mean_fp_text + (1 - p) * mean_fp_img
        table.append({"p": p, "mean_off": off, "mean_on": on, "on_over_off": on/off if off else None,
                      "fork_wins": on <= off})

    verdict = "PASS"
    reasons = []
    if p_star is None:
        verdict = "AMBIGUOUS"; reasons.append("G + C == 0; crossover undefined")
    elif not (0 <= p_star <= 1):
        verdict = "AMBIGUOUS"; reasons.append(f"p* = {p_star:.4f} outside [0,1]")

    L = [f"# R6.4 analytical crossover — **{verdict}**", ""]
    L.append("> Rep-level arithmetic means; p* = C / (G+C). Bootstrap CI on p*.")
    L.append("")
    L.append("## Inputs (rep-level mean_ttft_ms)")
    L.append(f"- stock_default text (R6.2): n={len(sd_text)}, mean={mean_sd_text:.4f}, values={sd_text}")
    L.append(f"- fork_pcg     text (R6.2): n={len(fp_text)}, mean={mean_fp_text:.4f}, values={fp_text}")
    L.append(f"- stock_default image (R6.3a): n={len(sd_img)}, mean={mean_sd_img:.4f}, values={sd_img}")
    L.append(f"- fork_pcg     image (R6.3a): n={len(fp_img)}, mean={mean_fp_img:.4f}, values={fp_img}")
    L.append("")
    L.append("## Crossover")
    L.append(f"- G (retained text gain) = {G:.4f} ms")
    L.append(f"- C (image path cost)    = {C:.4f} ms")
    L.append(f"- p* (analytical)        = {p_star:.4f}" if p_star is not None else "- p* undefined")
    if boot:
        L.append(f"- bootstrap p* (2000 resamples): mean={boot['mean']:.4f}, median={boot['median']:.4f}, "
                 f"95% CI [{boot['ci95_lo']:.4f}, {boot['ci95_hi']:.4f}]")
    L.append("")
    L.append("## Ratio table (mixed workload mean TTFT)")
    L.append("| p (text fraction) | mean_off (stock) | mean_on (fork) | on/off | fork wins? |")
    L.append("|---|---|---|---|---|")
    for row in table:
        L.append(f"| {row['p']:.4f} | {row['mean_off']:.4f} | {row['mean_on']:.4f} | "
                 f"{row['on_over_off']:.4f} | {'✅' if row['fork_wins'] else '❌'} |")
    L.append("")
    L.append("## Interpretation")
    L.append("- This is an **analytical** crossover from independent per-run means.")
    L.append("- Not an empirical mixed-workload measurement (R6.5 does that).")
    L.append(f"- Verdict: **{verdict}**")
    for r in reasons: L.append(f"- reason: {r}")

    args.out_md.write_text("\n".join(L) + "\n")
    args.out_json.write_text(json.dumps({
        "verdict": verdict, "reasons": reasons,
        "G_ms": G, "C_ms": C, "p_star": p_star, "bootstrap": boot,
        "inputs": {"sd_text": sd_text, "fp_text": fp_text,
                   "sd_img": sd_img, "fp_img": fp_img,
                   "mean_sd_text": mean_sd_text, "mean_fp_text": mean_fp_text,
                   "mean_sd_img": mean_sd_img, "mean_fp_img": mean_fp_img},
        "table": table,
    }, indent=2, sort_keys=True))
    print(f"VERDICT={verdict} p_star={p_star}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
