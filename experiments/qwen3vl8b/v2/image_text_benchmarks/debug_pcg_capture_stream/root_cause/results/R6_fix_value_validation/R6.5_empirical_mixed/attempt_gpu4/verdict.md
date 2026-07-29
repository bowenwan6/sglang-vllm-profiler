# R6.5 empirical mixed-workload — **AMBIGUOUS**

- analytical p* (from R6.4): -3.9120377283468675
- ratios executed: ['ratio_0p2', 'ratio_0p5', 'ratio_0p8']

| ratio_id | text_ratio | mean_lat_stock | mean_lat_fork | fork/stock | empirical fork wins | predicted fork wins | agree? |
|---|---|---|---|---|---|---|---|
| `ratio_0p2` | 0.2 | 0.6170331189506396 | 0.6287572392861226 | 1.0190007958655796 | False | True | False |
| `ratio_0p5` | 0.5 | 0.6583166645291203 | 0.6613469882306526 | 1.004603139894233 | False | True | False |
| `ratio_0p8` | 0.8 | 0.737395544500032 | 0.6761194712799624 | 0.9169020294777942 | True | True | True |

## Reasons
- ratio_0p2/fork_pcg: request_failures=93
- ratio_0p8/stock_default: request_failures=94
- only 1/3 ratios agree with analytical direction

## Overall verdict: **AMBIGUOUS**
