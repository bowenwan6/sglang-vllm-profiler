# R6.4 analytical crossover — **AMBIGUOUS**

> Rep-level arithmetic means; p* = C / (G+C). Bootstrap CI on p*.

## Inputs (rep-level mean_ttft_ms)
- stock_default text (R6.2): n=5, mean=26.8620, values=[26.870109472365584, 25.341953179886332, 25.37769448965264, 27.625336036981025, 29.094949082791572]
- fork_pcg     text (R6.2): n=5, mean=17.6504, values=[18.277023779519368, 17.5296024405543, 17.38324292276957, 17.49006175032264, 17.571855669848446]
- stock_default image (R6.3a): n=3, mean=94.3477, values=[111.22890760772862, 85.31447166555154, 86.49983157800307]
- fork_pcg     image (R6.3a): n=3, mean=87.0114, values=[108.95013954504975, 75.3129492096923, 76.77113389738224]

## Crossover
- G (retained text gain) = 9.2117 ms
- C (image path cost)    = -7.3363 ms
- p* (analytical)        = -3.9120
- bootstrap p* (2000 resamples): mean=1.0827, median=0.4129, 95% CI [-12.3901, 15.4413]

## Ratio table (mixed workload mean TTFT)
| p (text fraction) | mean_off (stock) | mean_on (fork) | on/off | fork wins? |
|---|---|---|---|---|
| 0.5000 | 60.6049 | 52.3309 | 0.8635 | ✅ |
| 0.7000 | 47.1077 | 38.4587 | 0.8164 | ✅ |
| 0.8000 | 40.3592 | 31.5226 | 0.7811 | ✅ |
| 0.9000 | 33.6106 | 24.5865 | 0.7315 | ✅ |
| 0.9500 | 30.2363 | 21.1184 | 0.6984 | ✅ |
| 1.0000 | 26.8620 | 17.6504 | 0.6571 | ✅ |

## Interpretation
- This is an **analytical** crossover from independent per-run means.
- Not an empirical mixed-workload measurement (R6.5 does that).
- Verdict: **AMBIGUOUS**
- reason: p* = -3.9120 outside [0,1]
