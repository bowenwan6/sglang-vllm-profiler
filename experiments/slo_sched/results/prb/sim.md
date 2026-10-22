# Queue model — which timeout setting can win in which scenario

> Output of `scripts/prb_sim.py --seeds 8` (128 slots, chat 2 s of service and a 2 s first-token
> objective, batch 4 s of service and a 40 s deadline, 180 s per run). A model, not a measurement:
> service time does not depend on batch occupancy. Columns are SLO attainment per class and overall;
> "wasted work" is slot time spent on requests that missed their objective.

## A  both classes surge together, FCFS (U1 as first planned)

| arm | chat % | batch % | all % (sd) | dropped % | wasted work % |
|---|---|---|---|---|---|
| none | 33 | 100 | 66.7 (0.9) | 0 | 22 |
| global 1.5 | 77 | 77 | 77.1 (0.8) | 23 | 0 |
| global 5 | 51 | 83 | 66.7 (0.9) | 17 | 13 |
| global 20 | 33 | 100 | 66.7 (0.9) | 0 | 22 |
| global 30 | 33 | 100 | 66.7 (0.9) | 0 | 22 |
| per-request chat 1.5, batch 30 | 50 | 100 | 75.2 (0.7) | 25 | 0 |
| per-request chat 1.5 only | 50 | 100 | 75.2 (0.7) | 25 | 0 |

## B  steady chat + batch burst, FCFS (U2 of the 10-06 amendment)

| arm | chat % | batch % | all % (sd) | dropped % | wasted work % |
|---|---|---|---|---|---|
| none | 29 | 100 | 56.4 (0.8) | 0 | 32 |
| global 1.5 | 87 | 49 | 72.4 (0.7) | 28 | 0 |
| global 5 | 68 | 58 | 64.2 (0.7) | 23 | 13 |
| global 20 | 30 | 99 | 56.3 (0.7) | 0 | 31 |
| global 30 | 29 | 100 | 56.4 (0.8) | 0 | 32 |
| per-request chat 1.5, batch 30 | 56 | 100 | 72.9 (0.5) | 27 | 0 |
| per-request chat 1.5 only | 56 | 100 | 72.9 (0.5) | 27 | 0 |

## C  chat bursts above capacity + steady batch, priority to chat (U1 of the 10-06 amendment)

| arm | chat % | batch % | all % (sd) | dropped % | wasted work % |
|---|---|---|---|---|---|
| none | 32 | 42 | 34.9 (1.0) | 0 | 64 |
| global 1.5 | 78 | 67 | 75.1 (0.4) | 25 | 0 |
| global 5 | 37 | 65 | 44.2 (0.5) | 19 | 37 |
| global 20 | 32 | 68 | 41.5 (1.2) | 8 | 46 |
| global 30 | 32 | 73 | 43.0 (1.1) | 7 | 45 |
| per-request chat 1.5, batch 30 | 78 | 98 | 83.3 (0.5) | 17 | 0 |
| per-request chat 1.5 only | 78 | 100 | 83.6 (0.5) | 16 | 0 |

## D  steady overload 1.3x, both classes, FCFS (U2 as first planned)

| arm | chat % | batch % | all % (sd) | dropped % | wasted work % |
|---|---|---|---|---|---|
| none | 5 | 66 | 35.4 (1.2) | 0 | 54 |
| global 1.5 | 78 | 78 | 77.6 (0.6) | 22 | 0 |
| global 5 | 5 | 79 | 41.9 (0.5) | 21 | 31 |
| global 20 | 5 | 85 | 45.0 (0.5) | 14 | 32 |
| global 30 | 5 | 90 | 47.2 (0.6) | 10 | 32 |
| per-request chat 1.5, batch 30 | 33 | 100 | 66.6 (1.1) | 33 | 0 |
| per-request chat 1.5 only | 33 | 100 | 66.6 (1.1) | 33 | 0 |
