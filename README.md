# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/JohnyKucel/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/JohnyKucel/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A local-linear log-ratio forecaster for GTA VI's launch-day Twitch viewers, hour by hour, for the make.fun "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty ($250 in USDC). It reads one case file, fits its forecast from that case's own comparable launches, and writes five quantiles (5/25/50/75/95%) for hours 2–24 of release day.

## How it works

Launch days do not scale with hour 1. Across the comparable launches, the peak absolute audience of a big launch sits in a narrow band (roughly 300k–1.1M viewers) whatever hour 1 was, while the ratio to hour 1 spans 1x for the biggest launches to 2000x for the smallest. So instead of forecasting a flat ratio, the program regresses the log-ratio `r_h = ln(v_h / v_1)` on `ln(v_1)` per hour, locally: a Gaussian kernel in `ln(v_1)` (bandwidth 1.5) weighs launches near this launch's size most. The spread per hour is the kernel-weighted residual standard deviation of that regression, times a calibration factor of 0.8 (floor 0.05 in log units). Quantiles are `v_1 * exp(pred + z_tau * spread)`. Nothing from the practice answers is hardcoded — every number the program outputs is fitted from the case file it is given at run time. No network, no clock, no randomness: two runs on the same case write identical bytes.

## Practice result (local, kit cases scored against SullyGnome answers)

Mean pinball loss **0.178** over the four practice cases (house baseline 0.387; best leaderboard entry at the time of writing 0.169):

| case | loss | typical miss | 90% coverage |
|---|---|---|---|
| Marvel's Wolverine | 0.177 | 25% | 1.00 |
| Red Dead Redemption 2 | 0.212 | 46% | 0.96 |
| God of War Ragnarök | 0.140 | 26% | 1.00 |
| Marvel's Spider-Man 2 | 0.184 | 33% | 0.83 |

Model selection used leave-one-out over the 27 comparables with hour-1 ≥ 20k (the regime GTA VI will be in): local-linear 0.259 vs global OLS 0.332. The spread calibration (0.8) and bandwidth (1.5) sit at the joint optimum of the leave-one-out and practice scores, not at either one's knife edge.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only. Reads every `<id>.in` in the input directory and writes `<id>.out` (`{"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]}`) to the output directory. Runs in ~0.1s per case, well inside the 8s cap.

---

<sub>Entered on [make.fun](https://make.fun/?ref=jy3a6hm3&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
