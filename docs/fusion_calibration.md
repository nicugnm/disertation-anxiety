# Phase 1C — recalibration of the fusion+focal model

Temperature scaling (per target) + per-subreddit thresholds applied to the winning `fusion+focal` model (`src/models/fusion.py`), author-disjoint train/calib/test. `scripts/exp_fusion_calibration.py`.

- **anxiety per-subreddit macro-F1: 0.852 -> 0.888** (global vs per-subreddit thresholds)

| target | temperature | ECE before | ECE after |
|---|---:|---:|---:|
| anxiety | 0.618 | 0.0202 | 0.0062 |
| health_anxiety | 0.474 | 0.0148 | 0.001 |
| depression | 0.555 | 0.0182 | 0.0016 |
| suicidality | 0.513 | 0.0058 | 0.0003 |

![fusion calibration](figures/fusion_calibration.png)