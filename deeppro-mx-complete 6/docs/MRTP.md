# MRTP: multi-rate temporal profiles with stationarity-adaptive coordinates

This branch adds a detector design and the experiments that decide whether it works.
Nothing here changes the frozen `deeppro_lv_calibrated_seed46` path or the `lv_*` and
`lvh_*` training defaults.

## Problem: the velocity blind zone

A temporal-profile detector (DeepPro, DeepPro-LV) judges each pixel by its profile over a
window of T samples. A target of extent s moving at v px/frame covers a pixel for s / v
frames, or s / (k v) samples when the window takes every k-th frame. Two limits follow:

* **Blind zone.** Pixels covered for more than T / 2 samples go into the window median,
  i.e. into the background. Below k v T = 2 s, the variation keeps only slivers at most
  k v T / 2 px wide, near the window ends; at k v = 0 nothing is left.
* **Aliasing.** Above k v = s, the target can skip a pixel between two samples.

So one rate covers the band **2 s / (k T) ≤ v ≤ s / k**, a band T / 2 = 20 wide on a log
scale. The natural axis is the dwell ratio r = k v T / s, with the cliff predicted at
r ≈ 2. With T = 40 and s = 6 px, rate 1 starts near 0.3 px/frame. The seed-46 validation
runs show the cliff there (recording-matched 1 FA/frame, static-camera clips,
`code/rate_analysis.py`):

| Relative speed (px/frame) | Boxes | DeepPro-LV | + level Focal | LVH full | LVH dec |
|---|---:|---:|---:|---:|---:|
| < 0.05 | 578 | 0.3% | 1.7% | 0.2% | 1.0% |
| 0.05–0.2 | 1,844 | 5.8% | 18.0% | 2.4% | 3.3% |
| 0.2–0.5 | 843 | 47.0% | 57.5% | 36.2% | 37.7% |
| ≥ 0.5 | 1,475 | 88.5% | 86.2% | 83.7% | 82.8% |

DeepPro's authors state the same limit: "It is hard to detect stationary objects from
the temporal profile" (arXiv:2506.12766, Sec. 4.7). Their Table V finds no gain from
contiguous windows longer than 40 frames, on moving-target data.

A second assumption, a stationary background, fails under camera motion. Compensation
(LVH) fixes moving cameras but costs static ones:

| Macro recall over clips, 1 FA/frame | DeepPro-LV | + level Focal | LVH full | LVH dec |
|---|---:|---:|---:|---:|
| 5 static-camera clips | 58.8% | **63.1%** | 55.9% | 57.4% |
| 3 moving-camera clips | 72.3% | 71.0% | **87.1%** | 85.9% |

`DJI_0319_2` (all tracks category 0, ship, so a water scene) loses at every speed under
LVH, including 0.5–1 px/frame (97.6% → 80.0%). Registration on water, History, or both
are suspects; test T3 below separates them.

## Method

1. **Rates.** The same T-sample network scores phase-interleaved windows
   {φ, φ + k, …, φ + (T − 1) k}, φ = 0…k − 1, for k ∈ {1, 2, 4}. Sampled at rate k, a
   target at speed v has the sample-domain profile of a target at speed k v sampled at
   rate 1. The weights are shared across rates and no rate input is needed: the target
   is rate equivariant. Each rate costs about as much as rate 1, about n / 36 windows per
   clip. Rates {1, 2, 4} lower the band edge four-fold, to about 0.075 px/frame at s = 6.
2. **Coordinates.** A profile also assumes a stationary background, and a longer span
   gives it more time to change. Each window, at each rate, is scored in recorded pixels
   or in camera-compensated pixels, whichever changes less between the two halves of
   the window: the stationarity residual, the 90th percentile over edge pixels of
   |median(first half) − median(second half)| on 4 × 4 block means. Compensation is used
   when it lowers the residual by more than max(10%, 0.5 grey levels). Rate > 1 windows
   whose compensated view keeps under 10% of the pixels are skipped. This replaces
   always warping (LVH) and the displacement gate, which never fired.
3. **Fusion.** Rates differ in clutter, since the background is not rate equivariant.
   Each rate's logits are calibrated against their own local clutter level, CFAR-style:
   minus the (10 w)-th largest candidate score over the w frames within ±18. The fused
   score is the maximum over the calibrated rates.
4. **Training.** `mr_lv` is DeepPro-LV with the masked decomposition of `lvh.py` and a
   `cover` input, so one network handles both coordinate frames. Training draws rate 1
   with probability 1/2 and rates 2 and 4 with 1/4 each, in coordinates chosen by the
   same rule. It uses the level-focal loss of `gpu_lvfull_focal_s46`, the best
   static-camera baseline. Background at slow rates is what the network has to learn;
   the target itself needs no relearning.

### Relation to earlier work

| Work | Uses | Differs |
|---|---|---|
| DeepPro / DeepPro-Plus | 40 contiguous frames | One rate; blind to slow targets, by its own account |
| FeedbackSTS-Det (arXiv:2601.14690) | Interleaved sparse frames | Sparse frames for cheap long-range context; no speed or camera analysis |
| Adaptive spatial-temporal local feature difference (arXiv:2309.02054) | Fixed frame gap (5) | Hand-set gap; not learned or adaptive |
| SlowFast (Feichtenhofer et al., ICCV 2019) | Two frame rates, two pathways | Action recognition; separate pathways, no stationarity rule |
| DeepPro-LVH (this repo) | Always-on compensation, History | Hurts static cameras; History is a veto on stationary targets |

The claims here: the dwell-ratio account of where profile detectors fail; rate
equivariance, which allows shared weights across rates; per-window, per-rate coordinate
choice by background stationarity; CFAR fusion across rates. All four need the tests below.

## Files

| File | Content |
|---|---|
| `code/multirate.py` | window plans, stationarity residual, coordinate rule, CFAR calibration, fusion (numpy) |
| `code/infer_mr.py` | multi-rate inference for `lv_*`, `mr_lv`, `lvh_*`; per-rate and fused runs |
| `code/mr.py` | `mr_lv` registration |
| `code/mr_data.py` | multi-rate training samples (numpy) |
| `code/train.py` | `MultiRateTrainSet`, `--rates`, `--rate-probs`, `--coords` |
| `code/rate_analysis.py` | recall by speed, apparent speed k v, dwell ratio; per-clip macro; `--split cv` |
| `code/folds.py`, `code/splits.py` | grouped 5-fold cross-validation, `foldK/train` and `foldK/val` names |
| `code/test_multirate.py`, `test_mr_data.py`, `test_infer_mr.py`, `test_folds.py` | numpy tests |
| `code/test_lvh.py` | adds torch tests for `mr_lv` and multi-rate inference (not run on the authoring machine) |

## Step 1: decide without training (minutes per run)

```sh
python code/test_multirate.py && python code/test_mr_data.py && python code/test_infer_mr.py
python code/test_folds.py && python code/test_lvh.py
```

**T0, reproduction.** `infer_mr.py` at rate 1 on recorded frames uses DeepPro's windows,
so `focal_s46_mr__r1` must score exactly like `gpu_lvfull_focal_s46`. If it does not, stop.

**T1, the law.** Recall against the dwell ratio r = v T / s at rate 1:

```sh
python code/rate_analysis.py --split val --sizes \
  --score-root results/lvh_performance_s46/scores --score-root results/lvh_decoupled_s46/scores \
  --runs base=gpu_lvfull_s46 focal=gpu_lvfull_focal_s46 lvh=gpu_lvhfull_s46 dec=gpu_lvhdec_s46 \
  --speeds results/profile/targets.csv
```

Expected: on the 'dwell' axis the static-camera recall crosses 50% between r = 1 and
r = 4. On the 'speed' axis the crossing depends on target size.

**T2, multi-rate with the existing level-focal weights.**

```sh
python code/infer_mr.py --run focal_s46_mr --arch lv_full \
  --ckpt results/lvh_performance_s46/runs/gpu_lvfull_focal_s46/final.pth \
  --clips val --rates 1,2,4 --coords raw --fusion max,cal --device cuda
for r in r1 r2 r4 r1cal mrmax mrcal; do python code/score.py --run focal_s46_mr__$r --split val; done
python code/rate_analysis.py --split val --sizes --score-root results/scores \
  --runs r1=focal_s46_mr__r1@1 r2=focal_s46_mr__r2@2 r4=focal_s46_mr__r4@4 \
         r1cal=focal_s46_mr__r1cal max=focal_s46_mr__mrmax cal=focal_s46_mr__mrcal \
  --speeds results/profile/targets.csv --exclude-clip DJI_0319_2
```

Pre-registered reading, recording-matched 1 FA/frame:

* **Equivariance:** on the 'apparent' axis (k v), rates 2 and 4 stay within 10 points
  of rate 1 in each bin from 0.2 to 1 px/frame that holds at least 100 boxes.
* **Gain:** `cal` beats `r1` by at least 10 points in the static 0.05–0.2 band, on both
  `DJI_0117_2` and `DJI_0319_2`.
* **No loss:** at most −2 points in the static ≥ 0.5 band and on the moving clips;
  macro recall is not lower.
* `r1cal` versus `r1` shows what calibration alone does; `max` versus `cal` shows what
  it does for fusion.

If equivariance holds but gains are small, rate-augmented training is the next lever. If
equivariance fails, the network relies on window statistics that change with the rate,
and the method needs rethinking before training.

**T3, why LVH fails on DJI_0319_2.**

```sh
python code/infer_mr.py --run dec_s46_raw --arch lvh_dec \
  --ckpt results/lvh_decoupled_s46/runs/gpu_lvhdec_s46/final.pth --clips val --rates 1 --coords raw --fusion none --device cuda
python code/infer_mr.py --run dec_s46_nohist --arch lvh_dec \
  --ckpt results/lvh_decoupled_s46/runs/gpu_lvhdec_s46/final.pth --clips val --rates 1 --coords warp --no-history --fusion none --device cuda
python code/infer_mr.py --run dec_s46_auto --arch lvh_dec \
  --ckpt results/lvh_decoupled_s46/runs/gpu_lvhdec_s46/final.pth --clips val --rates 1 --coords auto --fusion none --device cuda
for r in dec_s46_raw__r1 dec_s46_nohist__r1 dec_s46_auto__r1; do python code/score.py --run $r --split val; done
```

The saved arms of `gpu_lvhdec_s46` (`cache/preds/gpu_lvhdec_s46__appearance` and the
others) score with `score.py --run gpu_lvhdec_s46__appearance --split val` as well.
`*__meta/summary.json` counts how often `auto` compensated each clip.

## Step 2: train and compare under cross-validation

```sh
python code/folds.py                          # needs results/profile/targets.csv; writes configs/folds.json
for s in 46 47 48; do for f in 0 1 2 3 4; do
  python code/train.py --run lvfocal_f${f}_s$s --arch lv_full --split fold$f/train --seed $s --device cuda \
    --level-focal 1 --persist hard --hard-neg 0
  python code/train.py --run lvh_f${f}_s$s --arch lvh_full --split fold$f/train --seed $s --device cuda
  python code/train.py --run mrraw_f${f}_s$s --arch lv_full --split fold$f/train --seed $s --device cuda \
    --level-focal 1 --persist hard --rates 1,2,4 --coords raw
  python code/train.py --run mrlv_f${f}_s$s --arch mr_lv --split fold$f/train --seed $s --device cuda
done; done
```

Inference uses `infer.py` for `lvfocal` and `lvh`, and `infer_mr.py` for `mrraw` (raw) and
`mrlv` (auto). Every held-out fold is scored with `--split fold$f/val`. The pooled view is
`rate_analysis.py --split cv --runs mr=mrlv_f{fold}_s46__mrcal ...`.

| Run | Question |
|---|---|
| `lvfocal` | best static baseline, retrained per fold |
| `lvh` | best moving-camera baseline, retrained per fold |
| `mrraw` | rates and rate-augmented training, recorded frames |
| `mrlv` | proposed: rates, stationarity-selected coordinates, CFAR fusion |
| `mrlv` with `--rates 1`, `1,2`, `1,2,4,8` | how many rates |
| `mrlv` with `--fusion max` | calibration |
| `mrlv` with `--coords raw` or `warp` at inference | coordinate rule |

Pre-registered success, mean of three seeds, recording-matched 1 FA/frame, `mrlv` with
`mrcal` against the better of `lvfocal` and `lvh` per group:

* static-camera 0.05–0.2 px/frame band: +15 points;
* static ≥ 0.5 band: no worse than −2 points; moving-camera macro: no worse than −2 points;
* macro recall over all 40 clips: +3 points, with the same sign in at least 4 of 5 folds;
* also reported: 0.25 FA/frame, pooled-threshold CPM and AP, per-clip tables.

The test split is scored once, after this table is fixed.

## Limits

* Targets slower than about 2 s / (k_max T), about 0.075 px/frame at s = 6 with k_max = 4,
  stay in the blind zone, apart from edge slivers. They are left to the level pathway; History, which helped
  `DJI_0117_2`, is an optional extension and is not part of `mr_lv`.
* A rate needs k T frames: rate 4 is unused on clips shorter than 160 frames (`DJI_0107_1`).
* A rate-k window spans (T − 1) k + 1 frames, 157 at rate 4. Like DeepPro's, the
  evaluation is offline; an online detector would wait up to that span.
* Homographies assume a distant or planar scene. The stationarity rule falls back to
  recorded frames where compensation does not help, but it cannot repair parallax.
* `test_lvh.py` and the training path were written without a torch install and have not
  been run yet.
