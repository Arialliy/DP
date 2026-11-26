# DeepPro-MXC: motion-calibrated level for profile-blind targets

DeepPro-MXC (`mxc_lv`) is the next version of DeepPro-MX (`mx_lv`, [MX.md](MX.md)). It keeps
the problem and the exemplar idea of MX. It changes what the exemplars are used for, and it
replaces M0 with M0c, because M0 was biased against the method.

**Changes in v7.**

* **No test annotations.** No test-time step reads test annotations any more. Camera
  motion comes from the frames alone (`motion_image.py`), not from box-masked homographies.
* **Exact invariance.** The calibration is exactly gain-invariant, thanks to a relative
  floor. The fixed floor is kept as an ablation, and it is only approximately invariant.
* **Controls stay controls.** Every control keeps its exemplar source in training,
  checkpoint selection and inference. There is a new no-motion control, which uses the
  clip's own blobs.
* **One raw contrast in M0c.** M0c uses a single raw contrast definition for every feature
  set. M0c is a diagnostic, not a gate.
* **Fingerprinted predictions.** Predictions are bound to the checkpoint and settings that
  made them.
* **Full official evaluation.** The scoring runs all 27 official thresholds and reports
  IoU, Pd, Fa and AUC for the same checkpoint.

## 1. Problem

The temporal profile cannot see a target that does not move relative to its background.
On RGB-T, 1,362 of 11,303 test targets stay missed by DeepPro-Plus even at threshold 1e-5.
On the validation clips, 89% of the near-stationary boxes belong to tracks that never
reach 0.5 px/frame. Only the level pathway can find them. It has to separate a parked
target from a ground hot spot in the window level L, mostly by local contrast.

**The contrast of a target depends on the scene.**

* On DJI_0319_2, 98% of the ships have positive contrast, median 14 (band 9–22), while the
  hot spots have median 37.
* On DJI_0117_2, 41% of the pedestrians are darker than their surroundings.

A detector trained on other scenes learns one absolute contrast rule, and that rule is
wrong for a scene like DJI_0319_2.

**The gap is measurable.** M0's positives and negatives (v5 run) were scored with a
logistic regression on [contrast, contrast²], fitted two ways:

| Clip | Fitted on the other clips (what a trained detector faces) | Fitted within the clip (knows the scene) | Gap |
|---|---:|---:|---:|
| DJI_0319_2 | 0.837 | 0.995 | 0.158 |
| DJI_0127_1 | 0.785 | 0.981 | 0.196 |
| DJI_0117_2 | 0.775 | 0.708 | none |

The information is in the clip, but only relative to the scene. The `mx_lv` encoder cannot
use it, because it starts with LocalNorm, which removes contrast amplitude.

**Why parked targets cannot simply be found by their own similarity.** Parked targets do
look alike, and the level pathway already uses their appearance. Hot spots, however, also
look alike, and in DJI_0319_2 they are brighter than the ships. Similarity groups the blobs
of a clip, but it does not say which group is the targets. The moving targets settle that:
the temporal profile marks them as targets, without labels. The table in §2 measures this:
calibrating on the clip's strongest blobs, which carries no motion, does not transfer.

**Why M0 said NO-GO.** M0 fitted its baseline within the clip, so the baseline already knew
the clip's contrast band. On DJI_0319_2 it reached 0.975. That left at most 0.025 to gain,
while M0 asked for a gain of 0.10.

## 2. Idea: the moving targets calibrate the scene

The temporal profile reliably finds the moving targets of a clip. They show what contrast
and polarity targets have in this scene now: the same sensor gain, altitude, time of day
and target kind.

MXC measures two contrasts:

* the local contrast c_e at each exemplar: the centre 3 × 3 minus the ring at radii 4–7,
  taking the signed value of largest magnitude within ±2 px;
* the contrast c(p) of L at each pixel.

From them it builds four calibrated channels for LevelNet (`mx.calibration_channels`):

| Channel | Definition | Meaning |
|---|---|---|
| R | log(\|c\| + ε) − median_j log(\|c_e,j\| + ε) | Contrast in exemplar units |
| \|R\| | | Distance from the typical target of this scene |
| D | min_j \|log(\|c\| + ε) − log(\|c_e,j\| + ε)\| | Distance from the nearest exemplar (clips with two target kinds) |
| P | sign(c) · mean_j sign(c_e,j) | Polarity agreement |

LevelNet's input becomes [L, E, cover, M_x, A_x, M_g, R, \|R\|, D, P], 10 channels instead
of 6. Everything else is `mx_lv`. Without exemplars the four channels are 0, and A_x = 0
marks such samples.

**Gain invariance.**

* **`mxc_lv`: exact.** The floor is relative: ε = 0.1 · median_j \|c_e,j\|. A gain g > 0 on
  the clip's contrast multiplies c, every c_e and ε alike, so R, \|R\|, D and P are exactly
  invariant: up to float rounding, and as long as the median exemplar contrast exceeds
  0.01 grey levels. `test_mx.py` checks this to 1e-4 on the whole map.
* **`mxc_lv_absfloor` (ablation): only approximate.** It uses a fixed ε of one grey level,
  which does not scale with the gain. For gains in [½, 2], R changes by up to 0.14 when
  every contrast lies in 5–60 grey levels, and by up to 0.27 in 2–60.
* **L is not invariant.** `train.py --gain-aug 2` (the default for `mxc_*`) applies a gain,
  drawn log-uniformly from [½, 2], to half of the samples. It is applied to the window and
  its exemplars together (`mx_data.gain_augment`). Absolute contrast then stops being a
  reliable cue in training, and the level pathway has to use the calibrated one.

**Proxy evidence.** These numbers come from the v5 M0 data, on three clips, with the other
clips used for fitting. The exemplars are simulated by 8–16 stationary targets from other
windows of the same clip, which is optimistic. Real exemplars are moving targets, and at
inference they are the network's own peaks. The three clips were looked at while designing
MXC, so M0c decides on the other clips.

| Reference of the calibration | DJI_0319_2 | DJI_0127_1 | DJI_0117_2 |
|---|---:|---:|---:|
| None: absolute [c, c²] | 0.837 | 0.785 | 0.775 |
| Targets, fixed floor (1 grey) | 0.920 | 0.977 | 0.728 |
| Targets, relative floor (`mxc_lv`) | 0.933 | 0.973 | 0.735 |
| The clip's strongest blobs: no labels, no motion | 0.836 | 0.897 | 0.675 |
| All candidate blobs, unlabelled | 0.809 | 0.440 | 0.729 |
| Within the clip (oracle) | 0.995 | 0.981 | 0.708 |

The calibration helps only when the reference is the targets, which motion selects. With
the clip's own blobs as the reference, nothing is gained on DJI_0319_2.

### Claims, and the controls that test them

Each control keeps its exemplar source in all three stages: training, the per-epoch
checkpoint selection, and inference.

| Run (`run_mx.sh controls`) | Exemplars | Tests |
|---|---|---|
| `mxc_lv` | Moving targets (training: GT, rel_speed ≥ 0.5; selection and inference: the network's own peaks, `outside`) | Proposed |
| `*_noex` (`--ex-p-none 1`) | None (selection bank `none`, `--bank none`) | Is it the exemplars? |
| `*_self` (`--ex-source self`) | The clip's strongest blobs outside the window, found without annotation or motion (`mx_data.self_bank`; selection bank and `--bank self`) | Is it motion, or would the clip's own blobs do? |
| `*_nogain` (`--gain-aug 1`) | Moving targets | Gain augmentation |
| `mx_s46` (`mx_lv`) | Moving targets, no calibration (v5) | Calibration |
| `mxc_lv_absfloor` | Moving targets, fixed floor | Exact invariance |
| `infer_mx.py --bank gt` / `random` | GT movers / random points | Upper bound / content |

## 3. M0c: a diagnostic before training

`code/calib_diag.py` (numpy, no GPU) runs M0's sampling on the training clips.

* **One raw contrast.** Every feature set uses the same contrast, c = `contrast_abs`: the
  signed value of largest magnitude within ±2 px. This is the contrast the calibration
  uses. M0's `contrast`, the largest signed value, differs from it wherever a dark and a
  bright value share the neighbourhood: of [−14, −2, +1] it takes +1, while `contrast_abs`
  takes −14. Mixing the two definitions would credit the calibration with a change of
  sampling. M0's definition is reported as `base_max` only.
* **Negatives.**
  * 'top': the strongest contrast peaks.
  * 'band': up to 50 peaks per window inside the 10–90% band of the window's exemplar
    contrast. This is the clutter that calibration alone cannot reject.
* **Fitting.** Every logistic regression is fitted leave-one-recording-out, with every clip
  and class weighing the same.
* **Measures.**
  * base = absolute [c, c²];
  * full = calibrated + NCC, the inputs of MXC;
  * cal_self = calibrated against the clip's strongest blobs;
  * oracle = absolute contrast fitted within the clip;
  * closure = (full − base) / (oracle − base);
  * motion_gain = cal − cal_self.

The rule is pre-registered and applied to the clips not in `--seen` (default: the
validation clips, which shaped this design). GO needs all of:

* (a) at least 2 calibration clips: ≥ 20 positives and negatives, and gap ≥ 0.02;
* (b) median closure ≥ 0.30;
* (c) full < base − 0.03 on at most 20% of the eligible clips;
* (d) with `--det-run`, median closure with detected exemplars ≥ 0.15;
* (e) median motion_gain ≥ 0.02.

**M0c does not decide the result.** By default, `run_mx.sh` records the verdict in
`runs/<run>/m0c.json` and trains anyway. The detection scores of §5 decide.
`M0_GATE=1` restores the stop.

## 4. Data, training, checkpoint and scoring: official or ours

| Part | Official DeepPro-Plus (`train.py`, `test.py`) | This code | Same? |
|---|---|---|---|
| Training windows | `TrainIRSeqDataLoader`: 40 frames, 128 × 128 crops centred on a target with p = 0.75, mean 85.0799, std 47.4845 | `train.py` re-implements this recipe | Same recipe, written again. MX/MXC add exemplars, compensated windows (`auto`) and, for MXC, the gain |
| Camera motion | None (raw frames) | `motion_image.py`: corners, LK and RANSAC, with blobs suppressed by the image itself; **no annotation** | Ours. The box-masked homographies of `profile_motion.py` now serve only the speeds in `targets.csv` |
| Labels | `segmentations/*.png`, uint8 / 255 > 0 | `paper_k6_floor`, drawn from the boxes | Same pixels (`check_labels.py`, 60 clips) |
| Main loss | SoftLoUloss | `soft_iou_loss` | Same |
| Extra losses | None | Level pathway (Soft-IoU + focal) for LV, MR, MX, MXC; match focal for MX, MXC | Ours |
| Optimiser | Adam 1e-3, β (0.9, 0.999), ε 1e-8, wd 1e-4; ×0.7 every 10 epochs (≥ 1e-5); 32 epochs; batch 4 | The same | Same |
| Checkpoint | After every epoch: every test window, sigmoid > 0.5, global IoU; `best_model.pth` when IoU ≥ best | `original_test_iou`, the default of `train.py` (`auto`) and of `run_mx.sh`; MX/MXC with the exemplars of inference | Same rule |
| Scoring | `test.py`: windows fused by their maximum; IoU at prob > 0.5; ShootingRules Pd/Fa at 27 thresholds; AUC (`write_results.py`) | `infer*.py` (maximum fusion; logits stored as float16); `score.py --author`: the 27 thresholds, ShootingRules with the official edge behaviour, IoU as `test.py`, the same AUC; plus box metrics | Re-implemented, because `test.py` cannot load these networks. It gives the same counts as the official loop at every threshold (tests) |

**Defaults of `train.py`.** `--checkpoint-selection auto` is the authors' rule,
`original_test_iou`. The research splits are the exception, so that their numbers never
choose on the test set:

* `--split train` selects on `val` (`val_iou`);
* `--split foldK/train` selects on `foldK/val`.

`val_iou` refuses clips that overlap the training or test clips, and `original_test_iou`
refuses training clips from the test set. `final` keeps only `final.pth`. The rule is in
`splits.selection_plan` and tested in `test_splits.py`.

**Test annotations.** With `original_test_iou`, the test masks choose the checkpoint, as
in the authors' protocol. Every other test-time step, from motion to exemplars to the
network, is label-free. For a claim that uses no test labels at all, use `val_iou`.

**The image edge.** `ShootingRules.py` slices `[i - 4 : i + 5]` around every target pixel.
For i < 4 the start is negative, so the slice is empty. As a result:

* a target in the first 4 rows or columns protects no pixel from counting as a false alarm;
* a target in the first row or column cannot be hit.

`score.py --author` does the same by default (`--author-edges official`). Thresholds are
compared in float32, as the authors' float32 maps are compared under NumPy 1. Two
residual differences remain:

* predictions are stored as float16 logits;
* pixels outside a compensated window get logit −30 (p ≈ 9e-14), which is positive only at
  the thresholds 0 and 1e-20.

**Caches.** `infer.py`, `infer_mr.py` and `infer_mx.py` skip a clip only if its stored
fingerprint matches the current one. The fingerprint covers the checkpoint's SHA-256, the
inference settings, the homography file (and `targets.csv` for `--bank gt`), the data root
and the code (`cachekey.py`). `score.py` records each clip's fingerprint, and it refuses a
run whose clips came from different checkpoints. After a new `best_model.pth`, the
homography change or a code change, the inference therefore runs again.

## 5. Evaluation and success, for the same `best_model.pth`

`score.py --author` reports, for one checkpoint:

* IoU, Pd and Fa at 0.5 (the authors' operating point);
* the Pd and Fa of all 27 thresholds, with their AUC.

This is everything `test.py` reports. The box metrics and the near-stationary recall come
on top, from the same predictions.

Pre-registered success, written before any MXC training:

* **Official protocol** (`official_train`, `original_test_iou`, `best_model.pth` for every
  method, `official_test`):
  * `mxc_lv` against DeepPro-Plus and `mx_lv`: a higher IoU, a higher Pd and a Fa that is
    not higher at 0.5;
  * a higher AUC.
* **Validation protocol** (`train` / `val_iou` / `val`, mean of seeds 46–48):
  * static-camera near-stationary recall +10 points against `mx_lv`;
  * the ≥ 0.5 px/frame band no worse than −2 points;
  * `*_noex` and `*_self` keep at most half of the gain.

## 6. Commands

```sh
export VT_ROOT=/home/md0/ly/DP/datasets/RGB-T
bash run_mx.sh all          # check, profile (speeds + annotation-free motion), m0, train, eval
bash run_mx.sh controls     # *_noex, *_self, *_nogain, mx_lv: the same recipe and selection
```

| Variable | Default | Meaning |
|---|---|---|
| `ARCH` | `mxc_lv` | `mxc_lv_absfloor`, `mx_lv` |
| `TRAIN_SPLIT` | `official_train` | The 45 official training clips |
| `SELECT` | `original_test_iou` | `val_iou` (with `SELECT_SPLIT`, default `val`) or `final` |
| `EVAL_SPLIT` | `official_test` | Clips for `infer_mx.py` and `score.py` |
| `M0_SPLIT` | `official_train` | M0c clips (test clips are refused) |
| `M0_GATE` | 0 | 1: M0c NO-GO stops training |
| `DET_RUN` | – | M0c also checks detected exemplars |

`profile` must run again in v7. It writes `results/motion/homographies`, and every
compensated window now reads that folder.

Validation protocol, for the claims:

```sh
TRAIN_SPLIT=train SELECT=val_iou SELECT_SPLIT=val EVAL_SPLIT=val bash run_mx.sh train && \
TRAIN_SPLIT=train SELECT=val_iou SELECT_SPLIT=val EVAL_SPLIT=val bash run_mx.sh eval
```

## 7. Limits

* **Never-moving kinds.** A kind of target that never moves in a clip gives no calibration
  there, and MXC then is `mx_lv` with gain augmentation.
* **Clutter exemplars.** Detected exemplars can be clutter. The median and D are robust to
  a minority of them, and training replaces 20% of the exemplars with clutter points.
* **Fixed ring.** The ring of the contrast is fixed at radii 4–7. Targets larger than about
  7 px get a smaller contrast, and so do their exemplars. R compares like with like, but a
  size change between moving and parked targets is not modelled.
* **Annotation-free motion.** It suppresses corners around blobs it finds itself, so dense
  moving targets can still bias a frame pair. `results/motion/clips.csv` reports, per clip,
  the difference to the box-masked homographies, as the largest grid displacement within a
  window. Check it before trusting compensated numbers.
* **Your earlier runs.** The LVH, MRTP and LVH-dec runs of the DP reproduction read the
  box-masked homographies at test time, through checkpoint selection and inference. For a
  label-free comparison, run their inference again with `results/motion/homographies`.
  The cache fingerprints make this happen.

## 8. What was checked here, and what was not

* **numpy tests (113)** all pass, including:
  * `test_motion_image.py`: the pan is recovered without boxes, and is no worse than the
    box-masked estimate against the truth;
  * `test_cachekey.py`;
  * `test_splits.py`;
  * `test_shooting_rules.py`: the sweep equals the official loop at all 27 thresholds,
    and the AUC equals sklearn's;
  * `test_calib_diag.py`: one raw contrast, exact invariance, the motion check;
  * `test_run_mx.py`: the advisory M0c, `M0_GATE`, the recorded verdict.
* **Scoring:** `evaluate.shooting_rules` equals the authors' `ShootingRules.py`, run here
  with skimage, on 1,800 / 1,800 random frames.
* **The torch calibration functions** were run with a numpy stand-in for the torch
  operations they use:
  * both floors agree with `mx_data.calibrated_features` (difference 1.4e-7);
  * the relative floor changes by 1.8e-7 under a gain, and the fixed floor by up to 0.39.
* **The training data path** ran with a numpy stand-in for torch:
  * the gain path gives the same sample scaled by exactly g;
  * `--ex-source self` gives samples without fakes.
* **Not run here (no torch):** `test_mx.py`. It covers the MXC network, both floors under a
  gain, a training step, `val_iou` and `original_test_iou` with MX networks, the
  `noex` and `self` controls through selection and inference, the cache binding, and the
  check that the selection's exemplar candidates equal `infer_mx.py`'s.
  `bash run_mx.sh check` runs it first on the training machine.
