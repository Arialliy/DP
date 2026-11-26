# DeepPro-MTA: a single-frame appearance pathway taught by the temporal one

Current execution decision (2026-11-27): M1 is NO-GO. Do not start MTA training.
Rerun M1 only if its implementation is found incorrect and corrected; do not change
candidate parameters to obtain a GO. M0c is also NO-GO, so MXC training is not started.
The earlier custom-split training was stopped and retained as an interrupted historical
experiment. The later official MTA training plan was cancelled before launch. No
three-seed DeepPro-Plus retraining is requested. Current work is the M1 audit, early-clip
diagnosis, and training-free M2. These diagnostics are now complete; see
[`../DIAGNOSTICS.txt`](../DIAGNOSTICS.txt) and `runtime/current_authorization.json`.
Sections 2–3 retain the proposed mechanism, not a claim validated by these diagnostics.

DeepPro-MTA (`mta_lv`) follows DeepPro-MXC ([MXC.md](MXC.md)). It was redesigned after the
real-data diagnosis of the official DeepPro-Plus on the RGB-T test set. That diagnosis did
not support "near-stationary targets are the main source of misses". It did show where the
misses are.

## 1. What the official model misses (RGB-T test, `official_rgbt`, threshold 1e-5)

Counts are of the authors' ShootingRules components: 11,300 component-frames and 1,334
misses (`results/miss_attribution`, DP `deeppro-mx-complete 7`).

**Not mainly near-stationary targets.**

| Target speed (relative to the ground) | Misses / components | Miss rate | Share of misses |
|---|---:|---:|---:|
| < 0.2 px/frame | 344 / 1,999 | 17.2% | 25.8% |
| 0.2–0.5 | 211 / 1,334 | 15.8% | 15.8% |
| ≥ 0.5 | 770 / 7,878 | 9.8% | 57.7% |

Of the near-stationary misses, 92.2% are in two clips. In DJI_0237_2, near-stationary targets
are missed less often (14.6%) than moving ones (19.4%). At box level, isolated
near-stationary boxes (nearest other target ≥ 20 px) are missed as often as isolated
moving ones (9.9% vs 10.1%).

**Concentrated in time.** This is the strongest regularity found:

| | Frames 0–59 of a clip | Later frames |
|---|---:|---:|
| Share of the frames | ~15% | ~85% |
| Share of the misses (1e-5 / 0.5) | 40.9% / 34.7% | 59.1% / 65.3% |
| Miss rate (1e-5) | 25.8% | 8.6% |
| Miss rate of the same tracks (present early and later) | 26.1% | 10.1% |

By clip:

| Clip | Frames 0–39 | Later |
|---|---:|---:|
| DJI_0255_1 | 92.5% | 2.3% |
| DJI_0307_2 | 81% | 0% |
| DJI_0237_2 | 32.4% | 14.6% |
| DJI_0049_2 | 61% | 44% |
| DJI_0257_2 | 20% | 7.3% |

Every window uses the same model (`infer.py`, the authors' windows), but this does not
exclude an effect of window placement or context. A frozen-model check on DJI_0307_2
found that the same frames 36–39 were missed in [0,40) and detected in [36,76).
Context and relative temporal position have not been separated. There is no established
single cause across clips. Candidates:

* a background that changes at the clip start (camera motion: DJI_0237_2 moves 1.93 px/frame
  in frames 0–39, 0.86 later);
* targets that only start to stand out from their background later;
* an annotation effect.

`windows.csv` of M1 gives each window's stationarity residual next to its miss rate. Look at
the first windows of DJI_0255_1 and DJI_0307_2 before choosing an explanation.

**Other concentrations.**

* Pedestrians (category 3) hold 50.7% of the misses, with a miss rate of 23.2%.
* Four clips hold 87% of the misses.
* Box-level misses rise steeply when another target is within 8 px (43–76%). Most of that is
  an effect of box matching: the official components merge such targets.

## 2. Problem

DeepPro finds a target only through its temporal anomaly in a window. Wherever that evidence
is weak, the target is missed. This happens in the first frames of a clip, where the same
tracks are later found; on a moving background; for a pedestrian among others; or for a
target at rest. Yet in a single frame the missed target looks like the targets the detector
does find in the same video, often like itself in other windows. DeepPro never uses that.

This states the advisor's point as something that can be measured:

* **(a) Similar features.** Parked targets resemble the moving targets of their clip, more
  than clutter does. These are raw AUCs, with no fitting (v5 M0):
  * planes (DJI_0127_1): shape NCC 0.911;
  * pedestrians (DJI_0117_2): 0.644;
  * ships (DJI_0319_2): 0.524. For blob-like ships the similarity lies in the calibrated
    contrast (0.933, MXC.md), not in the shape.
* **(b) Detectable from a single frame.** The per-frame median similarity is 0.828 for
  planes. For every miss of the official model, M1 measures the same directly (§4).

## 3. Design (`mta_lv`, `code/mx.py`)

| Part | Does |
|---|---|
| Temporal pathway | DeepPro-Plus backbone on V = x − L, unchanged: l_v(t) |
| Level pathway | `mxc_lv`: LevelNet on L with the motion-calibrated channels |
| Teacher | At inference, the network's own confident detections in other windows of the clip (`infer_mx.py`, bank `outside`); in training, moving GT targets of other frames, 20% of them replaced by clutter |
| **Appearance pathway (new)** | For **every frame** x_t: a_t = AppearanceHead([LocalNorm(x_t), S_t, R_t, \|R_t\|, D_t, P_t, A_x]) |
| | S_t: the similarity of the frame to the teacher's exemplars (the `mx_lv` encoder) |
| | R_t … P_t: the frame's own local contrast in the exemplars' units, relative floor, exactly gain-invariant |
| Fusion | Noisy-OR of the temporal, level and appearance probabilities |
| Loss | DeepPro-Plus's Soft-IoU, plus the level and match losses of MXC, plus Soft-IoU + focal BCE of a_t against every frame's label |

**Why every frame.** MXC calibrated only L, the window median. A moving target is absent
from L, so MXC could never recover the 57.7% of misses that are moving. The appearance
pathway works per frame: moving or not, every target is in its frame.

**Innovation.** Inside one network, the temporal pathway labels, per video and without
annotations, the targets it is sure of. A single-frame appearance pathway is calibrated to
them and detects the targets whose temporal evidence is weak. This is test-time self-training
across two kinds of evidence of the same object:

* detection from motion supplies the labels;
* detection from appearance generalises them to frames where motion says nothing.

It differs from earlier work in two ways:

* motion is not only a training signal (Pathak et al. 2017; Jonschkowski & Stone 2019): here
  it teaches at test time;
* the exemplars are not given by a person (FamNet, LOCA): they come from the detector itself.

The calibration is exactly invariant to the scene's contrast gain. Check the literature
again before writing.

**Controls** (`bash run_mx.sh controls`, `ARCH=mta_lv`). Each one keeps its exemplar
source in training, checkpoint selection and inference.

| Run | Removes | Tests |
|---|---|---|
| `*_noex` (`--ex-p-none 1`) | The teacher: the head is a plain single-frame detector | Taught by motion, or one more CNN? |
| `*_self` (`--ex-source self`) | Motion: the exemplars are the clip's strongest blobs | Is it the temporal pathway's selection? |
| `*_noapp` (`mxc_lv`) | The appearance pathway | Is the gain per-frame appearance? |
| `*_nogain` | Gain augmentation | |
| `mx_s46` (`mx_lv`) | Calibration and appearance pathway (v5) | |

## 4. M1: recoverability, before any training (`code/miss_recovery.py`)

For each clip of the split, M1 uses the cached run (default `official_rgbt`), the frames and
the boxes:

* **Misses.** A box is missed at threshold th if no probability ≥ th lies on its mask or
  next to it, as ShootingRules hits a target. This is counted per box.
* **Teacher.** The run's peaks with probability ≥ 0.5, from frames outside the window.
* **Single frames.** The candidates are the local maxima of \|local contrast\| of each
  frame. A candidate within 3 px of a GT centre is a target; one beyond 8 px of every GT
  centre is clutter. Only actual proposed locations are scored; annotated centres are
  never substituted for missing proposals. Recovery uses one-to-one candidate/box matching.
* **Score.** Calibrated contrast and NCC to the teacher, from the single frame, combined by
  a logistic regression fitted leave-one-recording-out. A window without teacher
  detections recovers nothing.
* **Budget.** One threshold over all clips that admits 0.25, 1 or 2 clutter candidates per
  frame, conservatively excluding boundary ties. This is an artificial diagnostic working
  point, not noise added to the data, not official pixel Fa, and not a deployment requirement.
* **Output.** Misses, misses recovered and recall with recovery, per group: frames before
  and after 60, category, speed band, camera group, clip. `windows.csv` adds each window's
  stationarity residuals and miss rate.

M1 fits on the other test recordings' labels and calibrates its clutter threshold using
evaluated labels. It is a diagnostic of this candidate/feature recipe, not a model test
result. The geometric candidate ceiling and realized clutter count are reported separately.

**Gate and instruction history.** The original gate was at least 30% early and 20%
overall recovery at pooled 1 clutter candidate/frame. The measured gate is retained in
`summary.json`. An intermediate user instruction allowed training despite NO-GO; the
subsequent instruction on 2026-11-27 restored the gate and stopped those experiments.
The current decision is NO-GO, with no further MTA training. A failed M1 measures this
fixed candidate/feature recipe; it does not prove that every learned single-frame
pathway is ineffective.

**Completed implementation review (2026-11-27).** `frame_peaks` incorrectly counted
multiple pixels of one maximum plateau as separate blobs before top100 selection.
The correction retains one actual maximal pixel nearest the plateau centroid. All
parameters and the original gate were unchanged; 29 targeted tests passed. The full
rerun still recovers 1/644 early misses (0.1553%) and 4/1545 overall (0.2589%), below
30% and 20%. Independent CSV audits verified unchanged baseline hits and settings,
one-to-one matching, and the realized budget. The conclusion remains NO-GO. Corrected
results are in `results/miss_recovery/official_rgbt/test`; the preceding pixel-plateau
version and its audits are retained in `results/miss_recovery_pixel_plateaus`.

Final comparison directly reports Pd, Fa and IoU under the same evaluation protocol and
threshold. It is not replaced by recall at a fixed false-alarm budget. All 27 thresholds
and AUC remain supplementary outputs.

## 5. Order and commands

The training commands below preserve the historical recipe for reference. They are
not authorized by the current NO-GO execution decision and are not running.

```sh
export VT_ROOT=/home/md0/ly/DP/datasets/RGB-T
bash run_mx.sh profile            # again: plausibility-checked annotation-free homographies (v8)
bash run_mx.sh m1                 # needs cache/preds/official_rgbt (code/infer.py --run official_rgbt)
export ARCH=mta_lv SEED=46 M0_GATE=0
export RUN=mta_val_s46 MX_RUN=mta_val_s46_mx
export TRAIN_SPLIT=train M0_SPLIT=train SELECT=val_iou SELECT_SPLIT=val EVAL_SPLIT=val
bash run_mx.sh m0
bash run_mx.sh train && bash run_mx.sh eval && bash run_mx.sh controls

# Only after the validation experiment; separate names prevent checkpoint reuse.
export RUN=mta_official_s46 MX_RUN=mta_official_s46_mx
export TRAIN_SPLIT=official_train M0_SPLIT=official_train SELECT=original_test_iou EVAL_SPLIT=official_test
bash run_mx.sh m0
bash run_mx.sh train && bash run_mx.sh eval && bash run_mx.sh controls
```

Both protocols evaluate IoU after every epoch and save `best_model.pth` when IoU is at
least as high as the previous best. Final inference and Pd/Fa/IoU scoring use that best
checkpoint. Validation uses val labels for selection; the original official protocol
uses test labels, as the author implementation does.

**Success, written before training.** All of these use the same `best_model.pth` against
DeepPro-Plus on the official protocol:

* directly compare IoU, Pd and Fa at 0.5: IoU and Pd higher, with Fa not higher;
  report the 27-threshold curve and AUC as supplementary measurements;
* recall in frames 0–59 up by at least 10 points;
* `*_noex` and `*_self` keep at most half of the gain.

## 6. Also changed in v8

* **`multirate.choose_coords`.** At rate 1, `auto` uses the recorded frames when the
  compensated field keeps less than `min_cover`. The residual cannot be measured then;
  before, it warped anyway (cover 0 on two windows of DJI_0277_2). `warp` still compensates.
* **`motion_image.py`.**
  * **Plausibility check.** A homography is kept only if its tracks support it: inliers
    ≥ 30%, perspective ≤ 2e-4/px, median residual ≤ 1 px. Otherwise a partial affine, then
    a shift, then identity, counted per clip in `clips.csv`.
  * **Common view.** The comparison with the box-masked homographies uses only the common
    view. Far outside it, a projective map extrapolates without bound: this was the
    21,260 px on DJI_0277_2.
* **`run_mx.sh`.**
  * stage `m1`;
  * `controls` adds `*_noapp` for `mta_*`;
  * the `mx_lv` control keeps the main gain (your fix, also for `mta_*`).
* **Your two fixes** are included: `mxc_lv_absfloor` in the selection, and the gain of the
  `mx_lv` control.

## 7. Completed local checks and diagnostic limits

* `bash run_mx.sh check` on the original v8 passed 153 tests in 22 files, including
  torch CPU forward/backward, toy training, checkpoint selection and inference.
  See `logs/check.20261126.log`; this does not establish full-training performance.
* The plateau correction passed 18 `test_mx_data.py` and 11 `test_miss_recovery.py`
  tests, then the complete M1 rerun and independent result audit.
* Training-free M2 passed 15 synthetic tests and independent matching/provenance
  checks. Under the frozen raw-coordinate recipe, fixed-position backcasting recovered
  0/870 early misses at baseline 0.5, while constant velocity recovered 28/870 (3.22%)
  with 11.13 wrong points/frame. This is a box/location diagnostic, not official Pd/Fa/IoU.
  See `results/m2/summary.json` and `results/m2/notes.txt`.
* Early-clip evidence differs: DJI_0255_1 has weak initial contrast; DJI_0307_2 shows
  dependence on window context/position. Neither a single cause nor a new model's
  detection improvement has been established. See `results/frame_inspection/findings_zh.txt`
  and `results/early_cause/analysis_zh.txt`.
