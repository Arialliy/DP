# DeepPro-MXC: motion-calibrated level for profile-blind targets

DeepPro-MXC (`mxc_lv`) is the next version of DeepPro-MX (`mx_lv`, [MX.md](MX.md)). It keeps
the problem and the exemplar idea of MX. It changes what the exemplars are used for, and it
replaces M0 with M0c, because M0 was biased against the method.

## 1. Problem

The temporal profile cannot see a target that does not move relative to its background.
On RGB-T, 1,362 of 11,303 test targets stay missed by DeepPro-Plus even at threshold 1e-5.
On the validation clips, 89% of the near-stationary boxes belong to tracks that never
reach 0.5 px/frame. Only the level pathway can find them. It has to separate a parked
target from a ground hot spot in the window level L, mostly by local contrast.

**The contrast of a target depends on the scene.** On DJI_0319_2, 98% of the ships have
positive contrast, median 14 (band 9–22), while the hot spots have median 37. On DJI_0117_2,
41% of the pedestrians are darker than their surroundings. A detector trained on other
scenes learns one absolute contrast rule, and that rule is wrong for a scene like
DJI_0319_2.

**The gap is measurable.** M0's positives and negatives (v5 run) were scored with a logistic
regression on [contrast, contrast²], fitted in two ways:

| Clip | Fitted on the other clips (what a trained detector faces) | Fitted within the clip (knows the scene) | Gap |
|---|---:|---:|---:|
| DJI_0319_2 | 0.837 | 0.995 | 0.158 |
| DJI_0127_1 | 0.785 | 0.981 | 0.196 |
| DJI_0117_2 | 0.775 | 0.708 | none |

The information is in the clip, but only relative to the scene. The `mx_lv` encoder cannot
use it, because it starts with LocalNorm, which removes contrast amplitude.

**Why M0 said NO-GO.** M0 fitted its baseline within the clip, so the baseline already knew
the clip's contrast band. On DJI_0319_2 it reached 0.975, which leaves at most 0.025 to
gain, while M0 asked for a gain of 0.10. M0 therefore could not pass DJI_0319_2 for any
method. It also measured the wrong thing: a trained detector never sees the test clip's
band.

## 2. Idea: the moving targets calibrate the scene

The temporal profile reliably finds the moving targets of a clip. They show what contrast
and polarity targets have in this scene now: same sensor gain, altitude, time of day and
target kind. MXC measures the local contrast c_e at each exemplar (centre 3 × 3 minus the
ring at radii 4–7, the signed value of largest magnitude within ±2 px). It also measures
the contrast c(p) of L. LevelNet then gets four calibrated channels
(`mx.calibration_channels`):

| Channel | Definition | Meaning |
|---|---|---|
| R | log(\|c\| + ε) − median_j log(\|c_e,j\| + ε) | Contrast in exemplar units |
| \|R\| | | Distance from the typical target of this scene |
| D | min_j \|log(\|c\| + ε) − log(\|c_e,j\| + ε)\| | Distance from the nearest exemplar (clips with two target kinds) |
| P | sign(c) · mean_j sign(c_e,j) | Polarity agreement |

ε is one grey level. LevelNet's input becomes [L, E, cover, M_x, A_x, M_g, R, \|R\|, D, P],
10 channels instead of 6. Everything else is `mx_lv`. Without exemplars the four channels
are 0, and A_x = 0 marks such samples.

**Gain equivariance.** A contrast gain g of the clip multiplies c and every c_e alike, so R,
\|R\|, D and P do not change, while L does. `train.py --gain-aug 2` (the default for
`mxc_*`) applies such a gain, drawn log-uniformly from [1/2, 2], to half of the samples:
to the window and to its exemplars together (`mx_data.gain_augment`). Absolute contrast
then stops being a reliable cue in training, and the level pathway has to use the
calibrated one. `test_mx.py` checks that the channels follow a gain.

**Proxy evidence.** On the same three clips, exemplars were simulated by 4–16 positives
from other windows of the same clip. That is optimistic, because real exemplars are
moving targets and, at inference, the network's own peaks. Fitted on the other clips (nine
draws: K = 4, 8, 16 × three seeds):

| Clip | Absolute [c, c²] | + NCC (M0) | Calibrated [R, \|R\|, P] | Calibrated + NCC | Absolute + calibrated | Within clip |
|---|---:|---:|---:|---:|---:|---:|
| DJI_0319_2 | 0.837 | 0.861 | 0.904 (0.86–0.97) | 0.880 | 0.904 | 0.995 |
| DJI_0127_1 | 0.785 | 0.974 | 0.980 (0.97–0.99) | 0.987 | 0.727 (0.50–0.84) | 0.981 |
| DJI_0117_2 | 0.775 | 0.788 | 0.755 (0.72–0.78) | 0.766 | 0.773 | 0.708 |

Calibrated contrast transfers across scenes where absolute contrast does not. Adding the
absolute contrast back hurts transfer on DJI_0127_1, which is why MXC uses gain
augmentation rather than only adding channels. These are three clips, and they were
looked at while designing MXC. M0c below decides on the other clips.

### Claims, and the controls that test them

1. The profile-blind targets are a measurable ceiling on the official protocol (MX.md).
2. Absolute contrast does not transfer across scenes. The calibration gap
   (oracle − leave-one-recording-out AUC) is the measure.
3. The moving targets of the same clip, found by the temporal profile without labels at
   test time, calibrate the level pathway. This is a per-video, label-free reference.
4. The calibrated channels are gain-invariant by construction, and gain augmentation
   makes the network rely on them.

| Run | Removes | Tests claim |
|---|---|---|
| `mxc_lv` | – | Proposed |
| `mxc_lv --ex-p-none 1` (`*_noex`) | Exemplars (the calibration channels are 0), keeps gain augmentation | 3: is it the exemplars? |
| `mxc_lv --gain-aug 1` (`*_nogain`) | Gain augmentation | 4 |
| `mx_lv` | Calibration (v5) | 3, 4 |
| `mr_lv --rates 1`, DeepPro-LV | The exemplar branch | Baseline |
| `infer_mx.py --bank gt` / `random` / `none` | Detected exemplars | Upper bound / content / nothing |

## 3. M0c: decide before training

`code/calib_diag.py` (numpy, no GPU) runs M0's sampling on the training clips, with three
changes:

* **Negatives.** Besides the strongest contrast peaks ('top'), it keeps up to 50 'band'
  negatives per window. These are peaks inside the 10–90% band of the window's exemplar
  contrast, the clutter that calibration alone cannot reject.
* **Fitting.** Every logistic regression is fitted leave-one-recording-out (LORO): on the
  clips of all other recordings, with every clip and class weighing the same.
* **Measure.** It reports closure = (full − base) / (oracle − base), where full =
  calibrated + NCC (the inputs of MXC), base = absolute contrast, and oracle = absolute
  contrast fitted within the clip.

Pre-registered **GO** rule, applied to the clips not in `--seen`. The default for `--seen` is
the validation clips, whose M0 numbers shaped this design. The result on all clips is
reported as `decision_all`. GO needs all of:

* (a) at least 2 calibration clips: ≥ 20 positives and negatives, and gap ≥ 0.02;
* (b) median closure ≥ 0.30;
* (c) full < base − 0.03 on at most 20% of the eligible clips;
* (d) with `--det-run`, median closure with detected exemplars ≥ 0.15. Use a run that was
  not trained on these clips, for example a fold model.

`test_calib_diag.py` checks both directions. It says GO on synthetic scenes where
absolute contrast fails across scenes and calibration does not. It says NO-GO when the
moving targets have another contrast than the parked ones.

## 4. Training, checkpoint and scoring: official or ours

| Part | Official DeepPro-Plus (`train.py`, `test.py`) | This code | Same? |
|---|---|---|---|
| Training windows | `TrainIRSeqDataLoader`: 40 frames, 128 × 128 crops centred on a target with p = 0.75, mean 85.0799, std 47.4845 | `train.py` re-implements this recipe | Same recipe, written again. MX/MXC add exemplars, compensated windows (`auto`) and, for MXC, the gain |
| Labels | `segmentations/*.png`, uint8 / 255 > 0 | `paper_k6_floor`, drawn from the boxes | Same pixels: `check_labels.py`, 60 clips |
| Main loss | SoftLoUloss | `soft_iou_loss` | Same |
| Extra losses | None | Level pathway (Soft-IoU + focal) for LV, MR, MX, MXC; match focal for MX, MXC | Ours |
| Optimiser | Adam 1e-3, β (0.9, 0.999), ε 1e-8, wd 1e-4; ×0.7 every 10 epochs (≥ 1e-5); 32 epochs; batch 4 | The same | Same |
| Checkpoint | After every epoch: every test window, sigmoid > 0.5, global IoU; `best_model.pth` when IoU ≥ best | `--checkpoint-selection original_test_iou` (`checkpoint_selection.py`, your version) | Same rule. MX and MXC are now supported, with the exemplars of inference (two passes) |
| Scoring | `test.py`: windows fused by their maximum, IoU at prob > 0.5, ShootingRules Pd/Fa | `infer*.py` (maximum fusion; logits stored as float16), `score.py --author`: ShootingRules vectorised, with the official edge behaviour, and IoU as `test.py`; plus box metrics | Re-implemented, because `test.py` cannot load these networks. It gives the same counts as `ShootingRules.py` on 1,800 / 1,800 random frames, including targets at the edge (see below) |

**The edge of the image.** `ShootingRules.py` slices `[i - 4 : i + 5]` around every target
pixel. For i < 4 the start is negative, so the slice is empty. A target in the first 4 rows or
columns therefore protects no pixel from counting as a false alarm, and one in the first row
or column cannot be hit. `score.py --author` now does the same by default
(`--author-edges official`); `--author-edges clipped` gives the earlier, clipped numbers.
`test_shooting_rules.py` checks both against a transcription of the official loop.

**Which checkpoint.** For the table against DeepPro-Plus, train every method on
`official_train` with `original_test_iou` and use `best_model.pth`, as the authors did for
their released weights. Every method then gets the same advantage. For the claims and the
ablations, `--checkpoint-selection val_iou` chooses on `--selection-split` (default
`val`), so that the test set is not used to choose. It refuses to run if those clips
overlap the training clips or the test set. Train on `train`, select on `val`, and report
`val` and the cross-validation folds.

## 5. Commands

```sh
export VT_ROOT=/home/md0/ly/DP/datasets/RGB-T
bash run_mx.sh all          # check, profile, m0 (M0c), train, eval; stops at the first failure
bash run_mx.sh controls     # *_noex, *_nogain, mx_lv with the same recipe and selection
```

| Variable | Default | Meaning |
|---|---|---|
| `ARCH` | `mxc_lv` | Network |
| `TRAIN_SPLIT` | `official_train` | The 45 official training clips |
| `SELECT` | `original_test_iou` | `val_iou` (with `SELECT_SPLIT`, default `val`) or `final` |
| `EVAL_SPLIT` | `official_test` | Clips for `infer_mx.py` and `score.py` |
| `M0_SPLIT` | `official_train` | M0c clips (test clips are refused) |
| `DET_RUN` | – | M0c also checks detected exemplars |

Validation protocol, for the claims:

```sh
TRAIN_SPLIT=train SELECT=val_iou SELECT_SPLIT=val EVAL_SPLIT=val bash run_mx.sh train && \
TRAIN_SPLIT=train SELECT=val_iou SELECT_SPLIT=val EVAL_SPLIT=val bash run_mx.sh eval
```

Pre-registered success, written before any MXC training:

* M0c GO.
* On `val` (mean of seeds 46–48, recording-matched 1 FA/frame), `mxc_lv` against `mx_lv`
  and `mr_lv --rates 1`:
  * static-camera near-stationary recall +10 points;
  * the ≥ 0.5 px/frame band no worse than −2 points.
* `*_noex` keeps at most half of that gain.
* On the official protocol, Pd at Fa ≤ 6.28e-5 above DeepPro-Plus's 80.83%.

## 6. Limits

* A kind of target that never moves in a clip gives no calibration there, and MXC then is
  `mx_lv` with gain augmentation.
* Detected exemplars can be clutter. The median and D are robust to a minority of them.
  Training replaces 20% of the exemplars with clutter points.
* The ring of the contrast is fixed (radii 4–7). Targets larger than about 7 px get a
  smaller contrast, and so do their exemplars. R compares like with like, but an absolute
  size change between moving and parked targets is not modelled.
* M0c uses GT exemplars. Only `--det-run` tests the inference source, and only `--bank gt`
  versus `outside` tests it in the network.

## 7. What was checked here, and what was not

* **numpy tests:** all pass, including `test_calib_diag.py` (7) and `test_run_mx.py`
  (5: M0c GO and NO-GO through `run_mx.sh`, the validation protocol, the controls stage,
  and the stops).
* **Scoring:** `evaluate.shooting_rules` equals the authors' `ShootingRules.py`, run here
  with skimage, on 1,800 / 1,800 random frames with targets at every edge.
* **The torch functions** `local_contrast_t`, `exemplar_contrast` and
  `calibration_channels` were run with a numpy stand-in for the torch operations they
  use. They agree with `stabilize.local_contrast` and `mx_data.calibrated_features`
  (maximum difference 3e-7).
* **The gain path of `ExemplarTrainSet`** ran with a numpy stand-in for torch. The same
  sample, with gain g, is m + g (x − m) at the valid pixels, the labels do not change, and
  the exemplar contrast is multiplied by exactly g.
* **Not run here (no torch):** `test_mx.py`, that is the MXC network, a training step,
  `val_iou` and `original_test_iou` with an MX network, and the check that the selection's
  exemplar candidates equal `infer_mx.py`'s. `bash run_mx.sh check` runs them first on the
  training machine.
