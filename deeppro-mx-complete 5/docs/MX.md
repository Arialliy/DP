# DeepPro-MX: motion exemplars for targets the temporal profile cannot see

This branch adds a detector design, a training-free test that decides whether to train it,
and the code to train, run and score it. Nothing here changes the `lv_*`, `lvh_*` or
`mr_*` defaults. `--label-rule` defaults to the earlier masks.

## Problem: profile-blind targets

The authors' released RGB-T evaluation of DeepPro-Plus
(`eval_ShootingRules_LocLen2_4.txt`) shows a ceiling that thresholds cannot lift:

| Threshold | Pd | Missed | Fa |
|---|---:|---:|---:|
| 0.5 | 80.83% | 2,167 / 11,303 | 6.3e-5 |
| 1e-5 | 87.95% | 1,362 / 11,303 | 4.0e-4 |
| 1e-6 | 94.36% | 638 / 11,303 | 5.3e-2 (5% of all pixels) |

At 1e-5 the false-alarm rate is already six times higher and 1,362 targets are still
missed. The network gives them a score of about 0. 87% of these misses are in four clips:
DJI_0237_2 (518), DJI_0049_2 (311), DJI_0309_4 (259) and DJI_0257_2 (98). The authors
name the reason in Sec. 4.7: "It is hard to detect stationary objects from the temporal
profile." Whether these test misses are near-stationary should be confirmed with
`results/profile/targets.csv`.

On the validation clips the near-stationary boxes fall into two kinds with different
physics:

* **Slow movers.** On the static-camera clips, 1,844 boxes move at 0.05–0.2 px/frame.
  A longer span takes them out of the median, which is what MRTP does.
* **Static ones.** 578 boxes move under 0.05 px/frame. No window length, rate or History
  reaches them. Moreover, 2,146 of the 2,423 near-stationary validation boxes (89%)
  belong to tracks that never reach 0.5 px/frame anywhere in the clip, so remembering an
  earlier motion does not help either.

What remains is appearance. DeepPro-LV's level pathway has to separate a parked target
from a ground hot spot by local appearance alone. A small U-Net trained on 32 clips has to
learn, for every scene, what targets look like at that altitude, time of day and sensor
gain. On the static-camera validation clips it reaches 14.1% near-stationary recall at
1 FA/frame (focal loss, seed 46).

## Idea

Within one clip, targets of the same kind look alike whether they move or not, and the
temporal profile already finds the moving ones: 83–89% recall at ≥ 0.5 px/frame on the
static-camera validation clips, and 54% Pd at Fa 7.3e-6 for the released DeepPro-Plus
at threshold 1.0. DeepPro-MX lets those detections teach the detector what to look for in
this clip:

| Block | Input → output | Notes |
|---|---|---|
| Decomposition | x → L, V, E, cover | Masked median of `mr_lv` (recorded or compensated windows) |
| Profile | V → l_v(t) | DeepPro-Plus backbone, unchanged |
| Encoder A | each frame x_t → F_t (32-d unit vectors) | Local contrast normalisation, 5 dilated 3 × 3 convs, receptive field 39 px, **no normalisation layers** |
| Exemplars | K ≤ 16 patches (39 × 39) of moving targets of the same clip → e_j = A(patch) at the centre | Training: GT boxes with rel_speed ≥ 0.5 from frames outside the window. Inference: peaks of the network's own l_v from frames outside the window |
| Matching | M_t(p) = max_j ⟨F_t(p), e_j⟩; M_x = masked median over every 4th frame | M_g is the same with 8 learned global prototypes |
| Level | LevelNet on [L, E, cover, M_x, A_x, M_g] | A_x = 1 when exemplars are available |
| Fusion | Pointwise gate on V; p = 1 − (1 − σ(l_v))(1 − g σ(l_L)) | As `mr_lv` |

Because the encoder has no batch or group normalisation, a pixel's embedding depends only
on the input within 19 px. So a 39 × 39 exemplar patch gives exactly the vector the full
frame gives at that pixel (`test_mx.py` checks this).

Without valid exemplars, M_x = 0 and A_x = 0, and the network is `mr_lv` plus a learned
prototype channel. Training with `--ex-p-none 1` and inferring with `--bank none` gives the
control that separates "exemplars from this clip" from "a larger appearance branch". It
has the same parameters (about 229k, against 190k for `mr_lv`).

`mx_lv_abs` also matches the inverted frame, for targets whose contrast polarity differs
from the exemplars', for example a parked car colder than the moving ones.

### Relation to earlier work

| Work | Uses | Differs |
|---|---|---|
| DeepPro / DeepPro-Plus | Temporal profile | Blind to stationary targets, by its own account |
| DeepPro-LV / LVH / MRTP (this repository) | Level pathway, History, longer spans | No appearance reference; History vetoes resident targets |
| Jonschkowski & Stone, *Towards Object Detection from Motion* (arXiv:1909.12950); Pathak et al., *Learning Features by Watching Objects Move* (CVPR 2017) | Motion as a training signal for an appearance detector | Training time only; no per-video reference at test time |
| Exemplar-based class-agnostic counting (FamNet, LOCA) | User-given exemplars | Exemplars from a person, not from motion |
| SELSA, MEGA (video object detection) | Feature aggregation across frames | Proposal features, not motion-selected exemplars for static instances |
| Decoupled motion representation for moving IRSTD (arXiv:2606.15286) | Optical-flow motion cues | Moving targets only |

The claims here are as follows. The profile-blind ceiling is a measurable problem on the
official protocol. Motion-selected exemplars give a per-video, label-free appearance
reference at test time. The no-normalisation encoder makes patch and frame embeddings
identical. The no-exemplar control isolates the exemplar effect. Check the literature
again before writing.

## Files

| File | Content |
|---|---|
| `code/exemplar_diag.py` | M0, the training-free test (numpy) |
| `code/mx.py` | `mx_lv`, `mx_lv_abs`, `match_loss` (torch) |
| `code/mx_data.py` | Exemplar tables, selection, patches, training draws, inference banks (numpy) |
| `code/train.py` | `ExemplarTrainSet`, `--ex-*`, `--mx-aux`, `--label-rule`; `MultiRateTrainSet.__getitem__` split into `_draw` and `_sample` |
| `code/infer_mx.py` | Two-pass inference with banks `outside`, `causal`, `clip`, `gt`, `random`, `none` |
| `code/score.py` | `--label-rule` for the authors' protocol (`--author`) |
| `code/vtdata.py` | Rule `paper_k6_floor` (the masks `export_masks.py` writes); boxes from `VT_ANN_DIR`; reads `tracking_id` (RGBT-Tiny release) as `track_id` |
| `code/check_labels.py` | Checks that the training labels equal the PNGs in `segmentations/` |
| `code/models.py` | Registers the `mx` module |
| `code/fake_dataset.py` | A tiny dataset in a temporary folder, created and deleted by the end-to-end tests |
| `code/test_mx_data.py`, `test_exemplar_diag.py`, `test_infer_mx.py`, `test_dataset_layout.py` | numpy tests |
| `code/test_mx.py` | torch tests, including one short training epoch and inference on a temporary fake dataset |

## Data on the training machine

The official RGB-T folder holds frames, masks and clip lists:

```
RGB-T/train2017/<clip>/01/*.jpg   RGB-T/test2017/<clip>/01/*.jpg
RGB-T/segmentations/<clip>/*.png  RGB-T/train.txt  RGB-T/test.txt
```

It holds no boxes. Every step here needs the boxes with their track ids:

* training crops are centred on targets;
* exemplars are moving targets, and their speeds come from `profile_motion.py`;
* M0 needs the near-stationary annotations;
* `score.py` matches detections to boxes.

A mask does not give them back, because neighbouring targets merge and the track ids are
lost. So the frames and lists come from `RGB-T`, and the boxes from the release's
`annotations_coco`. Copy that folder into `RGB-T` (it is found there without further
settings) or point `VT_ANN_DIR` at it:

```sh
export VT_ROOT=/path/to/DP/datasets/RGB-T OFFICIAL_LISTS=/path/to/DP/datasets/RGB-T
# export VT_ANN_DIR=/path/to/DP/datasets/RGBT-Tiny/annotations_coco   # only if it is not in RGB-T/
python code/check_labels.py                 # training labels (paper_k6_floor) == RGB-T/segmentations, all 60 clips
NPROC=8 python code/profile_motion.py       # speeds and homographies of these annotations (back up results/profile first)
python code/folds.py                        # configs/folds.json, for cross-validation
```

What each step reads besides the dataset:

| Step | Needs |
|---|---|
| tests (`test_*.py`) | Nothing: they build their own data in a temporary folder |
| `check_labels.py` | Nothing |
| `profile_motion.py` | Nothing; it writes `results/profile/{targets.csv, clips.csv, homographies/}` |
| `folds.py` | `configs/splits.json`, `results/profile/targets.csv` |
| `exemplar_diag.py` | `targets.csv`, and the homographies unless `--coords raw` |
| `train.py --arch mx_lv` | `targets.csv`, the homographies (coordinates `auto`), `configs/splits.json` or `folds.json` |
| `infer_mx.py` | The homographies unless `--coords raw`; `targets.csv` for `--bank gt` |
| `score.py` | `configs/splits.json` or `folds.json` |

`train.py` does not read the PNGs. With `--label-rule paper_k6_floor`, now the default of
`train.py` and of `score.py --author`, it renders the same masks from the boxes, cropped or
warped with each frame. `check_labels.py` confirms that they equal the files on disk. Pass
`--label-rule sigma_sixth` only to reproduce runs made before DeepPro-MX, whose
`config.json` has no `label_rule`.

`targets.csv` must come from the current `profile_motion.py` and the same annotations.
`profile_motion.py` writes `targets.json` next to it, naming the track-id scheme.
`train.py`, `exemplar_diag.py` and `infer_mx.py --bank gt` stop unless that file names the
current scheme and every box of every clip they use has a row. Coverage is compared
unrounded, so 89.96% is not 90%. Requiring every box matters: a file keyed by the raw
`tracking_id` still matches every ship box (category 0, whose id does not change), so a
mostly-ship clip would pass any lower threshold. A `targets.csv` made before this change
is therefore refused: run `profile_motion.py` again (`bash run_mx.sh profile`), then
`folds.py`, then M0.

**Track ids.** The RGBT-Tiny release numbers `tracking_id` within each category, so a
pedestrian and a drone of one clip can both be track 0. `vtdata` therefore uses
track = category × 1,000,000 + tracking_id (`vtdata.track_of`, `split_track` inverts it).
Keyed by `tracking_id` alone, two targets would be merged into one track, and boxes of one
frame would overwrite each other's speed row.

`profile_motion.py` writes one row per box and stops if the row count differs from the
box count. A box that still shares its frame's track id with another box is ambiguous: it
gets nan speeds and is counted under `duplicate_keys` in `clips.csv`. For boxes with unique
keys the speeds are identical to those of the earlier version.

The VT-Tiny-MOT copy used for the earlier LV, LVH and MRTP experiments has clip-unique
`track_id`s (no duplicate keys among the 5,664 validation boxes, one category per track),
so those results are not affected. The official DeepPro training and evaluation read only
the masks, which do not depend on track ids.

## Running with hard stops

`run_mx.sh` (repository root) runs every step with `set -euo pipefail`. A failing test,
differing labels, a missing or stale input, and M0 NO-GO each end it before the next step,
and training starts only after a GO that is newer than `targets.csv`. Labels are always
`paper_k6_floor`.

```sh
export VT_ROOT=/path/to/DP/datasets/RGB-T
bash run_mx.sh all        # check, profile, m0, train, eval; stops at the first failure
```

The stages, each also usable on its own:

| Stage | Does | Stops when |
|---|---|---|
| `check` | All tests (torch ones too), labels vs PNGs; with `VT_OLD=...` also `compare_datasets.py` | A test fails, labels differ, the copies differ, or the track relation cannot be confirmed |
| `profile` | `targets.csv`, `targets.json`, homographies (an old `results/profile` is moved aside), `folds.json` | Rows ≠ boxes |
| `m0` | The go / no-go test | NO-GO (exit status 3) |
| `train` | `mx_lv` with `paper_k6_floor` | No GO, or M0 older than `targets.csv` |
| `eval` | `infer_mx.py`, `score.py --author --label-rule paper_k6_floor` | No checkpoint |

To run stages one by one, chain them so that a failure stops the rest:

```sh
bash run_mx.sh check && bash run_mx.sh profile && bash run_mx.sh m0 && bash run_mx.sh train && bash run_mx.sh eval
```

Pasted as separate lines, the stages would go on after a failure.

`code/test_run_mx.py` checks each stop condition.

## Step 0: tests

```sh
bash run_mx.sh check      # or each file: python code/test_<name>.py; test_mx.py and test_lvh.py need torch
```

The end-to-end tests write a small fake dataset to a temporary folder and delete it
afterwards. They do not read or change `RGB-T`, and the shell's `VT_ROOT` and
`VT_ANN_DIR` do not affect them.

## Step 1: M0, decide without training (minutes)

```sh
python code/exemplar_diag.py --split val          # exit status 0 on GO, 3 on NO-GO
python code/exemplar_diag.py --split fold0/val fold1/val fold2/val fold3/val fold4/val   # after code/folds.py
```

For each window, M0 uses the following samples:

* **Positives:** near-stationary annotations resident in the window.
* **Negatives:** the strongest |local contrast| peaks of L away from every annotation.
* **Exemplars:** up to 16 moving GT boxes from frames outside the window.

The script reports per-clip AUCs of contrast, NCC to exemplars on L, |NCC|, and the
median over frames of NCC on each frame. Every score is maximised over ±2 px, because
positives sit on box centres while negatives sit on contrast peaks; without this, NCC
would favour the positives by construction.

The decision uses held-out logistic regressions (folds alternate over windows):
[contrast, contrast²] against the same plus the exemplar scores, giving
gain = AUC(full) − AUC(base). A raw contrast AUC far below 0.5 is also information, which
this comparison credits to the baseline.

**GO** if gain ≥ 0.10 on DJI_0117_2 and DJI_0319_2 (when present) and on ≥ 60% of the
clips with ≥ 20 positives and negatives. Otherwise do not train MX. The synthetic tests
check both directions: GO when the hot spots are round blobs, NO-GO when they are
car-shaped.

`--det-run <run>` adds `ncc_det`, with exemplars from the peaks of a cached run, as
inference would use them. Use a run trained without the validation clips.

## Step 2: train and compare

Use the paper masks for every run in this comparison (the default; written out here), and
train only after a GO. The block runs in its own `bash -euo pipefail`, so a failure or a
NO-GO stops it without closing the terminal it is pasted into:

```sh
bash -euo pipefail <<'STEP2'
python code/exemplar_diag.py --split fold0/val fold1/val fold2/val fold3/val fold4/val   # stops here on NO-GO
for s in 46 47 48; do for f in 0 1 2 3 4; do
  python code/train.py --run mrlv1_f${f}_s$s --arch mr_lv --rates 1 --split fold$f/train --seed $s --device cuda \
    --label-rule paper_k6_floor
  python code/train.py --run mx_f${f}_s$s --arch mx_lv --split fold$f/train --seed $s --device cuda \
    --label-rule paper_k6_floor
  python code/train.py --run mxnone_f${f}_s$s --arch mx_lv --split fold$f/train --seed $s --device cuda \
    --label-rule paper_k6_floor --ex-p-none 1
done; done
python code/infer_mx.py --run mx_f0_s46 --ckpt runs/mx_f0_s46/final.pth --clips fold0/val --device cuda --save-arms
python code/score.py --run mx_f0_s46 --split fold0/val --author --label-rule paper_k6_floor
STEP2
```

| Run | Question |
|---|---|
| `mr_lv --rates 1` | The same base without the exemplar branch |
| `mx_lv`, `--bank outside` | Proposed |
| `mx_lv --ex-p-none 1`, `--bank none` | Is it the exemplars, or only a larger appearance branch? (main control) |
| `mx_lv`, `--bank gt` | Upper bound of the exemplar source |
| `mx_lv`, `--bank causal` | Without future frames |
| `mx_lv`, `--bank random` | Control: exemplars without content |
| `mx_lv_abs` | Polarity, if M0 shows |NCC| ≫ NCC |

Pre-registered success, with the mean of three seeds and recording-matched 1 FA/frame,
comparing `mx_lv`/`outside` with `mr_lv --rates 1` and with the control:

* static-camera near-stationary recall: +15 points;
* the ≥ 0.5 px/frame band and moving-camera macro recall: no worse than −2 points;
* the control and the random bank stay within half of that gain.

The official protocol is used once, on the test set, after this table is fixed: Pd at
Fa ≤ 6.28e-5 above DeepPro-Plus's 80.83% (`score.py --author --label-rule paper_k6_floor`).

## Limits

* A kind of target that never moves in a clip has no exemplars there. It is left to the
  global prototypes, which is about what the level pathway already does.
* Exemplars from detections can be clutter. Training replaces 20% of the exemplars with
  clutter points and jitters the others by ±1 px; M0's `ncc_det` measures the effect.
* `outside` and `clip` use frames after the window. Like DeepPro's windows, that is
  offline. `causal` is the online variant.
* Inference makes two passes: the variation pathway, then the full network.
* Warped windows match warped frames against recorded exemplar patches. A warp changes
  local appearance only slightly, but it is untested on real parallax.

## What was checked, and what was not

* **numpy tests pass here:** 84 in 14 files, including the older tests of this
  repository.
* **The training-data path ran here with a numpy stand-in for torch.** Its samples are
  identical to the code before this change for `MultiRateTrainSet` (auto, raw and warp;
  rates 1 and 1,2) and for `TrainSet`. `ExemplarTrainSet` and `StabilizedTrainSet` run
  with the new label rule.
* **The official RGB-T layout:** with the boxes from `VT_ANN_DIR`, the `paper_k6_floor`
  labels equal, pixel for pixel, the PNGs that `export_masks.py --k 6 --storage floor`
  writes, and the earlier `sigma_sixth` labels do not (`test_dataset_layout.py`).
* **Stops:** `run_mx.sh` ends at a failing test, differing labels, missing or stale inputs,
  and NO-GO; training starts only after a current GO and always with `paper_k6_floor`
  (`test_run_mx.py`, with the real `profile_motion.py` and M0 on synthetic clips).
* **`targets.csv`:** refused without the current `targets.json`, and for any box without a
  row (89.96% fails a 90% threshold; a raw-id file on a 95%-ship clip fails).
* **Track ids:** a pedestrian and a drone that are both `tracking_id` 0 stay two tracks with
  their own speeds; every box gets a row; ambiguous keys get nan; with unique keys
  `profile_motion.py` gives the same rows as before (`test_dataset_layout.py`).
* **Imports:** `mx.py` imports and registers in both this repository's `code/` and DP's
  `research_models/` package.
* **Lint:** ruff (pyflakes, pycodestyle, bugbear) and flake8 pass on all new and changed
  files.
* **Not run here:** `test_mx.py` (torch is not installed in the authoring environment),
  and therefore the network, the losses, a training step and `infer_mx.py` with a real
  model. Run `test_mx.py` first on the training machine.
