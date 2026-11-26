# DeepPro-MTA / MXC, complete bundle (v8)

This folder holds everything DeepPro-MTA and DeepPro-MXC need:

* the code: every file they import, and their tests;
* `run_mx.sh`;
* `configs/splits.json`;
* the design documents.

Unpack it anywhere and run the commands from this folder. It does not need the
`Arialliy/ly` repository.

* **Current design** (diagnosis of the official model, problem, `mta_lv`, M1, controls,
  success criteria): `docs/MTA.md`.
* **The calibration it builds on:** `docs/MXC.md`.

## What changed since v7

| | v7 | v8 |
|---|---|---|
| Problem | Near-stationary targets | The official model's misses: 57.7% moving, and 40.9% in the first 60 frames of the clips (the same tracks: 26% early, 10% later); in a single frame they look like the targets the detector finds in the same video |
| Model | `mxc_lv`: calibration on L, the window median, so moving targets cannot be recovered | `mta_lv`: `mxc_lv` plus a single-frame appearance pathway on every frame, taught by the network's own confident detections; noisy-OR fusion |
| Before training | M0c | Also M1 (`bash run_mx.sh m1`): which misses are recoverable from single frames at 0.25 / 1 / 2 clutter candidates per frame |
| `auto` coordinates | Warped even when cover < min_cover at rate 1 | Recorded frames (warp only for an explicit `warp` policy) |
| Annotation-free homographies | Any RANSAC result | Plausibility check, with partial affine / shift / identity fallback; comparison over the common view only |
| Controls | noex, self, nogain, mx_lv | Also `*_noapp` (`mxc_lv`) for `mta_*`; your two fixes included |

**Run `profile` again.** The homographies changed, so M0c, M1's window residuals and every
compensated window need the new ones.

```sh
bash run_mx.sh profile && bash run_mx.sh m1      # M1 needs cache/preds/official_rgbt
```

## Data

The official RGB-T folder, with the RGBT-Tiny boxes copied into it:

```
RGB-T/train2017/<clip>/01/*.jpg     RGB-T/test2017/<clip>/01/*.jpg
RGB-T/segmentations/<clip>/*.png    RGB-T/train.txt   RGB-T/test.txt
RGB-T/annotations_coco/instances_01_train2017.json   instances_01_test2017.json
```

```sh
export VT_ROOT=/home/md0/ly/DP/datasets/RGB-T     # OFFICIAL_LISTS defaults to it
unset VT_ANN_DIR HOMOGRAPHY_DIR                   # HOMOGRAPHY_DIR would override results/motion/homographies
```

## Order: one command, which stops at the first failure

```sh
bash run_mx.sh all
```

| Stage | Does |
|---|---|
| `check` | Every test (torch ones too), and labels vs PNGs. `VT_OLD=/path/to/VT-Tiny-MOT` also compares the copies |
| `profile` | `targets.csv` / `targets.json` (speeds, box-masked), `results/motion/homographies` (annotation-free, for every network input), `folds.json`. Old folders are moved aside |
| `m0` | M0c on `official_train` (no test clip): GO or NO-GO, recorded |
| `m1` | M1 (`miss_recovery.py`) on `M1_RUN` (default `official_rgbt`, cached predictions) and `M1_SPLIT` (test): recoverable misses per group; not part of `all` |
| `train` | `mxc_lv` on `official_train` with `--checkpoint-selection original_test_iou` → `runs/mxc_s46/best_model.pth` and `m0c.json` |
| `eval` | `infer_mx.py` with `best_model.pth` on `official_test`, then `score.py --author --label-rule paper_k6_floor` (27 thresholds) |
| `controls` | `*_noex`, `*_self`, `*_nogain`, `mx_s46`, and `*_noapp` (`mxc_lv`) for `ARCH=mta_lv`: the same recipe and selection |

To run the stages one at a time, chain them with `&&`. Never paste them as separate lines:

```sh
bash run_mx.sh check && bash run_mx.sh profile && bash run_mx.sh m0 && bash run_mx.sh train && bash run_mx.sh eval
```

**`profile` must run again in v7.** It writes the annotation-free homographies. M0c and
every compensated window read them, so M0c must run after `profile`.
`results/motion/clips.csv` reports how far they are from the box-masked ones.

A stage stops with a non-zero status, so that `all` or the chain does not go on, when:

* a test fails;
* the labels differ from `RGB-T/segmentations`;
* the copies differ, or their track relation cannot be confirmed (with `VT_OLD`);
* `profile_motion.py` writes another number of rows than there are boxes;
* an input is missing, or older than its source (M0c older than `targets.csv` or
  `motion.json`);
* training ends without its checkpoint;
* `score.py` finds clips predicted by different checkpoints.

An M0c NO-GO is printed and saved, but does not stop training unless `M0_GATE=1`.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ARCH` | `mxc_lv` | `mta_lv` (v8, the single-frame appearance pathway), `mxc_lv_absfloor` (fixed floor), `mx_lv` (v5) |
| `RUN` | `mxc_s$SEED` | Run name |
| `TRAIN_SPLIT` | `official_train` | 45 clips |
| `SELECT` | `original_test_iou` | `val_iou` (with `SELECT_SPLIT`, default `val`), or `final` |
| `EVAL_SPLIT` | `official_test` | |
| `M0_SPLIT` | `official_train` | |
| `M0_GATE` | 0 | 1: M0c NO-GO stops training |
| `DET_RUN` | – | M0c also checks detected exemplars; use a run that was not trained on these clips |
| `SEED`, `DEVICE`, `NPROC`, `FOLDS`, `PY` | 46, cuda, 8, 1, python | |

The validation protocol, for the ablations and claims, keeps the test set out of every
choice:

```sh
TRAIN_SPLIT=train SELECT=val_iou SELECT_SPLIT=val EVAL_SPLIT=val bash run_mx.sh train && \
TRAIN_SPLIT=train SELECT=val_iou SELECT_SPLIT=val EVAL_SPLIT=val bash run_mx.sh eval
```

## Outputs (all under the current folder)

| Path | Written by | Read by |
|---|---|---|
| `results/profile/targets.csv`, `targets.json` | `profile_motion.py` | M0c, `train.py`, `infer_mx.py --bank gt` |
| `results/profile/homographies/` | `profile_motion.py` (box-masked) | The speeds only |
| `results/motion/homographies/`, `clips.csv`, `motion.json` | `motion_image.py` (annotation-free) | Every compensated window: training, M0c, selection, inference |
| `results/calib_diag/summary.json` | M0c | `run_mx.sh train` (copied to `runs/<run>/m0c.json`) |
| `runs/<run>/best_model.pth`, `best_selection.json`, `eval_log.jsonl`, `progress.json` | `train.py` | `infer_mx.py --ckpt` |
| `cache/preds/<run>/`, `<run>__meta/` (fingerprints) | `infer_mx.py` | `score.py` |
| `results/scores/<run>/summary.json` | `score.py` | You: `author_headline` (IoU, Pd, Fa at 0.5, AUC), `author_protocol` (27 thresholds), `predictions` (checkpoint per clip) |

## DP tree

With `DP/research_models/mx.py` and `models.py`, `research_models.models.build('mxc_lv')`
builds the network inside `Arialliy/DP`. Training and inference use this bundle's `code/`.
