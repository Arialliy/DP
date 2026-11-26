# DeepPro-MXC, complete bundle (v6)

This folder holds everything DeepPro-MXC needs:

* the code (the MX/MXC files, every file they import, and their tests);
* `run_mx.sh`;
* `configs/splits.json`, which includes `official_train` and `official_test`;
* the design documents.

Unpack it anywhere and run the commands from this folder. It does not need the
`Arialliy/ly` repository. Design, evidence, decision rule and limits are in `docs/MXC.md`.
`docs/MX.md` describes v5 and why its M0 was replaced.

## What changed since v5

| | v5 (`mx_lv`) | v6 (`mxc_lv`, default) |
|---|---|---|
| Exemplars are used for | Appearance matching (NCC-like, amplitude removed) | Matching, and also contrast calibration: R, \|R\|, D, P of the level in exemplar units |
| Training | | `--gain-aug 2`: one contrast gain per sample, applied to the window and its exemplars |
| Go / no-go | M0 `exemplar_diag.py`: baseline fitted within the clip (biased) | M0c `calib_diag.py`: leave-one-recording-out, calibration gap and closure, band negatives |
| Checkpoint | `final.pth` | `best_model.pth` by the authors' rule (per-epoch test-window IoU), or `val_iou` |
| `score.py --author` | Clipped edges | ShootingRules.py's edge behaviour, plus IoU as `test.py` |

## Data

The official RGB-T folder, with the RGBT-Tiny boxes copied into it:

```
RGB-T/train2017/<clip>/01/*.jpg     RGB-T/test2017/<clip>/01/*.jpg
RGB-T/segmentations/<clip>/*.png    RGB-T/train.txt   RGB-T/test.txt
RGB-T/annotations_coco/instances_01_train2017.json   instances_01_test2017.json
```

```sh
export VT_ROOT=/home/md0/ly/DP/datasets/RGB-T     # OFFICIAL_LISTS defaults to it
unset VT_ANN_DIR                                  # not needed when annotations_coco is inside RGB-T
```

## Order: one command, which stops at the first failure

```sh
bash run_mx.sh all
```

| Stage | Does |
|---|---|
| `check` | Every test (torch ones too) and labels vs PNGs; `VT_OLD=/path/to/VT-Tiny-MOT` also compares copies |
| `profile` | `targets.csv`, `targets.json`, homographies (an old `results/profile` is moved aside), `folds.json` |
| `m0` | M0c on `official_train` (no test clip); exit status 3 on NO-GO |
| `train` | `mxc_lv` on `official_train`, `--checkpoint-selection original_test_iou` → `runs/mxc_s46/best_model.pth`; only after a current GO |
| `eval` | `infer_mx.py` with `best_model.pth` on `official_test`; `score.py --author --label-rule paper_k6_floor` |
| `controls` | `mxc_s46_noex` (`--ex-p-none 1`), `mxc_s46_nogain` (`--gain-aug 1`) and `mx_s46` (`mx_lv`), the same recipe and selection |

To run the stages one at a time, chain them with `&&`. Never paste them as separate lines:

```sh
bash run_mx.sh check && bash run_mx.sh profile && bash run_mx.sh m0 && bash run_mx.sh train && bash run_mx.sh eval
```

Each stage stops with a non-zero status, so that `all` or the chain does not go on, when:

* a test fails;
* the labels differ from `RGB-T/segmentations`;
* the two copies differ, or their track relation cannot be confirmed (with `VT_OLD`);
* `profile_motion.py` writes a different number of rows than there are boxes;
* an input is missing or older than its source;
* M0c says NO-GO;
* training ends without its checkpoint.

`targets.csv` from v5 is still valid, because the track ids did not change. `bash run_mx.sh
profile` is needed only if `results/profile` is missing. M0c must run once, because v5's
M0 result (`results/exemplar_diag`) is not read any more.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ARCH` | `mxc_lv` | `mx_lv` for v5 |
| `RUN` | `mxc_s$SEED` | Run name |
| `TRAIN_SPLIT` | `official_train` | 45 clips |
| `SELECT` | `original_test_iou` | `val_iou` (with `SELECT_SPLIT`, default `val`), or `final` |
| `EVAL_SPLIT` | `official_test` | |
| `M0_SPLIT` | `official_train` | |
| `DET_RUN` | – | M0c also checks detected exemplars; use a run not trained on these clips |
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
| `results/profile/targets.csv`, `targets.json`, `homographies/` | `profile_motion.py` | M0c, `train.py`, compensated windows |
| `results/calib_diag/summary.json`, `positives.csv`, `negatives.csv` | M0c | `run_mx.sh train` (GO) |
| `runs/<run>/best_model.pth`, `best_selection.json`, `eval_log.jsonl`, `progress.json` | `train.py` | `infer_mx.py --ckpt` |
| `runs/<run>/final.pth`, `last.pth` | `train.py` | Resume (`last.pth` keeps the selection mode and best IoU) |
| `cache/preds/<run>/` | `infer_mx.py` | `score.py` |
| `results/scores/<run>/` | `score.py` | You |

## DP tree

With `DP/research_models/mx.py` and `models.py`, `research_models.models.build('mxc_lv')`
builds the network inside `Arialliy/DP`. Training and inference use this bundle's `code/`.
