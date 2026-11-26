# DeepPro-MX, complete bundle

Everything DeepPro-MX needs, in one folder: the code (the MX files and every file they
import, with their tests), `run_mx.sh`, `configs/splits.json`, and the design documents.
Unpack it anywhere and run the commands from this folder. It does not need the
`Arialliy/ly` repository. Design, decision rule and limits are in `docs/MX.md`.

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

It runs the five stages in order:

| Stage | Does |
|---|---|
| `check` | Every test (torch ones too) and labels vs PNGs; with `VT_OLD=/path/to/VT-Tiny-MOT` also `compare` (same frames, boxes and track relation) |
| `profile` | `targets.csv`, `targets.json`, homographies (an old `results/profile` is moved aside), `folds.json` |
| `m0` | The go / no-go test; exit status 3 on NO-GO |
| `train` | Only after a GO newer than `targets.csv`; labels `paper_k6_floor` |
| `eval` | `infer_mx.py`; `score.py --author --label-rule paper_k6_floor` |

To run them one at a time, chain them with `&&`, never as separate pasted lines:

```sh
bash run_mx.sh check && bash run_mx.sh profile && bash run_mx.sh m0 && bash run_mx.sh train && bash run_mx.sh eval
```

Each stage stops with a non-zero status, so that `all` or the chain does not go on, when:

* a test fails;
* the labels differ from `RGB-T/segmentations`;
* the two copies differ, or their track relation cannot be confirmed (with `VT_OLD`);
* `profile_motion.py` writes a different number of rows than there are boxes;
* an input is missing or older than its source;
* M0 says NO-GO.

Settings come from environment variables: `SEED` (46), `DEVICE` (cuda), `RUN`
(mx_s$SEED), `NPROC` (8), `M0_SPLIT` (val), `FOLDS` (1).

## targets.csv must be generated again

The track ids changed: they are now category × 1,000,000 + tracking_id, because RGBT-Tiny
numbers `tracking_id` within each category. A `targets.csv` written before this change is
refused, because it has no `targets.json` naming the current scheme. One that misses any
box is refused too. Regenerate it, together with `configs/folds.json`, and run M0 again before
training:

```sh
bash run_mx.sh profile && bash run_mx.sh m0
```

## Outputs (all under the current folder)

| Path | Written by | Read by |
|---|---|---|
| `results/profile/targets.csv`, `targets.json` | `profile_motion.py` | `folds.py`, `exemplar_diag.py`, `train.py` (mx), `infer_mx.py --bank gt` |
| `results/profile/homographies/<clip>.npy` | `profile_motion.py` | Compensated windows (`--coords auto` / `warp`) |
| `configs/folds.json` | `folds.py` | `--split foldK/train`, `foldK/val` |
| `results/exemplar_diag/summary.json` | M0 | `run_mx.sh train` (GO) |
| `cache/video/<clip>.npy` | `train.py` | Frame cache |
| `runs/<run>/` | `train.py` | `infer_mx.py --ckpt` |
| `cache/preds/<run>/` | `infer_mx.py` | `score.py` |
| `results/scores/<run>/` | `score.py` | You |

## Labels

`train.py` and `score.py --author` default to `paper_k6_floor`, the masks in
`RGB-T/segmentations`. `--label-rule sigma_sixth` reproduces runs made before DeepPro-MX.

## DP tree

`DP/research_models/mx.py` and `models.py` let `research_models.models.build('mx_lv')`
build the network inside `Arialliy/DP`. Training and inference use this bundle's `code/`.
