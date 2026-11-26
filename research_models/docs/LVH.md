# DeepPro-LVH: Level–Variation–History

This branch extends DeepPro-LV for near-stationary targets. Nothing in the frozen
`deeppro_lv_calibrated_seed46` path changes; `lv_*` architectures train exactly as before
unless the new loss flags are passed.

## Why

Reading the seed-46 code and results gave four findings.

1. **The decomposition ignores camera motion.** `lv.decompose` takes the median of fixed
   image pixels, while near-stationary is defined by camera-compensated relative speed.
   The clip with 93% of the held-out false alarms, `DJI_0293_1`, pans 1.68 px/frame
   (≈67 px per window); `DJI_0237_2`, which holds 2,373 of the 2,856 test near-stationary
   boxes, pans 1.01 px/frame.
2. **The level pathway never learns to reject ground hot spots.** Soft-IoU has exactly zero
   gradient on a crop without positive pixels (checked in `test_lvh.py`), and three in four
   crops are centred on targets. A hot spot is penalised only when a persistent target
   shares its crop.
3. **Its inputs cannot separate the two.** In `[L, E]` a parked target and a hot spot are
   both bright in L with E ≈ 0, and the pointwise gate sees V ≈ 0 at both and stays open.
4. **Near-stationary evaluation rests on two clips.** `DJI_0319_2` holds 80% of the
   validation near-stationary boxes and `DJI_0237_2` 83% of the test ones.

## Idea

LV separates content by dwell time inside one window. A parked target and a hot spot have
the same dwell time; they differ in history. LVH adds that third time scale, all in the
coordinates of the window's middle frame:

| content | Variation V | Level L (window) | History B (outside window) |
|---|---|---|---|
| moving target | yes | no | no |
| parked target | weak | yes | **no** (arrived or leaves) |
| ground hot spot | no | yes | **yes** |

The level pathway receives `D = L − B` and a History-availability mask, and the gate gets
3×3 spatial context and D.

Limitation: a target present and motionless for the whole clip is in B as well, so History
cannot separate it. `diagnose_history.py` measures how many near-stationary boxes fall in
that case before any training.

## Files

| file | content |
|---|---|
| `code/stabilize.py` | homography chains, window warping, History median, warp-back, local contrast (numpy/OpenCV) |
| `code/diagnose_history.py` | training-free test of the hypothesis |
| `code/hotspots.py` | mines persistent bright spots as hard negatives |
| `code/lvh_data.py` | stabilized training sample (numpy) |
| `code/lvh.py` | `lvh_full`, `lvh_stab`, `lvh_pointgate`, `lvh_nogate`, focal BCE, soft persistence target |
| `code/train.py` | `StabilizedTrainSet`, `--level-focal`, `--main-focal`, `--persist`, `--hist-*`, `--hard-neg` |
| `code/infer.py` | stabilized inference with warp-back; `--save-arms` |
| `code/test_*.py` | tests; all but `test_lvh.py` run without torch |

## Run order

Homographies come from `code/profile_motion.py` (`results/profile/homographies/`,
`results/profile/targets.csv`). Set `VT_ROOT`.

```sh
# 0. tests
python code/test_stabilize.py && python code/test_lvh_data.py
python code/test_diagnose_history.py && python code/test_lvh.py

# 1. go / no-go, no training (CPU, minutes per clip)
python code/diagnose_history.py --split val test --nproc 4
```

Read `results/history_diag/summary.json` per clip, not pooled:

* `near_static_whole_clip_fraction` – share of near-stationary boxes History cannot help.
* `auc_L` vs `auc_D` – separation of resident near-stationary targets from persistent
  bright spots by local contrast of L and of L − B.
* `spots_per_window_at_half_recall_*` – bright spots per window that survive a threshold
  keeping half of the targets.

Train only if `auc_D` clearly beats `auc_L` on the clips that contain near-stationary
targets. If most near-stationary boxes are `static_whole_clip`, the problem is appearance,
not history, and LVH is the wrong lever.

```sh
# 2. ablation ladder, 3 seeds each (46 47 48)
python code/train.py --run lvfocal_s46   --arch lv_full   --seed 46 --device cuda --level-focal 1   # loss fix only
python code/train.py --run lvhstab_s46   --arch lvh_stab  --seed 46 --device cuda                  # + camera compensation
python code/train.py --run lvh_s46       --arch lvh_full  --seed 46 --device cuda                  # + History (proposed)
python code/train.py --run lvhpg_s46     --arch lvh_pointgate --seed 46 --device cuda              # gate ablation
python code/train.py --run lvhpast_s46   --arch lvh_full  --seed 46 --device cuda --hist-mode past # causal History

# 3. predictions and scores
python code/infer.py --run lvh_s46 --arch lvh_full --ckpt runs/lvh_s46/final.pth --clips val --device cuda --save-arms
python code/score.py --run lvh_s46 --split val
```

`--save-arms` writes `cache/preds/<run>__{variation,level,gated_level}/`, so the existing
calibrated fusion can be rebuilt on the new arms.

## Reporting

* Near-stationary recall per clip and its macro average; always also without
  `DJI_0319_2` (val) and `DJI_0237_2` (test).
* Leave-one-recording-out over the 40 non-test clips instead of the 8 validation clips.
* False alarms also as distinct locations, since one bright spot in the level map fires in
  every frame of its window.

## Known gaps

* Homographies assume a planar or distant scene; tall structures leave parallax edges.
  `cover` (fraction of valid frames per pixel) is the only alignment-quality input so far.
* History is unavailable where no out-of-window frame covers a pixel; there `hvalid = 0`
  and the level pathway falls back to LV behaviour.
* Mined hot spots can include unannotated real targets; inspect some before training.
* `test_lvh.py` was written without a torch install and has not been run yet.
