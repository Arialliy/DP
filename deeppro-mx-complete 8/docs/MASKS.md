# Gaussian masks for RGBT-Tiny, written to the DeepPro paper

## What the paper says, and what it leaves open

DeepPro (arXiv:2506.12766), Sec. 4.6, is the whole specification:

> "For training, we generated rough Gaussian masks as labels according to the bounding
> boxes provided by its authors."

The paper gives no sigma, no formula and no citation, and the authors released neither
the masks nor the code that made them. `code/gaussian_masks.py` implements the sentence
and nothing more. Every choice the sentence leaves open is a parameter you can state.

For each box (x, y, w, h), in COCO convention (pixel (u, v) covers [u, u+1) × [v, v+1)):

    g(u, v) = exp(−(u + 0.5 − cx)² / (2 sx²) − (v + 0.5 − cy)² / (2 sy²))
    cx = x + w/2,  cy = y + h/2,  sx = w/k,  sy = h/k

This is a Gaussian with peak 1 at the box centre. The mask of a frame is the per-pixel
maximum over its boxes. Boxes with w ≤ 0 or h ≤ 0 are skipped.

| Open choice | Parameter | Source |
|---|---|---|
| Spread | `k`, with sigma = box size / k | Not in the paper. `check_mask_protocol.py` measures it. |
| 8-bit storage | `storage`: `round` = rint(255 g), `floor` = uint8(255 g) | Not in the paper |
| Ignored boxes | `--ignored` | Not in the paper |
| Reading the mask | uint8 / 255, every value > 0 set to 1, native 640 × 512 | Authors' `TrainDataLoader.py`, `TestDataLoader.py` (`RGB-T`) |
| Counting targets | 8-connected components per frame | Authors' `ShootingRules.py` |

The loaders binarise, so a network trains on the **support** of the stored Gaussian.
That support is an ellipse with semi-axes a·w/k and a·h/k, where a = √(2 ln 510) = 3.53
for `round` and √(2 ln 255) = 3.33 for `floor`. At k = 6 the semi-axes are 0.589 and
0.555 times the box size.

A box whose Gaussian is narrower than a pixel can lose every pixel in 8 bits, for example
a 1 × 1 box centred on a pixel corner. It then has no label, as the formula implies.
Integer 1 × 1 and 2 × 2 boxes keep theirs. `export_masks.py` counts such boxes.

## The earlier `vtdata.gaussian_mask`

The training code in this repository uses `vtdata.gaussian_mask(…, 'sigma_sixth')`. It
adds two things the paper does not have:

* a sigma floor of 0.5 px;
* a cut of the Gaussian to a square of ceil(4 sigma) + 1 px.

For boxes of at least 3 px on both sides it equals `gaussian_masks.label(…, k=6,
storage='round')` pixel for pixel. That holds in a test and on 3,000 random 640 × 512
frames. Smaller boxes get a larger mask there. Its docstring also used to describe
`sigma_quarter` as the default although every caller passes `sigma_sixth`; it now says
what the code does. `check_mask_protocol.py` reports how many boxes are under 3 px.

## Use

Write masks into the RGBT-Tiny root in the layout of the official DeepPro code:

```sh
export VT_ROOT=/path/to/RGBT-Tiny OFFICIAL_LISTS=/path/to/official_deeppro/datasets/RGB-T
python code/check_mask_protocol.py                  # which k / storage / ignored match the authors' log
python code/export_masks.py --k 6 --storage round   # use the values the check supports
```

The layout is then:

```
$VT_ROOT/train2017/<clip>/01/*.jpg     from the release
$VT_ROOT/test2017/<clip>/01/*.jpg
$VT_ROOT/segmentations/<clip>/*.png    one 8-bit Gaussian per frame, same names as the frames
$VT_ROOT/segmentations/MASKS.json      parameters, paper sentence, per-clip box counts
$VT_ROOT/train.txt, test.txt           45 + 15 clips of the paper
```

The official `train.py --dataset RGB-T --datapath $VT_ROOT` reads this as it is.

## Choosing k

The authors' log (`eval_ShootingRules_LocLen2_4.txt`) gives the number of targets their
evaluation found in each of the 15 test clips, 11,303 in all. Their evaluation counted
components of their masks, and wider Gaussians merge neighbours, so these counts depend
on k, storage and ignored. `check_mask_protocol.py` tries k ∈ {2, 3, 4, 5, 6, 7, 8, 10,
12} with both storages, with and without ignored boxes, and prints how many clips each
setting reproduces exactly.

* **One setting matches all 15 clips:** use it, and say so in the paper.
* **Several settings match:** the counts cannot tell them apart. Use the one nearest
  the earlier `sigma_sixth` (k = 6, `round`) and report the range.
* **None matches:** report the closest setting and its per-clip differences. The
  authors' masks then differ in some other way, for example in the box set or in
  thresholding, and the paper sentence cannot settle it.

A sentence for a methods section:

> Following DeepPro, we train on rough Gaussian masks generated from the RGBT-Tiny boxes:
> g = exp(−Δx²/2(w/k)² − Δy²/2(h/k)²) around each box centre, per-pixel maximum, stored
> in 8 bits and binarised (> 0) by the official loader. With k = …, the masks reproduce
> the per-clip target counts of the authors' released evaluation on all 15 test clips.

## Tests

```sh
python code/test_gaussian_masks.py
```

These tests check four things: the formula against a dense evaluation; the centre
convention; the 8-bit support radius; and a PNG round trip through the authors' loader.
They also check the equality with the earlier rule, tiny boxes, 8-connected counting and
log parsing.
