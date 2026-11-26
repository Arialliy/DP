# Paper-strict Gaussian masks for RGBT-Tiny

Complete files, based on Arialliy/ly commit 6f0b443. code/vtdata.py is the same in that
commit and in the MRTP files, so this folder can be copied over either.

## Install

    cp -r paper-masks/code paper-masks/docs /path/to/ly/

## New files (5)

| File | Lines | Content |
|---|---:|---|
| `code/check_mask_protocol.py` | 106 | compares every k / storage / ignored setting with the 15 per-clip target counts of the authors' log |
| `code/export_masks.py` | 128 | writes segmentations/<clip>/*.png, train.txt, test.txt and MASKS.json for the official DeepPro RGB-T loaders |
| `code/gaussian_masks.py` | 142 | the paper sentence as code: Gaussian per box, per-pixel max, 8-bit storage; authors' loader binarisation and 8-connected counting; authors' per-clip test counts |
| `code/test_gaussian_masks.py` | 115 | numpy tests (8) |
| `docs/MASKS.md` | 107 | what the paper specifies, what is a parameter, how to export and choose k |

## Modified files (1)

| File | Lines | Change |
|---|---:|---|
| `code/vtdata.py` | 118 | gaussian_mask docstring and default now match the code (sigma_sixth); results unchanged |

## Run

    python code/test_gaussian_masks.py
    export VT_ROOT=/path/to/RGBT-Tiny OFFICIAL_LISTS=/path/to/official_deeppro/datasets/RGB-T
    python code/check_mask_protocol.py
    python code/export_masks.py --k 6 --storage round

docs/MASKS.md explains each step. SHA256SUMS lists the checksum of every file here.
