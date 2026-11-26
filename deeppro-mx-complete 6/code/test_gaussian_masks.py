"""Checks of the paper-strict Gaussian masks (numpy, PIL; no dataset needed).

Run: python code/test_gaussian_masks.py
"""
import io
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import gaussian_masks as gm  # noqa: E402
import vtdata  # noqa: E402

H, W = 120, 160


def dense(boxes, k):
    """The formula evaluated on every pixel, without any cut."""
    v, u = np.mgrid[0:H, 0:W] + 0.5
    m = np.zeros((H, W))
    for x, y, w, h, *_ in boxes:
        sx, sy = w / k, h / k
        m = np.maximum(m, np.exp(-(u - (x + w / 2)) ** 2 / (2 * sx * sx) - (v - (y + h / 2)) ** 2 / (2 * sy * sy)))
    return m


def random_boxes(rng, n, lo=3, hi=16):
    out = []
    for _ in range(n):
        w, h = rng.uniform(lo, hi), rng.uniform(lo, hi)
        out.append((rng.uniform(-4, W - w + 4), rng.uniform(-4, H - h + 4), w, h, 1, 0))
    return out


def test_heatmap_is_the_formula():
    rng = np.random.default_rng(0)
    for k in (2, 4, 6, 8):
        b = random_boxes(rng, 6)
        assert np.abs(gm.heatmap(b, H, W, k) - dense(b, k)).max() < 1e-7, k


def test_peak_and_centre_convention():
    g = gm.heatmap([(10, 20, 6, 4, 0, 0)], H, W, 6)        # centre (13, 22): a pixel corner
    assert np.isclose(g[21, 12], g[21, 13]) and np.isclose(g[21, 12], g[22, 12])
    assert np.unravel_index(np.argmax(g), g.shape) in {(21, 12), (21, 13), (22, 12), (22, 13)}
    g = gm.heatmap([(10, 20, 5, 5, 0, 0)], H, W, 6)        # centre (12.5, 22.5): the middle of pixel (12, 22)
    assert np.isclose(g[22, 12], 1.0) and np.unravel_index(np.argmax(g), g.shape) == (22, 12)


def test_support_matches_the_8_bit_levels():
    box = [(40, 20, 60, 60, 0, 0)]                          # sigma = 10 px at k = 6
    for storage in ('round', 'floor'):
        row = gm.label(box, H, W, 6, storage)[50]
        half = row.sum() / 2
        assert abs(half - 10 * gm.semi_axis(storage)) <= 1.0, (storage, half)
    assert math.isclose(gm.semi_axis('round'), math.sqrt(2 * math.log(510)))


def test_png_round_trip_through_the_authors_loader():
    rng = np.random.default_rng(1)
    b = random_boxes(rng, 5)
    img = gm.store(gm.heatmap(b, H, W, 6), 'round')
    buf = io.BytesIO()
    Image.fromarray(img, mode='L').save(buf, format='PNG')
    back = np.array(Image.open(io.BytesIO(buf.getvalue())), dtype=np.uint8) / 255.0   # TrainDataLoader.py
    back[back > 0] = 1.0
    assert np.array_equal(back, gm.label(b, H, W, 6, 'round'))
    assert img.max() == 255 or img.max() > 200                  # a Gaussian, not a binary image
    assert len(np.unique(img)) > 20


def test_legacy_rule_equals_k6_round_for_boxes_of_3px_or_more():
    rng = np.random.default_rng(2)
    for _ in range(200):
        b = random_boxes(rng, int(rng.integers(1, 6)))
        assert np.array_equal(vtdata.gaussian_mask(np.array(b), H, W, 'sigma_sixth', quant=True),
                              gm.label(b, H, W, 6, 'round'))


def test_tiny_boxes_follow_the_formula_not_a_floor():
    tiny = [(30.5, 40.5, 1.0, 1.0, 0, 0)]                       # centre at a pixel corner, sigma 1/6 px
    assert gm.label(tiny, H, W, 6).sum() == 0 and gm.vanished(tiny, H, W, 6) == 1
    on_pixel = [(30.0, 40.0, 1.0, 1.0, 0, 0)]                   # integer 1 x 1 box: centre on a pixel centre
    assert gm.label(on_pixel, H, W, 6).sum() == 1
    assert vtdata.gaussian_mask(np.array(tiny), H, W, 'sigma_sixth').sum() > 0   # legacy floor keeps it
    assert gm.label([(30.0, 40.0, 2.0, 2.0, 0, 0)], H, W, 6).sum() > 0
    assert gm.heatmap([(30, 40, 0, 5, 0, 0), (30, 40, 4, -1, 0, 0)], H, W, 6).max() == 0


def test_counting_merges_neighbours_as_sigma_grows():
    pair = [(50, 50, 4, 4, 0, 0), (58, 50, 4, 4, 0, 1)]         # centres 8 px apart
    assert gm.count_targets(gm.label(pair, H, W, 6)) == 2
    assert gm.count_targets(gm.label(pair, H, W, 2)) == 1
    diag = np.zeros((5, 5))
    diag[1, 1] = diag[2, 2] = 1
    assert gm.count_targets(diag) == 1                          # 8-connectivity, as ShootingRules.py


def test_author_counts_and_log_parsing():
    assert len(gm.AUTHOR_TEST_COUNTS) == 15 and sum(gm.AUTHOR_TEST_COUNTS.values()) == gm.AUTHOR_TEST_TOTAL
    log = ("DJI_0049_2results:\nTh_Seg = 0.000000e+00:\tPD:[602/602, 1.00000]\tFA:[1, 1e-01]\n"
           "Th_Seg = 5.000000e-01:\tPD:[226/602, 0.37542]\tFA:[22, 7.4e-07]\n"
           "DJI_0109_1results:\nTh_Seg = 5.000000e-01:\tPD:[90/90, 1.00000]\tFA:[3, 1e-07]\n"
           "\nFinal results:\nTh_Seg = 5.000000e-01:\tPD:[316/692, 0.4]\tFA:[25, 1e-06]\n")
    assert gm.parse_eval_log(log) == {'DJI_0049_2': 602, 'DJI_0109_1': 90}


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
