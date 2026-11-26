"""Exact RGB-T metrics from DeepPro-Plus ``test.py``/``ShootingRules.py``.

The caller supplies one final, overlap-fused float32 probability image and its
original PNG mask. This module does not fuse windows, resize, sigmoid, or choose
a checkpoint. RGB-T uses the same binary PNG for IoU and ShootingRules targets.

The 27 operating points, 8-connected targets, +/-1 detection neighborhoods,
+/-4 false-pixel exclusions, and even Python's unusual negative slice behavior
are retained. AUC integrates only these measured points, using sklearn's auc.
"""

from collections.abc import Iterable, Mapping

import numpy as np
from scipy import ndimage
from sklearn.metrics import auc as sklearn_auc


THRESHOLDS = np.array([
    0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
    1e-2, 1e-1, .2, .3, .35, .4, .45, .5, .55, .6, .65,
    .7, .8, .85, .9, .95, .99, 1,
], dtype=np.float64)
THRESHOLDS.setflags(write=False)


def _float32_boundaries():
    """Smallest float32 passing the original array >= scalar comparison.

    Broadcasting a float64 threshold array can promote float32 probabilities
    differently from the original individual NumPy scalar comparisons. Probe
    those comparisons directly, so histogramming retains the active NumPy
    version's behavior without casting the probabilities to float64.
    """
    boundaries = np.empty(len(THRESHOLDS), dtype=np.float32)
    for index, threshold in enumerate(THRESHOLDS):
        candidate = np.float32(threshold)
        if not (np.array([candidate], dtype=np.float32) >= threshold)[0]:
            candidate = np.nextafter(candidate, np.float32(np.inf))
        boundaries[index] = candidate
    boundaries.setflags(write=False)
    return boundaries


_BOUNDARIES = _float32_boundaries()
_CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)
_ARRAY_FIELDS = ('false_pixels', 'true_targets', 'targets')
_SCALAR_FIELDS = ('intersection', 'union', 'pixels', 'frames')


def _counts_at_thresholds(values):
    """Count >= each threshold in one histogram; values remain float32."""
    passed = np.searchsorted(_BOUNDARIES, values, side='right')
    histogram = np.bincount(passed, minlength=len(THRESHOLDS) + 1)
    return np.cumsum(histogram[:0:-1], dtype=np.int64)[::-1].copy()


def _slice_bounds(coordinates, radius, size):
    """Vectorized slice(i-radius, i+radius+1).indices(size), positive step."""
    start = coordinates - radius
    start = np.where(start < 0, start + size, start)
    return np.clip(start, 0, size), np.minimum(coordinates + radius + 1, size)


def _protected_pixels(shape, rows, columns):
    """Union of the exact original Box2_map slices, including empty slices."""
    height, width = shape
    if rows.size == 0:
        return np.zeros(shape, dtype=bool)
    r0, r1 = _slice_bounds(rows, 4, height)
    c0, c1 = _slice_bounds(columns, 4, width)
    valid = (r0 < r1) & (c0 < c1)
    r0, r1, c0, c1 = r0[valid], r1[valid], c0[valid], c1[valid]
    difference = np.zeros((height + 1, width + 1), dtype=np.int32)
    np.add.at(difference, (r0, c0), 1)
    np.add.at(difference, (r1, c0), -1)
    np.add.at(difference, (r0, c1), -1)
    np.add.at(difference, (r1, c1), 1)
    np.cumsum(difference, axis=0, dtype=np.int32, out=difference)
    np.cumsum(difference, axis=1, dtype=np.int32, out=difference)
    return difference[:-1, :-1] > 0


def _target_maxima(probability, labels, count, rows, columns):
    """Max probability in each component's union of exact +/-1 slices."""
    maxima = np.full(count, -np.inf, dtype=np.float32)
    if count == 0:
        return maxima
    nearby = ndimage.maximum_filter(probability, size=3, mode='constant', cval=-np.inf)
    values = nearby[rows, columns]
    # maximum_filter clips borders; the author's negative Python slice does
    # not. Only row/column zero can have a negative start for radius one.
    height, width = probability.shape
    for index in np.flatnonzero((rows == 0) | (columns == 0)):
        row, column = int(rows[index]), int(columns[index])
        r0, r1, _ = slice(row - 1, row + 2).indices(height)
        c0, c1, _ = slice(column - 1, column + 2).indices(width)
        values[index] = (probability[r0:r1, c0:c1].max()
                         if r0 < r1 and c0 < c1 else -np.inf)
    np.maximum.at(maxima, labels[rows, columns] - 1, values)
    return maxima


def score_frame(prob: np.ndarray, mask: np.ndarray) -> dict:
    """Return integer counts for one unique RGB-T frame, without averaging.

    ``prob`` must be finite float32 HxW probabilities in [0, 1]. ``mask`` is
    HxW; every positive value is foreground, as in the original PNG loader.
    Returned false_pixels/true_targets/targets are int64 arrays of length 27;
    intersection, union, pixels, and frames are Python integers. IoU uses
    strict ``prob > .5``. Pd/Fa use ``>=`` and preserve threshold-zero behavior:
    the original two assignment statements turn every probability into one.
    Inputs are never modified.
    """
    probability, mask = np.asarray(prob), np.asarray(mask)
    if probability.ndim != 2 or mask.shape != probability.shape or probability.size == 0:
        raise ValueError('prob and mask must have the same nonempty HxW shape')
    if probability.dtype != np.float32:
        raise TypeError('prob must be float32; perform sigmoid/conversion explicitly before scoring')
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError('prob must contain finite probabilities in [0, 1]')
    foreground = mask > 0
    labels, count = ndimage.label(foreground, structure=_CONNECTIVITY)
    rows, columns = np.nonzero(foreground)
    protected = _protected_pixels(probability.shape, rows, columns)
    maxima = _target_maxima(probability, labels, count, rows, columns)
    prediction = probability > .5
    return {
        'false_pixels': _counts_at_thresholds(probability[~protected]),
        'true_targets': _counts_at_thresholds(maxima),
        'targets': np.full(len(THRESHOLDS), count, dtype=np.int64),
        'intersection': int(np.count_nonzero(prediction & foreground)),
        'union': int(np.count_nonzero(prediction | foreground)),
        'pixels': int(probability.size),
        'frames': 1,
    }


def aggregate_results(results: Iterable[Mapping]) -> dict:
    """Sum frame or aggregate counts, then compute global Pd, Fa, AUC, IoU.

    Input may be a generator. Returning raw counts allows exact aggregation
    across clips without weighting clips equally. Undefined empty-target Pd
    or empty-union IoU remains NaN, rather than manufacturing a score. Empty
    input is rejected. No synthetic ROC endpoints or interpolation are added.
    """
    totals = {key: np.zeros(len(THRESHOLDS), dtype=np.int64) for key in _ARRAY_FIELDS}
    totals.update({key: 0 for key in _SCALAR_FIELDS})
    seen = False
    for result in results:
        seen = True
        for key in _ARRAY_FIELDS:
            values = np.asarray(result[key])
            if values.shape != THRESHOLDS.shape or values.dtype.kind not in 'iu' or (values < 0).any():
                raise ValueError(key + ' must be a nonnegative integer array of length 27')
            totals[key] += values.astype(np.int64, copy=False)
        for key in _SCALAR_FIELDS:
            value = result.get(key, 1) if key == 'frames' else result[key]
            if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 0:
                raise ValueError(key + ' must be a nonnegative integer')
            totals[key] += int(value)
    if not seen:
        raise ValueError('cannot aggregate an empty collection')
    if totals['pixels'] <= 0:
        raise ValueError('total pixels must be positive')
    pd = np.full(len(THRESHOLDS), np.nan, dtype=np.float64)
    np.divide(totals['true_targets'], totals['targets'], out=pd, where=totals['targets'] != 0)
    fa = totals['false_pixels'] / totals['pixels']
    iou = totals['intersection'] / totals['union'] if totals['union'] else float('nan')
    totals.update({
        'thresholds': THRESHOLDS.copy(),
        'pd': pd,
        'fa': fa,
        'auc': float(sklearn_auc(fa, pd)),
        'iou': float(iou),
        'iou_at_0_5': float(iou),
    })
    return totals


def aggregate_clip(results: Iterable[Mapping]) -> dict:
    """Alias for aggregating already overlap-fused unique frame counts."""
    return aggregate_results(results)
