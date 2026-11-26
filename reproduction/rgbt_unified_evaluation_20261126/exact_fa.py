"""Exact empirical Pd at fixed Fa budgets, using bounded-memory FP32 streams.

This is an additional diagnostic, independent of the original 27-point ROC.
Only original ShootingRules geometry is reused. No interpolation, added ROC
endpoints, score calibration, or threshold above one is allowed.

Usage::

    collector = ExactFaAccumulator()
    for probability, mask in unique_source_frames:
        collector.update(probability, mask)
    result = collector.finalize()  # JSON serializable, including empty-GT cases

If geometry is already available, call update_scores(fp_scores, target_maxima,
pixels=H*W, frames=1). Target maxima use -inf for an empty original detection
neighborhood. All target maxima are retained; only the highest K+1 background
scores plus the exact multiplicity of their boundary score are retained.
"""

import math

import numpy as np
from scipy import ndimage

from official_metrics import _protected_pixels, _target_maxima


DEFAULT_PIXELS_TOTAL = 1958215680
DEFAULT_FA_BUDGETS = (
    4.467025818116215e-5,
    4.5281018278844545e-4,
    6.136428240631799e-4,
)


def extract_frame_scores(probability, mask):
    """Extract FP pixels and component maxima with the exact official slices."""
    probability, mask = np.asarray(probability), np.asarray(mask)
    if probability.dtype != np.float32:
        raise TypeError('probability must be float32')
    if probability.ndim != 2 or probability.size == 0 or mask.shape != probability.shape:
        raise ValueError('probability and mask must have the same nonempty HxW shape')
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError('probability must be finite and within [0, 1]')
    foreground = mask > 0
    labels, count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    rows, columns = np.nonzero(foreground)
    protected = _protected_pixels(probability.shape, rows, columns)
    maxima = _target_maxima(probability, labels, count, rows, columns)
    return dict(fp_scores=probability[~protected], target_maxima=maxima,
                pixels=int(probability.size), frames=1)


def _nonnegative_integer(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < int(positive):
        raise ValueError(name + (' must be a positive integer' if positive else ' must be a nonnegative integer'))
    return int(value)


class ExactFaAccumulator:
    """Stream unique-frame scores and compute exact attainable budget points.

    A budget's integer cap is floor(float64(Fa) * pixels_total), allowing one
    upward product ULP to recover an integer represented just below itself.
    The supplied
    RGB-T defaults give caps 87,474, 886,700 and 1,201,645 exactly. If a budget
    admits several thresholds with the same best Pd, the smallest feasible
    float32 threshold is returned. Both neighboring Fa levels are actual
    measured points, with ties counted in full.

    Memory is O(max_fp_budget + buffer_size + number_of_targets), not pixels.
    The source probabilities must already be overlap-fused FP32 scores in
    [0,1]. The caller owns score generation and must not count frames twice.
    """

    def __init__(self, *, pixels_total=DEFAULT_PIXELS_TOTAL,
                 fa_budgets=DEFAULT_FA_BUDGETS, expected_targets=11300,
                 buffer_size=1048576):
        self.pixels_total = _nonnegative_integer(pixels_total, 'pixels_total', positive=True)
        self.fa_budgets = tuple(float(value) for value in fa_budgets)
        if not self.fa_budgets or any(not math.isfinite(value) or not 0 <= value <= 1
                                      for value in self.fa_budgets):
            raise ValueError('fa_budgets must be a nonempty collection within [0, 1]')
        self.fp_budgets = tuple(math.floor(math.nextafter(value * self.pixels_total, math.inf))
                                for value in self.fa_budgets)
        self.expected_targets = (None if expected_targets is None else
                                 _nonnegative_integer(expected_targets, 'expected_targets'))
        self.buffer_size = _nonnegative_integer(buffer_size, 'buffer_size', positive=True)
        self.keep = max(self.fp_budgets) + 1
        self.pixels = self.frames = self.background_pixels = self.targets = 0
        self._top = np.empty(0, dtype=np.float32)
        self._pending = []
        self._pending_size = 0
        self._cutoff = None
        self._boundary_extra = 0
        self._target_chunks = []
        self._finalized = False
        self._result = None
        self._peak_buffered_scores = 0

    @property
    def target_maxima(self):
        """Copy of all component maxima, in stream order; may contain -inf."""
        return (np.concatenate(self._target_chunks) if self._target_chunks else
                np.empty(0, dtype=np.float32))

    def update(self, probability, mask):
        """Consume one final FP32 probability image and its original PNG mask."""
        self.update_scores(**extract_frame_scores(probability, mask))

    def update_scores(self, fp_scores, target_maxima, *, pixels, frames=1):
        """Consume precomputed exact geometry without retaining the image.

        ``pixels`` counts all image pixels, including excluded target regions;
        ``fp_scores`` contains only pixels outside the original +/-4 slices.
        ``target_maxima`` has one value per original 8-connected component.
        Positive/negative finite values outside [0,1] are rejected; -inf is
        allowed only for target neighborhoods that were empty due to slicing.
        """
        if self._finalized:
            raise RuntimeError('cannot append scores after finalize')
        fp_scores, maxima = np.asarray(fp_scores), np.asarray(target_maxima)
        if fp_scores.dtype != np.float32 or maxima.dtype != np.float32:
            raise TypeError('fp_scores and target_maxima must remain float32')
        if fp_scores.ndim != 1 or maxima.ndim != 1:
            raise ValueError('fp_scores and target_maxima must be one-dimensional')
        pixels = _nonnegative_integer(pixels, 'pixels', positive=True)
        frames = _nonnegative_integer(frames, 'frames', positive=True)
        if len(fp_scores) > pixels:
            raise ValueError('background score count cannot exceed all image pixels')
        if self.pixels + pixels > self.pixels_total:
            raise ValueError('more pixels received than the declared complete evaluation')
        if not np.isfinite(fp_scores).all() or (fp_scores < 0).any() or (fp_scores > 1).any():
            raise ValueError('background scores must be finite and within [0,1]')
        valid_maxima = np.isneginf(maxima) | (np.isfinite(maxima) & (maxima >= 0) & (maxima <= 1))
        if not valid_maxima.all():
            raise ValueError('target maxima must be within [0,1] or -inf for empty neighborhoods')
        if self.expected_targets is not None and self.targets + len(maxima) > self.expected_targets:
            raise ValueError('more target components received than expected')
        self.pixels += pixels
        self.frames += frames
        self.background_pixels += len(fp_scores)
        self.targets += len(maxima)
        if len(maxima):
            self._target_chunks.append(maxima.copy())

        for start in range(0, len(fp_scores), self.buffer_size):
            values = fp_scores[start:start + self.buffer_size]
            if len(self._top) == self.keep:
                # These values cannot displace the retained top K+1. Count
                # ties separately so a huge boundary tie is never truncated.
                self._boundary_extra += int(np.count_nonzero(values == self._cutoff))
                values = values[values > self._cutoff]
            offset = 0
            while offset < len(values):
                take = min(self.buffer_size - self._pending_size, len(values) - offset)
                self._pending.append(values[offset:offset + take].copy())
                self._pending_size += take
                self._peak_buffered_scores = max(self._peak_buffered_scores, self._pending_size)
                offset += take
                if self._pending_size == self.buffer_size:
                    self._flush()

    def _flush(self):
        if not self._pending_size:
            return
        work = np.concatenate([self._top, *self._pending])
        old_cutoff, old_extra = self._cutoff, self._boundary_extra
        self._pending = []
        self._pending_size = 0
        if len(work) > self.keep:
            split = len(work) - self.keep
            work.partition(split)
            cutoff = work[split]
            discarded_ties = int(np.count_nonzero(work[:split] == cutoff))
            self._boundary_extra = discarded_ties + (old_extra if old_cutoff == cutoff else 0)
            self._top = work[split:].copy()
            self._cutoff = np.float32(cutoff)
        else:
            self._top = work
            self._cutoff = np.float32(work.min()) if len(work) else None
            self._boundary_extra = 0

    def _false_count(self, threshold):
        if threshold == 0:
            # The original sequential assignments turn all scores into one.
            return self.background_pixels
        if self.background_pixels > len(self._top) and threshold < self._cutoff:
            raise ValueError('threshold below retained score range cannot be counted exactly')
        count = len(self._top) - int(np.searchsorted(self._top, threshold, side='left'))
        if self._cutoff is not None and threshold <= self._cutoff:
            count += self._boundary_extra
        return count

    def _point(self, threshold, sorted_targets):
        threshold = np.float32(threshold)
        if not 0 <= threshold <= 1:
            raise ValueError('only actual operating thresholds within [0,1] are allowed')
        false_pixels = self._false_count(threshold)
        true_targets = len(sorted_targets) - int(np.searchsorted(sorted_targets, threshold, side='left'))
        return dict(threshold=float(threshold), false_pixels=int(false_pixels),
                    fa=float(false_pixels / self.pixels_total), true_targets=int(true_targets),
                    targets=int(self.targets), pd=(float(true_targets / self.targets) if self.targets else None))

    def finalize(self):
        """Return exact, JSON-safe points; reject an incomplete frame stream.

        ``lower_fa_point`` is the highest-Pd feasible point. ``upper_fa_point``
        is the adjacent higher-Fa level reached by including the boundary tie.
        If even threshold=1 exceeds the budget, feasible is False and the
        threshold=1 point is returned as the smallest attainable Fa. No
        threshold >1 or fictional zero-detection endpoint is manufactured.
        """
        if self._result is not None:
            return self._result
        if self.pixels != self.pixels_total:
            raise ValueError(f'incomplete pixels: received {self.pixels}, expected {self.pixels_total}')
        if self.expected_targets is not None and self.targets != self.expected_targets:
            raise ValueError(f'incomplete targets: received {self.targets}, expected {self.expected_targets}')
        self._flush()
        self._top.sort()
        sorted_targets = np.sort(self.target_maxima)
        minimum = self._point(np.float32(1), sorted_targets)
        rows = []
        for fa_budget, cap in zip(self.fa_budgets, self.fp_budgets):
            row = dict(fa_budget=fa_budget, fp_budget=cap, feasible=False,
                       best_point=None, lower_fa_point=None, upper_fa_point=None,
                       exact_fp_budget_reached=False, reason=None)
            if minimum['false_pixels'] > cap:
                row.update(upper_fa_point=dict(minimum),
                           reason='threshold_one_still_exceeds_budget_due_to_float32_scores_equal_one')
            elif self.background_pixels <= cap:
                point = self._point(np.float32(0), sorted_targets)
                row.update(feasible=True, best_point=point, lower_fa_point=dict(point),
                           exact_fp_budget_reached=point['false_pixels'] == cap,
                           reason='all_background_pixels_fit_budget; no_higher_fa_point_exists')
            else:
                # K+1 retained scores suffice to obtain the (cap+1)-th largest
                # value. Full multiplicity at the retained minimum is kept
                # separately for the upper point, even if millions tie there.
                boundary = self._top[-(cap + 1)]
                threshold = np.nextafter(boundary, np.float32(np.inf))
                point = self._point(threshold, sorted_targets)
                upper = self._point(boundary, sorted_targets)
                if not point['false_pixels'] <= cap < upper['false_pixels']:
                    raise AssertionError('exact empirical points do not bracket the FP budget')
                row.update(feasible=True, best_point=point, lower_fa_point=dict(point),
                           upper_fa_point=upper, boundary_score=float(boundary),
                           exact_fp_budget_reached=point['false_pixels'] == cap)
            row['achieved_fa_minus_budget'] = (row['best_point']['fa'] - fa_budget
                                                if row['best_point'] is not None else None)
            row['upper_fa_minus_budget'] = (row['upper_fa_point']['fa'] - fa_budget
                                             if row['upper_fa_point'] is not None else None)
            rows.append(row)
        self._finalized = True
        self._result = dict(
            method='exact_empirical_scores_no_interpolation',
            comparison='FP32 score >= FP32 threshold; original threshold-zero behavior retained',
            threshold_domain=[0.0, 1.0],
            tie_break='smallest_feasible_float32_threshold_maximizes_detection_probability',
            fp_budget_rounding='floor(nextafter(float64(fa_budget) * pixels_total, +inf)); one_product_ULP_tolerance',
            complete=True, pixels=self.pixels, frames=self.frames, targets=self.targets,
            pd_defined=self.targets > 0, background_pixels=self.background_pixels,
            minimum_attainable_point=minimum, budgets=rows,
            memory=dict(retained_background_scores=len(self._top), keep_limit=self.keep,
                        boundary_score=float(self._cutoff) if self._cutoff is not None else None,
                        discarded_boundary_ties=self._boundary_extra,
                        retained_target_maxima=self.targets,
                        peak_buffered_scores=self._peak_buffered_scores,
                        buffer_size=self.buffer_size),
        )
        return self._result


ExactFaCollector = ExactFaAccumulator
