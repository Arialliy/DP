"""CPU witnesses for the original test-window checkpoint selection protocol.

Uses the original TestIRSeqDataLoader for the window and PNG reference. The scorer
is replaced with deterministic logits so that threshold, overlap and RNG behavior
can be checked without a GPU or a full neural-network evaluation.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def same_state(before, after):
    return (before[0] == after[0]
            and before[1][0] == after[1][0]
            and np.array_equal(before[1][1], after[1][1])
            and before[1][2:] == after[1][2:]
            and torch.equal(before[2], after[2]))


def arguments(**overrides):
    values = dict(arch='lv_full', hist_len=80, hist_step=4, hist_mode='both',
                  motion_gate=0.0, coords='raw', stat_margin=0.1, abs_tol=0.5,
                  min_cover=0.1, lowres=4, selection_max_windows=None)
    values.update(overrides)
    return SimpleNamespace(**values)


@contextmanager
def dataset_root(selection, root):
    # Scene caches COCO annotations by split; isolate synthetic and real fixtures.
    with patch.object(selection.vtdata, 'DATA_ROOT', str(root)), \
            patch.object(selection.vtdata, 'OFFICIAL_LISTS', str(root)), \
            patch.object(selection.vtdata, 'ANN_DIR', str(root / 'annotations_coco')), \
            patch.object(selection.vtdata, '_ANN_CACHE', {}):
        yield


def original_windows(loader):
    return [[tuple(row[2] for row in window) for window in loader[i].samplelist]
            for i in range(len(loader))]


def make_synthetic(root):
    clips = [('fixture_A', 90, 0), ('fixture_B', 41, 100)]
    (root / 'test.txt').write_text('\n'.join(c for c, _, _ in clips) + '\n')
    (root / 'train.txt').write_text('')
    images, reference = [], {}
    for clip, n, offset in clips:
        frame_dir = root / 'test2017' / clip / '01'
        mask_dir = root / 'segmentations' / clip
        frame_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        labels = []
        for t in range(n):
            name = f'{t + offset:05d}'
            im = np.full((8, 8), 60 + t % 100, np.uint8)
            mask = np.zeros((8, 8), np.uint8)
            # Different values deliberately verify label > 0, not label == 255.
            mask[t % 8, (t // 8) % 8] = (1, 128, 255)[t % 3]
            if t % 5 == 0:
                mask[(t + 3) % 8, (t // 8 + 2) % 8] = 1
            Image.fromarray(im).save(frame_dir / (name + '.jpg'))
            Image.fromarray(mask).save(mask_dir / (name + '.png'))
            images.append(dict(id=len(images) + 1, file_name=f'{clip}/01/{name}.jpg',
                               width=8, height=8))
            labels.append(mask > 0)
        reference[clip] = np.stack(labels)
    ann = root / 'annotations_coco'
    ann.mkdir()
    # No COCO boxes: recomputing labels instead of reading supplied PNGs must fail.
    (ann / 'instances_01_test2017.json').write_text(json.dumps(dict(images=images, annotations=[])))
    return clips, reference


def synthetic_witness(selection, loader_class):
    with tempfile.TemporaryDirectory(prefix='deeppro-selection-test-') as temp:
        root = Path(temp)
        clips, labels = make_synthetic(root)
        original = loader_class('RGB-T', str(root), 40, 4)
        plans = original_windows(original)
        expected_calls = [(clip, frames) for (clip, _, _), windows in zip(clips, plans)
                          for frames in windows]
        reference = []
        for index, (clip, frames) in enumerate(expected_calls):
            target = labels[clip][list(frames)]
            # Include positive logits that sigmoid rounds to exactly 0.5 in FP32.
            levels = np.array([-1.0, 0.0, 1e-8, 1e-6, 1.0], np.float32)
            logits = levels[(np.arange(target.size).reshape(target.shape) + index) % 5]
            predicted = torch.sigmoid(torch.from_numpy(logits)).numpy() > 0.5
            reference.append(dict(logits=logits,
                                  intersection=int(np.count_nonzero(predicted & target)),
                                  union=int(np.count_nonzero(predicted | target))))
        expected_i = sum(row['intersection'] for row in reference)
        expected_u = sum(row['union'] for row in reference)
        calls = []
        model = torch.nn.Linear(1, 1)
        model.train()

        def factory(_model, _needs, _device, _video, _chain, _hist_cfg, _no_history):
            def score(frames, coords):
                index = len(calls)
                require(not model.training, 'Selection did not switch model to eval mode')
                require(coords == 'raw', 'LV selection unexpectedly changed coordinates')
                require(tuple(frames) == expected_calls[index][1], 'Window differs from original loader')
                random.random()
                np.random.random()
                torch.rand(1)
                calls.append(tuple(frames))
                return reference[index]['logits'].copy()
            return score

        before = state()
        with dataset_root(selection, root), patch.object(selection, 'make_scorer', factory):
            result = selection.evaluate_test_window_iou(model, ('valid',), arguments())
        require(same_state(before, state()), 'Selection changed training RNG state')
        require(model.training, 'Selection did not restore training mode')
        require(len(calls) == len(expected_calls) == 5, 'Unexpected synthetic window count')
        require(result['windows'] == 5 and result['window_frames'] == 200,
                'Overlapping frames were deduplicated')
        require(result['intersection'] == expected_i and result['union'] == expected_u,
                'Global intersection/union differ from original PNG/threshold reference')
        require(abs(result['iou'] - expected_i / expected_u) < 1e-12,
                'IoU must be total intersection / total union')
        require(result['complete'], 'Full synthetic evaluation marked incomplete')

        def failing_factory(*_args, **_kwargs):
            def score(_frames, _coords):
                random.random()
                np.random.random()
                torch.rand(1)
                raise RuntimeError('intentional scorer failure')
            return score

        model.eval()
        before_failure = state()
        with dataset_root(selection, root), patch.object(selection, 'make_scorer', failing_factory):
            try:
                selection.evaluate_test_window_iou(model, ('valid',), arguments())
            except RuntimeError as exc:
                require(str(exc) == 'intentional scorer failure', repr(exc))
            else:
                raise AssertionError('The failing scorer exception was swallowed')
        require(same_state(before_failure, state()), 'Failure path changed training RNG state')
        require(not model.training, 'Failure path did not restore eval mode')
        return dict(status='passed', windows=5, window_frames=200, unique_frames=131,
                    intersection=expected_i, union=expected_u, iou=expected_i / expected_u,
                    threshold_edge='sigmoid(float32(+1e-8)) == 0.5 excluded',
                    labels='PNG nonzero values including 1 and 128; empty COCO annotations',
                    rng_and_mode_restored_on_success_and_failure=True)


def real_witness(selection, loader_class, root):
    original = loader_class('RGB-T', str(root), 40, 4)
    plans = original_windows(original)
    names = original.seq_names
    require(len(names) == 15, 'Expected all 15 official RGB-T test clips')
    require(sum(map(len, plans)) == 172, 'Unexpected official test-window count')
    for index, windows in enumerate(plans):
        frame_count = windows[-1][-1] + 1
        implemented = [tuple(range(a, b)) for a, b in selection.test_windows(frame_count)]
        require(implemented == windows, f'{names[index]}: selection window plan differs from original')
        expected_n = len(original[index].samplelist[-1])
        require(expected_n == 40, 'Official data unexpectedly has a short final window')
        require(all(len(w) == 40 for w in windows), 'Official window is not 40 frames')
    images, labels, _centroids, _ends = original[0][0]
    target = labels.numpy() > 0
    logits = np.full(target.shape, -1.0, np.float32)
    logits[target] = 1.0
    # Add one definite false positive and one definite false negative.
    background = np.argwhere(~target)[0]
    foreground = np.argwhere(target)[0]
    logits[tuple(background)] = 1.0
    logits[tuple(foreground)] = 0.0
    expected_i = int(target.sum()) - 1
    expected_u = int(target.sum()) + 1
    model = torch.nn.Linear(1, 1)
    score_calls = []

    def factory(_model, _needs, _device, video, _chain, _hist_cfg, _no_history):
        normalized = (np.asarray(video[:40], np.float32) - 85.0799) / 47.4845
        require(np.array_equal(normalized, images.numpy()[0]),
                'Real frame order or image pixels differ from original loader')

        def score(frames, coords):
            require(tuple(frames) == plans[0][0], 'First real window differs from original loader')
            require(coords == 'raw', 'LV should keep original coordinates')
            score_calls.append(tuple(frames))
            return logits.copy()
        return score

    with dataset_root(selection, root), patch.object(selection, 'make_scorer', factory):
        result = selection.evaluate_test_window_iou(
            model, ('valid',), arguments(selection_max_windows=1))
    require(len(score_calls) == 1, 'selection_max_windows did not stop after one window')
    require(result['expected_windows'] == 172 and result['windows'] == 1,
            'Preflight did not retain full evaluation scope')
    require(not result['complete'], 'Partial evaluation must not select a checkpoint')
    require(result['intersection'] == expected_i and result['union'] == expected_u,
            'Selected labels differ from real original TestIRSeqDataLoader PNG labels')
    return dict(status='passed', clips=15, windows=172, window_frame_instances=6880,
                actual_reference_loader='data_utils.TestDataLoader.TestIRSeqDataLoader',
                first_clip=names[0], first_window_png_positive_pixels=int(target.sum()),
                witnessed_windows=1, intersection=expected_i, union=expected_u,
                partial_evaluation_marked_incomplete=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-source', type=Path,
                        default=Path(os.environ.get('OFFICIAL_SOURCE', '/official-source')))
    parser.add_argument('--data', type=Path,
                        default=Path(os.environ.get('VT_ROOT', '/data/RGB-T')))
    args = parser.parse_args()
    sys.path.insert(0, str(args.official_source))
    from data_utils.TestDataLoader import TestIRSeqDataLoader
    import checkpoint_selection as selection
    report = dict(status='passed', synthetic=synthetic_witness(selection, TestIRSeqDataLoader),
                  real=real_witness(selection, TestIRSeqDataLoader, args.data))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
