"""Test-window IoU for selecting research-model checkpoints like official train.py.

Only checkpoint selection belongs here. Training data, losses, optimizers, and the
models' final inference methods are not changed. Each test window is counted
separately, including its overlapping frames; there is no temporal max fusion.

LVH keeps its native History, compensation, and motion gate. MRTP uses its
rate-one, auto-coordinate network predictions as a selection probe. Its training
rates and final multi-rate/CFAR inference are unaffected. CFAR scores are not
interpreted as probabilities for the template's 0.5 threshold.
"""

from contextlib import contextmanager
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch

from infer_mr import make_scorer
import multirate
import stabilize
import vtdata


SEQUENCE_LENGTH = 40
OVERLAP = 4
THRESHOLD = 0.5
SUPPORTED = {
    'lv_full': ('valid',),
    'lvh_full': ('valid', 'pvalid', 'hist', 'hvalid'),
    'mr_lv': ('valid', 'pvalid'),
    'lvh_dec': ('valid', 'pvalid', 'hist', 'hvalid'),
}


def test_windows(n_frames):
    """Official TestIRSeqDataLoader windows as half-open [start, end) pairs.

    The final window is shifted to end at the last frame. Its overlap can
    therefore exceed four frames: a 90-frame clip yields (0,40), (36,76), (50,90).
    """
    if n_frames < SEQUENCE_LENGTH:
        raise ValueError('Selection needs at least 40 frames per test clip; the original '
                         'TPro cannot evaluate a shorter unpadded test window.')
    count = math.ceil((n_frames - OVERLAP) / (SEQUENCE_LENGTH - OVERLAP))
    windows = []
    for i in range(count):
        end = min(n_frames, (i + 1) * (SEQUENCE_LENGTH - OVERLAP) + OVERLAP)
        windows.append((max(0, end - SEQUENCE_LENGTH), end))
    return windows


def _setting(args, name, default):
    value = getattr(args, name, None)
    return default if value is None else value


@contextmanager
def _evaluation_state(model):
    """Restore RNGs and every module's training flag, also on evaluation failure.

    Evaluation never accesses a training Dataset, DataLoader, or its Generator.
    CUDA RNGs are read only if CUDA was already initialized by the caller.
    cuDNN, TF32, determinism, thread counts, and parameter storage are untouched.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    training_flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, flag in training_flags:
            module.training = flag
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _test_layout(root):
    """Read the literal official test list and sorted image/mask pairs, read-only."""
    clips = list(dict.fromkeys(row.split('/')[0]
                               for row in (root / 'test.txt').read_text().split()))
    if not clips:
        raise ValueError(f'{root / "test.txt"} contains no test clips')
    layout = []
    for clip in clips:
        image_dir = root / 'test2017' / clip / '01'
        mask_dir = root / 'segmentations' / clip
        # The original loaders sort os.listdir for both directories independently.
        images = sorted(image_dir.iterdir(), key=lambda p: p.name)
        masks = sorted(mask_dir.iterdir(), key=lambda p: p.name)
        if len(images) != len(masks):
            raise ValueError(f'{clip}: {len(images)} images but {len(masks)} masks')
        if any(not p.is_file() for p in images + masks):
            raise ValueError(f'{clip}: image/mask directory contains a non-file entry')
        if any(p.suffix.lower() != '.png' for p in masks):
            raise ValueError(f'{clip}: selection labels must be the existing PNG masks')
        if any(im.stem != mask.stem for im, mask in zip(images, masks)):
            raise ValueError(f'{clip}: sorted image and mask stems do not match')
        layout.append((clip, images, masks, test_windows(len(images))))
    return layout


def _load_video(paths):
    frames = []
    for path in paths:
        with Image.open(path) as im:
            frame = np.asarray(im, dtype=np.float32)
        if frame.ndim == 3:
            frame = frame[:, :, 0]
        if frame.ndim != 2:
            raise ValueError(f'{path}: expected a 2-D infrared image, got {frame.shape}')
        frames.append(frame)
    # Float32 raw grey matches the native infer.py / infer_mr.py video input.
    return np.stack(frames)


def _load_labels(paths, expected_shape):
    labels = []
    for path in paths:
        with Image.open(path) as im:
            label = np.asarray(im, dtype=np.uint8)
        if label.ndim == 3:
            label = label[:, :, 0]
        if label.shape != expected_shape:
            raise ValueError(f'{path}: mask shape {label.shape}, expected {expected_shape}')
        # Equivalent to the original uint8 / 255, followed by lab[lab > 0] = 1.
        labels.append(label > 0)
    return np.stack(labels)


def _counts(logits, target, device=None):
    """Count the exact strict probability threshold, without float16 storage.

    Do not replace sigmoid(logit) > .5 by logit > 0: a tiny positive FP32
    logit can round to an exact .5 sigmoid and must be negative in this rule.
    """
    logits = np.asarray(logits)
    if logits.dtype != np.float32:
        raise TypeError(f'Selection requires FP32 logits, got {logits.dtype}')
    if logits.shape != target.shape:
        raise ValueError(f'Prediction shape {logits.shape} differs from labels {target.shape}')
    if not np.isfinite(logits).all():
        raise ValueError('Non-finite logits in checkpoint selection')
    # Match the official trainer's sigmoid device as well as its FP32 dtype.
    tensor = torch.from_numpy(np.ascontiguousarray(logits))
    if device is not None:
        tensor = tensor.to(device)
    prob = torch.sigmoid(tensor).cpu().numpy()
    predicted = prob > THRESHOLD
    intersection = int(np.count_nonzero(predicted & target))
    union = int(np.count_nonzero(predicted | target))
    return intersection, union


def evaluate_test_window_iou(model, needs, args, *, progress=None):
    """Return selection metrics; the caller compares IoU using >= and saves weights.

    Parameters:
      model, needs: the unchanged model and needs tuple returned by models.build.
      args: the research trainer's argparse.Namespace, including arch and its
            native history/coordinate settings. Optional selection_max_windows
            limits a preflight only; complete=False forbids formal selection.
      progress: optional callback receiving one dict after each evaluated window.

    Test paths come from vtdata.DATA_ROOT, including its test.txt. No cache,
    checkpoint, log, source data, or annotation is written by this function.
    """
    arch = args.arch
    needs = tuple(needs)
    if arch not in SUPPORTED or set(needs) != set(SUPPORTED[arch]):
        raise ValueError(f'Unsupported selection interface: arch={arch!r}, needs={needs!r}')
    limit = getattr(args, 'selection_max_windows', None)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError('selection_max_windows must be None or a positive integer')
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device('cpu')
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise TypeError('Checkpoint selection expects the research model in FP32')
    root = Path(vtdata.DATA_ROOT)
    hist_cfg = {
        'hist_len': int(_setting(args, 'hist_len', 80)),
        'hist_step': int(_setting(args, 'hist_step', 4)),
        'hist_mode': _setting(args, 'hist_mode', 'both'),
    }
    motion_gate = float(_setting(args, 'motion_gate', 1.0 if arch == 'lvh_dec' else 0.0))
    lowres = int(_setting(args, 'lowres', 4))
    margin = float(_setting(args, 'stat_margin', 0.1))
    abs_tol = float(_setting(args, 'abs_tol', 0.5))
    min_cover = float(_setting(args, 'min_cover', 0.1))
    if hist_cfg['hist_step'] < 1 or lowres < 1:
        raise ValueError('hist_step and lowres must be positive')
    protocol = {
        'name': 'official_test_window_global_iou',
        'split': 'test',
        'test_list': str(root / 'test.txt'),
        'threshold': THRESHOLD,
        'comparison': 'sigmoid(float32_logits) > threshold',
        'sequence_length': SEQUENCE_LENGTH,
        'nominal_overlap': OVERLAP,
        'stride': SEQUENCE_LENGTH - OVERLAP,
        'overlapping_frames': 'counted once per window; no fusion',
        'labels': 'existing segmentations PNGs, uint8 > 0, original frame coordinates',
        'aggregation': 'sum(intersection) / sum(union), not mean window or clip IoU',
        'checkpoint_update': 'iou >= best_iou',
        'arch': arch,
        'selection_rates': [1],
        'coordinates': 'auto' if arch == 'mr_lv' else ('native_warp' if 'hist' in needs else 'raw'),
        'history': hist_cfg if 'hist' in needs else None,
        'motion_gate': motion_gate if 'hist' in needs else None,
        'mr_selection_probe': 'rate1 + auto; no CFAR; multi-rate training/final inference unchanged'
                              if arch == 'mr_lv' else None,
        'stationarity': dict(lowres=lowres, margin=margin, abs_tol=abs_tol, min_cover=min_cover)
                        if arch == 'mr_lv' else None,
    }
    started = time.monotonic()
    total_intersection = total_union = done = window_frames = unique_frames = 0
    per_clip = {}

    with _evaluation_state(model), torch.autocast(device_type=device.type, enabled=False):
        layout = _test_layout(root)
        expected_windows = sum(len(plan) for _, _, _, plan in layout)
        expected_unique_frames = sum(len(images) for _, images, _, _ in layout)
        for clip, image_paths, mask_paths, plan in layout:
            if limit is not None and done >= limit:
                break
            video = _load_video(image_paths)
            if 'pvalid' in needs:
                chain = stabilize.load_chain(clip, len(video))
                if not np.isfinite(chain).all():
                    raise ValueError(f'{clip}: non-finite homography chain')
            else:
                chain = np.broadcast_to(np.eye(3), (len(video), 3, 3)).copy()
            low = multirate.lowres(video, lowres) if arch == 'mr_lv' else None
            scorer = (make_scorer(model, needs, device, video, chain, hist_cfg, False)
                      if 'hist' not in needs else None)
            ci = cu = cw = cf = 0
            seen = set()
            coordinate_counts = {'raw': 0, 'warp': 0, 'gated': 0}
            for a, b in plan:
                if limit is not None and done >= limit:
                    break
                frames = list(range(a, b))
                if 'hist' in needs:
                    ref = stabilize.reference_frame(a, b)
                    history_frames = stabilize.history_frames(
                        len(video), a, b, hist_cfg['hist_len'], hist_cfg['hist_step'], hist_cfg['hist_mode'])
                    effective_chain, gated = stabilize.gate_chain(
                        chain, frames + history_frames, ref, motion_gate, video.shape[1:])
                    scorer = make_scorer(model, needs, device, video, effective_chain, hist_cfg, False)
                    # Identity-chain warping is also the native motion-gated LVH path.
                    coords = 'warp'
                    coordinate_counts['gated'] += int(gated)
                elif arch == 'mr_lv':
                    stats = multirate.stationarity(low[frames], chain, frames, lowres)
                    coords = multirate.choose_coords('auto', stats, 1, margin, abs_tol, min_cover)
                    if coords == 'skip':
                        raise RuntimeError('The native auto-coordinate rule skipped a rate-one window')
                else:
                    coords = 'raw'
                logits = scorer(frames, coords)
                target = _load_labels(mask_paths[a:b], video.shape[1:])
                intersection, union = _counts(logits, target, device=device)
                del logits, target
                ci += intersection
                cu += union
                cw += 1
                cf += b - a
                seen.update(frames)
                coordinate_counts[coords] += 1
                total_intersection += intersection
                total_union += union
                done += 1
                window_frames += b - a
                if progress is not None:
                    progress(dict(eval_clip=clip, eval_windows_done=done,
                                  eval_windows_total=expected_windows, eval_clip_windows_done=cw,
                                  eval_clip_windows_total=len(plan), first_frame=a, end_frame=b - 1))
            unique_frames += len(seen)
            per_clip[clip] = dict(intersection=ci, union=cu, iou=ci / cu if cu else None,
                                  windows=cw, window_frames=cf, unique_frames=len(seen),
                                  source_frames=len(video), coordinates=coordinate_counts)
            del scorer, video, chain, low

    if total_union == 0:
        raise ValueError('Checkpoint-selection IoU is undefined: total union is zero')
    return dict(iou=total_intersection / total_union, intersection=total_intersection,
                union=total_union, windows=done, window_frames=window_frames,
                unique_frames=unique_frames, expected_windows=expected_windows,
                expected_unique_frames=expected_unique_frames, complete=done == expected_windows,
                selection_max_windows=limit, seconds=time.monotonic() - started,
                test_clips=[clip for clip, _, _, _ in layout], per_clip=per_clip, protocol=protocol)
