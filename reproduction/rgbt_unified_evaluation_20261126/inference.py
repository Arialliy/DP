"""FP32, frame-unique inference for the predeclared RGB-T comparison.

load_model(...) returns (model, needs). predict_clip(...) yields (variant, array),
one full-clip FP32 array at a time. Consume/save each array before requesting the
next. CFAR's sigmoid-transformed output is a ranking score, not a probability.
"""
import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
import models
import multirate
import stabilize
from infer_mr import make_decider, make_scorer
import torch_compat

RATES = (1, 2, 4)
T = 40


def canonical_arch(arch):
    return 'deeppro_plus' if arch in ('DeepPro-Plus', 'deeppro_plus') else arch


def _official_module():
    source = Path(os.environ.get('OFFICIAL_SOURCE', '/official-source')).resolve()
    path = source / 'networks/models/DeepPro-Plus.py'
    if not path.is_file():
        raise FileNotFoundError(f'{path}: mount the original DP source read-only and set OFFICIAL_SOURCE')
    for name, module in list(sys.modules.items()):
        if name == 'networks' or name.startswith('networks.'):
            location = getattr(module, '__file__', None)
            if location is not None and not Path(location).resolve().is_relative_to(source):
                raise RuntimeError(f'{name} was already imported from a different source: {location}')
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('_original_deeppro_plus_unified_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, str(path)


def load_model(arch, checkpoint, device):
    """Strict-load the chosen checkpoint; Plus always uses the original network."""
    arch = canonical_arch(arch)
    device = torch.device(device)
    if arch == 'deeppro_plus':
        original, source = _official_module()
        model, needs = original.detector(1, T, T), ()
    elif arch in ('lv_full', 'lvh_full', 'mr_lv', 'lvh_dec'):
        model, needs = models.build(arch, seqlen=T)
        source = str(ROOT / 'code')
    else:
        raise ValueError(f'Unsupported architecture: {arch}')
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError('Checkpoint must contain a state dictionary')
    state = payload.get('model_state_dict', payload.get('model', payload))
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise TypeError('Checkpoint model state is not a nonempty tensor dictionary')
    if any(v.is_floating_point() and v.dtype != torch.float32 for v in state.values()):
        raise TypeError('Unified evaluation requires FP32 checkpoint tensors')
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Checkpoint contains non-finite tensor values')
    model.load_state_dict(state, strict=True)
    model.to(device)
    # Preserve the original Plus inference layout; research models used this layout.
    if arch != 'deeppro_plus' and device.type == 'cuda':
        torch_compat.install()
        torch.nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
    model.eval()
    model._evaluation_arch = arch
    model._evaluation_source = source
    model._evaluation_checkpoint = str(checkpoint)
    return model, tuple(needs)


def prediction_protocol(arch, train_cfg=None):
    arch = canonical_arch(arch)
    cfg = train_cfg or {}
    protocol = dict(arch=arch, dtype='float32', sequence_length=T, overlap=4,
                    frame_fusion='maximum over overlapping windows; each source frame output once',
                    sigmoid='FP32 torch.sigmoid on the model device; no float16 intermediate',
                    primary_variant='multirate_max' if arch == 'mr_lv' else 'native')
    if arch == 'mr_lv':
        protocol.update(rates=list(RATES), coords='auto', cfar_m=10, cfar_half=18,
                        variants={
                            'rate1_auto': 'sigmoid of rate-1 frame logits in adaptive coordinates',
                            'multirate_max': 'sigmoid of maximum uncalibrated logits across rates 1,2,4',
                            'multirate_cfar_score': 'sigmoid(max_k(logit_k - native CFAR reference_k)); '
                                                     'monotonic ranking score, NOT a calibrated probability'},
                        short_clip_rule='A rate k is omitted when N < 40*k; no padding or rate substitution',
                        cfar_quantization='Native formula on FP32 maps; original CLI float16 storage omitted',
                        stationarity=dict(lowres=cfg.get('lowres', 4), margin=cfg.get('stat_margin', 0.1),
                                          abs_tol=cfg.get('abs_tol', 0.5), min_cover=cfg.get('min_cover', 0.1)))
    else:
        protocol['variants'] = {'native': 'sigmoid of native frame logits, without score calibration'}
        if arch in ('lvh_full', 'lvh_dec'):
            protocol.update(history={key: cfg.get(key, value) for key, value in
                                     (('hist_len', 80), ('hist_step', 4), ('hist_mode', 'both'))},
                            coords='native stabilized, with original motion gate over window AND history',
                            motion_gate=cfg.get('motion_gate', 1.0 if arch == 'lvh_dec' else 0.0))
        else:
            protocol['coords'] = 'raw'
    return protocol


@contextmanager
def _fp32_inference(device):
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        yield


def _validate(model, video, device):
    if not isinstance(video, np.ndarray) or video.dtype != np.float32 or video.ndim != 3:
        raise TypeError('video must be a raw-grey float32 numpy array shaped (N,H,W)')
    if len(video) < T or not np.isfinite(video).all():
        raise ValueError('At least 40 finite source frames are required')
    if model.training:
        raise ValueError('Inference requires model.eval()')
    parameter = next(model.parameters(), None)
    if parameter is not None:
        actual, requested = parameter.device, torch.device(device)
        if actual.type != requested.type or (requested.index is not None and actual.index != requested.index):
            raise ValueError(f'Model device {actual} differs from requested {requested}')
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise TypeError('Model parameters must be FP32')


def _setup(model, needs, video, clip, cfg, device):
    arch = canonical_arch(getattr(model, '_evaluation_arch', cfg.get('arch')))
    if arch not in ('deeppro_plus', 'lv_full', 'lvh_full', 'lvh_dec', 'mr_lv'):
        raise ValueError('Architecture must come from load_model or train_cfg[arch]')
    needs = tuple(needs)
    hist_cfg = {key: cfg.get(key, default) for key, default in
                (('hist_len', 80), ('hist_step', 4), ('hist_mode', 'both'))}
    chain = (stabilize.load_chain(clip, len(video)) if 'pvalid' in needs else
             np.broadcast_to(np.eye(3), (len(video), 3, 3)).copy())
    if not np.isfinite(chain).all():
        raise ValueError(f'{clip}: non-finite homography chain')
    if 'hist' in needs:
        gate = cfg.get('motion_gate', 1.0 if arch == 'lvh_dec' else 0.0)

        def score(frames, coords):
            a, b = int(frames[0]), int(frames[-1]) + 1
            ref = stabilize.reference_frame(a, b)
            hframes = stabilize.history_frames(len(video), a, b, hist_cfg['hist_len'],
                                               hist_cfg['hist_step'], hist_cfg['hist_mode'])
            effective, _ = stabilize.gate_chain(chain, list(frames) + hframes, ref,
                                                gate, video.shape[1:])
            scorer = make_scorer(model, needs, device, video, effective, hist_cfg, False)
            return scorer(frames, 'warp')

        decide = lambda frames: ('warp', {})
    else:
        score = make_scorer(model, needs, device, video, chain, hist_cfg, False)
        if arch == 'mr_lv':
            factor = cfg.get('lowres', 4)
            low = multirate.lowres(video, factor)
            decide = make_decider('auto', low, chain, factor, cfg.get('stat_margin', 0.1),
                                  cfg.get('abs_tol', 0.5), cfg.get('min_cover', 0.1))
        else:
            decide = lambda frames: ('raw', {})
    return arch, score, decide


def _rate_logits(video, clip, rate, score, decide, device, progress):
    plan = multirate.plan_windows(len(video), rate, T)
    if not plan:
        if progress:
            progress(dict(clip=clip, rate=rate, used=False, reason='clip shorter than 40*rate'))
        return None
    out = np.full(video.shape, -np.inf, np.float32)
    with _fp32_inference(device):
        for index, (phase, frames) in enumerate(plan):
            coords, _ = decide(frames)
            if coords == 'skip':
                if rate == 1:
                    raise RuntimeError('Rate-one inference cannot skip a source window')
            else:
                logits = score(frames, coords)
                if logits.dtype != np.float32 or logits.shape != (len(frames), *video.shape[1:]):
                    raise ValueError('Native scorer returned an unexpected dtype or shape')
                if not np.isfinite(logits).all():
                    raise ValueError(f'{clip}: non-finite window logits')
                out[frames] = np.maximum(out[frames], logits)
            if progress:
                progress(dict(clip=clip, rate=rate, used=True, window=index + 1,
                              windows_total=len(plan), phase=int(phase), coords=coords,
                              first_frame=int(frames[0]), last_frame=int(frames[-1])))
    return out


def _sigmoid_inplace(array, device):
    """Transform a sole-owned FP32 output in bounded chunks on the native device."""
    with _fp32_inference(device):
        for start in range(0, len(array), T):
            tensor = torch.from_numpy(np.ascontiguousarray(array[start:start + T])).to(device)
            array[start:start + T] = torch.sigmoid(tensor).cpu().numpy()
    return array


def predict_clip(model, needs, video, clip, train_cfg, device, progress=None):
    """Yield fixed (variant_name, FP32 array) results; no test-dependent selection.

    CFAR output is explicitly named multirate_cfar_score: sigmoid is only a fixed
    monotonic score transform. Use prediction_protocol() when saving metadata.
    """
    _validate(model, video, device)
    cfg = train_cfg or {}
    arch, score, decide = _setup(model, needs, video, clip, cfg, device)
    if arch != 'mr_lv':
        logits = _rate_logits(video, clip, 1, score, decide, device, progress)
        if not np.isfinite(logits).all():
            raise ValueError('Native inference left a source frame uncovered')
        yield 'native', _sigmoid_inplace(logits, device)
        return

    fused = calibrated = None
    for rate in RATES:
        logits = _rate_logits(video, clip, rate, score, decide, device, progress)
        if logits is None:
            continue
        reference = multirate.cfar_reference(
            [multirate.peak_scores(frame) for frame in logits], m=10, half=18)
        if rate == 1:
            if not np.isfinite(logits).all():
                raise ValueError('Rate-one inference left a source frame uncovered')
            fused = logits
            calibrated = logits - reference[:, None, None]
            rate_one = _sigmoid_inplace(logits.copy(), device)
            yield 'rate1_auto', rate_one
            del rate_one
        else:
            np.maximum(fused, logits, out=fused)
            logits -= reference[:, None, None]
            np.maximum(calibrated, logits, out=calibrated)
        del logits, reference
    yield 'multirate_max', _sigmoid_inplace(fused, device)
    del fused
    yield 'multirate_cfar_score', _sigmoid_inplace(calibrated, device)


def benchmark_window(model, needs, video, clip, train_cfg, device='cpu'):
    """Time one full-resolution rate-one window, retaining genuine outside History."""
    _validate(model, video, device)
    started = time.monotonic()
    arch, score, decide = _setup(model, needs, video, clip, train_cfg or {}, device)
    frames = multirate.plan_windows(len(video), 1, T)[0][1]
    prep = time.monotonic() - started
    with _fp32_inference(device):
        coords, _ = decide(frames)
        if coords == 'skip':
            raise RuntimeError('First rate-one window was skipped')
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
        forward_start = time.monotonic()
        logits = score(frames, coords)
        probability = _sigmoid_inplace(logits, device)
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
    if probability.dtype != np.float32 or not np.isfinite(probability).all():
        raise ValueError('Invalid benchmark output')
    return dict(arch=arch, device=str(device), clip=clip, shape=list(probability.shape),
                coordinates=coords, setup_seconds=prep,
                window_seconds=time.monotonic() - forward_start,
                total_seconds=time.monotonic() - started,
                probability_min=float(probability.min()), probability_max=float(probability.max()),
                threads=torch.get_num_threads(), status='passed')


def main():
    parser = argparse.ArgumentParser(description='Benchmark one full-resolution inference window.')
    parser.add_argument('--arch', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--video', required=True, help='Raw-grey cached clip .npy')
    parser.add_argument('--clip', required=True)
    parser.add_argument('--train-config', required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, needs = load_model(args.arch, args.checkpoint, 'cpu')
    cfg = json.loads(Path(args.train_config).read_text())
    video = np.asarray(np.load(args.video, mmap_mode='r'), dtype=np.float32)
    print(json.dumps(benchmark_window(model, needs, video, args.clip, cfg), indent=2), flush=True)


if __name__ == '__main__':
    main()
