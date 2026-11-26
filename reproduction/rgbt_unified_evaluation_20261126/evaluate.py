"""Score immutable best checkpoints on all 15 official RGB-T test clips."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

from common import ROOT, config, read_json, save_json, sha256, verify_source
from inference import load_model, predict_clip, prediction_protocol
from official_metrics import THRESHOLDS, score_frame, aggregate_results


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


def score_dict(raw):
    """Normalize scorer output for the report and preserve exact integer counts."""
    thresholds = []
    for i, threshold in enumerate(THRESHOLDS):
        fp, tp, targets = (int(raw[k][i]) for k in ('false_pixels', 'true_targets', 'targets'))
        thresholds.append(dict(threshold=float(threshold), false_pixels=fp,
                               true_targets=tp, targets=targets, pd=tp / targets,
                               fa=fp / int(raw['pixels'])))
    return dict(auc=float(raw['auc']), iou_at_0_5=float(raw['iou']),
                intersection=int(raw['intersection']), union=int(raw['union']),
                pixels=int(raw['pixels']), thresholds=thresholds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job_id')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-clips', type=int, default=None, help='preflight only; partial metrics excluded')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = config()
    job = read_json(ROOT / 'jobs' / (args.job_id + '.json'))
    status_path = ROOT / 'jobs' / (args.job_id + '.status.json')
    status = dict(job_id=args.job_id, model=job['model'], state='starting', stage='loading',
                  started_at=datetime.now(timezone.utc).isoformat())

    def update(**values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        save_json(status_path, status)

    try:
        verify_source()
        checkpoint = ROOT / job['checkpoint']
        assert sha256(checkpoint) == job['checkpoint_sha256']
        model, needs = load_model(job['arch'], checkpoint, args.device)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        best_epoch = int(payload.get('best_epoch', int(payload['epoch']) + 1))
        train_cfg = read_json(ROOT / job['train_config'])
        protocol = prediction_protocol(job['arch'], train_cfg)
        protocol.update(evaluation='original ShootingRules, all 27 thresholds; unique source frames',
                        detection_threshold_comparison='>= (original sequential numpy assignments)',
                        iou_threshold_comparison='> 0.5', connectivity=8, loc_len1=1, loc_len2=4,
                        border_behavior='original NumPy negative/overflow slice behavior retained',
                        auc='sklearn.metrics.auc(Fa, Pd), original 27 points, no added endpoints',
                        labels='existing uint8 segmentations PNG masks binarized >0')
        dataset = Path(os.environ.get('VT_ROOT', '/data/RGB-T'))
        clips = list(dict.fromkeys(x.split('/')[0] for x in (dataset / 'test.txt').read_text().split()))
        assert clips == cfg['test_clips']
        if args.max_clips is not None:
            clips = clips[:args.max_clips]
        variants = job['variants']
        assert set(variants) == set(protocol['variants'])
        started = time.monotonic()
        update(state='running', best_epoch=best_epoch, checkpoint_sha256=job['checkpoint_sha256'],
               device=args.device, clips_done=0, clips_total=len(clips))
        frame_count = 0
        clip_results = {variant: [] for variant in variants}
        for clip_index, clip in enumerate(clips):
            images = sorted((dataset / 'test2017' / clip / '01').iterdir())
            masks = sorted((dataset / 'segmentations' / clip).iterdir())
            assert len(images) == len(masks) and all(a.stem == b.stem for a, b in zip(images, masks))
            video = np.asarray(np.load('/artifacts/cache/video/' + clip + '.npy', mmap_mode='r'), dtype=np.float32)
            assert len(video) == len(images)
            for index in sorted({0, len(video) // 2, len(video) - 1}):
                with Image.open(images[index]) as image:
                    original = np.asarray(image, dtype=np.float32)
                if original.ndim == 3:
                    original = original[..., 0]
                assert np.array_equal(original, video[index]), 'Cached frame mismatch: ' + clip
            update(stage='inference', clip=clip, clips_done=clip_index, frames_done=frame_count)
            for variant, probabilities in predict_clip(
                    model, needs, video, clip, train_cfg, args.device,
                    progress=lambda values: update(stage='inference', **values)):
                assert variant in variants and probabilities.dtype == np.float32
                assert probabilities.shape == video.shape
                update(stage='scoring', variant=variant, scoring_frame=0)
                frame_results = []
                for frame, path in enumerate(masks):
                    with Image.open(path) as image:
                        target = np.asarray(image, dtype=np.uint8)
                    if target.ndim == 3:
                        target = target[..., 0]
                    frame_results.append(score_frame(probabilities[frame], target))
                    if (frame + 1) % 100 == 0:
                        update(scoring_frame=frame + 1)
                totals = aggregate_results(frame_results)
                clip_results[variant].append(totals)
                record = dict(clip=clip, checkpoint_sha256=job['checkpoint_sha256'],
                              variant=variant, **serializable(totals))
                save_json(ROOT / 'results' / args.job_id / variant / 'clips' / (clip + '.json'), record)
                del probabilities, frame_results
            frame_count += len(video)
            del video
            update(clips_done=clip_index + 1, frames_done=frame_count)
            print(f'{job["model"]} {clip} done ({clip_index + 1}/{len(clips)})', flush=True)
        complete = clips == cfg['test_clips']
        results = []
        for variant in variants:
            raw = aggregate_results(clip_results[variant])
            if complete:
                assert frame_count == cfg['expected_frames']
                assert raw['pixels'] == cfg['expected_pixels']
                assert all(int(n) == cfg['expected_targets'] for n in raw['targets'])
            label = job['model'] + ((' / ' + variant) if len(variants) > 1 else '')
            metrics = dict(job_id=args.job_id, model=job['model'], model_id=job['model_id'],
                           label=label, arch=job['arch'], variant=variant,
                           primary_variant=job['primary_variant'],
                           checkpoint_sha256=job['checkpoint_sha256'], best_epoch=best_epoch,
                           training_epochs_completed=job['training_epochs_completed'],
                           final_checkpoint=job['final_checkpoint'],
                           checkpoint_status='final' if job['final_checkpoint'] else 'interim',
                           complete=complete, clips=clips, frames=frame_count,
                           device=args.device, dtype='float32', protocol=protocol,
                           elapsed_seconds=time.monotonic() - started,
                           source='Frozen best-IoU checkpoint; native model inference; common original detector metrics',
                           **score_dict(raw))
            path = ROOT / 'results' / args.job_id / variant / 'metrics.json'
            save_json(path, metrics)
            results.append(str(path.relative_to(ROOT)))
        update(state='completed', stage='completed', complete=complete, results=results,
               elapsed_seconds=time.monotonic() - started)
    except BaseException as exc:
        update(state='failed', stage='failed', error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
