"""Build exact uint8 video caches, copy verified homographies, then mine hot spots.

Only preprocessing artifacts are written; frozen research code is imported unchanged.
"""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np

ROOT = Path('/experiment')
SOURCE_HOMOGRAPHIES = Path('/homographies_source')
sys.path.insert(0, str(ROOT / 'code'))
import vtdata


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def one_clip(clip):
    scene = vtdata.Scene(clip)
    n, h, w = len(scene), scene.H, scene.W
    cache = ROOT / 'cache/video' / f'{clip}.npy'
    tmp = cache.with_suffix('.preparing.npy')
    if not cache.exists():
        out = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(n, h, w))
        for i in range(n):
            frame = scene.image(i)
            assert frame.shape == (h, w), (clip, i, frame.shape)
            out[i] = np.clip(frame, 0, 255).astype(np.uint8)
        out.flush()
        del out
        tmp.replace(cache)
    cached = np.load(cache, mmap_mode='r')
    assert cached.dtype == np.uint8 and cached.shape == (n, h, w), (clip, cached.shape, cached.dtype)
    witnesses = []
    for i in sorted({0, n // 2, n - 1}):
        expected = np.clip(scene.image(i), 0, 255).astype(np.uint8)
        assert np.array_equal(cached[i], expected), (clip, i, 'cache mismatch')
        witnesses.append({'index': i, 'frame': scene.frames[i], 'image_sha256': sha256(scene.paths[i]),
                          'decoded_uint8_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
                          'exact_equal': True})
    original = SOURCE_HOMOGRAPHIES / f'{clip}.npy'
    homography = ROOT / 'results/profile/homographies' / original.name
    pairs = np.load(original)
    assert pairs.shape == (n - 1, 3, 3), (clip, pairs.shape, n)
    assert np.isfinite(pairs).all(), clip
    source_hash = sha256(original)
    if homography.exists():
        assert sha256(homography) == source_hash, ('Existing homography differs', clip)
    else:
        shutil.copy2(original, homography)
    assert sha256(homography) == source_hash
    return {'clip': clip, 'split': scene.split, 'frames': n, 'height': h, 'width': w,
            'cache': str(cache.relative_to(ROOT)), 'dtype': str(cached.dtype),
            'cache_bytes': cache.stat().st_size, 'cache_sha256': sha256(cache),
            'sampled_frames': witnesses, 'homography': str(homography.relative_to(ROOT)),
            'homography_sha256': source_hash, 'homography_shape': list(pairs.shape),
            'homography_dtype': str(pairs.dtype), 'homography_finite': True}


def main():
    os.chdir(ROOT)
    cfg = json.loads((ROOT / 'experiment.json').read_text())
    train, test = cfg['train_clips'], cfg['test_clips']
    clips = train + test
    assert len(train) == 45 and len(test) == 15 and len(set(clips)) == 60
    assert train == vtdata.scene_list('train') and test == vtdata.scene_list('test')
    for folder in ['cache/video', 'cache/hotspots', 'results/profile/homographies', 'logs']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    started = utcnow()
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for record in pool.map(one_clip, clips):
            records.append(record)
            print(f"CACHE {len(records)}/60 {record['clip']} frames={record['frames']} samples=PASS", flush=True)
    # The historical targets.csv was audited as invalid and must never enter this run.
    assert not (ROOT / 'results/profile/targets.csv').exists()
    manifest = {
        'status': 'ready', 'started_utc': started, 'completed_utc': utcnow(),
        'image': cfg['image'], 'dataset_host': cfg['dataset'], 'dataset_container': vtdata.DATA_ROOT,
        'train_clips': 45, 'test_clips': 15, 'clips': 60,
        'frames': sum(r['frames'] for r in records),
        'train_frames': sum(r['frames'] for r in records if r['clip'] in train),
        'test_frames': sum(r['frames'] for r in records if r['clip'] in test),
        'cache_bytes': sum(r['cache_bytes'] for r in records),
        'cache_rule': 'Per-frame np.clip(Scene.image(i), 0, 255).astype(np.uint8); exact same pixels as train.clip_array',
        'validation': 'Shape/dtype for every clip; first/middle/last decoded frame equality for all 60 clips',
        'homography_source_host': '/home/md0/ly/DP/deeppro-mx-complete/results/profile/homographies',
        'homography_copy_scope': 'Only the 60 official clip .npy files, byte-identical; shape and finite validation repeated',
        'invalid_targets_csv_copied': False,
        'targets_csv_present': False,
        'inputs_sha256': {str(p): sha256(p) for p in [ROOT / 'experiment.json', ROOT / 'configs/splits.json',
             Path(vtdata.DATA_ROOT) / 'train.txt', Path(vtdata.DATA_ROOT) / 'test.txt',
             Path(vtdata.annotation_path('train2017')), Path(vtdata.annotation_path('test2017'))]},
        'records': records,
    }
    atomic_json(ROOT / 'preprocessing_provenance.json', manifest)
    atomic_json(ROOT / 'data_ready.json', manifest)
    atomic_json(ROOT / 'cache_ready.json', {k: v for k, v in manifest.items() if k != 'records'})
    print('ALL CACHES AND HOMOGRAPHIES READY; starting unchanged hotspot miner', flush=True)
    command = [sys.executable, '-u', 'code/hotspots.py', '--split', 'official_train', '--nproc', '4']
    hotspot_start = utcnow()
    atomic_json(ROOT / 'hotspots_status.json', {'status': 'running', 'started_utc': hotspot_start, 'command': command})
    with (ROOT / 'logs/hotspots.log').open('w') as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'hotspot miner exited {completed.returncode}; see logs/hotspots.log')
    hot_records = []
    for clip in train:
        path = ROOT / 'cache/hotspots' / f'{clip}.npz'
        with np.load(path) as data:
            assert set(data.files) == {'ref', 'x', 'y', 'contrast'}, (clip, data.files)
            count = len(data['ref'])
            assert all(data[k].shape == (count,) and np.isfinite(data[k]).all() for k in data.files), clip
            hot_records.append({'clip': clip, 'candidates': count, 'path': str(path.relative_to(ROOT)),
                                'sha256': sha256(path), 'bytes': path.stat().st_size})
    result = {'status': 'ready', 'started_utc': hotspot_start, 'completed_utc': utcnow(), 'command': command,
              'exit_code': completed.returncode, 'split': 'official_train', 'clips': 45,
              'candidates': sum(r['candidates'] for r in hot_records), 'mining_behavior': 'Unmodified code/hotspots.py defaults',
              'records': hot_records}
    atomic_json(ROOT / 'hotspots_ready.json', result)
    atomic_json(ROOT / 'hotspots_status.json', {k: v for k, v in result.items() if k != 'records'})
    print(f"HOTSPOTS READY: 45 clips, {result['candidates']} candidates", flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        atomic_json(ROOT / 'preprocessing_failed.json', {'status': 'failed', 'time_utc': utcnow(),
                    'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
