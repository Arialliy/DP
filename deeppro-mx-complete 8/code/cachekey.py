"""Prediction caches bound to what made them: weights, settings, inputs and code.

The inference scripts (infer.py, infer_mr.py, infer_mx.py) used to skip a clip whenever a
cache file of the same run name existed, so evaluating again after best_model.pth changed
could score the old predictions. Each now stores a fingerprint with its predictions
(cache/preds/<run>__meta/<clip>.json) and skips a clip only if that fingerprint equals the
one it would write now; otherwise it computes the clip again and overwrites the files.

The fingerprint holds
  ckpt_sha256   SHA-256 of the checkpoint file
  settings      the inference settings that change predictions (not threads or device)
  inputs        SHA-256 of per-clip inputs, e.g. the homographies and targets.csv
  data          vtdata.DATA_ROOT and the clip's frame count
  code_sha256   SHA-256 of every module of this folder that the script has loaded
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_SHA = {}
IGNORED = ('threads', 'device', 'run', 'clips', 'nproc')


def sha256(path):
    """SHA-256 of a file, cached by path, size and modification time."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        _SHA[key] = h.hexdigest()
    return _SHA[key]


def code_sha256():
    """One digest of every loaded module whose file lies in this folder."""
    files = sorted({os.path.abspath(m.__file__) for m in list(sys.modules.values())
                    if getattr(m, '__file__', None) and os.path.dirname(os.path.abspath(m.__file__)) == HERE})
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.basename(f).encode())
        h.update(sha256(f).encode())
    return h.hexdigest()


def settings_of(args, ignored=IGNORED):
    """The settings of an argparse.Namespace (or dict) that can change predictions."""
    d = vars(args) if not isinstance(args, dict) else args
    d = {k: (sorted(v) if isinstance(v, set) else v) for k, v in d.items() if k not in ignored}
    return json.loads(json.dumps(d, sort_keys=True, default=str))


def fingerprint(ckpt, settings, clip, n_frames, inputs=(), code=None):
    """inputs: paths of per-clip input files (missing ones are recorded as None). code: a
    code_sha256() taken once before the first clip, so that modules imported while a clip is
    processed do not change the fingerprint of the next one."""
    import vtdata
    return dict(ckpt=os.path.abspath(ckpt), ckpt_sha256=sha256(ckpt), settings=settings, clip=clip,
                data=dict(root=os.path.abspath(vtdata.DATA_ROOT), frames=int(n_frames)),
                inputs={os.path.relpath(p) if os.path.exists(p) else p: (sha256(p) if os.path.exists(p) else None)
                        for p in inputs},
                code_sha256=code if code is not None else code_sha256())


def current(meta_path, fp, outputs):
    """True if meta_path holds this fingerprint and every output file exists."""
    if not os.path.exists(meta_path) or not all(os.path.exists(o) for o in outputs):
        return False
    try:
        old = json.load(open(meta_path)).get('fingerprint')
    except (OSError, ValueError):
        return False
    return old == json.loads(json.dumps(fp))


def digest(fp):
    """Short SHA-256 of a fingerprint, for score summaries."""
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]


def stored(run, clip, root='cache/preds'):
    """The fingerprint stored with cache/preds/<run>/<clip>.npy, or None. A run derived by
    infer_mr.py (<base>__mrmax, <base>__r2, ...) shares the meta folder of <base>."""
    names = [run] + ([run.rsplit('__', 1)[0]] if '__' in run else [])
    for name in names:
        path = os.path.join(root, f'{name}__meta', f'{clip}.json')
        if os.path.exists(path):
            try:
                return json.load(open(path)).get('fingerprint')
            except (OSError, ValueError):
                return None
    return None


def invalidate(meta_path):
    """Remove a stale fingerprint before outputs are overwritten: if the run stops half way,
    the old fingerprint cannot vouch for the new, partial files."""
    if os.path.exists(meta_path):
        os.remove(meta_path)


def write_json(path, value):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(value, f, indent=1)
    os.replace(tmp, path)
