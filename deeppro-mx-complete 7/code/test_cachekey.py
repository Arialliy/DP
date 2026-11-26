"""Checks of cachekey.py: predictions are reused only for the same weights, settings, inputs
and code (numpy only).

Run: python code/test_cachekey.py
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cachekey as ck  # noqa: E402


def test_fingerprint_follows_weights_settings_and_inputs():
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        os.chdir(d)
        try:
            open('a.pth', 'wb').write(b'weights 1')
            open('H.npy', 'wb').write(b'homographies')
            open('out.npy', 'wb').write(b'maps')
            args = argparse.Namespace(arch='mxc_lv', bank='outside', threads=4, device='cuda', run='r', clips='val')
            s = ck.settings_of(args)
            assert set(s) == {'arch', 'bank'}                                  # threads / device do not count
            code = ck.code_sha256()
            fp = ck.fingerprint('a.pth', s, 'c', 90, ['H.npy'], code)
            meta = os.path.join(d, 'meta.json')
            assert not ck.current(meta, fp, ['out.npy'])                       # nothing stored yet
            ck.write_json(meta, dict(fingerprint=fp))
            assert ck.current(meta, fp, ['out.npy'])
            assert not ck.current(meta, fp, ['out.npy', 'missing.npy'])        # an output is missing
            open('a.pth', 'wb').write(b'weights 2, a new best_model.pth')       # same path, new weights
            assert not ck.current(meta, ck.fingerprint('a.pth', s, 'c', 90, ['H.npy'], code), ['out.npy'])
            open('a.pth', 'wb').write(b'weights 1')
            assert ck.current(meta, ck.fingerprint('a.pth', s, 'c', 90, ['H.npy'], code), ['out.npy'])
            s2 = ck.settings_of(argparse.Namespace(arch='mxc_lv', bank='none'))
            assert not ck.current(meta, ck.fingerprint('a.pth', s2, 'c', 90, ['H.npy'], code), ['out.npy'])
            open('H.npy', 'wb').write(b'annotation-free homographies')
            assert not ck.current(meta, ck.fingerprint('a.pth', s, 'c', 90, ['H.npy'], code), ['out.npy'])
            ck.invalidate(meta)
            assert not os.path.exists(meta)
            os.makedirs('cache/preds/base__meta')
            json.dump(dict(fingerprint=fp), open('cache/preds/base__meta/c.json', 'w'))
            assert ck.stored('base__mrmax', 'c') == json.loads(json.dumps(fp)) and ck.stored('other', 'c') is None
        finally:
            os.chdir(old)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
