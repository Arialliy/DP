"""Checks of the DeepPro-MX network, its training path and inference (needs torch).

Run on the training machine: python code/test_mx.py
The last test trains one short epoch and runs infer_mx.py on a fake dataset written to a
temporary directory (JPEG frames, COCO json with 'tracking_id', identity homographies).
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # absolute: a test changes directory
import models  # noqa: E402
import mx  # noqa: E402
import mx_data  # noqa: E402

torch.manual_seed(0)


def _inputs(b=2, T=40, H=48, W=48, K=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 1, T, H, W, generator=g)
    valid = torch.ones(b, T, dtype=torch.bool)
    pvalid = torch.ones(b, 1, T, H, W, dtype=torch.bool)
    S = mx.Encoder().patch
    ex = torch.randn(b, K, 1, S, S, generator=g)
    ex_valid = torch.zeros(b, K, dtype=torch.bool)
    ex_valid[0, :5] = True                                    # sample 1 has no exemplars
    return x, valid, pvalid, ex, ex_valid


def test_registered_forward_backward():
    for name in ('mx_lv', 'mx_lv_abs'):
        model, needs = models.build(name)
        assert tuple(needs) == mx.NEEDS and getattr(model, 'exemplar', False)
        x, valid, pvalid, ex, ex_valid = _inputs()
        _, logit = model(x, valid=valid, pvalid=pvalid, ex=ex, ex_valid=ex_valid)
        assert logit.shape == (2, 40, 48, 48) and torch.isfinite(logit).all(), name
        assert model.last_ll.shape == (2, 1, 48, 48) and model.last_lm.shape == (2, 1, 48, 48)
        assert model.last_avail.tolist() == [True, False]
        assert (model.last['mx'][1] == 0).all()               # no exemplars -> M_x = 0
        lab = (torch.rand(2, 40, 48, 48) > 0.995).float()
        loss = logit.mean() + model.last_ll.mean() + mx.match_loss(model, lab, pvalid & valid[:, None, :, None, None])
        loss.backward()
        assert model.enc.net[0].weight.grad is not None and torch.isfinite(model.enc.net[0].weight.grad).all()
        assert model.protos.grad is not None
    print('  parameters mx_lv:', models.n_params(models.build('mx_lv')[0]))


def test_encoder_patch_equals_full_frame():
    enc = mx.Encoder()
    m, S = enc.margin, enc.patch
    frame = torch.randn(1, 1, 90, 100)
    full = enc(frame)
    for y, x in ((45, 50), (m, m), (90 - m - 1, 100 - m - 1), (30, 70)):
        patch = frame[:, :, y - m:y + m + 1, x - m:x + m + 1]
        assert patch.shape[-2:] == (S, S)
        assert torch.allclose(enc(patch)[0, :, m, m], full[0, :, y, x], atol=1e-4), (y, x)
    # the same through mx_data.extract_patches (numpy, normalised like the network input)
    video = np.random.default_rng(1).normal(85, 40, (3, 90, 100)).astype(np.float32)
    mean, std = 85.0799, 47.4845
    ex, ok = mx_data.extract_patches(video, np.array([[2, 50.0, 45.0]]), S, mean, std, 1)
    model = mx.DeepProMX()
    e = model.embed_exemplars(torch.from_numpy(ex)[None])[0, 0]
    f = model.enc(torch.from_numpy((video[2:3] - mean) / std)[:, None])[0, :, 45, 50]
    assert ok[0] and torch.allclose(e, f, atol=1e-4)


def test_invalid_exemplars_are_ignored():
    model, _ = models.build('mx_lv')
    model.eval()
    x, valid, pvalid, ex, ex_valid = _inputs()
    off = torch.zeros_like(ex_valid)
    with torch.no_grad():
        a = model(x, valid=valid, pvalid=pvalid, ex=ex, ex_valid=off)[1]
        b = model(x, valid=valid, pvalid=pvalid, ex=torch.randn_like(ex), ex_valid=off)[1]
        c = model(x, valid=valid, pvalid=pvalid)[1]
    assert torch.equal(a, b) and torch.allclose(a, c)
    # an invalid slot next to valid ones does not change the result either
    ex2 = ex.clone()
    ex2[0, 10] = 99.0
    with torch.no_grad():
        d = model(x, valid=valid, pvalid=pvalid, ex=ex, ex_valid=ex_valid)[1]
        e = model(x, valid=valid, pvalid=pvalid, ex=ex2, ex_valid=ex_valid)[1]
    assert torch.allclose(d, e, atol=1e-6)


def test_self_exemplar_matches_with_similarity_one():
    model, _ = models.build('mx_lv')
    model.eval()
    H = W = 64
    frame = torch.randn(1, 1, 1, H, W) * 0.3
    frame[..., 30:33, 28:36] += 3.0                           # a car-like bar at (31, 32)
    x = frame.expand(1, 1, 40, H, W).contiguous()
    m = model.enc.margin
    ex = torch.zeros(1, 16, 1, model.patch, model.patch)
    ex[0, 0] = frame[0, 0, :, 31 - m:31 + m + 1, 32 - m:32 + m + 1]
    ok = torch.zeros(1, 16, dtype=torch.bool)
    ok[0, 0] = True
    with torch.no_grad():
        Mx, Mg, avail, idx = model.match_maps(x, torch.ones_like(x, dtype=torch.bool), ex, ok)
    assert avail.tolist() == [True] and idx == list(range(0, 40, 4))
    assert abs(float(Mx[0, 31, 32]) - 1.0) < 1e-5
    assert float(Mx.max()) <= 1.0 + 1e-5 and float(Mg.abs().max()) <= 1.0 + 1e-5


def test_polarity_both_matches_an_inverted_target():
    H = W = 64
    frame = torch.randn(1, 1, 1, H, W) * 0.3
    frame[..., 30:33, 28:36] += 3.0
    x = frame.expand(1, 1, 40, H, W).contiguous()
    res = {}
    for name in ('mx_lv', 'mx_lv_abs'):
        torch.manual_seed(3)
        model, _ = models.build(name)
        model.eval()
        m = model.enc.margin
        ex = torch.zeros(1, 16, 1, model.patch, model.patch)
        ex[0, 0] = -frame[0, 0, :, 31 - m:31 + m + 1, 32 - m:32 + m + 1]   # a dark exemplar
        ok = torch.zeros(1, 16, dtype=torch.bool)
        ok[0, 0] = True
        with torch.no_grad():
            res[name] = float(model.match_maps(x, torch.ones_like(x, dtype=torch.bool), ex, ok)[0][0, 31, 32])
    assert abs(res['mx_lv_abs'] - 1.0) < 1e-5 and res['mx_lv'] < 1.0 - 1e-3, res


def test_invalid_frames_do_not_change_the_output():
    model, _ = models.build('mx_lv')
    model.eval()
    x, valid, pvalid, ex, ex_valid = _inputs()
    valid[:, :10] = False                                     # a zero-padded short window
    x1, x2 = x.clone(), x.clone()
    x1[:, :, :10] = 50.0
    x2[:, :, :10] = -7.0
    with torch.no_grad():
        a = model(x1, valid=valid, pvalid=pvalid, ex=ex, ex_valid=ex_valid)[1]
        b = model(x2, valid=valid, pvalid=pvalid, ex=ex, ex_valid=ex_valid)[1]
    assert torch.allclose(a[:, 10:], b[:, 10:], atol=1e-5)


def test_masked_median_and_erosion():
    m = torch.tensor([[[[1.0]], [[5.0]], [[3.0]], [[100.0]]]]).view(1, 4, 1, 1)
    v = torch.tensor([True, True, True, False]).view(1, 4, 1, 1)
    assert float(mx.masked_median_t(m, v)) == 3.0
    assert float(mx.masked_median_t(m, torch.zeros_like(v))) == 0.0
    pv = torch.ones(1, 20, 20, dtype=torch.bool)
    pv[0, :, :5] = False
    e = mx.erode_valid(pv, 3)
    assert not e[0, :, :8].any() and e[0, :, 8:].all()        # the image border is not invalid


def test_match_loss_without_exemplars_is_zero():
    model, _ = models.build('mx_lv')
    x, valid, pvalid, ex, ex_valid = _inputs()
    model(x, valid=valid, pvalid=pvalid, ex=ex, ex_valid=torch.zeros_like(ex_valid))
    loss = mx.match_loss(model, torch.zeros(2, 40, 48, 48), pvalid)
    assert float(loss) == 0.0
    loss.backward()


# ---------------------------------------------------------------- end to end on a fake dataset
def _fake_clip(rng, n=90, H=160, W=200):
    yy, xx = np.mgrid[0:H, 0:W]
    base = 70 + 0.1 * xx + 20 * np.exp(-((xx - 150) ** 2 + (yy - 40) ** 2) / 50.0)
    video, boxes = [], []
    for t in range(n):
        img = base + rng.normal(0, 2, (H, W))
        b = []
        for i, y in enumerate((30, 100)):                    # two cars driving at 1 px/frame
            x = 10 + (t + 50 * i) % 170
            img[y - 1:y + 2, x - 3:x + 4] += 60
            b.append((x - 3, y - 1, 7, 3, 1, 10 + i))
        for j, (x, y) in enumerate(((60, 70), (140, 120))):  # two parked cars
            img[y - 1:y + 2, x - 3:x + 4] += 40
            b.append((x - 3, y - 1, 7, 3, 1, 20 + j))
        video.append(img)
        boxes.append(np.array(b, float))
    return np.stack(video), boxes


def test_training_and_inference_end_to_end():
    import fake_dataset
    import infer_mx
    import splits
    import train
    import vtdata
    splits.SPLIT_FILE = os.path.abspath(splits.SPLIT_FILE)  # splits.clips reads it after the chdir below
    old_root, old_ann, old_cwd, old_argv = vtdata.DATA_ROOT, vtdata.ANN_DIR, os.getcwd(), sys.argv
    rng = np.random.default_rng(0)
    clips = {name: _fake_clip(rng) for name in ('fakeA', 'fakeB')}
    with tempfile.TemporaryDirectory() as d:
        try:
            fake_dataset.write(d, {k: (v, b, 'train2017') for k, (v, b) in clips.items()}, track_key='tracking_id')
            speed = {}
            for k, (_, b) in clips.items():
                for t, bb in enumerate(b):
                    for row in bb:
                        speed[(k, t, int(row[5]))] = 1.0 if row[5] < 20 else 0.0
            fake_dataset.write_targets(os.path.join(d, 'targets.csv'), speed, {k: b for k, (v, b) in clips.items()})
            os.makedirs(os.path.join(d, 'results/profile/homographies'))
            for k, (v, _) in clips.items():                      # a static camera
                np.save(os.path.join(d, f'results/profile/homographies/{k}.npy'),
                        np.repeat(np.eye(3)[None], len(v) - 1, 0))
            vtdata.DATA_ROOT, vtdata.ANN_DIR = d, None           # the fake data, not VT_ANN_DIR of the shell
            vtdata._ANN_CACHE.clear()
            os.chdir(d)
            # the dataset
            needs = models.build('mx_lv')[1]
            ds = train.ExemplarTrainSet(['fakeA'], 'targets.csv', ex_k=16, ex_patch=39, rates=[1], coords='auto',
                                        needs=needs, seed=0, sample_rate=0.05, rule='paper_k6_floor')
            s = ds[0]
            assert s['img'].shape == (1, 40, 128, 128) and s['lab'].shape == (40, 128, 128)
            assert s['pvalid'].shape == (1, 40, 128, 128) and s['pvalid'].dtype == torch.bool
            assert s['ex'].shape == (16, 1, 39, 39) and s['ex_valid'].dtype == torch.bool
            assert s['ex_t'].dtype == torch.int64 and len(ds.moving['fakeA']) == 2 * 90
            n_with = sum(bool(ds[i]['ex_valid'].any()) for i in range(20))
            assert 5 <= n_with <= 20                                     # p_none = 0.3
            # one short training run
            sys.argv = ['train.py', '--run', 'mx_test', '--arch', 'mx_lv', '--split', 'fakeA,fakeB', '--epochs', '1',
                        '--batch', '2', '--sample_rate', '0.05', '--workers', '0', '--threads', '2',
                        '--ex-targets', 'targets.csv']                   # no --label-rule: the default
            train.main()
            cfg = json.load(open('runs/mx_test/config.json'))
            assert cfg['label_rule'] == 'paper_k6_floor'
            assert cfg['coords'] == 'auto' and cfg['level_focal'] == 1.0 and cfg['persist'] == 'hard'
            assert cfg['ex_k'] == 16 and os.path.exists('runs/mx_test/final.pth')
            assert 'exemplars:' in open('runs/mx_test/train_log.txt').read()
            # inference with every bank
            for bank in ('outside', 'causal', 'gt', 'none'):
                infer_mx.main(['--run', f'mx_test_{bank}', '--ckpt', 'runs/mx_test/final.pth', '--clips', 'fakeA',
                               '--bank', bank, '--targets', 'targets.csv', '--threads', '2', '--save-arms'])
                p = np.load(f'cache/preds/mx_test_{bank}/fakeA.npy')
                assert p.shape == (90, 160, 200) and p.dtype == np.float16 and np.isfinite(p).all()
                for arm in ('variation', 'level', 'match'):
                    assert np.load(f'cache/preds/mx_test_{bank}__{arm}/fakeA.npy').shape == (90, 160, 200)
                meta = json.load(open(f'cache/preds/mx_test_{bank}__meta/fakeA.json'))
                assert meta['summary']['windows'] == 3
                if bank == 'none':
                    assert meta['summary']['mean_exemplars'] == 0
                if bank == 'gt':
                    assert meta['summary']['mean_exemplars'] > 0
            import evaluate
            import score
            r = evaluate.eval_clip(np.load('cache/preds/mx_test_outside/fakeA.npy'), vtdata.Scene('fakeA'))
            assert len(r['gt']) == 4 * 90
            sys.argv = ['score.py', '--run', 'mx_test_outside', '--split', 'fakeA', '--author', '--nproc', '1']
            score.main()                                                   # no --label-rule: the default
            summ = json.load(open('results/scores/mx_test_outside/fakeA/summary.json'))
            assert summ['author_label_rule'] == 'paper_k6_floor' and summ['overall']['gt'] == 4 * 90
        finally:
            os.chdir(old_cwd)
            sys.argv = old_argv
            vtdata.DATA_ROOT, vtdata.ANN_DIR = old_root, old_ann
            vtdata._ANN_CACHE.clear()


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
