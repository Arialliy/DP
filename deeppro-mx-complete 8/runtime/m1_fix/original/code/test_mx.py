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
    for name in ('mx_lv', 'mx_lv_abs', 'mxc_lv'):
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
    print('  parameters mx_lv:', models.n_params(models.build('mx_lv')[0]),
          'mxc_lv:', models.n_params(models.build('mxc_lv')[0]))


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


# ---------------------------------------------------------------- DeepPro-MXC: motion calibration
def test_local_contrast_matches_stabilize():
    import stabilize as st
    rng = np.random.default_rng(0)
    img = rng.normal(0, 1, (37, 45)).astype(np.float32)
    img[10:13, 20:23] += 4
    ok = rng.random((37, 45)) > 0.15
    ref = st.local_contrast(np.where(ok, img, np.nan))
    c, defined = mx.local_contrast_t(torch.from_numpy(img)[None, None], torch.from_numpy(ok)[None, None])
    c, defined = c[0, 0].numpy(), defined[0, 0].numpy()
    assert np.array_equal(defined, np.isfinite(ref))
    assert np.allclose(c[defined], ref[defined], atol=1e-4) and np.all(c[~defined] == 0)


def test_exemplar_contrast_matches_numpy_and_the_full_frame():
    import stabilize as st
    rng = np.random.default_rng(1)
    frame = rng.normal(80, 2, (90, 120)).astype(np.float32)
    frame[40:43, 60:63] += 30
    frame[20:23, 30:33] -= 12
    rows = np.array([(0, 61.3, 41.0), (0, 31, 21), (0, 90, 70)])
    S = mx.Encoder().patch
    ex, exv = mx_data.extract_patches(frame[None], rows, S, 85.0799, 47.4845, 16)
    ce = mx.exemplar_contrast(torch.from_numpy(ex)[None])[0].numpy()
    ref = mx_data.exemplar_contrast(ex[:3, 0])
    assert np.allclose(ce[:3], ref, atol=1e-5)
    full = st.local_contrast((frame - 85.0799) / 47.4845)
    assert np.isclose(ce[0], np.nanmax(full[39:44, 59:64]), atol=1e-5)
    assert np.isclose(ce[1], np.nanmin(full[19:24, 29:34]), atol=1e-5)


def test_calibration_channels_match_numpy():
    rng = np.random.default_rng(2)
    c = rng.normal(0, 0.5, (2, 1, 9, 11)).astype(np.float32)
    defined = rng.random((2, 1, 9, 11)) > 0.1
    ce = rng.normal(0.3, 0.2, (2, 6)).astype(np.float32)
    cev = np.array([[1, 1, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0]], bool)
    args = [torch.from_numpy(v) for v in (c, defined, ce, cev)]
    for floor, kw in (('relative', dict(eps_min=mx.EPS_MIN)), ('absolute', dict(eps=mx.EPS_C))):
        out = mx.calibration_channels(*args, floor=floor).numpy()
        assert out.shape == (2, 4, 9, 11)
        ref = mx_data.calibrated_features(c[0, 0].ravel(), ce[0][cev[0]], floor, **kw).reshape(9, 11, 4)
        ref = np.moveaxis(ref, -1, 0) * defined[0]
        assert np.allclose(out[0], ref, atol=1e-5), floor
        assert np.all(out[1] == 0)                               # no valid exemplar: all zero


def _gain_pair(arch, g=1.8):
    """last_cal of a window with a parked target and exemplars of the same contrast, before
    and after a contrast gain g about the window mean (window and exemplars together)."""
    x, valid, pvalid, ex, ex_valid = _inputs(b=1, H=40, W=40)
    x = x * 0.2
    x[:, :, :, 18:21, 18:21] += 1.5                              # a parked target
    ex = ex * 0.2
    S = ex.shape[-1]
    ex[:, :8, :, S // 2 - 1:S // 2 + 2, S // 2 - 1:S // 2 + 2] += 1.5
    ex_valid[:, :8] = True
    model, _ = models.build(arch)
    model.eval()
    with torch.no_grad():
        model(x, valid, pvalid, ex, ex_valid)
        a = model.last_cal.clone()
        m = float(x.mean())
        model(m + g * (x - m), valid, pvalid, m + g * (ex - m), ex_valid)
        b = model.last_cal.clone()
    return a, b


def test_mxc_channels_are_exactly_gain_invariant():
    """mxc_lv (relative floor): R, |R|, D do not change under a gain, anywhere; P neither
    where the contrast is clearly signed."""
    for g in (0.5, 1.8):
        a, b = _gain_pair('mxc_lv', g)
        assert a.shape == (1, 4, 40, 40) and a.abs().sum() > 0
        assert torch.allclose(a[:, :3], b[:, :3], atol=1e-4), g
        centre = (slice(None), slice(3, 4), slice(16, 23), slice(16, 23))
        assert torch.equal(a[centre], b[centre])
    assert abs(float(a[0, 0, 19, 19])) < 0.3 and float(a[0, 3, 19, 19]) > 0.9


def test_mxc_absfloor_channels_are_only_approximately_invariant():
    """mxc_lv_absfloor (fixed floor of one grey level): close at the target, not exact."""
    a, b = _gain_pair('mxc_lv_absfloor')
    centre = (slice(None), slice(None), slice(18, 21), slice(18, 21))
    assert torch.allclose(a[centre], b[centre], atol=0.05)
    assert not torch.allclose(a[:, :3], b[:, :3], atol=1e-4)     # the fixed floor does not scale


def test_mxc_forward_backward_and_no_exemplars():
    x, valid, pvalid, ex, ex_valid = _inputs()
    ex_valid[0, :5] = True
    model, needs = models.build('mxc_lv')
    assert needs == mx.NEEDS and model.level.e1[0].in_channels == 10
    f, logit = model(x, valid, pvalid, ex, ex_valid)
    assert logit.shape == (2, 40, 48, 48) and torch.isfinite(logit).all()
    cal = model.last_cal
    assert cal[0].abs().sum() > 0 and torch.all(cal[1] == 0)    # sample 1 has no exemplar
    (logit.mean() + mx.match_loss(model, (torch.rand(2, 40, 48, 48) > 0.99).float(), pvalid)).backward()
    assert all(p.grad is not None for p in model.level.parameters())
    model(x, valid, pvalid, None, None)
    assert torch.all(model.last_cal == 0)
    base, _ = models.build('mx_lv')
    assert base.level.e1[0].in_channels == 6                    # mx_lv checkpoints still load


def test_gain_augment_keeps_the_relative_contrast():
    rng = np.random.default_rng(3)
    img = rng.normal(0, 1, (1, 4, 16, 16)).astype(np.float32)
    pv = np.ones(img.shape, bool)
    pv[:, :, :2] = False
    ex = rng.normal(0, 1, (5, 1, 39, 39)).astype(np.float32)
    exv = np.array([1, 1, 1, 0, 0], bool)
    out, eo, g = mx_data.gain_augment(np.random.default_rng(0), img, pv, ex, exv, 2.0, 1.0)
    assert 0.5 <= g <= 2.0 and g != 1.0
    m = img[pv].mean()
    assert np.allclose(out[pv], m + g * (img[pv] - m), atol=1e-5) and np.array_equal(out[~pv], img[~pv])
    assert np.allclose(eo[:3], m + g * (ex[:3] - m), atol=1e-5) and np.array_equal(eo[3:], ex[3:])
    same = mx_data.gain_augment(np.random.default_rng(0), img, pv, ex, exv, 2.0, 0.0)
    assert same[2] == 1.0 and same[0] is img


# ---------------------------------------------------------------- DeepPro-MTA: single-frame appearance
def test_mta_appearance_pathway():
    x, valid, pvalid, ex, ex_valid = _inputs()
    pvalid[1, :, :, :, :4] = False                               # an invalid strip in sample 1
    model, needs = models.build('mta_lv')
    assert needs == mx.NEEDS and getattr(model, 'appearance', False)
    f, logit = model(x, valid, pvalid, ex, ex_valid)
    la = model.last_la
    assert la.shape == logit.shape == (2, 40, 48, 48) and torch.isfinite(logit).all()
    assert torch.all(la[1, :, :, :4] == -30.0)                   # no appearance outside the field
    assert torch.all(logit >= la - 1e-4)                         # noisy-OR: never below a component
    lab = (torch.rand(2, 40, 48, 48) > 0.99).float()
    loss = logit.mean() + mx.appearance_loss(model, lab, pvalid) + mx.match_loss(model, lab, pvalid)
    loss.backward()
    assert all(p.grad is not None for p in model.app.parameters())
    # without exemplars the head still runs, as a plain single-frame detector
    model.zero_grad()
    _, logit2 = model(x, valid, pvalid, None, None)
    assert torch.isfinite(logit2).all() and torch.isfinite(model.last_la).all()
    # chunking the encoder does not change the similarity
    with torch.no_grad():
        model.eval()
        a = model.appearance_logits(x, pvalid, ex, ex_valid)
        model.app_chunk = 3
        b = model.appearance_logits(x, pvalid, ex, ex_valid)
    assert torch.allclose(a, b, atol=1e-5)


def test_mta_trains_and_infers_with_the_appearance_arm():
    import fake_dataset
    import infer_mx
    import splits
    import train
    import vtdata
    import export_masks
    splits.SPLIT_FILE = os.path.abspath(splits.SPLIT_FILE)
    old = vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS, os.getcwd(), sys.argv
    rng = np.random.default_rng(3)
    data = {name: _fake_clip(rng) for name in ('fakeA', 'fakeB')}
    with tempfile.TemporaryDirectory() as d:
        try:
            fake_dataset.write(d, {k: (v, b, 'train2017') for k, (v, b) in data.items()}, track_key='tracking_id')
            speed = {(k, t, int(row[5])): (1.0 if row[5] < 20 else 0.0)
                     for k, (_, b) in data.items() for t, bb in enumerate(b) for row in bb}
            fake_dataset.write_targets(os.path.join(d, 'targets.csv'), speed, {k: b for k, (_, b) in data.items()})
            os.makedirs(os.path.join(d, 'results/motion/homographies'))
            for k, (v, _) in data.items():
                np.save(os.path.join(d, f'results/motion/homographies/{k}.npy'),
                        np.repeat(np.eye(3)[None], len(v) - 1, 0))
            vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = d, None, d
            vtdata._ANN_CACHE.clear()
            for k in data:
                export_masks.export_clip((k, os.path.join(d, 'segmentations'), 6.0, 'floor', False))
            os.chdir(d)
            sys.argv = ['train.py', '--run', 'mta_val', '--arch', 'mta_lv', '--split', 'fakeA', '--epochs', '1',
                        '--batch', '2', '--sample_rate', '0.05', '--workers', '0', '--threads', '2',
                        '--ex-targets', 'targets.csv', '--checkpoint-selection', 'val_iou',
                        '--selection-split', 'fakeB']
            train.main()
            cfg = json.load(open('runs/mta_val/config.json'))
            assert cfg['arch'] == 'mta_lv' and cfg['gain_aug'] == 2.0 and cfg['app_aux'] == 1.0
            sel = json.load(open('runs/mta_val/best_selection.json'))
            assert sel['protocol']['arch'] == 'mta_lv' and sel['complete']
            infer_mx.main(['--run', 'mta_inf', '--ckpt', 'runs/mta_val/best_model.pth', '--clips', 'fakeB',
                           '--threads', '2', '--save-arms'])
            a = np.load('cache/preds/mta_inf__appearance/fakeB.npy')
            fused = np.load('cache/preds/mta_inf/fakeB.npy')
            assert a.shape == fused.shape == (90, 160, 200) and np.isfinite(fused).all()
        finally:
            vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = old[:3]
            os.chdir(old[3])
            sys.argv = old[4]
            vtdata._ANN_CACHE.clear()


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
            os.makedirs(os.path.join(d, 'results/motion/homographies'))
            for k, (v, _) in clips.items():                      # a static camera
                np.save(os.path.join(d, f'results/motion/homographies/{k}.npy'),
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
                        '--ex-targets', 'targets.csv',                   # no --label-rule: the default
                        '--checkpoint-selection', 'final']               # the fake root has no test list
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


def test_mxc_absfloor_checkpoint_selection():
    """The fixed-floor ablation completes real selection windows, including calibration."""
    from argparse import Namespace
    import checkpoint_selection
    import export_masks
    import fake_dataset
    import vtdata
    old = vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS, os.getcwd()
    video = np.full((90, 48, 48), 70, np.float32)
    video[:, 22:25, 22:25] = 110
    boxes = [np.array([(22, 22, 3, 3, 1, 0)], float) for _ in video]
    with tempfile.TemporaryDirectory() as d:
        try:
            fake_dataset.write(d, {'fakeB': (video, boxes, 'train2017')})
            vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = d, None, d
            vtdata._ANN_CACHE.clear()
            export_masks.export_clip(('fakeB', os.path.join(d, 'segmentations'), 6.0, 'floor', False))
            os.makedirs(os.path.join(d, 'results/motion/homographies'))
            np.save(os.path.join(d, 'results/motion/homographies/fakeB.npy'),
                    np.repeat(np.eye(3)[None], len(video) - 1, 0))
            os.chdir(d)
            model, needs = models.build('mxc_lv_absfloor')
            args = Namespace(arch='mxc_lv_absfloor', selection_bank='self')
            result = checkpoint_selection.evaluate_window_iou(model, needs, args, clips=['fakeB'], split='val')
            assert result['complete'] and result['windows'] == result['expected_windows'] == 3
            assert np.isfinite(result['iou']) and result['protocol']['arch'] == 'mxc_lv_absfloor'
            assert result['protocol']['exemplars']['bank'] == 'self'
            assert result['per_clip']['fakeB']['exemplars']['exemplars'] > 0
            assert torch.isfinite(model.last_cal).all() and model.last_cal.abs().sum() > 0
        finally:
            vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = old[:3]
            os.chdir(old[3])
            vtdata._ANN_CACHE.clear()


def test_mxc_training_with_checkpoint_selection():
    """mxc_lv trains with --gain-aug and both per-epoch selections; the selection scores MX
    windows with inference's exemplars (same candidates as infer_mx.py)."""
    import checkpoint_selection
    import export_masks
    import fake_dataset
    import infer_mx
    import splits
    import train
    import vtdata
    splits.SPLIT_FILE = os.path.abspath(splits.SPLIT_FILE)
    old = vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS, os.getcwd(), sys.argv
    rng = np.random.default_rng(1)
    clips = {'fakeA': 'train2017', 'fakeB': 'train2017', 'fakeC': 'test2017'}
    data = {name: _fake_clip(rng) for name in clips}
    with tempfile.TemporaryDirectory() as d:
        try:
            fake_dataset.write(d, {k: (v, b, clips[k]) for k, (v, b) in data.items()}, track_key='tracking_id')
            open(os.path.join(d, 'train.txt'), 'w').write('fakeA\nfakeB\n')
            open(os.path.join(d, 'test.txt'), 'w').write('fakeC\n')
            speed = {(k, t, int(row[5])): (1.0 if row[5] < 20 else 0.0)
                     for k, (_, b) in data.items() for t, bb in enumerate(b) for row in bb}
            fake_dataset.write_targets(os.path.join(d, 'targets.csv'), speed, {k: b for k, (_, b) in data.items()})
            os.makedirs(os.path.join(d, 'results/motion/homographies'))
            for k, (v, _) in data.items():
                np.save(os.path.join(d, f'results/motion/homographies/{k}.npy'),
                        np.repeat(np.eye(3)[None], len(v) - 1, 0))
            vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = d, None, d
            vtdata._ANN_CACHE.clear()
            for k in data:
                export_masks.export_clip((k, os.path.join(d, 'segmentations'), 6.0, 'floor', False))
            os.chdir(d)
            needs = models.build('mxc_lv')[1]
            ds = train.ExemplarTrainSet(['fakeA'], 'targets.csv', ex_k=16, ex_patch=39, rates=[1], coords='auto',
                                        needs=needs, seed=0, sample_rate=0.05, rule='paper_k6_floor',
                                        gain=2.0, gain_p=1.0)
            gains = [float(ds[i]['gain']) for i in range(6)]
            assert all(0.5 <= g <= 2.0 and g != 1.0 for g in gains)
            common = ['--arch', 'mxc_lv', '--split', 'fakeA', '--batch', '2', '--sample_rate', '0.05', '--workers', '0',
                      '--threads', '2', '--ex-targets', 'targets.csv']
            sys.argv = ['train.py', '--run', 'mxc_val', '--epochs', '2', '--checkpoint-selection', 'val_iou',
                        '--selection-split', 'fakeB'] + common
            train.main()
            cfg = json.load(open('runs/mxc_val/config.json'))
            assert cfg['gain_aug'] == 2.0 and cfg['arch'] == 'mxc_lv' and cfg['coords'] == 'auto'
            log = [json.loads(line) for line in open('runs/mxc_val/eval_log.jsonl')]
            assert len(log) == 2 and all(r['split'] == 'fakeB' and r['complete'] for r in log)
            assert log[0]['protocol']['name'] == 'window_global_iou' and log[0]['protocol']['clips'] == ['fakeB']
            assert log[0]['windows'] == log[0]['expected_windows'] == 3
            assert log[0]['per_clip']['fakeB']['coordinates']['raw'] == 3          # static camera: auto -> raw
            best = torch.load('runs/mxc_val/best_model.pth', map_location='cpu', weights_only=False)
            assert best['selection_mode'] == 'val_iou' and best['best_epoch'] in (1, 2)
            assert 'gain: changed' in open('runs/mxc_val/train_log.txt').read()
            # the selection's exemplar candidates are infer_mx.py's for the same weights
            sys.argv = ['train.py', '--run', 'mxc_test', '--epochs', '1', '--checkpoint-selection',
                        'original_test_iou'] + common
            train.main()
            sel = json.load(open('runs/mxc_test/best_selection.json'))
            assert sel['protocol']['name'] == 'official_test_window_global_iou' and sel['test_clips'] == ['fakeC']
            assert sel['protocol']['exemplars']['bank'] == 'outside' and sel['windows'] == 3
            infer_mx.main(['--run', 'mxc_test_inf', '--ckpt', 'runs/mxc_test/best_model.pth', '--clips', 'fakeC',
                           '--threads', '2'])
            meta = json.load(open('cache/preds/mxc_test_inf__meta/fakeC.json'))
            assert meta['settings']['arch'] == 'mxc_lv'
            assert meta['summary']['candidates'] == sel['per_clip']['fakeC']['exemplars']['candidates']
            assert sum(w['exemplars'] for w in meta['windows']) == sel['per_clip']['fakeC']['exemplars']['exemplars']
            # predictions are bound to the checkpoint: the same weights are reused, new ones recomputed
            stamp = os.path.getmtime('cache/preds/mxc_test_inf__meta/fakeC.json')
            infer_mx.main(['--run', 'mxc_test_inf', '--ckpt', 'runs/mxc_test/best_model.pth', '--clips', 'fakeC',
                           '--threads', '2'])
            assert os.path.getmtime('cache/preds/mxc_test_inf__meta/fakeC.json') == stamp
            import shutil
            shutil.copy('runs/mxc_val/best_model.pth', 'runs/mxc_test/best_model.pth')   # a new best, same path
            infer_mx.main(['--run', 'mxc_test_inf', '--ckpt', 'runs/mxc_test/best_model.pth', '--clips', 'fakeC',
                           '--threads', '2'])
            new = json.load(open('cache/preds/mxc_test_inf__meta/fakeC.json'))
            assert new['fingerprint']['ckpt_sha256'] != meta['fingerprint']['ckpt_sha256']
            # the no-exemplar control sees no exemplars in selection or (by default) in inference
            sys.argv = ['train.py', '--run', 'mxc_noex', '--epochs', '1', '--checkpoint-selection', 'val_iou',
                        '--selection-split', 'fakeB', '--ex-p-none', '1'] + common
            train.main()
            sel = json.load(open('runs/mxc_noex/best_selection.json'))
            assert sel['protocol']['exemplars']['bank'] == 'none'
            assert sel['per_clip']['fakeB']['exemplars'] == dict(candidates=0, exemplars=0, windows_without_exemplars=3)
            infer_mx.main(['--run', 'mxc_noex_inf', '--ckpt', 'runs/mxc_noex/best_model.pth', '--clips', 'fakeB',
                           '--threads', '2'])
            meta = json.load(open('cache/preds/mxc_noex_inf__meta/fakeB.json'))
            assert meta['settings']['bank'] == 'none' and meta['summary']['mean_exemplars'] == 0
            # the no-motion control keeps its own bank (the clip's strongest blobs) in all three stages
            sys.argv = ['train.py', '--run', 'mxc_self', '--epochs', '1', '--checkpoint-selection', 'val_iou',
                        '--selection-split', 'fakeB', '--ex-source', 'self'] + common
            train.main()
            sel = json.load(open('runs/mxc_self/best_selection.json'))
            assert sel['protocol']['exemplars']['bank'] == 'self'
            assert sel['per_clip']['fakeB']['exemplars']['exemplars'] > 0
            assert sel['per_clip']['fakeB']['exemplars']['candidates'] == 0          # no pass 1
            infer_mx.main(['--run', 'mxc_self_inf', '--ckpt', 'runs/mxc_self/best_model.pth', '--clips', 'fakeB',
                           '--threads', '2'])
            meta = json.load(open('cache/preds/mxc_self_inf__meta/fakeB.json'))
            assert meta['settings']['bank'] == 'self' and meta['summary']['mean_exemplars'] > 0
            # val_iou refuses to select on training clips
            sys.argv = ['train.py', '--run', 'mxc_bad', '--epochs', '1', '--checkpoint-selection', 'val_iou',
                        '--selection-split', 'fakeA'] + common
            try:
                train.main()
                raise AssertionError('val_iou on a training clip must fail')
            except SystemExit as e:
                assert 'overlaps the training clips' in str(e)
            assert set(checkpoint_selection.SUPPORTED['mxc_lv']) == set(needs)
        finally:
            vtdata.DATA_ROOT, vtdata.ANN_DIR, vtdata.OFFICIAL_LISTS = old[:3]
            os.chdir(old[3])
            sys.argv = old[4]
            vtdata._ANN_CACHE.clear()


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
