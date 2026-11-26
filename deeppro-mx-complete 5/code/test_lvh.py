"""Checks of the LVH network and losses (needs torch; run on the training machine).

Run: python code/test_lvh.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import lv  # noqa: E402
import lvh  # noqa: E402
import models  # noqa: E402


def _batch(b=2, T=40, H=32, W=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 1, T, H, W, generator=g)
    pvalid = torch.ones_like(x, dtype=torch.bool)
    pvalid[..., :, :3] = False                     # a warped border
    hist = torch.randn(b, 1, H, W, generator=g)
    hvalid = torch.ones(b, 1, H, W, dtype=torch.bool)
    hvalid[:, :, :, -4:] = False
    valid = torch.ones(b, T, dtype=torch.bool)
    valid[0, :5] = False                           # padded short window
    return x, valid, pvalid, hist, hvalid


def test_masked_decomposition_matches_lv():
    x = torch.randn(2, 1, 40, 8, 8)
    L0, V0 = lv.decompose(x)
    L1, V1, count = lvh.decompose_masked(x, torch.ones_like(x, dtype=torch.bool))
    assert torch.allclose(L0, L1) and torch.allclose(V0, V1) and (count == 40).all()


def test_masked_decomposition_ignores_invalid():
    x = torch.zeros(1, 1, 10, 1, 1)
    x[0, 0, :7] = 5.0                              # invalid frames hold garbage
    m = torch.zeros_like(x, dtype=torch.bool)
    m[0, 0, 7:] = True
    L, V, count = lvh.decompose_masked(x, m)
    assert L.item() == 0.0 and V[0, 0, :7].abs().sum() == 0 and count.item() == 3


def test_variants_forward_backward():
    for name in ('lvh_full', 'lvh_stab', 'lvh_pointgate', 'lvh_nogate', 'lvh_dec'):
        model, needs = models.build(name)
        assert set(needs) == set(lvh.NEEDS), needs
        x, valid, pvalid, hist, hvalid = _batch()
        _, logit = model(x, valid=valid, pvalid=pvalid, hist=hist, hvalid=hvalid)
        assert logit.shape == (2, 40, 32, 32) and torch.isfinite(logit).all(), name
        assert model.last_ll.shape == (2, 1, 32, 32)
        (logit.mean() + model.last_ll.mean()).backward()


def test_history_changes_level_input():
    torch.manual_seed(0)
    model, _ = models.build('lvh_full')
    model.eval()
    x, valid, pvalid, hist, hvalid = _batch()
    with torch.no_grad():
        model(x, valid=valid, pvalid=pvalid, hist=hist, hvalid=hvalid)
        a = model.last['ll'].clone()
        model(x, valid=valid, pvalid=pvalid, hist=hist + 3.0, hvalid=hvalid)
        b = model.last['ll']
    assert (a - b).abs().max() > 0


def test_focal_has_gradient_without_targets():
    logit = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.zeros(2, 1, 16, 16)
    lvh.soft_iou_loss(logit, target).backward()
    assert logit.grad.abs().max() == 0             # Soft-IoU is blind on empty crops
    logit.grad = None
    lvh.focal_bce(logit, target).backward()
    assert logit.grad.abs().max() > 0


def test_persistent_target_ramp():
    lab = torch.zeros(1, 10, 1, 3)
    lab[0, :2, 0, 0] = 1    # 20 %
    lab[0, :5, 0, 1] = 1    # 50 %
    lab[0, :8, 0, 2] = 1    # 80 %
    v = torch.ones(1, 10, dtype=torch.bool)
    r = lvh.persistent_target(lab, v, 'ramp')[0, 0]
    h = lvh.persistent_target(lab, v, 'hard')[0, 0]
    assert torch.allclose(r, torch.tensor([0.0, 0.5, 1.0])), r
    assert torch.equal(h, torch.tensor([0.0, 0.0, 1.0])), h


def test_stabilized_inference_runs():
    import numpy as np
    import infer
    import stabilize
    model, _ = models.build('lvh_full')
    model.eval()
    n, H, W = 50, 32, 32
    video = np.random.default_rng(0).normal(85, 10, size=(n, H, W)).astype(np.float32)
    pairs = np.repeat(stabilize.translation(-0.5, 0.0)[None], n - 1, 0)
    chain = stabilize.chain_from_pairs(pairs)
    fused, arms = infer.predict_clip_stabilized(model, video, chain, 40, 20, 2, 'both', arms=True)
    assert fused.shape == (n, H, W) and np.isfinite(fused).all()
    assert set(arms) == {'variation', 'level', 'gated_level'}
    assert all(a.shape == (n, H, W) and np.isfinite(a).all() for a in arms.values())
    dec, _ = models.build('lvh_dec')
    dec.eval()
    for gate in (0.0, 1.0):   # the 0.5 px/frame pan exceeds 1 px, so both paths warp; 50 px gates
        fused, arms = infer.predict_clip_stabilized(dec, video, chain, 40, 20, 2, 'both', arms=True,
                                                    motion_gate=gate)
        assert set(arms) == {'variation', 'level', 'gated_level', 'appearance', 'arrival'}
        assert np.isfinite(fused).all()
    fused_gated, _ = infer.predict_clip_stabilized(dec, video, chain, 40, 20, 2, 'both', motion_gate=50.0)
    fused_plain, _ = infer.predict_clip_stabilized(dec, video, stabilize.chain_from_pairs(
        np.repeat(np.eye(3)[None], n - 1, 0)), 40, 20, 2, 'both')
    assert np.allclose(fused_gated, fused_plain, atol=1e-4)  # gated window == identity chain


def test_mr_lv_equals_lv_full_on_recorded_frames_up_to_cover():
    """mr_lv on all-valid windows sees [L, E, cover=1]; its decomposition matches lv_full."""
    model, needs = models.build('mr_lv')
    assert set(needs) == {'valid', 'pvalid'}
    x, valid, pvalid, _, _ = _batch()
    _, logit = model(x, valid=valid, pvalid=pvalid)
    assert logit.shape == (2, 40, 32, 32) and torch.isfinite(logit).all()
    (logit.mean() + model.last_ll.mean()).backward()
    assert model.level.e1[0].weight.shape[1] == 3 and isinstance(model.gate, lv.Gate)
    L0, V0 = lv.decompose(x)
    L1, V1, _ = lvh.decompose_masked(x, torch.ones_like(x, dtype=torch.bool))
    assert torch.allclose(L0, L1) and torch.allclose(V0, V1)


def test_multirate_inference_with_real_networks():
    import numpy as np
    import tempfile
    from argparse import Namespace
    import infer_mr
    import stabilize
    n, H, W = 170, 32, 48
    rng = np.random.default_rng(0)
    video = rng.normal(85, 10, size=(n, H, W)).astype(np.float32)
    # 0.05 px/frame: 8 px over a rate-4 window, so compensated windows keep most of the view
    chain = stabilize.chain_from_pairs(np.repeat(stabilize.translation(-0.05, 0.0)[None], n - 1, 0))
    cases = (('mr_lv', 'auto', False), ('mr_lv', 'warp', False), ('lv_full', 'raw', False), ('lvh_dec', 'raw', False),
             ('lvh_dec', 'warp', True))
    for arch, coords, no_hist in cases:
        model, needs = models.build(arch)
        model.eval()
        score = infer_mr.make_scorer(model, needs, 'cpu', video, chain if coords != 'raw' else
                                     infer_mr.identity_chain(n),
                                     dict(hist_len=20, hist_step=2, hist_mode='both'), no_hist)
        args = Namespace(rates=[1, 4], seqlen=40, coords=coords, lowres=4, margin=0.1, abs_tol=0.5,
                         min_cover=0.1, fusion={'max', 'cal'}, cal_m=10, cal_half=18)
        with tempfile.TemporaryDirectory() as tmp:
            dirs = infer_mr.outputs(f'{tmp}/run', args.rates, args.fusion)
            for d in dirs.values():
                os.makedirs(d)
            meta = infer_mr.process_clip('FAKE', video, chain, score, args, dirs)
            for key, d in dirs.items():
                a = np.load(f'{d}/FAKE.npy')
                assert a.shape == (n, H, W) and np.isfinite(a).all(), (arch, key)
        assert meta['rates']['4']['used'], arch


def test_dec_heads_and_losses():
    model, _ = models.build('lvh_dec')
    assert getattr(model, 'decoupled', False)
    x, valid, pvalid, hist, hvalid = _batch()
    _, logit = model(x, valid=valid, pvalid=pvalid, hist=hist, hvalid=hvalid)
    la, lr = model.last_la, model.last_lr
    assert la.shape == lr.shape == (2, 1, 32, 32) and set(model.last) >= {'la', 'lr', 'll', 'lq', 'lv'}
    pm = torch.zeros(2, 32, 32)
    pm[:, 10:13, 10:13] = 1
    hlab = torch.zeros(2, 1, 32, 32)
    hlab[1, :, 10:13, 10:13] = 0.9                 # sample 1: the target is resident
    tgt, w = lvh.arrival_target(pm, hlab, hvalid)
    loss = (lvh.soft_iou_loss(la, pm[:, None]) + lvh.focal_bce(la, pm[:, None])
            + lvh.masked_soft_iou_loss(lr, tgt[:, None], w[:, None])
            + lvh.masked_focal_bce(lr, tgt[:, None], w[:, None]) + lvh.soft_iou_loss(logit, (x[:, 0] > 2).float()))
    loss.backward()
    assert model.appear.head.weight.grad.abs().sum() > 0
    assert model.arrive.head.weight.grad.abs().sum() > 0


def test_arrival_target_ignores_resident_targets():
    pm = torch.tensor([[[1.0, 1.0, 1.0, 0.0]]])        # (1,1,4) as (b,H,W)
    hlab = torch.tensor([[[[0.1, 0.9, 0.1, 0.0]]]])
    hvalid = torch.tensor([[[[True, True, False, True]]]])
    t, w = lvh.arrival_target(pm, hlab, hvalid)
    assert torch.equal(t[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0])), t
    assert torch.equal(w[0, 0], torch.tensor([1.0, 0.0, 0.0, 1.0])), w


def test_noisy_or_heads_do_not_veto_each_other():
    la, lr = torch.tensor([-10.0, 10.0]), torch.tensor([10.0, -10.0])
    q = lv.noisy_or_logit(la, torch.nn.functional.logsigmoid(lr))
    assert (q > 9).all(), q


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
