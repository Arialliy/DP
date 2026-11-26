"""CPU tests for native model identity, frame fusion and the three MR outputs."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import inference as api
import infer as native
import infer_mr


class Dummy(torch.nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self._evaluation_arch = arch
        self.eval()


def identity(n):
    return np.broadcast_to(np.eye(3), (n, 3, 3)).copy()


def sigmoid(array):
    return torch.sigmoid(torch.from_numpy(array)).numpy()


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_plus_strict_load_and_forward(self):
        original, path = api._official_module()
        reference = original.detector(1, 40, 40).eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'best_model.pth'
            torch.save({'model_state_dict': reference.state_dict()}, checkpoint)
            loaded, needs = api.load_model('deeppro_plus', checkpoint, 'cpu')
            self.assertEqual(needs, ())
            self.assertIn('_original_deeppro_plus_unified_eval', type(loaded).__module__)
            self.assertEqual(loaded._evaluation_source, path)
            x = torch.randn(1, 1, 40, 8, 8)
            with torch.no_grad():
                self.assertTrue(torch.equal(loaded(x)[1], reference(x)[1]))
            broken = dict(reference.state_dict())
            broken.pop(next(iter(broken)))
            torch.save({'model_state_dict': broken}, checkpoint)
            with self.assertRaises(RuntimeError):
                api.load_model('deeppro_plus', checkpoint, 'cpu')

    def test_research_checkpoint_wrapper_and_fp32_enforcement(self):
        model, expected_needs = api.models.build('lv_full')
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'best_model.pth'
            torch.save({'model': model.state_dict(), 'epoch': 3}, checkpoint)
            loaded, needs = api.load_model('lv_full', checkpoint, 'cpu')
            self.assertEqual(tuple(expected_needs), needs)
            for key, value in loaded.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]))
            half = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
            torch.save(half, checkpoint)
            with self.assertRaises(TypeError):
                api.load_model('lv_full', checkpoint, 'cpu')

    def test_raw_native_logits_max_and_fp32_storage(self):
        video = np.zeros((90, 8, 8), np.float32)
        expected = np.full_like(video, -np.inf)

        def score(frames, coords):
            self.assertEqual(coords, 'raw')
            value = np.float32(-0.00013 + int(frames[0]) / 1000)
            return np.full((len(frames), 8, 8), value, np.float32)

        for _, frames in api.multirate.plan_windows(90, 1):
            expected[frames] = np.maximum(expected[frames], score(frames, 'raw'))
        with patch.object(api, 'make_scorer', return_value=score):
            result = list(api.predict_clip(Dummy('lv_full'), ('valid',), video, 'mock', {}, 'cpu'))
        self.assertEqual([name for name, _ in result], ['native'])
        self.assertEqual(result[0][1].dtype, np.float32)
        np.testing.assert_array_equal(result[0][1], sigmoid(expected))
        self.assertFalse(np.array_equal(result[0][1], result[0][1].astype(np.float16).astype(np.float32)))

    def test_lv_matches_native_full_clip_inference(self):
        model, needs = api.models.build('lv_full')
        model.eval()
        model._evaluation_arch = 'lv_full'
        video = np.random.default_rng(46).uniform(0, 255, (41, 8, 8)).astype(np.float32)
        expected = sigmoid(native.predict_clip(model, video))
        actual = next(api.predict_clip(model, needs, video, 'mock', {}, 'cpu'))[1]
        np.testing.assert_array_equal(actual, expected)

    def test_lvh_and_dec_match_native_history_and_gate(self):
        video = np.random.default_rng(46).uniform(0, 255, (41, 8, 8)).astype(np.float32)
        chain = identity(len(video))
        for arch, gate in [('lvh_full', 0.0), ('lvh_dec', 1.0)]:
            model, needs = api.models.build(arch)
            model.eval()
            model._evaluation_arch = arch
            expected_logits, _ = native.predict_clip_stabilized(model, video, chain, motion_gate=gate)
            with patch.object(api.stabilize, 'load_chain', return_value=chain):
                actual = next(api.predict_clip(model, needs, video, 'mock', {'motion_gate': gate}, 'cpu'))[1]
            np.testing.assert_array_equal(actual, sigmoid(expected_logits), err_msg=arch)

    def test_gate_considers_outside_history_and_no_fake_history(self):
        video = np.zeros((90, 8, 8), np.float32)
        chain = identity(90)
        captured = []

        def gate(full, frames, ref, threshold, shape):
            captured.append(list(frames))
            self.assertEqual(threshold, 1.0)
            return full, True

        def factory(model, needs, device, vid, effective, hist_cfg, no_history):
            self.assertFalse(no_history)
            self.assertEqual(hist_cfg, dict(hist_len=80, hist_step=4, hist_mode='both'))
            return lambda frames, coords: np.zeros((len(frames), 8, 8), np.float32)

        with patch.object(api.stabilize, 'load_chain', return_value=chain), \
                patch.object(api.stabilize, 'gate_chain', side_effect=gate), \
                patch.object(api, 'make_scorer', side_effect=factory):
            list(api.predict_clip(Dummy('lvh_dec'), ('valid', 'pvalid', 'hist', 'hvalid'),
                                  video, 'mock', {}, 'cpu'))
        self.assertTrue(any(frame >= 40 for frame in captured[0]))

    def test_mr_fixed_variants_and_native_cfar(self):
        video = np.zeros((90, 8, 8), np.float32)
        chain = identity(90)

        def score(frames, coords):
            rate = int(frames[1] - frames[0])
            base = np.arange(64, dtype=np.float32).reshape(8, 8) / 64
            return np.stack([base + np.float32(rate + t / 1000) for t in frames])

        decide = lambda frames: ('raw', {})
        expected_rates = []
        expected_cal = []
        for rate in (1, 2):
            logits, _ = api.multirate.assemble(90, (8, 8), api.multirate.plan_windows(90, rate), decide, score)
            ref = api.multirate.cfar_reference([api.multirate.peak_scores(frame) for frame in logits], 10, 18)
            expected_rates.append(logits)
            expected_cal.append(logits - ref[:, None, None])
        with patch.object(api.stabilize, 'load_chain', return_value=chain), \
                patch.object(api, 'make_scorer', return_value=score), \
                patch.object(api, 'make_decider', return_value=decide):
            output = dict(api.predict_clip(Dummy('mr_lv'), ('valid', 'pvalid'), video, 'mock', {}, 'cpu'))
        self.assertEqual(list(output), ['rate1_auto', 'multirate_max', 'multirate_cfar_score'])
        np.testing.assert_array_equal(output['rate1_auto'], sigmoid(expected_rates[0]))
        np.testing.assert_array_equal(output['multirate_max'], sigmoid(np.maximum(*expected_rates)))
        np.testing.assert_array_equal(output['multirate_cfar_score'], sigmoid(np.maximum(*expected_cal)))
        for value in output.values():
            self.assertEqual(value.shape, video.shape)
            self.assertEqual(value.dtype, np.float32)
            self.assertTrue(np.isfinite(value).all())
        protocol = api.prediction_protocol('mr_lv')
        self.assertEqual(protocol['primary_variant'], 'multirate_max')
        self.assertIn('NOT a calibrated probability', protocol['variants']['multirate_cfar_score'])

    def test_invalid_input_is_rejected_before_scoring(self):
        for video in (np.zeros((40, 8, 8), np.float16), np.zeros((39, 8, 8), np.float32)):
            with self.assertRaises((TypeError, ValueError)):
                next(api.predict_clip(Dummy('lv_full'), ('valid',), video, 'mock', {}, 'cpu'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
