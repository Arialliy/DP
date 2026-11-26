"""End-to-end run of diagnose_history.process on the synthetic panning scene.

Run: python code/test_diagnose_history.py
"""
import os
import sys
from argparse import Namespace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import diagnose_history as dh  # noqa: E402
import stabilize as st  # noqa: E402
import test_stabilize as ts  # noqa: E402
import vtdata  # noqa: E402


class FakeScene:
    def __init__(self, video, boxes):
        self.name, self.boxes, self._v = 'FAKE_0001_1', boxes, video
        self.H, self.W = video.shape[1:]

    def __len__(self):
        return len(self._v)


def test_process():
    video, boxes, pairs = ts.scene()
    # add a second hot spot so the negatives are not a single location
    video = video.copy()
    for t in range(len(video)):
        x = int(round(260 - ts.PAN * t))
        if 3 <= x < ts.W - 3:
            video[t, 20:23, x:x + 3] += 50
    sc = FakeScene(video, boxes)
    vtdata.Scene = lambda name: sc
    st.load_video_u8 = lambda clip, scene=None: video
    st.load_chain = lambda clip, n=None: st.chain_from_pairs(pairs)
    speed = {(sc.name, t, 7): 0.0 for t, b in enumerate(boxes) if len(b)}
    args = Namespace(seqlen=40, hist_len=60, hist_step=2, mode='both', min_hist=3)
    clip, n, pos, neg = dh.process((sc.name, args, speed))
    s = dh.summarize(pos, neg, len(st.windows(n)))
    assert s['near_stationary'] > 20, s
    assert s['near_categories'] == {'arrives_or_moved_before': s['near_stationary']}, s
    assert s['auc_D'] > s['auc_L'], s
    assert s['auc_D'] > 0.9, s
    print(s)


if __name__ == '__main__':
    test_process()
    print('ok test_process')
