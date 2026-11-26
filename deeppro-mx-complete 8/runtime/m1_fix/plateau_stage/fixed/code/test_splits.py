"""Checks of splits.selection_plan, how train.py chooses best_model.pth (numpy only).

Run: python code/test_splits.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splits  # noqa: E402


def _raises(f, text):
    try:
        f()
    except ValueError as e:
        assert text in str(e), e
        return
    raise AssertionError('no error')


def test_auto_follows_the_authors_on_the_official_split():
    train45 = splits.clips('official_train')
    assert splits.selection_plan('official_train', 'auto', None, train45) == ('original_test_iou', 'val', None)


def test_auto_selects_on_validation_clips_for_research_splits():
    mode, sel, chosen = splits.selection_plan('train', 'auto', None, splits.clips('train'))
    assert (mode, sel, chosen) == ('val_iou', 'val', splits.clips('val'))
    if os.path.exists(splits.FOLD_FILE):
        mode, sel, chosen = splits.selection_plan('fold2/train', 'auto', None, splits.clips('fold2/train'))
        assert (mode, sel) == ('val_iou', 'fold2/val') and chosen == splits.clips('fold2/val')


def test_selection_never_sees_training_or_test_clips():
    tr = splits.clips('train')
    _raises(lambda: splits.selection_plan('train', 'val_iou', 'train', tr), 'overlaps the training clips')
    _raises(lambda: splits.selection_plan('train', 'val_iou', 'test', tr), 'must not select on test clips')
    _raises(lambda: splits.selection_plan('all', 'auto', None, splits.clips('all')), 'include test clips')
    assert splits.selection_plan('train', 'final', None, tr) == ('final', 'val', None)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
