"""Summarize new profile and existing M1 window diagnostics without model/image work."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/motion_review'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    window_path = ROOT / 'results/miss_recovery_oracle_location/official_rgbt/test/windows.csv'
    motion_path = ROOT / 'results/motion/clips.csv'
    geometry_path = OUT / 'geometry_summary.json'
    source = list(csv.DictReader(window_path.open()))
    geometry = {(r['clip'], int(r['window'])): r
                for r in csv.DictReader((OUT / 'window_geometry.csv').open())}
    choices, counts, by_clip = [], collections.Counter(), collections.defaultdict(collections.Counter)
    max_cover_difference = 0.0
    low_cover_agrees = True
    for row in source:
        cover = float(row['cover'])
        raw, warp = float(row['stationarity_raw']), float(row['stationarity_warp'])
        finite = math.isfinite(raw) and math.isfinite(warp)
        if cover < .1:
            choice, reason = 'raw', 'cover_below_0.1'
        elif not finite:
            choice, reason = 'raw', 'nonfinite_residual'
        elif raw - warp > max(.1 * raw, .5):
            choice, reason = 'warp', 'residual_reduction_exceeds_margin'
        else:
            choice, reason = 'raw', 'residual_reduction_insufficient'
        projected_cover = float(geometry[row['clip'], int(row['window'])]['lowres_cover'])
        max_cover_difference = max(max_cover_difference, abs(cover - projected_cover))
        low_cover_agrees &= (cover < .1) == (projected_cover < .1)
        counts[choice] += 1
        by_clip[row['clip']][choice] += 1
        choices.append(dict(clip=row['clip'], window=int(row['window']), first=int(row['first']),
                            last=int(row['last']), cover=cover, raw=raw if math.isfinite(raw) else None,
                            warp=warp if math.isfinite(warp) else None, default_auto_choice=choice,
                            reason=reason))
    with (OUT / 'test_default_auto.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(choices[0]))
        writer.writeheader()
        writer.writerows(choices)
    motion = list(csv.DictReader(motion_path.open()))
    geom = json.loads(geometry_path.read_text())
    test = set(json.loads((ROOT / 'configs/splits.json').read_text())['test'])
    selected = ['DJI_0277_2', 'DJI_0237_2', 'DJI_0221_2', 'DJI_0257_2', 'DJI_0163_1', 'DJI_0309_2']
    significant = []
    for clip in selected:
        r = next(r for r in motion if r['scene'] == clip)
        significant.append(dict(clip=clip, test=clip in test,
                                fallback_pairs=sum(int(r['pairs_' + m]) for m in ['affine', 'shift', 'identity']),
                                pair_diff_p95=float(r['pair_diff_p95']),
                                window_diff_p95=float(r['window_diff_p95']),
                                window_diff_max=float(r['window_diff_max'])))
    result = dict(
        scope='New v8 profile only. No profile rerun, no old homographies reused, no model execution.',
        sources={str(p.relative_to(ROOT)): sha(p) for p in [window_path, motion_path, geometry_path]},
        pair_methods=geom['pair_methods'], pairs=geom['frame_pairs'], clips=geom['clips'],
        selected_clip_disagreements=significant,
        default_auto=dict(scope='Default choose_coords(auto, rate=1) evaluated on already-computed M1 stationarity; not a learned-model inference result',
                          margin=.1, abs_tol=.5, min_cover=.1, windows=len(choices),
                          counts=dict(counts), per_clip={k: dict(v) for k, v in by_clip.items()},
                          low_cover_windows=[r for r in choices if r['cover'] < .1]),
        geometry_crosscheck=dict(max_absolute_cover_difference=max_cover_difference,
                                 low_cover_classification_agrees_on_all_test_windows=low_cover_agrees,
                                 authoritative_cover='Measured M1 cover is used for the default-auto reconstruction; independent host NumPy geometry differs slightly near pixel boundaries.'),
        limitations=[
            'Pair/window differences compare annotation-free and box-masked estimates, not registration ground truth.',
            'Inlier residuals are not saved per pair/window. The <=1 px gate applies to homographies; affine and shift fallbacks only have their separate consensus checks.',
            'Across all 720 windows, 34 reference-to-frame comparisons have no common grid visibility: 25 in the last two 0277_2 windows and 9 in 0309_2 window 504..543. Source compare() records zero for an empty comparison.',
            'The two 0277_2 windows with zero stationarity coverage correctly select raw. The large 0237_2 residual at window 324..363 also selects raw.',
            'Background stationarity and disagreement do not by themselves identify the cause of baseline misses.',
            'M1 computes its candidate features from raw frames; the homography chain is used for these window diagnostics, not the diagnostic recovery scores.'
        ])
    (OUT / 'review.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    notes = '''新版 profile 审查（只读主代码和新结果；未重跑 profile、未替换旧 homography）

60 个 clip、25,051 个帧对均完成。模型计数：homography 24,956、partial affine 6、shift 1、identity 88；共 95 对 fallback（0.379%）。

clips.csv 的差异是与遮框估计的共同视野分歧，不是相对真值的配准误差。全体 window_diff_p95 中位数为 3.534 px。较大的分歧包括：0163_1 p95=201.391/max=477.000、0309_2 p95=174.231/max=636.425；test 中 0237_2 为 87.103/181.175、0221_2 为31.390/76.615、0257_2 为26.917/60.084 px。单凭这些差异不能确定哪一估计更准确。

0277_2：第334个帧对退为 identity，window_diff_p95=13.857/max=14.218 px，旧的视野外极端值已排除。但最后两窗[324,363]、[326,365]分别有13、12个参考帧到其他帧的比较没有共同视野，不能把较小的报告差异理解为全窗可靠。全60个clip共有34个这样的空比较，源码将空比较记为0。

用已完成 M1 的真实 stationarity 指标代入默认 auto 规则，172个 test 窗会选97 warp、75 raw。0277_2最后两窗cover=0，raw/warp残差均不可计算，正确回raw；其他9窗选warp。0237_2窗[324,363]的raw=19.775、warp=37.026、cover=0.9788，因此也正确拒绝补偿。上述是默认规则复核，不是训练后网络的推论结果。

0255_1首窗raw=1.938、warp=1.490，改善未超过0.5而选raw；0307_2首窗16.812降至9.688而选warp。这些背景指标不能单独证明片头漏检成因。

实现限制：clips.csv没有保存逐对内点残差；≤1 px只约束接受的homography，affine fallback不继承该残差门槛。矩阵独立覆盖率复核与M1测量的最大绝对差为0.001758，所有低覆盖分类一致；最终auto统计使用M1实际测量值。M1的恢复分数基于原始帧特征，homography仅用于这里的窗口诊断。

详细数值和来源SHA-256见review.json、geometry_summary.json；各窗口见test_default_auto.csv、window_geometry.csv。
'''
    (OUT / 'notes.txt').write_text(notes)
    print(json.dumps(dict(pair_methods=result['pair_methods'], auto_counts=dict(counts),
                          low_cover_agrees=low_cover_agrees, max_cover_difference=max_cover_difference,
                          output=str(OUT / 'review.json')), indent=2))


if __name__ == '__main__':
    main()
