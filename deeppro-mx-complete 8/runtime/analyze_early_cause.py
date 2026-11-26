#!/usr/bin/env python3
"""Descriptive analysis of existing M1 records only. No image/model/torch inference."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/miss_recovery/official_rgbt/test'
OUT = ROOT/'results/early_cause'
EARLY = 60


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def save_csv(name, rows):
    with (OUT/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: '' if value is None else value for key, value in row.items()})


def safe(value):
    if isinstance(value, dict): return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(values)) if values else None


def median(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.median(values)) if values else None


def divide(a, b):
    return a/b if b else None


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rank(values):
    values = np.asarray(values)
    result = np.empty(len(values), float)
    order = np.argsort(values, kind='stable')
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start+end-1)/2 + 1
        start = end
    return result


def owner_indices(n, windows):
    owner, best = np.full(n, -1, int), np.full(n, np.inf)
    for row in windows:
        a, b = row['first'], row['last']+1
        distances = abs(np.arange(a, b)-(a+b-1)/2)
        use = distances < best[a:b]
        owner[a:b][use] = row['window']
        best[a:b][use] = distances[use]
    assert np.all(owner >= 0)
    return owner


def window_stats(name, rows):
    targets = sum(row['targets'] for row in rows)
    miss_low = sum(row['misses'] for row in rows)
    miss_half = sum(row['misses_0_5'] for row in rows)
    return dict(group=name, windows=len(rows), windows_with_targets=sum(row['targets']>0 for row in rows),
        owned_frames=sum(row['owned_all_count'] for row in rows), targets=targets,
        misses_1e_5=miss_low, miss_rate_1e_5=divide(miss_low, targets),
        misses_0_5=miss_half, miss_rate_0_5=divide(miss_half, targets),
        raw_finite_windows=sum(math.isfinite(row['stationarity_raw']) for row in rows),
        raw_mean=mean([row['stationarity_raw'] for row in rows]), raw_median=median([row['stationarity_raw'] for row in rows]),
        warp_mean=mean([row['stationarity_warp'] for row in rows]), warp_median=median([row['stationarity_warp'] for row in rows]),
        cover_mean=mean([row['cover'] for row in rows]), cover_median=median([row['cover'] for row in rows]),
        teacher_min=min(row['teacher'] for row in rows), teacher_max=max(row['teacher'] for row in rows),
        no_teacher_windows=sum(row['teacher']==0 for row in rows))


def target_stats(name, rows, lookup):
    misses_low = sum(not row['hit_1e-05'] for row in rows)
    misses_half = sum(not row['hit_0.5'] for row in rows)
    return dict(group=name, targets=len(rows), misses_1e_5=misses_low, miss_rate_1e_5=divide(misses_low,len(rows)),
        misses_0_5=misses_half, miss_rate_0_5=divide(misses_half,len(rows)),
        teacher_zero_targets=sum(row['teacher']==0 for row in rows),
        assigned_raw_mean=mean([lookup[row['clip'],row['window']]['stationarity_raw'] for row in rows]),
        assigned_warp_mean=mean([lookup[row['clip'],row['window']]['stationarity_warp'] for row in rows]),
        residual_missing_targets=sum(not math.isfinite(lookup[row['clip'],row['window']]['stationarity_raw']) for row in rows))


def associated(rows, field, threshold, control):
    y_key = 'miss_rate_'+threshold
    selected = [r for r in rows if r['targets'] and math.isfinite(r[field])]
    x = np.array([r[field] for r in selected])
    y = np.array([r[y_key] for r in selected])
    clips = sorted({r['clip'] for r in selected})
    if control != 'pooled':
        design = np.array([[float(r['clip']==clip) for clip in clips] for r in selected])
        if control == 'clip_and_early_owner_window':
            design = np.column_stack((design, [float(r['owned_early_count']>0) for r in selected]))
        x = x-design@np.linalg.lstsq(design,x,rcond=None)[0]
        y = y-design@np.linalg.lstsq(design,y,rcond=None)[0]
    return dict(field=field, threshold=threshold, control=control, windows=len(selected), clips=len(clips),
                pearson_r=correlation(x,y), spearman_r=correlation(rank(x),rank(y)) if control=='pooled' else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary=json.loads((SOURCE/'summary.json').read_text())
    raw_windows=read_csv(SOURCE/'windows.csv')
    targets=read_csv(SOURCE/'targets.csv')
    for row in targets:
        for name in ('t','window','teacher','box_index','track','cat','candidate_count'): row[name]=int(row[name])
        for name in ('hit_1e-05','hit_0.5'): row[name]=row[name]=='True'
    grouped=defaultdict(list)
    for row in targets: grouped[row['clip'],row['window']].append(row)
    windows=[]
    for original in raw_windows:
        row=dict(original)
        for name in ('window','first','last','teacher','targets','misses'): row[name]=int(row[name])
        for name in ('owned_first','owned_last'): row[name]=int(row[name]) if row[name] else None
        for name in ('stationarity_raw','stationarity_warp','cover'): row[name]=float(row[name])
        records=grouped[row['clip'],row['window']]
        assert len(records)==row['targets']
        assert sum(not r['hit_1e-05'] for r in records)==row['misses']
        assert all(r['teacher']==row['teacher'] for r in records)
        assert min((r['t'] for r in records),default=None)==row['owned_first']
        assert max((r['t'] for r in records),default=None)==row['owned_last']
        for period,predicate in [('early',lambda r:r['t']<EARLY),('later',lambda r:r['t']>=EARLY)]:
            selected=[r for r in records if predicate(r)]
            row[period+'_targets']=len(selected)
            row[period+'_misses_1e_5']=sum(not r['hit_1e-05'] for r in selected)
            row[period+'_misses_0_5']=sum(not r['hit_0.5'] for r in selected)
        row['misses_0_5']=sum(not r['hit_0.5'] for r in records)
        row['miss_rate_1e_5']=divide(row['misses'],row['targets'])
        row['miss_rate_0_5']=divide(row['misses_0_5'],row['targets'])
        row['raw_minus_warp']=row['stationarity_raw']-row['stationarity_warp']
        windows.append(row)
    by_clip={clip:sorted([r for r in windows if r['clip']==clip],key=lambda r:r['window']) for clip in summary['frames']}
    for clip,rows in by_clip.items():
        owners=owner_indices(summary['frames'][clip],rows)
        for target in [r for r in targets if r['clip']==clip]:assert owners[target['t']]==target['window']
        for row in rows:
            indices=np.flatnonzero(owners==row['window'])
            row.update(owned_all_first=int(indices.min()),owned_all_last=int(indices.max()),owned_all_count=len(indices),
                       owned_early_count=int((indices<EARLY).sum()),owned_later_count=int((indices>=EARLY).sum()),
                       source_early_count=max(0,min(EARLY,row['last']+1)-row['first']))
    lookup={(r['clip'],r['window']):r for r in windows}
    window_groups=[window_stats(name,rows) for name,rows in [
        ('all',windows),('first_source_window',[r for r in windows if r['window']==0]),
        ('after_first_source_window',[r for r in windows if r['window']>0]),
        ('owners_touching_frames_0_59',[r for r in windows if r['owned_early_count']>0]),
        ('owners_only_frames_60_onward',[r for r in windows if r['owned_early_count']==0])]]
    target_groups=[target_stats(name,rows,lookup) for name,rows in [
        ('all',targets),('exact_frames_0_59',[r for r in targets if r['t']<EARLY]),
        ('exact_frames_60_onward',[r for r in targets if r['t']>=EARLY])]]
    clip_stats=[]
    for clip,rows in sorted(by_clip.items()):
        entry={'clip':clip,'frames':summary['frames'][clip]}
        records=[r for r in targets if r['clip']==clip]
        for period,subset in [('early',[r for r in records if r['t']<EARLY]),('later',[r for r in records if r['t']>=EARLY])]:
            stats=target_stats(period,subset,lookup)
            for key in ('targets','misses_1e_5','miss_rate_1e_5','misses_0_5','miss_rate_0_5','residual_missing_targets'):
                entry[period+'_'+key]=stats[key]
        for period,subset in [('first',rows[:1]),('afterfirst',rows[1:]),
                              ('early_owner',[r for r in rows if r['owned_early_count']>0]),
                              ('later_owner',[r for r in rows if r['owned_early_count']==0])]:
            stats=window_stats(period,subset)
            for key in ('windows','raw_median','warp_median','cover_median','targets','misses_1e_5','miss_rate_1e_5','raw_finite_windows'):
                entry[period+'_'+key]=stats[key]
        entry['early_minus_later_miss_rate_1e_5']=entry['early_miss_rate_1e_5']-entry['later_miss_rate_1e_5']
        entry['early_minus_later_raw_median']=entry['early_owner_raw_median']-entry['later_owner_raw_median']
        entry['early_minus_later_warp_median']=entry['early_owner_warp_median']-entry['later_owner_warp_median']
        clip_stats.append(entry)
    associations=[associated(windows,field,threshold,control)
        for threshold in ('1e_5','0_5') for field in ('stationarity_raw','stationarity_warp','cover','teacher')
        for control in ('pooled','clip_intercepts','clip_and_early_owner_window')]
    within=[]
    for clip,rows in sorted(by_clip.items()):
        for field in ('stationarity_raw','stationarity_warp','cover','teacher'):
            result=associated(rows,field,'1e_5','pooled')
            within.append(dict(clip=clip,**result))
    missing=[r for r in windows if not math.isfinite(r['stationarity_raw']) or not math.isfinite(r['stationarity_warp'])]
    teacher_groups=[]
    for name,predicate in [('full_teacher_16',lambda r:r['teacher']==16),('teacher_1_to_15',lambda r:0<r['teacher']<16),
                           ('cover_zero',lambda r:r['cover']==0),('cover_below_0_1',lambda r:r['cover']<.1),
                           ('cover_at_least_0_1',lambda r:r['cover']>=.1)]:
        selected=[r for r in windows if predicate(r)]
        if selected: teacher_groups.append(window_stats(name,selected))
    sources=[SOURCE/'windows.csv',SOURCE/'targets.csv',SOURCE/'summary.json',
             ROOT/'code/miss_recovery.py',ROOT/'code/multirate.py',ROOT/'code/diagnose_history.py',
             ROOT/'results/frame_inspection/inspection_data.json',
             ROOT/'results/frame_inspection/findings_zh.txt',ROOT/'results/frame_inspection/DJI_0307_2_overlap_control.json',
             ROOT/'results/frame_inspection/DJI_0307_2_shift18_control.json',Path(__file__)]
    inspection=json.loads((ROOT/'results/frame_inspection/inspection_data.json').read_text())
    overlap=json.loads((ROOT/'results/frame_inspection/DJI_0307_2_overlap_control.json').read_text())
    shifted=json.loads((ROOT/'results/frame_inspection/DJI_0307_2_shift18_control.json').read_text())
    local_evidence=dict(
        DJI_0255_1=dict(rates=inspection['clips']['DJI_0255_1']['rates'],
            selected_contrast=[dict(frame=r['frame'],max_abs_local_contrast=r['max_abs_local_contrast_near_selected_point'])
                for r in inspection['clips']['DJI_0255_1']['selected'] if r['frame'] in (0,10,90,120)]),
        DJI_0307_2=dict(first_annotated_frame=inspection['clips']['DJI_0307_2']['first_annotated_frame'],
            overlap=overlap['overlap'],roi=overlap['roi_xyxy'],
            target_hit_agreement=overlap['equivalence']['target_hit_agreement_both_thresholds'],
            whole_roi_tolerance_passed=overlap['equivalence']['all_interior_probability_tolerance_passed'],
            shift18=shifted['summary'],limitations=shifted['limitations']))
    report=dict(type='descriptive_existing_records_only',new_model_forwards=0,new_training_runs=0,
        early_frame_definition='0 <= t < 60; no threshold selection or fitting of new detector',
        outcome='per-box ShootingRules hit from already max-fused official cached probabilities, not official merged-component Pd',
        residual_definition='90th percentile absolute difference between the two half-window median images after removing global offset; 4x4 block means, high-gradient pixels, shared valid mask',
        ownership='Nearest source-window center among covering windows; strict tie goes to earlier window. Each target counted once. owned_first/last in M1 CSV refers only to labelled target times.',
        groups_window=window_groups,groups_exact_target_time=target_groups,per_clip=clip_stats,
        associations=associations,association_method='Unit=owned window miss rate; unweighted Pearson and pooled Spearman. Clip-intercepts and clip+early-owner indicator removed by least squares from both variables. No p values, CI, causal interpretation or independent-window assumption.',
        missing_residual_windows=missing,coverage_teacher_groups=teacher_groups,
        existing_local_evidence=local_evidence,
        teacher_count_histogram=dict(Counter(r['teacher'] for r in windows)),
        checks=dict(windows=len(windows),clips=len(by_clip),owned_frames=sum(r['owned_all_count'] for r in windows),
                    targets=len(targets),all_window_counts_and_ownership_recomputed=True,
                    no_teacher_targets=sum(r['teacher']==0 for r in targets)),
        sources={str(path.relative_to(ROOT)):digest(path) for path in sources})
    save_csv('window_analysis.csv',windows)
    save_csv('clip_comparison.csv',clip_stats)
    save_csv('window_groups.csv',window_groups)
    save_csv('target_time_groups.csv',target_groups)
    save_csv('window_associations.csv',associations)
    save_csv('within_clip_correlations.csv',within)
    (OUT/'analysis.json').write_text(json.dumps(safe(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(safe({key:report[key] for key in ['groups_window','groups_exact_target_time','associations','coverage_teacher_groups','checks']}),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
