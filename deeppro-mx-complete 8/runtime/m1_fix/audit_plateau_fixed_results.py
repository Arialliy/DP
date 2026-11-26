"""Audit completed plateau-corrected M1 vs frozen pixel-plateau outputs."""
import ast,csv,json,math,hashlib,sys
from pathlib import Path
from collections import Counter,defaultdict
import audit_results
OUT=Path('results/miss_recovery/official_rgbt/test')
OLD=Path('results/miss_recovery_pixel_plateaus/official_rgbt/test')

def target_rows(path):
    out={}
    for r in csv.DictReader(path.open()):
        key=(r['clip'],int(r['t']),int(r['box_index']),int(r['track']))
        assert key not in out
        out[key]=r
    return out

def candidates(path):
    counts=Counter();teachers=Counter();labels=Counter();finite=0
    for r in csv.DictReader(path.open()):
        counts[r['clip']]+=1;teachers[int(r['teacher'])]+=1;labels[int(r['label'])]+=1;finite+=math.isfinite(float(r['score']))
    return dict(count=sum(counts.values()),per_clip=dict(counts),teacher_counts=dict(teachers),labels=dict(labels),finite_scores=finite)

def recovered(rows,summary):
    thresholds=summary['measured_clutter']['pooled budget 1']['thresholds']
    return {k for k,r in rows.items() if r['hit_1e-05']=='False' and int(r['teacher'])>0 and math.isfinite(float(r['score'])) and float(r['score'])>=thresholds[r['clip']]}

def main():
    current=json.loads((OUT/'summary.json').read_text());old=json.loads((OLD/'summary.json').read_text())
    assert current['settings']==old['settings'], 'Settings changed'
    assert current['frames']==old['frames']
    audit_results.main()  # Recompute actual budgets, matches, gates and maximum candidate ceiling.
    report=json.loads((OUT/'independent_audit.json').read_text())
    nowrows=target_rows(OUT/'targets.csv');oldrows=target_rows(OLD/'targets.csv');assert nowrows.keys()==oldrows.keys()
    fixed_fields=['clip','t','box_index','track','cat','x','y','rel_speed','camera','window','teacher','hit_1e-05','hit_0.5']
    for key,r in nowrows.items():
        assert all(r[field]==oldrows[key][field] for field in fixed_fields),(key,'baseline/metadata/teacher changed')
    wins=lambda path:{(r['clip'],r['window']):r for r in csv.DictReader(path.open())}
    assert wins(OUT/'windows.csv')==wins(OLD/'windows.csv'), 'Window diagnostic or teacher count changed'
    current_c=candidates(OUT/'candidates.csv');old_c=candidates(OLD/'candidates.csv')
    plateau=json.loads((OLD/'candidate_plateau_review.json').read_text())['all']
    expected=old_c['count']-plateau['duplicate_slots']+plateau['available_extra_blobs_after_dedup']
    assert old_c['count']==594289 and current_c['count']==expected==592988
    previous_recovered=recovered(oldrows,old);now_recovered=recovered(nowrows,current)
    oldmain=old['results']['pooled budget 1 / official th 1e-05'];newmain=current['results']['pooled budget 1 / official th 1e-05']
    changes=dict(archived_old_directory=str(OLD),all_settings_exact_equal=True,settings_fields=list(current['settings']),
        all_frames_exact_equal=True,box_rows=len(nowrows),baseline_hits_both_thresholds_exact_equal=True,baseline_target_metadata_exact_equal=True,
        teacher_count_per_target_exact_equal=True,all_window_diagnostics_and_teacher_counts_exact_equal=True,
        teacher_bank_identity_note='Full teacher banks were not serialized. Teacher count equality is directly verified; teacher extraction/selection source paths were not part of the plateau change.',
        original_candidates=old_c,current_candidates=current_c,expected_candidate_count_from_independent_plateau_audit=expected,
        old_groups={k:oldmain[k] for k in ['all','frames < 60','frames >= 60']},new_groups={k:newmain[k] for k in ['all','frames < 60','frames >= 60']},
        recovered_boxes_added=sorted(now_recovered-previous_recovered),recovered_boxes_removed=sorted(previous_recovered-now_recovered),
        old_clutter_threshold=old['measured_clutter']['pooled budget 1']['thresholds'],new_clutter_threshold=current['measured_clutter']['pooled budget 1']['thresholds'],
        sole_proposal_change='frame_peaks now merges each 8-connected maximum plateau and picks nearest-to-centroid actual peak pixel; equal-distance y/x tie order; floor3/nms9/k100 unchanged',
        shared_selfpeaks_affected=True,training_performed=False)
    report['plateau_fix_comparison']=changes
    report['fixed_cutoffs']={k:current['decision'][k] for k in ['budget','baseline_threshold','early_frames','early_minimum','overall_minimum']}
    report['old_sources']={name:hashlib.sha256((OLD/name).read_bytes()).hexdigest() for name in ['summary.json','targets.csv','candidates.csv']}
    (OUT/'independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    early,all_=newmain['frames < 60'],newmain['all'];oe,oa=oldmain['frames < 60'],oldmain['all']
    text=(OUT/'review_zh.txt').read_text().replace('修正版 M1 獨立結果核對（box 診斷口徑）','Plateau 合併修正版 M1 獨立結果核對（box 診斷口徑）').replace('不可把分母和前版1334次component漏檢混用。','不可與官方合併component分母混用。')
    text+='\nPlateau 修復前後對照：\n'
    text+=f"全部{len(nowrows)}個box的兩個baseline hit標識、目標座標/分類/速度/窗口/teacher數完全一致；所有windows診斷與teacher數相同。settings逐字段完全一致，沒有更改floor=3、NMS=9、top100、r3、預算或預定30%/20%門檻。\n"
    text+=f"候選數 {old_c['count']} → {current_c['count']}；此數等於舊候選−3135個重複plateau名額+1834個可補入的獨立blob，與先前全5976幀獨立幾何審計一致。\n"
    text+=f"pooled1片頭恢復 {oe['recovered']}/{oe['misses']} → {early['recovered']}/{early['misses']}；全部 {oa['recovered']}/{oa['misses']} → {all_['recovered']}/{all_['misses']}。候選上界片頭 {oe['candidate_ceiling']} → {early['candidate_ceiling']}，全部 {oa['candidate_ceiling']} → {all_['candidate_ceiling']}。\n"
    text+='唯一候選生成修復是每8連通同分plateau只取一個最近質心的實際峰值點；教師抽取路徑未改。公共SelfPeaks也使用此函數，因此self樣本提議會改變，但沒有訓練任何模型。\n'
    text+='去重改變positive/clutter擬合樣本，LORO權重及由原有規則算出的閾值可隨之改變；這是相同診斷配方的重算，不是用新結果調參。舊結果及審計完整保留在'+str(OLD)+'。\n'
    text+='1個噪聲候選/frame是預先指定的平均背景候選預算，沒有向資料加噪聲，也不是官方pixel Fa或資料集自帶常數；本次固定門檻只作記錄，不授權訓練。\n'
    (OUT/'review_zh.txt').write_text(text)
    print(json.dumps({'status':'passed','settings_equal':True,'baseline_rows_unchanged':len(nowrows),'candidate_count_old':old_c['count'],'candidate_count_new':current_c['count'],'old_early_recovered':oe['recovered'],'new_early_recovered':early['recovered'],'old_all_recovered':oa['recovered'],'new_all_recovered':all_['recovered'],'added':len(now_recovered-previous_recovered),'removed':len(previous_recovered-now_recovered)},indent=2))
if __name__=='__main__':main()
