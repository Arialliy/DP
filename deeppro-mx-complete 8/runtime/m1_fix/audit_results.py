"""Read-only audit of completed candidate M1; writes only its review and audit ledger."""
import ast
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

OUT = Path('results/miss_recovery/official_rgbt/test')
KEY = 'pooled budget 1 / official th 1e-05'
CATEGORIES = {0: '船', 1: '車', 2: '騎行者', 3: '行人', 4: '巴士', 5: '無人機', 6: '飛機'}

def flag(value):
    if value not in ('True', 'False'):
        raise AssertionError(value)
    return value == 'True'

def main():
    summary = json.loads((OUT/'summary.json').read_text())
    assert 'decision' in summary, 'Candidate M1 is not complete; original oracle output cannot be audited here'
    dec = summary['decision']
    assert dec['baseline_threshold'] == 1e-5 and dec['budget'] == 1.0
    assert dec['early_frames'] == 60 and dec['early_minimum'] == .30 and dec['overall_minimum'] == .20
    assert dec['diagnostic_only'] and dec['mode'] == 'pooled'
    frames = summary['frames']
    budgets = summary['measured_clutter']
    thresholds = budgets['pooled budget 1']['thresholds']
    targets = []
    assigned, needed = {}, set()
    with (OUT/'targets.csv').open() as f:
        for raw in csv.DictReader(f):
            r = dict(raw)
            for k in ('t','box_index','track','cat','teacher'):
                r[k] = int(r[k])
            for k in ('x','y','score'):
                r[k] = float(r[k])
            r['hit'] = flag(r['hit_1e-05'])
            r['candidate_ids'] = ast.literal_eval(r['candidate_ids'])
            r['recovered'] = (not r['hit'] and r['teacher'] > 0 and math.isfinite(r['score'])
                              and r['score'] >= thresholds[r['clip']])
            for cid in r['candidate_ids']:
                needed.add((r['clip'],r['t'],cid))
            cid = r['matched_candidate']
            if cid:
                key = (r['clip'],r['t'],int(cid))
                assert key not in assigned, 'One candidate assigned to multiple targets'
                assigned[key] = r
            else:
                assert r['score'] == -math.inf
            targets.append(r)
    admitted = {name:defaultdict(int) for name in budgets}
    found, positive_ids = set(), set()
    with (OUT/'candidates.csv').open() as f:
        for raw in csv.DictReader(f):
            clip, t, cid = raw['clip'], int(raw['t']), int(raw['candidate_id'])
            key = clip,t,cid
            score, teacher, label = float(raw['score']), int(raw['teacher']), int(raw['label'])
            if key in needed:
                assert label == 1, 'Target candidate is not labelled positive'
                positive_ids.add(key)
            if key in assigned:
                r=assigned[key]
                assert teacher>0 and r['teacher']>0 and math.isfinite(score) and score==r['score']
                assert max(abs(float(raw['x'])-r['x']),abs(float(raw['y'])-r['y'])) <= summary['settings']['r_pos']
                assert r['box_index'] in ast.literal_eval(raw['target_indices'])
                found.add(key)
            if label==0 and teacher>0 and math.isfinite(score):
                for name, report in budgets.items():
                    if score >= report['thresholds'][clip]:
                        admitted[name][clip]+=1
    assert found==set(assigned) and positive_ids==needed
    for name, report in budgets.items():
        per={c:admitted[name][c] for c in frames}
        assert per==report['per_clip'] and sum(per.values())==report['admitted']
        assert math.isclose(sum(per.values())/sum(frames.values()),report['per_frame'],abs_tol=1e-14)
        budget=float(name.rsplit(' ',1)[1])
        if name.startswith('pooled'):
            assert sum(per.values())<=math.floor(budget*sum(frames.values()))
        else:
            assert all(per[c]<=math.floor(budget*frames[c]) for c in frames)

    def ceiling(rows):
        misses=[r for r in rows if not r['hit'] and r['teacher']>0]
        cols={}
        rr,cc=[],[]
        for i,r in enumerate(misses):
            for cid in r['candidate_ids']:
                key=(r['clip'],r['t'],cid)
                j=cols.setdefault(key,len(cols))
                rr.append(i);cc.append(j)
        if not rr:
            return 0
        matrix=csr_matrix((np.ones(len(rr),np.int8),(rr,cc)),shape=(len(misses),len(cols)))
        match=maximum_bipartite_matching(matrix,perm_type='column')
        assert len(match)==len(misses)
        return int(np.count_nonzero(match>=0))

    def counts(rows):
        misses=sum(not r['hit'] for r in rows)
        recovered=sum(r['recovered'] for r in rows)
        return dict(targets=len(rows),misses=misses,recovered=recovered,
                    recovered_fraction=recovered/max(misses,1),candidate_ceiling=ceiling(rows))
    groups={'all':targets,'frames < 60':[r for r in targets if r['t']<60],
            'frames >= 60':[r for r in targets if r['t']>=60]}
    groups.update({f'clip {c}':[r for r in targets if r['clip']==c] for c in frames})
    groups.update({f'category {cat}':[r for r in targets if r['cat']==cat] for cat in sorted({r['cat'] for r in targets})})
    audited={name:counts(rows) for name,rows in groups.items()}
    for name, observed in audited.items():
        expected=summary['results'][KEY][name]
        for field in ('targets','misses','recovered','candidate_ceiling'):
            assert observed[field]==expected[field],(name,field,observed,expected)
        assert math.isclose(observed['recovered_fraction'],expected['recovered_fraction'],abs_tol=1e-14)
        assert observed['recovered']<=observed['candidate_ceiling']<=observed['misses']
    all_,early=audited['all'],audited['frames < 60']
    go=(all_['misses']>0 and early['misses']>0 and early['recovered_fraction']>=.3
        and all_['recovered_fraction']>=.2 and budgets['pooled budget 1']['per_frame']<=1.)
    assert go==dec['go']
    per_clip=sorted([(k.removeprefix('clip '),v) for k,v in audited.items() if k.startswith('clip ')],key=lambda x:-x[1]['misses'])
    per_cat=sorted([(int(k.removeprefix('category ')),v) for k,v in audited.items() if k.startswith('category ')],key=lambda x:-x[1]['misses'])
    lines=['修正版 M1 獨立結果核對（box 診斷口徑）','',
           f"固定 gate：pooled 1 clutter candidate/frame、基線閾值 1e-5、前60幀；結果 {'GO' if go else 'NO-GO'}。",
           f"片頭恢復 {early['recovered']}/{early['misses']} = {early['recovered_fraction']:.2%}（門檻30%）；全部恢復 {all_['recovered']}/{all_['misses']} = {all_['recovered_fraction']:.2%}（門檻20%）。",
           f"實際 clutter {budgets['pooled budget 1']['admitted']}/{sum(frames.values())} = {budgets['pooled budget 1']['per_frame']:.6f}/frame，未超預算。",
           f"幾何候選上界：片頭 {early['candidate_ceiling']}/{early['misses']}；全部 {all_['candidate_ceiling']}/{all_['misses']}，已用獨立 SciPy 最大二分匹配核對。",
           f"已核對全部 {len(assigned)} 個已分配候選：存在於真實候選表、有 teacher、有限分數、位置在 r_pos 範圍、每個候選至多對應一個目標。",
           '六組 pooled/clip 噪聲預算已從 candidates.csv 逐項重算；固定 gate 的全部、片頭、片後、逐clip與逐類別分母、恢復數和上界一致。','',
           '漏檢與恢復的集中處（描述性分組，不是失效成因證明）：']
    for c,v in per_clip:
        lines.append(f"  {c}: 漏檢 {v['misses']}，恢復 {v['recovered']} ({v['recovered_fraction']:.1%})，剩餘 {v['misses']-v['recovered']}。")
    lines.append('類別：')
    for cat,v in per_cat:
        lines.append(f"  {cat} {CATEGORIES.get(cat,'未知')}: 漏檢 {v['misses']}，恢復 {v['recovered']} ({v['recovered_fraction']:.1%})，剩餘 {v['misses']-v['recovered']}。")
    lines += ['', '口徑與結論限制：',
              '這裡每個標註框獨立計數，基線用逐框 Gaussian mask 鄰域命中；不是官方合併component，也不是9x9 NMS的box配對指標。不可把分母和前版1334次component漏檢混用。',
              '恢復必須來自實際contrast峰值候選，沒有用GT中心充當檢測提議；candidate ceiling忽略分數和噪聲預算，各子組上界獨立計算，不能直接相加。',
              'LORO分類器使用其他test錄像的標註，噪聲閾值也使用被評估資料的clutter標註；此結果只供診斷，不是未見測試標註的模型性能。',
              '保留原命中是baseline OR recovered的定義，不是已驗證新網路訓練後不退化。',
              'NO-GO只表示這個候選／特徵／教師診斷沒有達到預定門檻，不能證明所有可學習單幀路徑都無效；GO同樣不保證mta_lv訓練後有效。']
    (OUT/'review_zh.txt').write_text('\n'.join(lines)+'\n')
    sources={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['summary.json','targets.csv','candidates.csv']}
    report=dict(status='passed',decision=go,groups=audited,assigned_candidates=len(assigned),
                all_budget_tables_verified=True,independent_ceiling='scipy.sparse.csgraph.maximum_bipartite_matching',sources=sources)
    (OUT/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','decision':go,'all':all_,'early':early,'clutter':budgets['pooled budget 1']['per_frame']},indent=2))

if __name__=='__main__':
    main()
