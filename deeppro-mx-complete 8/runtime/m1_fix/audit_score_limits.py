"""Inspect existing M1 scores only; no new thresholds, fits or proposals."""
import ast,csv,json,math
from pathlib import Path
from collections import defaultdict
out=Path('results/miss_recovery/official_rgbt/test')
s=json.loads((out/'summary.json').read_text())
assert 'decision' in s
rows=[];needed=set()
for r in csv.DictReader(open(out/'targets.csv')):
 r=dict(r,t=int(r['t']),cat=int(r['cat']),teacher=int(r['teacher']),score=float(r['score']))
 r['hit']=r['hit_1e-05']=='True';r['ids']=ast.literal_eval(r['candidate_ids'])
 rows.append(r)
 if not r['hit']:
  needed.update((r['clip'],r['t'],i) for i in r['ids'])
scores={};fit=defaultdict(lambda:dict(candidates=0,teacher_rows=0,finite_teacher_scores=0,positive=0,clutter=0,ambiguous=0))
for r in csv.DictReader(open(out/'candidates.csv')):
 key=(r['clip'],int(r['t']),int(r['candidate_id']));value=float(r['score']);teacher=int(r['teacher'])
 z=fit[r['clip']];z['candidates']+=1;z['teacher_rows']+=teacher>0;z['finite_teacher_scores']+=(teacher>0 and math.isfinite(value))
 z[{1:'positive',0:'clutter',-1:'ambiguous'}[int(r['label'])]]+=1
 if key in needed:scores[key]=(value,teacher)
results={}
for mode in ('pooled','clip'):
 threshold=s['measured_clutter'][f'{mode} budget 1']['thresholds']
 stats=defaultdict(lambda:dict(misses=0,has_geometric_candidate=0,has_finite_candidate_score=0,has_threshold_passing_candidate=0,recovered_after_one_to_one=0))
 for r in rows:
  if r['hit']:continue
  candidates=[scores[r['clip'],r['t'],i] for i in r['ids']]
  finite=[score for score,teacher in candidates if teacher>0 and math.isfinite(score)]
  passed=any(score>=threshold[r['clip']] for score in finite)
  recovered=r['teacher']>0 and math.isfinite(r['score']) and r['score']>=threshold[r['clip']]
  for group in ('all','early' if r['t']<60 else 'later','clip '+r['clip'],'category '+str(r['cat'])):
   z=stats[group];z['misses']+=1;z['has_geometric_candidate']+=any(teacher>0 for score,teacher in candidates);z['has_finite_candidate_score']+=bool(finite);z['has_threshold_passing_candidate']+=passed;z['recovered_after_one_to_one']+=recovered
 results[mode]=dict(stats)
ledger=dict(no_new_fit_or_threshold=True,fit_score_availability=dict(fit),existing_budget_1_results=results)
(out/'score_limit_audit.json').write_text(json.dumps(ledger,indent=2)+'\n')
print(json.dumps({'all_candidates':sum(z['candidates'] for z in fit.values()),'teacher_rows':sum(z['teacher_rows'] for z in fit.values()),'finite_teacher_scores':sum(z['finite_teacher_scores'] for z in fit.values()),'pooled':results['pooled']['all'],'early_pooled':results['pooled']['early'],'clip':results['clip']['all'],'early_clip':results['clip']['early']},indent=2))
