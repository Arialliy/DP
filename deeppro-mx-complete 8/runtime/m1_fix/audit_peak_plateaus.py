"""Quantify M1 plateau slots; no feature extraction, logistic fit, threshold or model run."""
import csv,json,sys,hashlib,datetime
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
import stabilize as st,mx_data as mxd
OUT=Path('results/miss_recovery/official_rgbt/test');DATA=Path('/data/RGB-T')
summary=json.loads((OUT/'summary.json').read_text());by_frame=defaultdict(list)
for r in csv.DictReader((OUT/'candidates.csv').open()):by_frame[r['clip'],int(r['t'])].append((int(r['candidate_id']),float(r['x']),float(r['y'])))
misses=defaultdict(list)
for r in csv.DictReader((OUT/'candidate_generation_review.csv').open()):misses[r['clip'],int(r['t'])].append(r)
frames=[];miss_rows=[]
for clip,nframes in sorted(summary['frames'].items()):
    paths=sorted((DATA/'test2017'/clip/'01').glob('*.jpg'));assert len(paths)==nframes
    print('plateaus',clip,nframes,flush=True)
    for t,path in enumerate(paths):
        im=np.asarray(Image.open(path),np.float32)
        if im.ndim==3:im=im[:,:,0]
        a=np.abs(np.nan_to_num(st.local_contrast(im),nan=0.))
        peak=(a==ndimage.maximum_filter(a,size=9,mode='nearest'))&(a>3.)
        labels,n=ndimage.label(peak,structure=np.ones((3,3),bool))
        yy,xx=np.nonzero(peak);values=a[yy,xx];labs=labels[yy,xx]
        order=np.argsort(-values,kind='stable');oldpoints=np.stack([xx[order[:100]],yy[order[:100]]],1)
        saved=sorted(by_frame[clip,t]);assert np.array_equal(oldpoints,np.asarray([(x,y) for _,x,y in saved]).reshape(-1,2)),(clip,t)
        # Any adjacent retained local maxima see each other inside their 9x9
        # neighborhoods, hence must have the same value; verify actual components.
        if n:
            maxv=np.full(n+1,-np.inf);minv=np.full(n+1,np.inf);np.maximum.at(maxv,labs,values);np.minimum.at(minv,labs,values);assert np.array_equal(maxv[1:],minv[1:])
            first=np.sort(np.unique(labs,return_index=True)[1])
            comporder=first[np.argsort(-values[first],kind='stable')]
            repcoords=np.stack([xx[comporder],yy[comporder]],1)
            complabs=labs[comporder]
        else:
            repcoords=np.zeros((0,2),int);complabs=np.zeros(0,int)
        ranks={int(v):i+1 for i,v in enumerate(complabs)}
        selected_labs=labs[order[:100]]
        total_comp_size=np.bincount(labs,minlength=n+1)
        selected_size=np.bincount(selected_labs,minlength=n+1)
        occupied=int((selected_size>0).sum());selected_count=len(oldpoints)
        dup=selected_count-occupied
        selected_multicomponents=int(((selected_size>0)&(total_comp_size>1)).sum())
        multi_points=int(selected_size[total_comp_size>1].sum())
        frames.append(dict(clip=clip,t=t,selected_points=selected_count,selected_plateaus=occupied,duplicate_slots=dup,
            points_in_multimember_plateaus=multi_points,selected_multimember_plateaus=selected_multicomponents,
            all_peak_pixels=len(xx),all_plateaus=int(n),largest_plateau=int(total_comp_size.max()) if n else 0,
            top100_truncated_plateaus=int(((selected_size>0)&(selected_size<total_comp_size)).sum()),
            available_extra_blobs_after_dedup=min(100,n)-occupied))
        for r in misses[clip,t]:
            centre=np.array([float(r['x']),float(r['y'])]);dist=np.max(abs(np.stack([xx,yy],1)-centre),axis=1) if len(xx) else np.zeros(0)
            nearlabels=set(labs[dist<=3].tolist());component_rank=min((ranks[int(l)] for l in nearlabels),default=None)
            newpoints=repcoords[:100];newcount=int((np.max(abs(newpoints-centre),axis=1)<=3).sum()) if len(newpoints) else 0
            oldcount=int(r['candidate_count'])
            miss_rows.append(dict(clip=clip,t=t,box_index=int(r['box_index']),track=int(r['track']),early=t<60,
                original_loss_stage=r['loss_stage'],original_candidate_count=oldcount,selected_duplicate_slots=dup,
                best_nearby_component_rank=component_rank,
                nearby_component_would_fit_dedup_top100=component_rank is not None and component_rank<=100,
                first_row_major_representative_r3_count=newcount,
                gained_r3_coverage=(oldcount==0 and newcount>0),lost_r3_coverage=(oldcount>0 and newcount==0)))

def summarize_frames(rows):
    keys=['selected_points','selected_plateaus','duplicate_slots','points_in_multimember_plateaus','selected_multimember_plateaus','all_peak_pixels','all_plateaus','top100_truncated_plateaus','available_extra_blobs_after_dedup']
    out={k:sum(r[k] for r in rows) for k in keys}
    out.update(frames=len(rows),frames_with_duplicate_slots=sum(r['duplicate_slots']>0 for r in rows),largest_plateau=max((r['largest_plateau'] for r in rows),default=0),median_duplicate_slots=float(np.median([r['duplicate_slots'] for r in rows])))
    out['duplicate_fraction_of_selected_points']=out['duplicate_slots']/max(out['selected_points'],1)
    return out

def summarize_misses(rows):
    out=dict(misses=len(rows),original_with_r3=sum(r['original_candidate_count']>0 for r in rows),gained_r3_coverage=sum(r['gained_r3_coverage'] for r in rows),lost_r3_coverage=sum(r['lost_r3_coverage'] for r in rows))
    top=[r for r in rows if r['original_loss_stage']=='outside_top100']
    out['original_top100_excluded']=len(top)
    out['top100_excluded_near_component_within_first100_after_dedup']=sum(r['nearby_component_would_fit_dedup_top100'] for r in top)
    out['top100_excluded_actual_representative_within_r3_after_dedup']=sum(r['first_row_major_representative_r3_count']>0 for r in top)
    return out
report=dict(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit_only=True,no_new_features_fits_thresholds_or_models=True,
    unchanged_parameters=dict(floor=3.,nms=9,topk=100,positive_radius=3.),
    definition='8-connected components of the existing local-max mask, verified to have identical contrast values within each component; count excess stored top100 points per component.',
    hypothetical_representative='first row-major actual member of each component; then stable descending existing absolute-contrast score. Pure geometry/ranking audit, no scoring or recovery.',
    all=summarize_frames(frames),per_clip={c:summarize_frames([r for r in frames if r['clip']==c]) for c in summary['frames']},
    misses=summarize_misses(miss_rows),early_misses=summarize_misses([r for r in miss_rows if r['early']]),
    misses_per_clip={c:summarize_misses([r for r in miss_rows if r['clip']==c]) for c in summary['frames']},
    every_original_proposal_reproduced=True,source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['code/mx_data.py','code/evaluate.py','runtime/m1_fix/audit_peak_plateaus.py']})
for name,rows in [('candidate_plateau_frames.csv',frames),('candidate_plateau_misses.csv',miss_rows)]:
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'candidate_plateau_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'all':report['all'],'misses':report['misses'],'early':report['early_misses']},indent=2),flush=True)
