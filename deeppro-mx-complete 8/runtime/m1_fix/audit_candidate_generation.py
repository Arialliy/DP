"""Audit existing M1 proposals; no fitting, changed thresholds, predictions or training."""
import ast,csv,json,math,sys,hashlib,datetime
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy import ndimage
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
import vtdata,stabilize as st,mx_data as mxd,exemplar_diag as ed
OUT=Path('results/miss_recovery/official_rgbt/test')

def scalar(v):
    return float(v) if np.isfinite(v) else None

def stats(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    return dict(n=len(a),min=scalar(a.min()),p25=scalar(np.quantile(a,.25)),median=scalar(np.median(a)),p75=scalar(np.quantile(a,.75)),max=scalar(a.max())) if len(a) else dict(n=0)

def mask_hits_points(points,box,H,W):
    # Exact individual-box paper_k6_floor Gaussian support at candidate+-1,
    # keeping official first-row/first-column eligible-pixel convention.
    if not len(points): return np.zeros(0,bool)
    x,y,w,h=box[:4];cx,cy=x+w/2-.5,y+h/2-.5
    if w<=0 or h<=0: return np.zeros(len(points),bool)
    off=np.array([(dx,dy) for dy in [-1,0,1] for dx in [-1,0,1]])
    q=points[:,None,:]+off[None,:,:]
    d=(q-np.array([cx,cy]))/np.array([w/6,h/6])
    eligible=(q[:,:,0]>=1)&(q[:,:,0]<W)&(q[:,:,1]>=1)&(q[:,:,1]<H)
    return (eligible & (np.floor(np.exp(-.5*(d*d).sum(2))*255)>0)).any(1)

def synthetic_checks():
    tests={}
    frame=np.zeros((47,61),np.float32);frame[16:19,32:35]=90
    pk=mxd.frame_peaks(frame,7,100)
    assert len(pk) and tuple(pk[0,1:3])==(33.,17.) and pk[0,0]==7
    tests['xy_no_transposition']=True
    box=np.array([[32.,16.,3.,3.,6.,0.]])
    assert np.array_equal(mxd.box_centres(box),[[33.,17.]])
    tests['coco_pixel_centres_match_mask']=True
    for x,y in pk[:,1:3]:
        c=st.local_contrast(frame)
        assert abs(ed.sample_absmax(c,x,y,2))==abs(float(c[int(y),int(x)]))
    tests['shift2_does_not_relocate_9x9_peak_contrast']=True
    # Equality-based NMS keeps plateaus, then stable row-major ties consume the top-k.
    plateau=np.zeros((47,61),np.float32);plateau[23,30]=90
    pp=mxd.frame_peaks(plateau,0,100)
    tied=pp[pp[:,3]==pp[0,3]]
    assert len(tied)>1 and len(mxd.frame_peaks(plateau,0,1))==1
    tests['plateau_ties_preserved_then_topk_truncated']=dict(tied_highest_points=len(tied))
    for bx in ([32,16,3,3,6,0],[0,0,5,7,6,0],[11,12,17,9,6,0]):
        points=np.stack(np.meshgrid(np.arange(61),np.arange(47)),axis=-1).reshape(-1,2)
        m=vtdata.gaussian_mask(np.array([bx]),47,61,'paper_k6_floor')>0
        m[0,:]=False;m[:,0]=False
        actual=ndimage.binary_dilation(m,structure=np.ones((3,3)))[points[:,1],points[:,0]]
        assert np.array_equal(mask_hits_points(points,np.array(bx),47,61),actual)
    tests['analytic_mask_geometry_matches_existing_mask_and_official_border']=True
    return tests

def main():
    summary=json.loads((OUT/'summary.json').read_text()); settings=summary['settings']
    assert (settings['per_frame'],settings['r_pos'],settings['r_neg'],settings['shift'])==(100,3.,8.,2)
    targets=[];by_frame=defaultdict(list)
    for r in csv.DictReader((OUT/'targets.csv').open()):
        for k in ['t','box_index','track','cat','candidate_count']:r[k]=int(r[k])
        for k in ['x','y','score']:r[k]=float(r[k])
        r['ids']=ast.literal_eval(r['candidate_ids']);targets.append(r)
        if r['hit_1e-05']=='False':by_frame[(r['clip'],r['t'])].append(r)
    selected=defaultdict(list);scores=defaultdict(lambda:defaultdict(list))
    for r in csv.DictReader((OUT/'candidates.csv').open()):
        clip=r['clip'];t=int(r['t']);label=int(r['label']);score=float(r['score'])
        scores[clip][label].append(score)
        if (clip,t) in by_frame:
            selected[(clip,t)].append(dict(cid=int(r['candidate_id']),x=float(r['x']),y=float(r['y']),score=score,label=label))
    ledger=[];cache_checks=[];verified_frames=0;scene=None;current=None
    for key,rows in sorted(by_frame.items()):
        clip,t=key
        if current!=clip:
            current=clip;scene=vtdata.Scene(clip)
            cached=Path('cache/video')/(clip+'.npy');cache=np.load(cached,mmap_mode='r') if cached.exists() else None
            print('audit',clip,'miss frames',sum(c==clip for c,_ in by_frame),flush=True)
        frame=scene.image(t);H,W=frame.shape
        if cache is not None:
            if not np.array_equal(frame,np.asarray(cache[t],np.float32)):raise AssertionError(('cache mismatch',clip,t))
            cache_checks.append((clip,t))
        cmap=st.local_contrast(frame);a=np.abs(np.nan_to_num(cmap,nan=0.))
        peaks=(a==ndimage.maximum_filter(a,size=mxd.SELF_NMS,mode='nearest'))&(a>mxd.SELF_FLOOR)
        yy,xx=np.nonzero(peaks);vals=a[yy,xx];order=np.argsort(-vals,kind='stable')
        allpoints=np.stack([xx[order],yy[order]],1);allvalues=vals[order]
        chosen=allpoints[:100];saved=sorted(selected[key],key=lambda v:v['cid'])
        assert np.array_equal(chosen,np.array([[r['x'],r['y']] for r in saved])),('proposal mismatch',clip,t)
        verified_frames+=1
        for r in rows:
            box=scene.boxes[t][r['box_index']];centre=mxd.box_centres(np.asarray(box)[None])[0]
            assert tuple(centre)==(r['x'],r['y']) and int(box[5])==r['track'] and int(box[4])==r['cat']
            cx,cy=centre;dist=np.max(abs(allpoints-centre),axis=1)
            idx=np.flatnonzero(dist<=3)
            ids=[int(i) for i in idx if i<100]
            assert ids==r['ids'],('matched candidate ids differ',clip,t)
            x0,x1=max(0,math.ceil(cx-3)),min(W,math.floor(cx+3)+1)
            y0,y1=max(0,math.ceil(cy-3)),min(H,math.floor(cy+3)+1)
            local=a[y0:y1,x0:x1];maxlocal=float(local.max()) if local.size else 0.
            reason=('retained' if ids else 'below_contrast_floor' if maxlocal<=3 else 'suppressed_by_9x9_nms' if len(idx)==0 else 'outside_top100')
            mask=mask_hits_points(chosen,box,H,W)
            within=np.max(abs(chosen-centre),axis=1)<=3
            score=[saved[i]['score'] for i in ids]
            mask_ids=np.flatnonzero(mask).tolist()
            ledger.append(dict(clip=clip,t=t,box_index=r['box_index'],track=r['track'],cat=r['cat'],early=t<60,
                x=float(cx),y=float(cy),box_w=float(box[2]),box_h=float(box[3]),border_distance=float(min(cx,cy,W-1-cx,H-1-cy)),
                candidate_count=len(ids),loss_stage=reason,max_abs_contrast_within_r3=maxlocal,
                localmax_above_floor_count=len(allpoints),best_localmax_r3_rank1=int(idx[0]+1) if len(idx) else None,
                contrast_rank100=float(allvalues[99]) if len(allvalues)>=100 else None,
                nearest_retained_chebyshev=float(np.min(np.max(abs(chosen-centre),axis=1))) if len(chosen) else None,
                selected_mask_eligible_count=int(mask.sum()),selected_r3_not_mask_count=int((within&~mask).sum()),
                selected_mask_but_not_r3_count=int((mask&~within).sum()),
                max_existing_candidate_score=max(score) if score else None,
                passed_existing_pooled1=any(s>=summary['measured_clutter']['pooled budget 1']['thresholds'][clip] for s in score),
                passed_existing_clip1=any(s>=summary['measured_clutter']['clip budget 1']['thresholds'][clip] for s in score)))
    groups={}
    def summarize(rows):
        return dict(misses=len(rows),loss_stage=dict(Counter(r['loss_stage'] for r in rows)),
            has_r3_candidate=sum(r['candidate_count']>0 for r in rows),has_mask_eligible_selected_candidate=sum(r['selected_mask_eligible_count']>0 for r in rows),
            mask_eligible_but_no_r3=sum(r['selected_mask_eligible_count']>0 and r['candidate_count']==0 for r in rows),
            r3_eligible_but_no_mask=sum(r['candidate_count']>0 and r['selected_mask_eligible_count']==0 for r in rows),
            within_7px_image_border=sum(r['border_distance']<7 for r in rows),
            local_abs_contrast_r3=stats([r['max_abs_contrast_within_r3'] for r in rows]),
            nearest_retained_distance=stats([r['nearest_retained_chebyshev'] for r in rows if r['nearest_retained_chebyshev'] is not None]),
            retained_candidate_scores=stats([r['max_existing_candidate_score'] for r in rows if r['max_existing_candidate_score'] is not None]),
            passed_existing_pooled1=sum(r['passed_existing_pooled1'] for r in rows),passed_existing_clip1=sum(r['passed_existing_clip1'] for r in rows))
    groups['all']=summarize(ledger);groups['early']=summarize([r for r in ledger if r['early']]);groups['later']=summarize([r for r in ledger if not r['early']])
    for clip in sorted({r['clip'] for r in ledger}):groups[clip]=summarize([r for r in ledger if r['clip']==clip])
    score_summary={c:{'positive':stats(v[1]),'clutter':stats(v[0]),'ambiguous':stats(v[-1]),'existing_pooled1_threshold':summary['measured_clutter']['pooled budget 1']['thresholds'][c],'existing_clip1_threshold':summary['measured_clutter']['clip budget 1']['thresholds'][c]} for c,v in scores.items()}
    output=dict(reviewed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit_only=True,no_model_inference_training_or_refitting=True,
        parameters_unchanged=dict(per_frame=100,r_pos=3.,r_neg=8.,contrast_floor=3.,nms_size=9,shift=2),
        frame_checks=dict(missed_target_frames=verified_frames,proposals_exact_csv_match=True,box_centres_tracks_categories_exact_match=True,existing_video_cache_frames_equal_source_images=len(cache_checks)),
        synthetic_checks=synthetic_checks(),groups=groups,existing_score_distributions=score_summary,
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['code/miss_recovery.py','code/mx_data.py','code/stabilize.py','code/vtdata.py','code/gaussian_masks.py','code/exemplar_diag.py','runtime/m1_fix/audit_candidate_generation.py']},
        caveat='Mask-geometry counts use unchanged stored proposals only, not rescored/rethresholded recovery; one-to-one/component merges are not applied to these diagnostic geometry counts.')
    (OUT/'candidate_generation_review.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (OUT/'candidate_generation_review.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    print(json.dumps({'frames':verified_frames,'all':groups['all'],'DJI_0309_4':groups['DJI_0309_4'],'DJI_0255_1':groups['DJI_0255_1']},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
