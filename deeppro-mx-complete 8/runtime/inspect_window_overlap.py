"""Two CPU forwards: compare identical target frames in overlapping official windows."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import ndimage
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import models
import vtdata
import gaussian_masks as gm


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def sigmoid(x):
    with np.errstate(over='ignore'):
        return np.float32(1)/(np.float32(1)+np.exp(-np.asarray(x,np.float32)))


def support(box, H, W):
    m=gm.label([box],H,W,6,'floor')>0
    m[0,:]=False;m[:,0]=False
    return ndimage.binary_dilation(m,structure=np.ones((3,3)))


def main():
    torch.set_num_threads(4)
    clip='DJI_0307_2';sc=vtdata.Scene(clip)
    roi=(480,376,608,504);x0,y0,x1,y1=roi
    # Spatial support radius: conv_in3 + two residual blocks*(TD1+SD2) =9px.
    # TPro and all remaining layers only mix time/channels, with eval-mode batchnorm.
    radius=9
    margins=[]
    for boxes in sc.boxes[:76]:
        for x,y,w,h,*_ in boxes:
            margins.append(min(x-x0,y-y0,x1-x-w,y1-y-h))
    assert min(margins)>=32
    video=np.stack([sc.image(t)[y0:y1,x0:x1] for t in range(76)])
    ckpt=ROOT/'inputs/official_rgbt_best_model.pth'
    model,needs=models.build('deeppro_plus');assert not needs
    sd=torch.load(ckpt,map_location='cpu',weights_only=False)
    if isinstance(sd,dict) and ('model_state_dict' in sd or 'model' in sd):
        sd=sd.get('model_state_dict',sd.get('model'))
    model.load_state_dict(sd,strict=True);model.eval()
    out=[];timings=[]
    with torch.no_grad():
        for a,b in ((0,40),(36,76)):
            x=torch.from_numpy(((video[a:b]-vtdata.RGBT_MEAN)/vtdata.RGBT_STD)[None,None])
            start=time.monotonic();p=model(x)[1][0].numpy();timings.append(time.monotonic()-start)
            assert p.shape==(40,128,128) and np.isfinite(p).all()
            out.append(p)
    fused=np.full((76,128,128),-np.inf,np.float32)
    fused[:40]=out[0];fused[36:]=np.maximum(fused[36:],out[1])
    cache_path=ROOT/f'cache/preds/official_rgbt/{clip}.npy'
    cached=np.load(cache_path,mmap_mode='r')
    # Frames72..75 can receive a third window [72,112), so omit them from fused equivalence.
    cache=cached[:72,y0:y1,x0:x1].astype(np.float32)
    interior=(slice(None),slice(radius,-radius),slice(radius,-radius))
    prob_err=np.abs(sigmoid(fused[:72][interior])-sigmoid(cache[interior]))
    logit_err=np.abs(fused[:72][interior]-cache[interior])
    rows=[];equiv=[]
    for t in range(19,72):
        for box in sc.boxes[t]:
            local=box.copy();local[0]-=x0;local[1]-=y0
            m=support(local,128,128)
            pc=float(sigmoid(cache[t])[m].max());pf=float(sigmoid(fused[t])[m].max())
            same=all((np.float32(pc)>=np.float32(th))==(np.float32(pf)>=np.float32(th)) for th in (1e-5,.5))
            equiv.append(dict(frame=t,track=int(box[5]),cached_probability=pc,crop_probability=pf,
                              both_threshold_hits_match=bool(same),
                              probability_within_tolerance=bool(np.isclose(pf,pc,rtol=.02,atol=1e-7))))
            if 36<=t<40:
                pa=float(sigmoid(out[0][t])[m].max());pb=float(sigmoid(out[1][t-36])[m].max())
                rows.append(dict(frame=t,source_file=sc.frames[t],track=int(box[5]),
                    first_window_index=t,second_window_index=t-36,
                    first_window_probability=pa,second_window_probability=pb,
                    fused_crop_probability=pf,cached_full_frame_probability=pc,
                    first_hit_1e_5=bool(np.float32(pa)>=np.float32(1e-5)),
                    second_hit_1e_5=bool(np.float32(pb)>=np.float32(1e-5)),
                    first_hit_0_5=bool(np.float32(pa)>=np.float32(.5)),
                    second_hit_0_5=bool(np.float32(pb)>=np.float32(.5))))
    assert all(r['both_threshold_hits_match'] for r in equiv)
    assert all(r['probability_within_tolerance'] for r in equiv)
    folder=ROOT/'results/frame_inspection'
    # Preserve the measurement even if the extra whole-interior tolerance fails.
    # Target-region equivalence is the primary comparison requested here.
    np.savez_compressed(folder/'DJI_0307_2_overlap_logits.npz', first=out[0], second=out[1],
                        fused=fused, cached_first72=cache)
    report=dict(clip=clip,device='cpu',forwards=2,windows=[[0,40],[36,76]],roi_xyxy=roi,
        coordinate_convention='zero-based frame indices; native640x512 COCO x/y; crop128x128',
        min_gt_box_margin_px=min(margins),spatial_receptive_field_radius_px=radius,
        architecture_basis='conv_in spatial radius3; two blocks each add1+2; other layers 1x1. TPro mixes all40 time indices independently at each spatial position. BatchNorm uses fixed running statistics in eval.',
        checkpoint_sha256=sha(ckpt),model_source_sha256=sha(ROOT/'code/models.py'),
        cached_prediction_sha256=sha(cache_path),
        input_frame_sha256={sc.frames[t]:sha(sc.paths[t]) for t in range(76)},
        forward_seconds=timings,overlap=rows,
        equivalence=dict(frames=[0,72],all_interior_max_probability_error=float(prob_err.max()),
            all_interior_mean_probability_error=float(prob_err.mean()),
            all_interior_max_logit_error=float(logit_err.max()),target_frames=equiv,
            target_hit_agreement_both_thresholds=True,relative_probability_tolerance=.02,
            absolute_probability_tolerance=1e-7,all_interior_probability_tolerance=.002,
            all_interior_probability_tolerance_passed=bool(float(prob_err.max())<.002),
            target_probability_tolerance_passed=True,
            initial_attempt='Two identical CPU forwards; target tests passed but extra whole-interior .002 check failed before saving. Repeated same two windows only to retain measurements; no new window/context experiment.'),
        limitations=['Compares the same frame pixels with different40-frame contexts AND different relative temporal indices; does not separate these two mechanisms.',
            'Original cache was GPU FP32 output stored as float16; this witness uses CPU FP32. Equivalence is checked numerically on the interior and at target regions.',
            'No full-frame CPU forward was rerun; spatial equivalence relies on the exact finite receptive field and is checked against the existing fused full-frame cache.'])
    (folder/'DJI_0307_2_overlap_control.json').write_text(json.dumps(report,indent=2)+'\n')
    with (folder/'DJI_0307_2_overlap_control.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(overlap=rows,equivalence={k:v for k,v in report['equivalence'].items() if k!='target_frames'},timings=timings),indent=2))


if __name__=='__main__':main()
