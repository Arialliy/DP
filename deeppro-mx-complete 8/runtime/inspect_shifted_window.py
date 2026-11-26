"""Exactly one extra CPU window [18,58); local target-recovery witness only."""
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import models
import vtdata
from inspect_window_overlap import sha,sigmoid,support


def main():
    folder=ROOT/'results/frame_inspection'
    original=json.loads((folder/'DJI_0307_2_overlap_control.json').read_text())
    assert sha(ROOT/'code/models.py')==original['model_source_sha256']
    ckpt=ROOT/'inputs/official_rgbt_best_model.pth'
    assert sha(ckpt)==original['checkpoint_sha256']
    sc=vtdata.Scene('DJI_0307_2')
    roi=original['roi_xyxy'];x0,y0,x1,y1=roi
    a,b=18,58
    video=np.stack([sc.image(t)[y0:y1,x0:x1] for t in range(a,b)])
    torch.set_num_threads(4)
    model,needs=models.build('deeppro_plus');assert not needs
    sd=torch.load(ckpt,map_location='cpu',weights_only=False)
    if isinstance(sd,dict) and ('model_state_dict' in sd or 'model' in sd):
        sd=sd.get('model_state_dict',sd.get('model'))
    model.load_state_dict(sd,strict=True);model.eval()
    x=torch.from_numpy(((video-vtdata.RGBT_MEAN)/vtdata.RGBT_STD)[None,None])
    start=time.monotonic()
    with torch.no_grad():logits=model(x)[1][0].numpy()
    duration=time.monotonic()-start
    np.save(folder/'DJI_0307_2_shift18_logits.npy',logits)
    assert logits.shape==(40,128,128) and np.isfinite(logits).all()
    cached=np.load(ROOT/'cache/preds/official_rgbt/DJI_0307_2.npy',mmap_mode='r')
    first=np.load(folder/'DJI_0307_2_overlap_logits.npz')['first']
    rows=[]
    for t in range(19,36):
        for box in sc.boxes[t]:
            local=box.copy();local[0]-=x0;local[1]-=y0
            mask=support(local,128,128)
            pc=float(sigmoid(cached[t,y0:y1,x0:x1])[mask].max())
            p0=float(sigmoid(first[t])[mask].max())
            ps=float(sigmoid(logits[t-a])[mask].max())
            rows.append(dict(frame=t,source_file=sc.frames[t],track=int(box[5]),
                baseline_cached_probability=pc,baseline_cpu_window0_probability=p0,
                shifted_window_probability=ps,shifted_window_index=t-a,
                baseline_miss_1e_5=bool(np.float32(pc)<np.float32(1e-5)),
                baseline_miss_0_5=bool(np.float32(pc)<np.float32(.5)),
                shifted_hit_1e_5=bool(np.float32(ps)>=np.float32(1e-5)),
                shifted_hit_0_5=bool(np.float32(ps)>=np.float32(.5))))
    summary={}
    for th,key in ((1e-5,'1e_5'),(.5,'0_5')):
        selected=[r for r in rows if r['baseline_miss_'+key]]
        n=sum(r['shifted_hit_'+key] for r in selected)
        summary[str(th)]=dict(original_misses=len(selected),recovered=n,
                              recovered_fraction=n/len(selected) if selected else None)
    report=dict(clip='DJI_0307_2',window=[a,b],roi_xyxy=roi,device='cpu',forwards=1,
        checkpoint_sha256=sha(ckpt),model_source_sha256=sha(ROOT/'code/models.py'),
        script_sha256=sha(__file__),original_control='DJI_0307_2_overlap_control.json',
        input_frame_sha256={sc.frames[t]:sha(sc.paths[t]) for t in range(a,b)},
        forward_seconds=duration,summary=summary,frames=rows,
        limitations=['Local diagnostic on17 previously missed annotated target-frames of one clip.',
            'Additional global false alarms, full-frame false alarms and whole-test benefit were not evaluated.',
            'Changing window placement changes both temporal context and relative output index; those mechanisms remain inseparable here.',
            'This is the same frozen official detector; no appearance model or training was used.'])
    (folder/'DJI_0307_2_shift18_control.json').write_text(json.dumps(report,indent=2)+'\n')
    with (folder/'DJI_0307_2_shift18_control.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(summary=summary,frames=rows,seconds=duration),indent=2))


if __name__=='__main__':main()
