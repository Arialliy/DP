"""Analytic validity audit of unchanged local-contrast windows on JPEG frames."""
import csv,json,math,datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
out=Path('results/miss_recovery/official_rgbt/test')
dataset=Path('/home/md0/ly/DP/datasets/RGB-T')
rows=list(csv.DictReader((out/'candidate_generation_review.csv').open()))
shapes={};valid={};global_counts={}
for clip in sorted({r['clip'] for r in rows}):
    first=sorted((dataset/'test2017'/clip/'01').glob('*.jpg'))[0]
    with Image.open(first) as im:
        W,H=im.size;dtype=str(np.asarray(im).dtype)
    assert dtype=='uint8'
    shapes[clip]=[H,W]
    if (H,W) not in valid:
        yy,xx=np.indices((H,W))
        def count(radius):
            return (np.minimum(xx+radius,W-1)-np.maximum(xx-radius,0)+1)*(np.minimum(yy+radius,H-1)-np.maximum(yy-radius,0)+1)
        inside=count(1);ring=count(7)-count(3)
        valid[H,W]=(inside>=1)&(ring>=4)
        global_counts[str((H,W))]=dict(inner_min=int(inside.min()),ring_min=int(ring.min()),invalid_pixels=int((~valid[H,W]).sum()),total_pixels=H*W)
    assert valid[H,W].all()
groups=defaultdict(lambda:dict(targets=0,fully_defined=0,partly_undefined=0,all_undefined=0,empty_region=0,r3_region_clipped_by_image_border=0))
for r in rows:
    H,W=shapes[r['clip']];cx,cy=float(r['x']),float(r['y'])
    xa,xb=math.ceil(cx-3),math.floor(cx+3)+1;ya,yb=math.ceil(cy-3),math.floor(cy+3)+1
    x0,x1=max(0,xa),min(W,xb);y0,y1=max(0,ya),min(H,yb)
    mask=valid[H,W][y0:y1,x0:x1]
    for k in ['all',r['loss_stage']]:
        z=groups[k];z['targets']+=1
        if mask.size==0:z['empty_region']+=1
        elif mask.all():z['fully_defined']+=1
        elif mask.any():z['partly_undefined']+=1
        else:z['all_undefined']+=1
        z['r3_region_clipped_by_image_border']+=(xa!=x0 or xb!=x1 or ya!=y0 or yb!=y1)
report=dict(method='For finite uint8 JPEG inputs, local_contrast validity depends only on counts of valid image pixels; exact rectangle-intersection counts reproduce cv2 BORDER_CONSTANT sums of an all-ones mask. No contrast features or proposals recalculated.',shapes=shapes,validity_counts=global_counts,groups=dict(groups),source_finite='JPEG uint8 pixels converted to float32; box-filter sums bounded by 225*255=57375; no input NaN/Inf or overflow',all_original_jpegs_rechecked=False,note='Image dimensions and finite uint8 format checked using first source JPEG of each clip; previous complete proposal audit read every 838 missed frame and matched stored points exactly.')
(out/'candidate_generation_review_validity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
s=json.loads((out/'candidate_generation_review.json').read_text());s['contrast_validity']=report
(out/'candidate_generation_review.json').write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
p=out/'candidate_generation_review.md';text=p.read_text();needle='3. 1545 个漏检只有 50 个中心距图像边界不足 7 px；边缘不能解释整体低覆盖。局部对比按边界有效像素数量归一化，官方命中规则的首行/首列限制亦已保留并由合成测试核对。'
z=groups['suppressed_by_9x9_nms'];text=text.replace(needle,needle+' 另核对了未定义对比：实际图像为 512×640 的有限 uint8 JPEG，最角落中心仍有 4 个有效像素、背景环有 48 个，均超过实现要求。因此全图无 NaN 区域；198 个低于门槛的例子全部是有限的低对比，0 个全无效、0 个部分无效。435 个局部极大/r3 排除的例子也为 0 个全/部分无效，其中 '+str(z['r3_region_clipped_by_image_border'])+' 个 r3 方窗被图像边界裁短（保留部分的对比仍完全有定义）。')
p.write_text(text)
print(json.dumps({'validity_counts':global_counts,'groups':dict(groups)},ensure_ascii=False,indent=2))
