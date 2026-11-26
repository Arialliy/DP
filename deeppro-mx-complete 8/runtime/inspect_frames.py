"""Faithful raw-frame and target-crop inspection; no inference or training."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import gaussian_masks as gm
import vtdata
import stabilize

OUT = ROOT / 'results/frame_inspection'
SELECTED = (0, 10, 19, 20, 30, 39, 40, 60, 90, 120)


def font(size):
    p = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    return ImageFont.truetype(p, size) if Path(p).exists() else ImageFont.load_default()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def box_best(logits, boxes, H, W):
    with np.errstate(over='ignore'):
        p = 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float32)))
    best = []
    for box in boxes:
        m = gm.label([box], H, W, 6, 'floor') > 0
        m[0, :] = False
        m[:, 0] = False
        near = ndimage.binary_dilation(m, structure=np.ones((3, 3)))
        best.append(float(p[near].max()) if near.any() else 0.0)
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_summary = {}
    for clip in ('DJI_0255_1', 'DJI_0307_2'):
        sc = vtdata.Scene(clip)
        pred_path = ROOT / f'cache/preds/official_rgbt/{clip}.npy'
        logits = np.load(pred_path, mmap_mode='r')
        assert logits.shape == (len(sc), sc.H, sc.W)
        annotated = [i for i, boxes in enumerate(sc.boxes) if len(boxes)]
        first_box = sc.boxes[annotated[0]][0]
        metric_rows = []
        for t, boxes in enumerate(sc.boxes):
            best = box_best(logits[t], boxes, sc.H, sc.W) if len(boxes) else []
            for box, p in zip(boxes, best):
                x, y, w, h, cat, track = box
                metric_rows.append(dict(frame=t, source_file=sc.frames[t], track=int(track), cat=int(cat),
                    x=x, y=y, width=w, height=h, nearest_image_border=min(x, y, sc.W-x-w, sc.H-y-h),
                    max_official_hit_probability=p, hit_1e_5=bool(np.float32(p)>=np.float32(1e-5)),
                    hit_0_5=bool(np.float32(p)>=np.float32(.5))))
        with (OUT/f'{clip}_target_frames.csv').open('w') as f:
            wr=csv.DictWriter(f, fieldnames=list(metric_rows[0])); wr.writeheader(); wr.writerows(metric_rows)
        per_frame = {r['frame']:r for r in metric_rows}
        selected = [t for t in SELECTED if t < len(sc)]
        # Full-frame sheet: nearest-neighbour 2x reduction, fixed 0..255 grayscale.
        full_w, full_h, pad = 320, 256, 58
        full = Image.new('RGB', (full_w*5, (full_h+pad)*2+56), '#16191d')
        fd = ImageDraw.Draw(full)
        fd.text((12,8), f'{clip} | actual thermal JPEGs; full frames at 1/2 scale | cyan = GT box', font=font(21), fill='white')
        # Each zoom tile includes one unmarked raw crop and one identical GT-marked crop.
        crop_size, scale = 64, 4
        tile_w, tile_h = 520, 330
        zoom = Image.new('RGB', (tile_w*5, tile_h*2+64), '#16191d')
        zd = ImageDraw.Draw(zoom)
        zd.text((12,8), f'{clip} | 64x64 raw crops, nearest-neighbour 4x; fixed grayscale 0..255', font=font(24), fill='white')
        zd.text((12,36), 'Left: unmarked raw pixels. Right: identical pixels with cyan GT box. Frame indices are zero-based.', font=font(20), fill='white')
        frame_meta=[]
        for k,t in enumerate(selected):
            image=Image.open(sc.paths[t])
            arr=np.asarray(image)
            if arr.ndim==3: arr=arr[:,:,0]
            raw=Image.fromarray(arr.astype(np.uint8),'L').convert('RGB')
            boxes=sc.boxes[t]
            annotated_raw=raw.copy(); d=ImageDraw.Draw(annotated_raw)
            for x,y,w,h,*rest in boxes:
                d.rectangle((x,y,x+w-1,y+h-1),outline='#00e5ff',width=1)
            xx=(k%5)*full_w; yy=(k//5)*(full_h+pad)+56
            full.paste(annotated_raw.resize((full_w,full_h),Image.Resampling.NEAREST),(xx,yy+pad))
            desc=f't={t}, {sc.frames[t]}'
            p=per_frame[t]['max_official_hit_probability'] if t in per_frame else None
            stat=f'GT={len(boxes)}; p(hit)={p:.3g}' if p is not None else 'GT=0; no target scored'
            fd.text((xx+5,yy+3),desc,font=font(15),fill='white');fd.text((xx+5,yy+25),stat,font=font(15),fill='white')
            box=boxes[0] if len(boxes) else first_box
            cx,cy=box[0]+box[2]/2-.5,box[1]+box[3]/2-.5
            # Keep crop inside image; no fabricated padding at the top edge.
            x0=max(0,min(sc.W-crop_size,int(round(cx))-crop_size//2))
            y0=max(0,min(sc.H-crop_size,int(round(cy))-crop_size//2))
            crop=raw.crop((x0,y0,x0+crop_size,y0+crop_size))
            marked=crop.copy();md=ImageDraw.Draw(marked)
            for x,y,w,h,*rest in boxes:
                md.rectangle((x-x0,y-y0,x+w-1-x0,y+h-1-y0),outline='#00e5ff',width=1)
            zx=(k%5)*tile_w;zy=(k//5)*tile_h+64
            title=f't={t} file={sc.frames[t]}'
            track=int(box[5]) if len(boxes) else None
            zd.text((zx+5,zy+3),title,font=font(19),fill='white')
            zd.text((zx+5,zy+26),stat+(f'; track={track}' if track else ''),font=font(17),fill='white')
            zd.text((zx+5,zy+48),f'crop x={x0}, y={y0}'+(' (first GT location; no GT now)' if not len(boxes) else ''),font=font(15),fill='white')
            zoom.paste(crop.resize((256,256),Image.Resampling.NEAREST),(zx+3,zy+72))
            zoom.paste(marked.resize((256,256),Image.Resampling.NEAREST),(zx+263,zy+72))
            cmap=stabilize.local_contrast(arr.astype(np.float32))
            ci=np.asarray([int(round(cy)),int(round(cx))]); yi,xi=ci
            c=cmap[max(0,yi-2):yi+3,max(0,xi-2):xi+3]
            frame_meta.append(dict(frame=t,file=sc.frames[t],file_sha256=sha(sc.paths[t]),gt_boxes=boxes.tolist(),
                crop=(x0,y0,crop_size,crop_size),has_gt=bool(len(boxes)),hit_probability=p,
                max_abs_local_contrast_near_selected_point=float(np.nanmax(np.abs(c)))))
        full.save(OUT/f'{clip}_full_frames.png')
        zoom.save(OUT/f'{clip}_target_crops.png')
        rates={}
        for cut in (40,60):
            for th,key in ((1e-5,'hit_1e_5'),(.5,'hit_0_5')):
                for label,part in [('early',[r for r in metric_rows if r['frame']<cut]),('later',[r for r in metric_rows if r['frame']>=cut])]:
                    rates[f'cut{cut}_th{th:g}_{label}']=dict(target_frames=len(part),misses=sum(not r[key] for r in part))
        all_summary[clip]=dict(frames=len(sc),first_annotated_frame=annotated[0],first_annotated_file=sc.frames[annotated[0]],
            categories=sorted({int(b[4]) for bs in sc.boxes for b in bs}),track_ids=sorted({int(b[5]) for bs in sc.boxes for b in bs}),
            rates=rates, selected=frame_meta,prediction_sha256=sha(pred_path),
            first_hit_1e_5=next((r['frame'] for r in metric_rows if r['hit_1e_5']),None),
            first_hit_0_5=next((r['frame'] for r in metric_rows if r['hit_0_5']),None))
    (OUT/'inspection_data.json').write_text(json.dumps(dict(method='Raw JPEG grayscale unchanged, fixed 0..255 display. Crops nearest-neighbour enlarged. Cyan GT overlay only. Per-box hit matches official eligible-mask dilation; no model was run.',clips=all_summary),indent=2)+'\n')
    print(json.dumps({c:{k:v for k,v in d.items() if k!='selected'} for c,d in all_summary.items()},indent=2))


if __name__=='__main__':
    main()
