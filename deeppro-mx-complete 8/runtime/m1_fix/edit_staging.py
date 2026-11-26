from pathlib import Path
import ast
path=Path(__file__).parent/'fixed/code/miss_recovery.py'
s=path.read_text()
def replace_function(source,name,body):
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+body.strip()+'\n'+''.join(lines[node.end_lineno:])

s=replace_function(s,'box_hits',r'''
def box_hits(prob, boxes, H, W, ths, rule='paper_k6_floor'):
    """Per-box ShootingRules hit, including official first-row/column eligibility."""
    import vtdata
    out = np.zeros((len(boxes), len(ths)), bool)
    near = ndimage.maximum_filter(np.asarray(prob, np.float32), size=3,
                                  mode='constant', cval=-np.inf)
    thresholds_ = np.asarray(ths, np.float32)
    for i, b in enumerate(boxes):
        eligible = vtdata.gaussian_mask(np.asarray(b, float)[None], H, W, rule) > 0
        eligible[0, :] = False
        eligible[:, 0] = False
        if eligible.any():
            out[i] = near[eligible].max() >= thresholds_
    return out
''')
s=replace_function(s,'process_clip',r'''
def process_clip(clip, video, boxes, logits, args, chain=None, cam_group='static', speed=None):
    """Actual single-frame candidates, target metadata, and owned-window diagnostics.

    GT centres are used only to label/match candidates. They are never scored as
    detection locations. Ambiguous candidates are scored but excluded from fitting.
    """
    import multirate
    n, H, W = video.shape
    speed = speed or {}
    ths = list(args.thresholds)
    teacher = np.concatenate([mxd.peak_rows(np.asarray(logits[t], np.float32), t, _peaks, logit(args.teacher_th))
                              for t in range(n)]) if n else np.zeros((0, 4))
    ws = st.windows(n, args.seqlen)
    owner = closest_windows(n, ws)
    S, shift = args.patch, args.shift
    banks = []
    for a, b in ws:
        rows = mxd.select_bank(teacher, a, b - 1, 'outside', args.k)
        ce = ed.exemplar_contrasts(video, rows, shift)
        E = (np.stack([mxd.patch2d(video[int(t)], x, y, S) for t, x, y, _ in rows])
             if len(rows) else np.zeros((0, S, S)))
        banks.append((rows, ce, E))
    low = multirate.lowres(video, 4) if chain is not None else None
    targets, cands, wins = [], [], []
    for t in range(n):
        frame = np.asarray(video[t], np.float32)
        cmap = st.local_contrast(frame)
        rows, ce, E = banks[owner[t]]
        with np.errstate(over='ignore'):
            prob = 1.0 / (1.0 + np.exp(-np.asarray(logits[t], np.float32)))
        b = np.asarray(boxes[t], float).reshape(-1, 6)
        cen = mxd.box_centres(b) if len(b) else np.zeros((0, 2))
        hits = box_hits(prob, b, H, W, ths) if len(b) else np.zeros((0, len(ths)), bool)
        pk = mxd.frame_peaks(frame, t, args.per_frame)
        pts = pk[:, 1:3]
        distance = (np.max(np.abs(pts[:, None, :] - cen[None, :, :]), axis=2)
                    if len(cen) else np.zeros((len(pts), 0)))
        f_c = features(frame, cmap, [tuple(p) for p in pts], ce, E, S, shift)
        for i, row in enumerate(b):
            ids = np.flatnonzero(distance[:, i] <= args.r_pos).tolist()
            targets.append(dict(clip=clip, t=t, box_index=i, track=int(row[5]), cat=int(row[4]),
                                x=float(cen[i, 0]), y=float(cen[i, 1]),
                                rel_speed=speed.get((clip, t, int(row[5])), math.nan),
                                camera=cam_group, window=int(owner[t]), teacher=len(rows),
                                candidate_ids=ids, candidate_count=len(ids), score=-math.inf,
                                **{f'hit_{th:g}': bool(hits[i, j]) for j, th in enumerate(ths)}))
        for i, (x, y) in enumerate(pts):
            d = float(distance[i].min()) if len(cen) else math.inf
            matched = np.flatnonzero(distance[i] <= args.r_pos).tolist()
            label = 1 if d <= args.r_pos else 0 if d > args.r_neg else -1
            cands.append(dict(clip=clip, t=t, candidate_id=i, x=float(x), y=float(y),
                              window=int(owner[t]), teacher=len(rows), label=label,
                              target_indices=matched, target_distances=distance[i, matched].tolist(),
                              **{k: float(f_c[i, j]) for j, k in enumerate(FEATURES)}))
    for k, (a, b) in enumerate(ws):
        frames = list(range(a, b))
        stats = (multirate.stationarity(low[frames], chain, frames, 4) if chain is not None
                 else dict(raw=math.nan, warp=math.nan, cover=math.nan))
        mine = [r for r in targets if r['window'] == k]
        th0 = f'hit_{ths[0]:g}'
        wins.append(dict(clip=clip, window=k, first=a, last=b - 1, teacher=len(banks[k][0]),
                         targets=len(mine), misses=sum(not r[th0] for r in mine),
                         owned_first=min((r['t'] for r in mine), default=None),
                         owned_last=max((r['t'] for r in mine), default=None),
                         stationarity_raw=stats.get('raw', math.nan), stationarity_warp=stats.get('warp', math.nan),
                         cover=stats.get('cover', math.nan)))
    return targets, cands, wins
''')
s=replace_function(s,'loro_scores',r'''
def loro_scores(targets, cands):
    """Fit actual positive/clutter candidates on other recordings; score all candidates.

    Candidate labels of the held-out recording never enter the fit or normalization.
    Target scores are assigned afterwards by deterministic one-to-one spatial matching.
    """
    if cands:
        y = np.array([r['label'] for r in cands], float)
        X = np.array([[r[k] for k in FEATURES] for r in cands], float)
        clips = np.array([r['clip'] for r in cands])
        groups = np.array([cd.recording(c) for c in clips])
        has = np.array([r['teacher'] > 0 for r in cands])
        score = np.full(len(cands), -np.inf)
        for c in sorted(set(clips)):
            te = (clips == c) & has
            tr = (groups != cd.recording(c)) & has & (y >= 0)
            if not te.any() or len(np.unique(y[tr])) < 2:
                continue
            w = cd.clip_weights(clips[tr], y[tr])
            mu = (w[:, None] * X[tr]).sum(0)
            sd = np.sqrt((w[:, None] * (X[tr] - mu) ** 2).sum(0)) + 1e-9
            beta = cd.weighted_logistic((X[tr] - mu) / sd, y[tr], w)
            score[te] = np.c_[(X[te] - mu) / sd, np.ones(int(te.sum()))] @ beta
        for r, value in zip(cands, score):
            r['score'] = float(value)
    assign_candidates(targets, cands)
''')
s=replace_function(s,'thresholds',r'''
def thresholds(cands, frames, budget, mode):
    """Conservative threshold: finite labelled clutter admissions never exceed budget.

    Boundary ties are excluded together when including them would exceed the cap.
    Candidate labels are used for diagnostic threshold calibration, not deployment.
    """
    def thr(rows, n):
        s = np.sort(np.asarray([r['score'] for r in rows
                                if r['label'] == 0 and r['teacher'] > 0
                                and np.isfinite(r['score'])], float))[::-1]
        k = int(math.floor(budget * n))
        if k <= 0 or not len(s):
            return math.inf if k <= 0 else -math.inf
        if k >= len(s):
            return -math.inf
        return float(np.nextafter(s[k], math.inf))
    if mode == 'pooled':
        t = thr(cands, sum(frames.values()))
        return {c: t for c in frames}
    return {c: thr([r for r in cands if r['clip'] == c], frames[c]) for c in frames}
''')
s=s.replace("        rec = np.array([r['score'] >= thr[r['clip']] for r in rows])", "        rec = np.array([r['teacher'] > 0 and np.isfinite(r['score'])\n                        and r['score'] >= thr[r['clip']] for r in rows], bool)\n        ceiling = candidate_ceiling([r for r in rows if not r[key]])")
s=s.replace("                         hits_covered=int((hit & rec).sum()), recall=float(hit.mean()),", "                         candidate_ceiling=ceiling,\n                         candidate_ceiling_fraction=ceiling / max(int((~hit).sum()), 1),\n                         hits_covered=int((hit & rec).sum()), recall=float(hit.mean()),")
s=s.replace("    return clip, process_clip(clip, video, sc.boxes, logits, args, chain, cam_group, speed)", "    return clip, len(sc), process_clip(clip, video, sc.boxes, logits, args, chain, cam_group, speed)")
s=s.replace("    targets, cands, wins = [], [], []\n    with Pool", "    targets, cands, wins, frames = [], [], [], {}\n    with Pool")
s=s.replace("        for clip, (t, c, w) in pool.imap_unordered(_one, tasks):", "        for clip, nframes, (t, c, w) in pool.imap_unordered(_one, tasks):\n            frames[clip] = nframes")
s=s.replace("    nframes = {}\n    for r in targets + cands:\n        nframes.setdefault(r['clip'], set()).add(r['t'])\n    frames = {c: len(nframes.get(c, ())) for c in clips}\n", "")
s=s.replace("    write_rows(os.path.join(out, 'windows.csv'), wins)\n    result = {}", "    write_rows(os.path.join(out, 'windows.csv'), wins)\n    write_rows(os.path.join(out, 'candidates.csv'), cands)\n    result, budgets_observed = {}, {}")
s=s.replace("            thr = thresholds(cands, frames, budget, mode)\n            for th", "            thr = thresholds(cands, frames, budget, mode)\n            budgets_observed[f'{mode} budget {budget:g}'] = clutter_admissions(cands, frames, thr, budget, mode)\n            for th")
s=s.replace("    json.dump(dict(settings=vars(args), frames=frames, candidates=len(cands), results=result),", "    gate = decision(result, budgets_observed, args.early)\n    json.dump(dict(settings=vars(args), frames=frames, candidates=len(cands), results=result,\n                   candidate_labels={'positive': 1, 'clutter': 0, 'ambiguous': -1},\n                   measured_clutter=budgets_observed, decision=gate,\n                   metric='Per-box official mask hits; recovery from actual contrast peaks with one-to-one matching',\n                   fitting='LORO actual positive/clutter candidates; threshold uses evaluated labelled clutter; diagnostic only'),")
s=s.replace("    return result\n\n\nif __name__", "    print('M1 decision:', 'GO' if gate['go'] else 'NO-GO', gate, flush=True)\n    return result\n\n\nif __name__")
helpers=r'''
def assign_candidates(targets, cands):
    """Greedy score-descending matching, one real candidate and target at most once."""
    from collections import defaultdict
    by_frame, target_map = defaultdict(list), {}
    for r in targets:
        r['score'], r['matched_candidate'] = -math.inf, None
        target_map[(r['clip'], r['t'], r['box_index'])] = r
    for r in cands:
        if r['teacher'] > 0 and np.isfinite(r.get('score', -math.inf)):
            by_frame[(r['clip'], r['t'])].append(r)
    for key, rows in by_frame.items():
        used = set()
        for r in sorted(rows, key=lambda v: (-v['score'], v['candidate_id'])):
            choices = [(distance, box) for box, distance in zip(r['target_indices'], r['target_distances'])
                       if box not in used and (*key, box) in target_map]
            if choices:
                _, box = min(choices)
                target = target_map[(*key, box)]
                if target['teacher'] > 0:
                    target['score'], target['matched_candidate'] = r['score'], r['candidate_id']
                    used.add(box)


def candidate_ceiling(targets):
    """Maximum one-to-one matches among these misses before score/FA constraints.

    This is a geometric ceiling, independently evaluated within each reported group.
    A teacher is required. Augmenting paths prevent greedy matching lowering the ceiling.
    """
    from collections import defaultdict
    frames = defaultdict(list)
    for r in targets:
        if r['teacher'] > 0:
            frames[(r['clip'], r['t'])].append(r.get('candidate_ids', []))
    total = 0
    for edges in frames.values():
        matched = {}
        def visit(target, seen):
            for candidate in edges[target]:
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate not in matched or visit(matched[candidate], seen):
                    matched[candidate] = target
                    return True
            return False
        total += sum(visit(i, set()) for i in range(len(edges)))
    return total


def clutter_admissions(cands, frames, thr, budget, mode):
    admitted = {c: sum(r['clip'] == c and r['label'] == 0 and r['teacher'] > 0
                        and np.isfinite(r['score']) and r['score'] >= thr[c] for r in cands)
                for c in frames}
    if mode == 'pooled':
        ok = sum(admitted.values()) <= math.floor(budget * sum(frames.values()))
    else:
        ok = all(admitted[c] <= math.floor(budget * frames[c]) for c in frames)
    if not ok:
        raise AssertionError('Realized clutter exceeds the diagnostic budget')
    return dict(admitted=sum(admitted.values()), frames=sum(frames.values()),
                per_frame=sum(admitted.values()) / max(sum(frames.values()), 1),
                per_clip=admitted, thresholds=thr, within_budget=ok)


def decision(results, observed, early=60):
    """The predeclared gate, fixed before reading data: pooled 1/frame, p=1e-5."""
    key, budget_key = 'pooled budget 1 / official th 1e-05', 'pooled budget 1'
    groups = results.get(key, {})
    all_, start = groups.get('all', {}), groups.get('frames < 60', {})
    measured = observed.get(budget_key, {})
    available = early == 60 and bool(all_) and bool(start) and bool(measured)
    enough = available and all_.get('misses', 0) > 0 and start.get('misses', 0) > 0
    all_fraction, early_fraction = all_.get('recovered_fraction'), start.get('recovered_fraction')
    budget_ok = bool(measured.get('within_budget', False)) and measured.get('per_frame', math.inf) <= 1.0
    go = bool(enough and budget_ok and early_fraction >= 0.30 and all_fraction >= 0.20)
    return dict(go=go, diagnostic_only=True, applicable=bool(enough), mode='pooled', budget=1.0,
                baseline_threshold=1e-5, early_frames=60,
                early_minimum=0.30, overall_minimum=0.20,
                early_recovered_fraction=early_fraction, overall_recovered_fraction=all_fraction,
                realized_clutter_per_frame=measured.get('per_frame'), budget_satisfied=budget_ok,
                existing_hits_retained='Logical OR with baseline; appearance hits_covered is reported separately',
                reason=('Thresholds met' if go else 'Required gate outputs/misses absent' if not enough
                        else 'Realized clutter exceeds budget' if not budget_ok else 'Recovery thresholds not met'))


'''
s=s.replace('def main(argv=None):',helpers+'def main(argv=None):')
path.write_text(s)
print(path)
