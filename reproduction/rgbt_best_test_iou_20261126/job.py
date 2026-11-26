"""Execute one research-model rerun and verify all 32 selection evaluations."""
from datetime import datetime, timezone
import json
import math
import re
import subprocess
import sys
import traceback

from runlib import ROOT, read_config, read_json, save_json, sha256, train_command, verify_artifacts, verify_source


def verify_resume(arch, out, resume):
    """Only continue the explicitly archived, verified checkpoint boundary."""
    import torch
    count = resume['completed_epochs']
    assert not (out / 'final.pth').exists()
    for name, expected in resume['files'].items():
        assert sha256(out / name) == expected, 'Resume artifact changed: ' + name
    last = torch.load(out / 'last.pth', map_location='cpu', weights_only=False)
    best = torch.load(out / 'best_model.pth', map_location='cpu', weights_only=False)
    assert last['epoch'] + 1 == count
    assert last['best_epoch'] == best['best_epoch']
    assert last['best_iou'] == best['best_iou']
    assert last['opt']['state']
    evaluations = [json.loads(line) for line in (out / 'eval_log.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in evaluations] == list(range(1, count + 1))
    assert all(row['complete'] and row['windows'] == 172 for row in evaluations)
    selected = max(evaluations, key=lambda row: (row['iou'], row['epoch']))
    assert (best['best_epoch'], best['best_iou']) == (selected['epoch'], selected['iou'])
    epochs = [int(ep) for ep in re.findall(r'^epoch (\d+)/32 ', (out / 'train_log.txt').read_text(), re.M)]
    assert epochs == list(range(1, count + 1)), epochs
    sys.path.insert(0, str(ROOT / 'code'))
    import models
    net, _ = models.build(arch)
    for state in (last['model'], best['model']):
        net.load_state_dict(state, strict=True)
        assert all(torch.isfinite(value).all() for value in state.values())
    return dict(completed_epochs=count, next_epoch=count + 1, best_epoch=best['best_epoch'],
                best_iou=best['best_iou'], optimizer_restored=True,
                rng_note='Original trainer resumes model/optimizer/epoch; worker RNG streams are reinitialized.')


def verify_completed_run(arch, out, cfg):
    import torch
    sys.path.insert(0, str(ROOT / 'code'))
    import models
    count = cfg['epochs']
    history = (out / 'train_log.txt').read_text()
    rows = re.findall(r'^epoch (\d+)/(\d+) lr \S+ loss (\S+) iters (\d+)', history, re.M)
    assert [int(row[0]) for row in rows] == list(range(1, count + 1)), rows
    assert all(int(row[1]) == count for row in rows)
    assert all(int(row[3]) == 236 for row in rows), rows
    assert all(math.isfinite(float(row[2])) for row in rows), rows
    evaluations = [json.loads(line) for line in (out / 'eval_log.jsonl').read_text().splitlines() if line.strip()]
    assert len(evaluations) == count, len(evaluations)
    assert [row['epoch'] for row in evaluations] == list(range(1, count + 1))
    protocol = cfg['checkpoint_selection_protocol']
    for row in evaluations:
        assert row['complete'] is True, row
        assert row['split'] == protocol['split'] and float(row['threshold']) == protocol['threshold'], row
        assert math.isfinite(row['iou']) and 0 <= row['iou'] <= 1, row
        assert row['windows'] == protocol['windows_per_epoch'], row
        assert row['window_frames'] == protocol['window_frames_per_epoch'], row
        assert row['unique_frames'] == protocol['unique_test_frames'], row
        assert 0 <= row['intersection'] <= row['union'] and row['union'] > 0, row
        assert math.isclose(row['iou'], row['intersection'] / row['union'], rel_tol=0, abs_tol=1e-12), row
    selected = max(evaluations, key=lambda row: (row['iou'], row['epoch']))
    best = torch.load(out / 'best_model.pth', map_location='cpu', weights_only=False)
    assert isinstance(best['selection'], dict) and best['selection']
    assert best['selection']['complete'] is True
    assert best['selection']['epoch'] == selected['epoch']
    for key in ('windows', 'window_frames', 'unique_frames', 'intersection', 'union'):
        assert best['selection'][key] == selected[key], key
    assert math.isclose(float(best['selection']['iou']), selected['iou'], rel_tol=0, abs_tol=1e-12)
    assert best['best_epoch'] == selected['epoch']
    assert best['epoch'] == selected['epoch'] - 1
    assert math.isclose(float(best['best_iou']), selected['iou'], rel_tol=0, abs_tol=1e-12)
    assert isinstance(best['opt'], dict) and best['opt'].get('state')
    final = torch.load(out / 'final.pth', map_location='cpu', weights_only=False)
    last = torch.load(out / 'last.pth', map_location='cpu', weights_only=False)
    assert last['epoch'] == count - 1
    assert last['best_epoch'] == best['best_epoch']
    assert math.isclose(float(last['best_iou']), float(best['best_iou']), rel_tol=0, abs_tol=1e-12)
    assert isinstance(last['opt'], dict) and last['opt'].get('state')
    net, _ = models.build(arch)
    for state in (best['model'], final, last['model']):
        net.load_state_dict(state, strict=True)
        assert all(torch.isfinite(tensor).all() for tensor in state.values())
    assert set(final) == set(last['model'])
    assert all(torch.equal(final[key], last['model'][key]) for key in final)
    if selected['epoch'] == count:
        assert all(torch.equal(final[key], best['model'][key]) for key in final)
    return {'epochs_completed': count, 'eval_epochs_completed': count,
            'optimizer_updates': sum(int(row[3]) for row in rows), 'final_loss': float(rows[-1][2]),
            'best_iou': float(best['best_iou']), 'best_epoch': int(best['best_epoch']),
            'best_checkpoint': str((out / 'best_model.pth').relative_to(ROOT)),
            'best_sha256': sha256(out / 'best_model.pth'),
            'final_checkpoint': str((out / 'final.pth').relative_to(ROOT)),
            'final_sha256': sha256(out / 'final.pth'), 'last_sha256': sha256(out / 'last.pth'),
            'parameters': models.n_params(net), 'strict_load': 'passed', 'all_tensors_finite': True,
            'selection_verification': '32 full test evaluations; latest maximum IoU epoch'}


def main():
    arch = sys.argv[1]
    cfg = read_config()
    model = next(model for model in cfg['models'] if model['arch'] == arch)
    status = {'arch': arch, 'name': model['name'], 'state': 'starting',
              'started_at': datetime.now(timezone.utc).isoformat()}

    def update(**values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        save_json(ROOT / 'jobs' / (arch + '.json'), status)

    try:
        verify_source()
        reused = verify_artifacts()
        assert read_json(ROOT / 'gpu_preflight.json', {}).get('status') == 'passed', 'New GPU preflight required'
        out = ROOT / 'runs' / model['run']
        resume = cfg.get('resume_runs', {}).get(arch)
        if resume:
            update(resume=verify_resume(arch, out, resume))
        else:
            for name in ('last.pth', 'final.pth', 'best_model.pth', 'train_log.txt', 'eval_log.jsonl'):
                if (out / name).exists():
                    raise RuntimeError('Existing run artifact; refusing implicit resume/overwrite: ' + str(out / name))
        command = train_command(model)
        update(state='running', stage='train', command=command, run=model['run'], reused_artifacts=reused)
        subprocess.run(command, check=True, cwd=ROOT)
        update(state='verifying_checkpoint', stage='verifying')
        result = verify_completed_run(arch, out, cfg)
        update(state='completed', stage='completed', **result)
    except BaseException as exc:
        update(state='failed', error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
