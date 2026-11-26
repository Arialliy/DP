"""Run all 32 training epochs and verify the resulting checkpoint."""
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
import sys
import traceback

from runlib import ROOT, read_config, save_json, train_command, verify_source


def main():
    arch = sys.argv[1]
    cfg = read_config()
    model = next(m for m in cfg['models'] if m['arch'] == arch)
    status = {'arch': arch, 'name': model['name'], 'state': 'starting',
              'started_at': datetime.now(timezone.utc).isoformat()}

    def update(**values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        save_json(ROOT / 'jobs' / (arch + '.json'), status)

    try:
        verify_source()
        assert json.loads((ROOT / 'gpu_preflight.json').read_text())['status'] == 'passed'
        out = ROOT / 'runs' / model['run']
        if (out / 'last.pth').exists() or (out / 'final.pth').exists():
            raise RuntimeError('Existing checkpoint: refusing implicit resume or overwrite')
        command = train_command(model)
        update(state='training', command=command, run=model['run'])
        subprocess.run(command, check=True, cwd=ROOT)
        update(state='verifying_checkpoint')
        import torch
        sys.path.insert(0, str(ROOT / 'code'))
        import models
        history = (out / 'train_log.txt').read_text()
        epochs = [int(ep) for ep in re.findall(r'^epoch (\d+)/32 ', history, re.M)]
        assert epochs == list(range(1, 33)), epochs
        iterations = [int(n) for n in re.findall(r'iters (\d+)', history)]
        assert iterations == [236] * 32, iterations
        losses = [float(s) for s in re.findall(r'loss (\S+)', history)]
        assert torch.isfinite(torch.tensor(losses)).all(), losses
        final = out / 'final.pth'
        sd = torch.load(final, map_location='cpu', weights_only=False)
        net, needs = models.build(arch)
        net.load_state_dict(sd, strict=True)
        assert all(torch.isfinite(t).all() for t in sd.values())
        last = torch.load(out / 'last.pth', map_location='cpu', weights_only=False)
        assert last['epoch'] == 31
        assert all(torch.equal(sd[key], last['model'][key]) for key in sd)
        update(state='completed', epochs_completed=32, optimizer_updates=sum(iterations),
               final_loss=losses[-1], final_checkpoint=str(final.relative_to(ROOT)),
               final_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
               parameters=models.n_params(net), strict_load='passed', all_tensors_finite=True)
    except BaseException as exc:
        update(state='failed', error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
