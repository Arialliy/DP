"""Checked orchestration of the isolated original train/test entrypoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback

RUN = Path('/run')


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def parse_metrics(text):
    final = text.rsplit('Final results:', 1)[-1]
    auc = re.search(r'AUC:([0-9.]+)', final)
    iou = re.search(r'Eval avg class IoU of prediction: ([0-9.]+)', final)
    rows = re.findall(r'Th_Seg = ([\deE+.-]+):\s*PD:\[(\d+)/(\d+), ([\d.]+)\]\s*FA:\[(\d+), ([\deE+.-]+)\]', final)
    if auc is None or iou is None or len(rows) != 27:
        raise RuntimeError('Final evaluation did not produce all 27 threshold rows, AUC and IoU')
    return {
        'auc': float(auc.group(1)), 'iou_at_0_5': float(iou.group(1)),
        'thresholds': [{'threshold': float(th), 'true_targets': int(tp), 'targets': int(n),
                        'pd': int(tp) / int(n), 'false_pixels': int(fp), 'fa': float(fa)}
                       for th, tp, n, pd, fp, fa in rows],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['train', 'released'])
    args = parser.parse_args()
    name = 'retrained_rgbt_deepproplus' if args.phase == 'train' else 'released_rgbt_deepproplus'
    status_path = RUN / (args.phase + '_status.json')
    status = {'phase': args.phase, 'started_at': datetime.now(timezone.utc).isoformat(),
              'log_dir': name, 'state': 'starting'}

    def update(**values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        save(status_path, status)

    try:
        config = json.loads((RUN / 'config.json').read_text())
        manifest = json.loads((RUN / 'source_manifest.json').read_text())
        for rel, record in manifest['files'].items():
            assert hashlib.sha256((Path('/work') / rel).read_bytes()).hexdigest() == record['run_sha256'], rel
        if args.phase == 'train':
            # A fresh run must not silently resume the original best-only checkpoint.
            if (RUN / 'log/sem_seg' / name / 'checkpoints/best_model.pth').exists():
                raise RuntimeError('Training checkpoint already exists; refusing an implicit best-only resume')
            import train
            argv = ['train.py', '--datapath', '/data/RGB-T', '--savepath', '/run/log/',
                    '--log_dir', name, '--gpu', '0']
            for key in ['model', 'dataset', 'epoch', 'seqlen', 'patch_size', 'batch_size',
                        'sample_rate', 'learning_rate', 'optimizer', 'decay_rate', 'step_size',
                        'lr_decay', 'threshold_eval']:
                argv += ['--' + key, str(config[key])]
            sys.argv = argv
            update(state='training', train_command=argv)
            train.main(train.parse_args())
        checkpoint = RUN / 'log/sem_seg' / name / 'checkpoints/best_model.pth'
        assert checkpoint.is_file(), checkpoint
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if args.phase == 'released':
            assert checkpoint_hash == manifest['checkpoint_sha256'], 'Released checkpoint hash changed'
        command = [sys.executable, '-u', '/work/test.py', '--gpu', '0', '--seqlen', str(config['seqlen']),
                   '--dataset', config['dataset'], '--datapath', '/data/RGB-T',
                   '--logpath', '/run/log/', '--log_dir', name,
                   '--threshold_eval', str(config['threshold_eval'])]
        update(state='evaluating', test_command=command,
               checkpoint_sha256=checkpoint_hash)
        subprocess.run(command, check=True, cwd='/work')
        metrics = parse_metrics((RUN / 'log/sem_seg' / name / 'eval.txt').read_text())
        metrics.update(dataset='RGB-T', phase=args.phase, checkpoint_sha256=status['checkpoint_sha256'],
                       mask_protocol=config['mask_protocol'], paper_comparison_threshold=.99)
        save(RUN / (args.phase + '_metrics.json'), metrics)
        update(state='completed', metrics_file=args.phase + '_metrics.json')
    except BaseException as exc:
        update(state='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        subprocess.run([sys.executable, '/run/report.py'], check=False)


if __name__ == '__main__':
    sys.path.insert(0, '/work')
    main()
