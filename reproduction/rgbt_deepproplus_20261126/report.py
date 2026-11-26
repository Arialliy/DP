"""Render recorded results without guessing unmeasured training outcomes."""
from datetime import datetime
import json
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo

RUN = Path(__file__).resolve().parent


def main():
    lines = ['DeepPro-Plus / RGB-T 复现记录',
             '更新：' + datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
             '配置：32 epochs / batch4 / seq40 / patch128 / sample_rate .05',
             '本地重建mask：11,300测试目标；作者日志11,303。',
             '原实现以测试集IoU选择best checkpoint；随机种子未固定。', '']
    for phase in ['released', 'train']:
        path = RUN / (phase + '_status.json')
        status = json.loads(path.read_text()) if path.exists() else {'state': 'not_started'}
        lines.append(phase + ' 阶段：' + status['state'])
        if 'error' in status:
            lines.append('错误：' + status['error'])
        if phase == 'train':
            log = RUN / 'log/sem_seg/retrained_rgbt_deepproplus/logs/DeepPro-Plus.txt'
            if log.exists():
                text = log.read_text()
                epochs = re.findall(r'\*\*\*\* Epoch \d+ \((\d+)/32\)', text)
                losses = re.findall(r'Training mean loss: ([0-9.]+)', text)
                best = re.findall(r'Best mIoU_mid: ([0-9.]+)', text)
                lines.append('已进入epoch：' + (epochs[-1] if epochs else '0') + '/32；完成评估轮数：' + str(len(best)))
                if losses:
                    lines.append('最近训练loss：' + losses[-1])
                if best:
                    lines.append('当前best窗口IoU：' + best[-1])
        metrics = RUN / (phase + '_metrics.json')
        if metrics.exists() and status['state'] == 'completed':
            value = json.loads(metrics.read_text())
            lines.append('AUC=%.5f, IoU@0.5=%.6f' % (value['auc'], value['iou_at_0_5']))
            for th in [.5, .99]:
                row = next(r for r in value['thresholds'] if abs(r['threshold'] - th) < 1e-9)
                lines.append('threshold=%.2f: Pd=%.4f%% (%d/%d), Fa=%.7g (%d false pixels)' %
                             (th, 100 * row['pd'], row['true_targets'], row['targets'], row['fa'], row['false_pixels']))
        lines.append('')
    lines += ['参照：论文Pd=77.83%, Fa=4.47e-5, AUC=.9730；',
              '作者存档曲线AUC重算=.96912177，.99阈值Pd/Fa与论文表格对应。',
              '完整配置、指令、环境和限制见README.txt、config.json、protocol_review.txt。']
    text = '\n'.join(lines) + '\n'
    temp = RUN / ('REPORT.txt.%d.tmp' % os.getpid())
    temp.write_text(text)
    temp.replace(RUN / 'REPORT.txt')
    print(text)


if __name__ == '__main__':
    main()
