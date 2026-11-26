"""Summarize real ROC evaluations and export editable matplotlib figures.

No inference, GPU access, smoothing, interpolation, synthetic points, or downloads.
"""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path

POINT_FIELDS = ('threshold', 'pd', 'fa', 'true_targets', 'targets', 'false_pixels')
SUMMARY_FIELDS = ('result_id', 'model', 'label', 'arch', 'variant', 'checkpoint_status',
                  'checkpoint_sha256', 'best_epoch', 'training_epochs_completed',
                  'final_checkpoint', 'complete', 'auc', 'iou_at_0_5', 'operating_points',
                  'pd_at_0_5', 'fa_at_0_5', 'pd_at_0_99', 'fa_at_0_99',
                  'targets', 'source', 'protocol', 'metrics_path')
PALETTE = ['#557A9E', '#B68046', '#668D83', '#8A73A1', '#B36770', '#6D8892', '#8D875B', '#6F6F6F']
STATUS_STYLE = {'final': ('-', 'o'), 'interim': ('--', 's'), 'released': (':', '^')}


def read_json(path):
    return json.loads(Path(path).read_text())


def text_value(value):
    if value is None:
        return ''
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)


def finite(value, name, *, unit_interval=False):
    if isinstance(value, bool):
        raise ValueError(name + ' must be numeric, not bool')
    number = float(value)
    if not math.isfinite(number) or (unit_interval and not 0 <= number <= 1):
        raise ValueError(name + ' must be finite' + (' and within [0,1]' if unit_interval else ''))
    return number


def integer(value, name):
    number = finite(value, name)
    if number < 0 or int(number) != number:
        raise ValueError(name + ' must be a nonnegative integer')
    return int(number)


def normalize_points(raw):
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError('thresholds must contain at least two measured operating points')
    points = []
    for index, item in enumerate(raw):
        if isinstance(item, (list, tuple)):
            if len(item) != len(POINT_FIELDS):
                raise ValueError('thresholds row must have six values in the documented order')
            item = dict(zip(POINT_FIELDS, item))
        if not isinstance(item, dict):
            raise ValueError('thresholds row must be an object or six-value array')
        point = {key: finite(item[key], key, unit_interval=True) for key in ('threshold', 'pd', 'fa')}
        point.update({key: integer(item[key], key) for key in ('true_targets', 'targets', 'false_pixels')})
        if point['targets'] <= 0 or point['true_targets'] > point['targets']:
            raise ValueError('Invalid target counts at operating point ' + str(index))
        # Published text logs may round Pd to five decimal places.
        if not math.isclose(point['pd'], point['true_targets'] / point['targets'], rel_tol=0, abs_tol=1e-5):
            raise ValueError('Pd disagrees with target counts at operating point ' + str(index))
        points.append(point)
    if len({point['targets'] for point in points}) != 1:
        raise ValueError('Target denominator changes across thresholds')
    return points


def operating_point(points, threshold):
    """Return the measured point at this threshold; never interpolate or use nearest."""
    matches = [point for point in points
               if math.isclose(point['threshold'], threshold, rel_tol=0, abs_tol=1e-12)]
    if len(matches) > 1 and any(point != matches[0] for point in matches[1:]):
        raise ValueError('Conflicting measured points at threshold ' + str(threshold))
    return matches[0] if matches else None


def checkpoint_status(data):
    explicit = data.get('checkpoint_status')
    if explicit is not None:
        if explicit not in STATUS_STYLE:
            raise ValueError('checkpoint_status must be released, final, or interim')
        return explicit
    source = data.get('source')
    source_kind = source.get('kind', source.get('type', '')) if isinstance(source, dict) else source
    if data.get('released') is True or source_kind in ('released', 'author_released', 'released_checkpoint'):
        return 'released'
    return 'final' if data.get('final_checkpoint') is True else 'interim'


def result_path(root, value):
    path = Path(str(value))
    if not path.is_absolute():
        direct = root / path
        path = direct if direct.exists() or str(path).startswith('results/') else root / 'results' / path
    return path if path.name == 'metrics.json' else path / 'metrics.json'


def select_paths(root, config):
    active = None
    explicit = False
    if 'active_results' in config:
        active, explicit = config['active_results'], True
    else:
        manifest_path = root / 'manifest.json'
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if 'active_results' in manifest:
                active, explicit = manifest['active_results'], True
    if not explicit:
        return [{'path': path} for path in sorted((root / 'results').rglob('metrics.json'))], 'all_recursive_results'
    if active is None:
        raise ValueError('active_results must be a list or object; use [] to select no results')
    if isinstance(active, dict):
        converted = []
        for key, value in active.items():
            if isinstance(value, dict):
                converted.append(dict(value, job_id=value.get('job_id', key)))
            else:
                converted.append({'path': value, 'job_id': key})
        active = converted
    if not isinstance(active, list):
        raise ValueError('active_results must be a list or object')
    selections = []
    for item in active:
        spec = {'path': item} if isinstance(item, str) else dict(item)
        location = spec.get('path', spec.get('metrics_path', spec.get('job_id')))
        if not location:
            raise ValueError('Each active_results entry needs path, metrics_path, or job_id')
        spec['path'] = result_path(root, location)
        selections.append(spec)
    return selections, 'explicit_active_results'


def load_results(root, config):
    selections, mode = select_paths(root, config)
    records, excluded, seen = [], [], set()
    for spec in selections:
        path = Path(spec['path'])
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        try:
            data = read_json(path)
            expected_hash = spec.get('checkpoint_sha256')
            if expected_hash and data.get('checkpoint_sha256') != expected_hash:
                raise ValueError('Checkpoint SHA256 does not match active_results selection')
            if data.get('complete') is not True:
                raise ValueError('Evaluation is incomplete; excluded from curves and summary metrics')
            for required in ('model', 'label', 'arch', 'checkpoint_sha256', 'variant'):
                if required not in data:
                    raise ValueError('Missing required field: ' + required)
            digest = str(data['checkpoint_sha256'])
            if len(digest) != 64 or any(char not in '0123456789abcdefABCDEF' for char in digest):
                raise ValueError('checkpoint_sha256 must be a 64-digit hexadecimal digest')
            points = normalize_points(data['thresholds'])
            auc = finite(data['auc'], 'auc', unit_interval=True)
            iou = data.get('iou_at_0_5')
            if iou is not None:
                iou = finite(iou, 'iou_at_0_5', unit_interval=True)
            status = checkpoint_status(data)
            try:
                relative = str(path.relative_to(root))
            except ValueError:
                relative = str(path)
            result_id = data.get('result_id', spec.get('job_id', str(Path(relative).parent)))
            row = {key: data.get(key) for key in SUMMARY_FIELDS}
            row.update(result_id=result_id, checkpoint_status=status, auc=auc, iou_at_0_5=iou,
                       operating_points=len(points), targets=points[0]['targets'], metrics_path=relative,
                       final_checkpoint=data.get('final_checkpoint') is True, complete=True,
                       source=data.get('source'), protocol=data.get('protocol'), points=points)
            for threshold, suffix in ((0.5, '0_5'), (0.99, '0_99')):
                point = operating_point(points, threshold)
                row['pd_at_' + suffix] = point['pd'] if point is not None else None
                row['fa_at_' + suffix] = point['fa'] if point is not None else None
            records.append(row)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            excluded.append({'path': str(path), 'reason': str(error), 'job_id': spec.get('job_id')})
    return records, excluded, mode


def pending_items(config):
    pending = config.get('pending_results', config.get('pending', config.get('pending_models', [])))
    if isinstance(pending, dict):
        pending = [dict(value, label=value.get('label', key)) if isinstance(value, dict)
                   else {'label': key, 'reason': value} for key, value in pending.items()]
    if not isinstance(pending, list):
        raise ValueError('pending_results must be a list or object')
    out = []
    for item in pending:
        if isinstance(item, str):
            out.append(item)
        else:
            name = item.get('label', item.get('model', item.get('job_id', 'Pending result')))
            reason = item.get('reason', item.get('status', 'awaiting evaluation'))
            out.append(str(name) + ': ' + str(reason))
    return out


def write_csv(path, fields, rows):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: text_value(row.get(key)) for key in fields})
    temporary.replace(path)


def plot_label(record):
    status = record['checkpoint_status']
    status_text = {'released': 'released', 'final': 'final training', 'interim': 'interim training'}[status]
    if record.get('best_epoch') is not None:
        status_text += ', best e' + str(record['best_epoch'])
    if status == 'interim' and record.get('training_epochs_completed') is not None:
        status_text += '/' + str(record['training_epochs_completed'])
    variant = str(record.get('variant') or '')
    variant_text = '' if variant in ('', 'default', 'native') or variant in record['label'] else ' · ' + variant
    return '%s%s [%s]  AUC %.4f' % (record['label'], variant_text, status_text, record['auc'])


def render_figure(records, output, pending_count):
    try:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except ImportError as exc:
        raise RuntimeError('Python figure backend unavailable: ' + str(exc) +
                           '. Run with the supplied compatible Python image; no alternate backend was used.') from exc
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7,
                         'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.7,
                         'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
                         'legend.frameon': False, 'savefig.facecolor': 'white'})
    legend_rows = max(1, len(records))
    height_mm = max(112, 78 + legend_rows * 5)
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, height_mm / 25.4),
                             gridspec_kw={'width_ratios': [1, 1.45]}, sharey=True)
    bottom = min(0.49, (16 + legend_rows * 4.6) / height_mm)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.88, bottom=bottom, wspace=0.24)
    families = list(dict.fromkeys(str(record['model']) for record in records))
    colors = {family: PALETTE[index % len(PALETTE)] for index, family in enumerate(families)}
    handles, labels = [], []
    variants = {}
    marker_sets = {'final': ('o', 'D', 'P', 'h', '*'), 'interim': ('s', 'd', 'X', 'p', '8'),
                   'released': ('^', 'v', '<', '>')}
    for record in records:
        points = record['points']
        x = [point['fa'] for point in points]
        y = [point['pd'] for point in points]
        linestyle, marker = STATUS_STYLE[record['checkpoint_status']]
        group = (str(record['model']), record['checkpoint_status'])
        variant = str(record.get('variant') or '')
        if variant not in variants.setdefault(group, []):
            variants[group].append(variant)
        marker = marker_sets[record['checkpoint_status']][variants[group].index(variant) % len(marker_sets[record['checkpoint_status']])]
        for index, axis in enumerate(axes):
            line, = axis.plot(x, y, color=colors[str(record['model'])], linestyle=linestyle,
                              marker=marker, markersize=2.5, linewidth=1.15, markeredgewidth=0.6,
                              markerfacecolor='white' if record['checkpoint_status'] == 'interim' else colors[str(record['model'])],
                              clip_on=True)
            if index == 0:
                handles.append(line)
                labels.append(plot_label(record))
    for index, (axis, title, maximum) in enumerate(zip(axes,
            ('Full false-alarm range', 'Low false-alarm range'), (1.0, 1e-3))):
        axis.set_xlim(0, maximum)
        axis.set_ylim(0, 1.025)
        axis.set_xlabel('False-alarm pixel rate, $F_a$')
        axis.set_title(title, fontsize=8, pad=9)
        axis.text(-0.14, 1.085, chr(ord('a') + index), transform=axis.transAxes,
                  fontsize=8, fontweight='bold', va='top')
        axis.grid(axis='y', color='#D9DDE0', linewidth=0.45, alpha=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=6.5, length=3)
        if not records:
            axis.text(0.5, 0.5, 'Awaiting completed evaluations', transform=axis.transAxes,
                      ha='center', va='center', color='#666666', fontsize=8)
    axes[0].set_ylabel('Detection probability, $P_d$')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, -3))
    axes[1].xaxis.set_major_formatter(formatter)
    if handles:
        fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.075, 0.045),
                   fontsize=6.5, ncol=1, handlelength=2.8, labelspacing=0.6, borderaxespad=0)
    footer = 'Measured threshold points; lines only connect observations. No smoothing or added endpoints.'
    if pending_count:
        footer += '  Pending/excluded: %d.' % pending_count
    fig.text(0.085, 0.015, footer, fontsize=5.8, color='#555555')
    fig.canvas.draw()
    exported = []
    for extension in ('png', 'pdf', 'svg'):
        final = output / ('ROC.' + extension)
        temporary = output / ('ROC.tmp.' + extension)
        fig.savefig(temporary, format=extension, dpi=600, bbox_inches='tight', pad_inches=0.045)
        temporary.replace(final)
        exported.append(final.name)
    plt.close(fig)
    return {'backend': 'matplotlib', 'matplotlib_version': mpl.__version__, 'files': exported,
            'width_mm_before_tight_bbox': 183, 'height_mm_before_tight_bbox': height_mm,
            'raw_operating_points': sum(record['operating_points'] for record in records)}


def report_text(records, pending, excluded, selection_mode, rendering):
    lines = ['RGB-T unified ROC evaluation',
        'Updated: ' + datetime.now(timezone(timedelta(hours=8), 'Asia/Taipei')).isoformat(),
             'Selection: ' + selection_mode,
             'Only complete evaluations are plotted; checkpoint training status is explicit.',
             'AUC is copied from the evaluation result. No endpoints, samples, smoothing, or confidence intervals are added.',
             'Full Fa range: 0..1; low-Fa range: 0..0.001. Zero-Fa points are retained.',
             'Pd/Fa summaries use measured thresholds 0.5 and 0.99; missing points remain unavailable.', '']
    for record in records:
        iou = 'unavailable' if record['iou_at_0_5'] is None else '%.8f' % record['iou_at_0_5']
        lines.append('%s | %s | variant=%s | AUC=%.8f | IoU@0.5=%s | targets=%s | points=%s' % (
            record['label'], record['checkpoint_status'], record['variant'], record['auc'], iou,
            record['targets'], record['operating_points']))
        for threshold in (0.5, 0.99):
            point = operating_point(record['points'], threshold)
            if point is None:
                lines.append('  threshold=%.2f | Pd/Fa unavailable: no measured point at this threshold.' % threshold)
                continue
            lines.append('  threshold=%.2f | Pd=%.8f (%.4f%%), detected targets=%d/%d | '
                         'Fa=%.8e, false-positive pixels=%d' % (
                             threshold, point['pd'], 100 * point['pd'], point['true_targets'],
                             point['targets'], point['fa'], point['false_pixels']))
        lines.append('  best_epoch=%s; training_epochs_completed=%s; checkpoint_sha256=%s' % (
            record.get('best_epoch'), record.get('training_epochs_completed'), record['checkpoint_sha256']))
        lines.append('  source=' + text_value(record['source']))
        lines.append('  protocol=' + text_value(record['protocol']))
    if not records:
        lines.append('No complete, valid, active result is available. Figures contain no curves.')
    denominators = sorted({record['targets'] for record in records})
    if len(denominators) > 1:
        lines.extend(['', 'Protocol difference: target totals differ across curves: ' + ', '.join(map(str, denominators)) +
                      '. These results must not be treated as an identical-ground-truth comparison.'])
    lines.extend(['', 'Pending results:'])
    lines.extend(['  - ' + item for item in pending] or ['  None declared in config.json.'])
    lines.extend(['', 'Excluded or unavailable selected results:'])
    lines.extend(['  - ' + item['path'] + ': ' + item['reason'] for item in excluded] or ['  None.'])
    lines.extend(['', 'Exports: metrics.csv, roc_points.csv, ROC.png, ROC.pdf, ROC.svg.',
                  'Rendering: ' + text_value(rendering),
                  'QA: numeric validation and editable vector export settings applied; visual review of the exported figure is still required.'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, help='Output folder; default is --root')
    parser.add_argument('--no-plot', action='store_true', help='Write source tables/report without opening a graphics backend')
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root
    output.mkdir(parents=True, exist_ok=True)
    config_path = root / 'config.json'
    config = read_json(config_path) if config_path.exists() else {}
    records, excluded, mode = load_results(root, config)
    pending = pending_items(config)
    write_csv(output / 'metrics.csv', SUMMARY_FIELDS, records)
    point_rows = []
    for record in records:
        for index, point in enumerate(record['points']):
            point_rows.append({key: record.get(key) for key in ('result_id', 'model', 'label', 'arch', 'variant',
                                                              'checkpoint_status', 'checkpoint_sha256')} |
                              {'point_index': index, **point})
    write_csv(output / 'roc_points.csv', ('result_id', 'model', 'label', 'arch', 'variant', 'checkpoint_status',
                                          'checkpoint_sha256', 'point_index', *POINT_FIELDS), point_rows)
    rendering = {'status': 'not_requested'} if args.no_plot else render_figure(records, output, len(pending) + len(excluded))
    report = report_text(records, pending, excluded, mode, rendering)
    temporary = output / 'REPORT.txt.tmp'
    temporary.write_text(report)
    temporary.replace(output / 'REPORT.txt')
    print(json.dumps({'complete_results': len(records), 'pending': len(pending), 'excluded': len(excluded),
                      'selection': mode, 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
