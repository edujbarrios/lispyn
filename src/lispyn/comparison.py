"""Summarize and compare saved experiment runs without importing PyTorch.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .plots import _read_history, _values


def compare_experiments(directories, *, metric='loss', split='val', mode=None,
                        baseline=None):
    """Return comparable summaries for two or more saved Lispyn histories.

    Each run is labeled by its directory name. Minimize loss by default;
    maximize other metrics. Use mode='min'/'max' to override. With a
    baseline run name, add direction-aware gains (positive is better).
    """
    if split not in ('train', 'val'):
        raise ValueError("split must be 'train' or 'val'")
    if not isinstance(metric, str) or not metric or metric.startswith(('train_', 'val_')):
        raise ValueError('metric must be an unprefixed metric name')
    if mode not in (None, 'min', 'max'):
        raise ValueError("mode must be 'min', 'max' or None")
    paths = [Path(path).expanduser() for path in directories]
    if len(paths) < 2:
        raise ValueError('Compare at least two run directories')
    names = [path.name for path in paths]
    if len(set(names)) != len(names):
        raise ValueError('Run directories must have distinct names')
    direction = mode or ('min' if 'loss' in metric.lower() else 'max')
    key = f'{split}_{metric}'
    rows = []
    for directory in paths:
        values = _values(_read_history(directory / 'history.json'), key)
        if not values:
            raise ValueError(f'No {key} values recorded for {directory}')
        best = min(values) if direction == 'min' else max(values)
        rows.append({
            'run': directory.name,
            'epochs': len(values),
            'best_epoch': values.index(best) + 1,
            'best': best,
            'final': values[-1],
        })
    summary = {'metric': metric, 'split': split, 'mode': direction, 'runs': rows}
    if baseline is not None:
        reference = next((row for row in rows if row['run'] == baseline), None)
        if reference is None:
            raise ValueError(f"Unknown baseline run: {baseline!r}")
        sign = -1 if direction == 'min' else 1
        for row in rows:
            row['best_gain'] = sign * (row['best'] - reference['best'])
            row['final_gain'] = sign * (row['final'] - reference['final'])
        summary['baseline'] = baseline
    return summary


def format_comparison(summary):
    """Present the experiment summary as a Markdown table."""
    baseline = summary.get('baseline')
    suffix = f" vs {baseline}" if baseline is not None else ""
    header = '| Run | Epochs | Best epoch | Best | Final |'
    divider = '| --- | ---: | ---: | ---: | ---: |'
    if baseline is not None:
        header += ' Best gain | Final gain |'
        divider += ' ---: | ---: |'
    lines = [
        f'### {summary["split"]} {summary["metric"]} comparison ({summary["mode"]}){suffix}',
        '',
        header, divider,
    ]
    for row in summary['runs']:
        name = row['run'].replace('|', '\\|').replace('\n', ' ')
        line = (f'| {name} | {row["epochs"]} | {row["best_epoch"]} | '
                f'{row["best"]:.6g} | {row["final"]:.6g} |')
        if baseline is not None:
            line += f' {row["best_gain"]:+.6g} | {row["final_gain"]:+.6g} |'
        lines.append(line)
    return '\n'.join(lines)

def save_comparison(summary, output):
    """Export a comparison table to Markdown, JSON or CSV."""
    target = Path(output).expanduser()
    suffix = target.suffix.lower()
    if suffix not in ('.md', '.json', '.csv'):
        raise ValueError('Comparison output must be .md, .json or .csv')
    target.parent.mkdir(parents=True, exist_ok=True)
    if suffix == '.md':
        target.write_text(format_comparison(summary) + '\n', encoding='utf-8')
    elif suffix == '.json':
        target.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    else:
        with target.open('w', encoding='utf-8', newline='') as stream:
            columns = ['run', 'epochs', 'best_epoch', 'best', 'final']
            if summary.get('baseline') is not None:
                columns += ['best_gain', 'final_gain']
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(summary['runs'])
    return target
