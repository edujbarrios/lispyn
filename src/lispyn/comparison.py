"""Summarize and compare saved experiment runs without importing PyTorch.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .plots import _read_history, _values


def compare_experiments(directories, *, metric='loss', split='val', mode=None):
    """Return comparable summaries for two or more saved Lispyn histories.

    Each run is labeled by its directory name. Minimize loss by default;
    maximize other metrics. Use mode='min'/'max' to override.
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
    return {'metric': metric, 'split': split, 'mode': direction, 'runs': rows}


def format_comparison(summary):
    """Present the experiment summary as a Markdown table."""
    lines = [
        f'### {summary["split"]} {summary["metric"]} comparison ({summary["mode"]})',
        '',
        '| Run | Epochs | Best epoch | Best | Final |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for row in summary['runs']:
        name = row['run'].replace('|', '\\|').replace('\n', ' ')
        lines.append(
            f'| {name} | {row["epochs"]} | {row["best_epoch"]} | '
            f'{row["best"]:.6g} | {row["final"]:.6g} |'
        )
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
            writer = csv.DictWriter(
                stream, fieldnames=['run', 'epochs', 'best_epoch', 'best', 'final']
            )
            writer.writeheader()
            writer.writerows(summary['runs'])
    return target
