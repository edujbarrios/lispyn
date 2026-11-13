"""Compare recorded experiment runs without requiring PyTorch or Matplotlib."""

import csv
import json

import pytest

from lispyn.comparison import compare_experiments, format_comparison, save_comparison


def make_run(root, name, data):
    path = root/name
    path.mkdir(parents=True)
    (path/'history.json').write_text(json.dumps(data))
    return path


def test_multi_run_comparison_with_unequal_epochs_and_exports(tmp_path):
    one = make_run(tmp_path, 'baseline', {'val_loss': [1.0, 0.8, 0.9]})
    two = make_run(tmp_path, 'finetuned', {'val_loss': [0.7, 0.5]})
    result = compare_experiments([one, two])
    assert result['mode'] == 'min'
    assert result['runs'][0] == {
        'run': 'baseline', 'epochs': 3, 'best_epoch': 2, 'best': 0.8, 'final': 0.9
    }
    assert result['runs'][1]['best'] == 0.5
    assert '| baseline |' in format_comparison(result)
    md = save_comparison(result, tmp_path/'summary.md')
    js = save_comparison(result, tmp_path/'summary.json')
    csv_path = save_comparison(result, tmp_path/'summary.csv')
    assert 'baseline' in md.read_text()
    assert json.loads(js.read_text()) == result
    with csv_path.open(newline='') as stream:
        assert list(csv.DictReader(stream))[0]['best_epoch'] == '2'


def test_metric_mode_auto_and_override(tmp_path):
    one = make_run(tmp_path, 'a', {'train_accuracy': [0.4, 0.8, 0.7]})
    two = make_run(tmp_path, 'b', {'train_accuracy': [0.5, 0.6]})
    best = compare_experiments([one, two], metric='accuracy', split='train')
    assert best['mode'] == 'max'
    assert best['runs'][0]['best'] == 0.8
    assert best['runs'][0]['best_epoch'] == 2
    minimize = compare_experiments([one, two], metric='accuracy',
                                   split='train', mode='min')
    assert minimize['runs'][0]['best_epoch'] == 1


def test_invalid_comparison_reports(tmp_path):
    a = make_run(tmp_path/'a', 'repeat', {'val_loss': [1.0]})
    b = make_run(tmp_path/'b', 'repeat', {'val_loss': [0.8]})
    with pytest.raises(ValueError, match='distinct names'):
        compare_experiments([a, b])
    with pytest.raises(ValueError, match='at least two'):
        compare_experiments([a])
    with pytest.raises(ValueError, match='split'):
        compare_experiments([a, b], split='test')
    with pytest.raises(ValueError, match='mode'):
        compare_experiments([a, b], mode='highest')
    with pytest.raises(ValueError, match='output'):
        save_comparison({'runs': []}, tmp_path/'other.zip')
    c = make_run(tmp_path, 'no_val', {'train_loss': [1.0]})
    with pytest.raises(ValueError, match='No val_loss'):
        compare_experiments([a, c])


def test_cli_comparison_table_and_plot(tmp_path, capsys):
    from lispyn.cli import main
    one = make_run(tmp_path, 'baseline', {'val_loss': [1.0, 0.7]})
    two = make_run(tmp_path, 'candidate', {'val_loss': [0.8, 0.5]})
    output = tmp_path/'comparison.md'
    assert main(['compare', str(one), str(two), '--output', str(output)]) == 0
    captured = capsys.readouterr()
    assert 'baseline' in captured.out and 'candidate' in captured.out
    assert output.exists()

    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    chart = tmp_path/'curves.svg'
    assert main(['compare', str(one), str(two), '--plot', str(chart),
                 '--palette', 'turquoise-spectrum']) == 0
    assert chart.stat().st_size > 1000


def test_cli_reports_validation_error(tmp_path, capsys):
    from lispyn.cli import main
    assert main(['compare', str(tmp_path)]) == 1
    assert 'at least two' in capsys.readouterr().err
