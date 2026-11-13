"""Native classification reports and reusable Hy macro integration."""

import json
from pathlib import Path

import pytest

from lispyn.classification import classification_report, format_classification_report


def test_precision_recall_f1_and_macro_weighted_averages(tmp_path):
    actual = ['cat', 'cat', 'dog', 'dog', 'dog']
    predicted = ['cat', 'dog', 'dog', 'cat', 'dog']
    report = classification_report(actual, predicted, classes=['cat', 'dog', 'bird'])
    assert report['accuracy'] == 3 / 5
    assert report['total'] == 5
    by_label = {row['label']: row for row in report['per_class']}
    assert by_label['cat']['precision'] == 0.5
    assert by_label['cat']['recall'] == 0.5
    assert by_label['dog']['precision'] == 2 / 3
    assert by_label['dog']['recall'] == 2 / 3
    assert by_label['dog']['f1'] == pytest.approx(2 / 3)
    assert by_label['dog']['support'] == 3
    assert by_label['bird'] == {
        'label': 'bird', 'precision': 0.0, 'recall': 0.0,
        'f1': 0.0, 'support': 0
    }
    assert report['macro_avg']['recall'] == pytest.approx((0.5 + 2/3) / 3)
    assert report['weighted_avg']['recall'] == pytest.approx(0.6)
    assert '| Macro average |' in format_classification_report(report)
    assert '| Weighted average |' in format_classification_report(report)

    js = tmp_path / 'metrics.json'
    md = tmp_path / 'metrics.md'
    classification_report(actual, predicted, classes=['cat', 'dog'], save=js)
    classification_report(actual, predicted, classes=['cat', 'dog'], save=md)
    assert json.loads(js.read_text())['total'] == 5
    assert 'Classification report' in md.read_text()


def test_no_predictions_of_a_class_and_markdown_label_escaping():
    result = classification_report(['a', 'b'], ['a', 'a'])
    assert result['per_class'][1]['precision'] == 0
    assert result['per_class'][1]['recall'] == 0
    assert result['per_class'][1]['f1'] == 0
    escaped = classification_report(['a|b'], ['a|b'])
    assert 'a\\|b' in format_classification_report(escaped)


def test_reject_unsupported_report_exports(tmp_path):
    with pytest.raises(ValueError, match='save path'):
        classification_report([1], [1], save=tmp_path/'x.csv')
    for digits in (True, -1, 9, 0.5):
        with pytest.raises(ValueError, match='digits'):
            format_classification_report(classification_report([1], [1]), digits=digits)


def test_macro_runs_from_hy_and_compiles_to_python(tmp_path):
    import shutil, subprocess
    from lispyn.notebooks import build_notebook
    hy = shutil.which('hy')
    assert hy
    src = tmp_path / 'classify.hy'
    src.write_text(
        '(require lispyn.macros [classification-report])\n'
        '(setv report (classification-report ["a" "a" "b"] ["a" "b" "b"]))\n'
        '(assert (= (get report "total") 3))\n'
        '(print "report-macro-ok")\n'
    )
    run = subprocess.run([hy, str(src)], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert 'report-macro-ok' in run.stdout
    nb = build_notebook(src)
    code = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    assert 'classification_report' in code
    assert all(c.get('execution_count') is None for c in nb.cells if c.cell_type=='code')
