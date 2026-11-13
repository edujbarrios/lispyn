"""Chronological train/val/test and walk-forward CV regression tests."""

import json
import shutil
import subprocess

import pytest

from lispyn.temporal import (
    temporal_three_way_indices,
    temporal_fold_indices,
    temporal_three_way_frame,
    temporal_three_way_dataset,
    temporal_frame_folds,
    temporal_dataset_folds,
    run_temporal_dataset_cv,
    run_temporal_frame_cv,
)


def test_chronological_three_way_holdout_preserves_order_with_gaps():
    training, validation, testing = temporal_three_way_indices(
        30, validation_fraction=0.2, test_fraction=0.1, gap=2
    )
    assert training == list(range(18))
    assert validation == list(range(20, 25))
    assert testing == list(range(27, 30))
    assert set(training).isdisjoint(validation)
    assert set(training).isdisjoint(testing)
    assert set(validation).isdisjoint(testing)
    assert validation[0] - training[-1] == 3
    assert testing[0] - validation[-1] == 3
    assert len(training) + len(validation) + len(testing) + 4 == 30


def test_three_way_without_gap_and_invalid_inputs():
    tr, va, te = temporal_three_way_indices(20, gap=0)
    assert max(tr) + 1 == min(va)
    assert max(va) + 1 == min(te)
    assert sorted(tr + va + te) == list(range(20))
    for invalid in [True, -1, 1.5]:
        with pytest.raises(ValueError, match="gap"):
            temporal_three_way_indices(20, gap=invalid)
    with pytest.raises(ValueError, match="Not enough samples"):
        temporal_three_way_indices(6, gap=2)
    with pytest.raises(ValueError, match="less than 1"):
        temporal_three_way_indices(40, validation_fraction=.5, test_fraction=.5)
    with pytest.raises(ValueError, match="validation_fraction"):
        temporal_three_way_indices(40, validation_fraction=0)


def test_expanding_temporal_windows_and_no_future_leakage():
    folds = list(temporal_fold_indices(30, n_splits=4, test_size=4, gap=2))
    assert len(folds) == 4
    assert [(len(tr), len(va)) for tr, va in folds] == [
        (12, 4), (16, 4), (20, 4), (24, 4)
    ]
    assert [va for _, va in folds] == [
        list(range(14, 18)),
        list(range(18, 22)),
        list(range(22, 26)),
        list(range(26, 30)),
    ]
    for tr, va in folds:
        assert tr == sorted(tr)
        assert va == sorted(va)
        assert max(tr) + 2 < min(va)
        assert not set(tr) & set(va)


def test_rolling_temporal_window_and_default_size():
    folds = list(temporal_fold_indices(
        30, n_splits=4, test_size=4, gap=2,
        min_train_size=5, max_train_size=8,
    ))
    assert len(folds) == 4
    assert all(len(tr) == 8 and len(va) == 4 for tr, va in folds)
    assert folds[0][0] == list(range(4, 12))
    assert folds[-1][0] == list(range(16, 24))
    auto = list(temporal_fold_indices(30, n_splits=5))
    assert len(auto) == 5
    assert all(len(val) == 5 for _, val in auto)
    assert auto[-1][1] == list(range(25, 30))


def test_temporal_fold_rejects_invalid_or_empty_windows():
    bad = [
        ({"n_splits": 1}, "n_splits"),
        ({"n_splits": True}, "n_splits"),
        ({"test_size": 0}, "test_size"),
        ({"test_size": 2.4}, "test_size"),
        ({"gap": -1}, "gap"),
        ({"min_train_size": 0}, "min_train_size"),
        ({"max_train_size": True}, "max_train_size"),
        ({"min_train_size": 6, "max_train_size": 5}, "max_train_size"),
        ({"n_splits": 5, "test_size": 6}, "Not enough history"),
        ({"n_splits": 4, "test_size": 5, "gap": 10}, "Not enough history"),
    ]
    for options, pattern in bad:
        with pytest.raises(ValueError, match=pattern):
            list(temporal_fold_indices(30, **options))


def test_ordered_dataframe_folds_sort_unique_timestamps_without_mutation():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        "time": [5, 1, 6, 3, 4, 2, 9, 8, 7, 10, 12, 11],
        "value": list(range(12)),
    })
    original = frame.copy(deep=True)
    splits = temporal_three_way_frame(
        frame, time_column="time", gap=1,
        validation_fraction=.2, test_fraction=.2,
    )
    assert frame.equals(original)
    times = [part.time.tolist() for part in splits]
    assert all(x == sorted(x) for x in times)
    assert max(times[0]) < min(times[1]) < min(times[2])
    assert len({item for seq in times for item in seq}) == 10
    folds = list(temporal_frame_folds(
        frame, n_splits=3, test_size=2, gap=1, time_column="time"
    ))
    assert len(folds) == 3
    for tr, va in folds:
        assert max(tr.time) < min(va.time)
        assert set(tr.time).isdisjoint(va.time)
        assert va.time.tolist() == sorted(va.time.tolist())


def test_dataframe_rejects_missing_duplicate_and_bad_time_values():
    pd = pytest.importorskip("pandas")
    for frame, error in [
        (pd.DataFrame({"t": [1, 2, 2, 4, 5, 6]}), "unique"),
        (pd.DataFrame({"t": [1, 2, None, 4, 5, 6]}), "missing"),
        (pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]}), "Missing time"),
        (pd.DataFrame({"t": [1, "2", 3, 4, 5, 6]}), "sortable"),
    ]:
        with pytest.raises(ValueError, match=error):
            list(temporal_frame_folds(
                frame, n_splits=2, test_size=2, time_column="t"
            ))


def test_pytorch_temporal_subsets_and_reports(tmp_path):
    torch = pytest.importorskip("torch")
    dataset = torch.utils.data.TensorDataset(torch.arange(40))
    tr, va, te = temporal_three_way_dataset(dataset, gap=2, test_fraction=.2)
    assert tr.dataset is dataset and va.dataset is dataset and te.dataset is dataset
    assert max(tr.indices) < min(va.indices)
    assert max(va.indices) < min(te.indices)
    records = []

    def score(train, valid, fold, repeat):
        assert train.dataset is dataset
        assert valid.dataset is dataset
        assert repeat == 0
        assert max(train.indices) + 1 < min(valid.indices)
        records.append((fold, len(train), len(valid)))
        return float(fold + 1)

    report = run_temporal_dataset_cv(
        dataset, score, n_splits=4, test_size=5, gap=1,
        max_train_size=10, min_train_size=5,
        metric="val_accuracy", output=tmp_path / "report.json",
    )
    assert report["strategy"] == "walk_forward"
    assert report["gap"] == 1
    assert report["weighted_mean"] == pytest.approx(2.5)
    assert report["mean"] == pytest.approx(2.5)
    assert report["folds"] == 4
    assert len(records) == 4
    assert all(train <= 10 and valid == 5 for _, train, valid in records)
    assert json.loads((tmp_path / "report.json").read_text()) == report
    assert "dataset" not in json.dumps(report)


def test_frame_temporal_cv_uses_only_past_records():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        "date": list(range(27, 0, -1)),
        "feature": list(range(27)),
    })
    original = frame.copy()
    seen = []

    def score(train, valid, fold, repeat):
        assert max(train.date) < min(valid.date)
        assert repeat == 0
        seen.append(fold)
        return len(valid) / 10

    report = run_temporal_frame_cv(
        frame, score, n_splits=3, test_size=4, gap=2,
        time_column="date", metric="forecast_score",
    )
    assert report["weighted_mean"] == pytest.approx(.4)
    assert seen == [0, 1, 2]
    assert frame.equals(original)


def test_temporal_hy_macros_are_compilable_and_executable(tmp_path):
    pytest.importorskip("hy")
    pytest.importorskip("torch")
    from lispyn.notebooks import build_notebook

    hy_file = tmp_path / "time_cv.hy"
    hy_file.write_text(
        '(require lispyn.macros [temporal-train-val-test temporal-folds\n'
        ' cross-validate-temporal temporal-table-split temporal-table-folds\n'
        ' cross-validate-temporal-table])\n'
        '(import torch)\n'
        '(setv dataset (torch.utils.data.TensorDataset (torch.arange 30)))\n'
        '(setv splits (temporal-train-val-test dataset :gap 1))\n'
        '(assert (= (len splits) 3))\n'
        '(setv folds (list (temporal-folds dataset :n-splits 3\n'
        '  :test-size 4 :gap 2)))\n'
        '(assert (= (len folds) 3))\n'
        '(setv report (cross-validate-temporal dataset\n'
        '  (fn [tr va fold rep] (+ 1.0 fold))\n'
        '  :n-splits 3 :test-size 4 :gap 2 :metric "accuracy"))\n'
        '(assert (= (get report "mean") 2.0))\n'
        '(print "TEMPORAL MACROS OK")\n',
        encoding="utf-8",
    )
    nb = build_notebook(hy_file)
    generated = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "temporal_three_way_dataset" in generated
    assert "temporal_dataset_folds" in generated
    assert "run_temporal_dataset_cv" in generated
    hy = shutil.which("hy")
    assert hy
    completed = subprocess.run(
        [hy, str(hy_file)], cwd=tmp_path, capture_output=True,
        text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "TEMPORAL MACROS OK" in completed.stdout


def test_tabular_temporal_macro_compilation(tmp_path):
    pytest.importorskip("hy")
    from lispyn.notebooks import build_notebook

    source = tmp_path / "time_table.hy"
    source.write_text(
        '(require lispyn.macros [temporal-table-split temporal-table-folds\n'
        ' cross-validate-temporal-table])\n'
        '(setv splits (temporal-table-split frame :time-column "date"))\n'
        '(setv folds (temporal-table-folds frame :n-splits 3 :gap 2))\n'
        '(setv report (cross-validate-temporal-table frame scorer\n'
        '  :time-column "date" :n-splits 3))\n',
        encoding="utf-8",
    )
    nb = build_notebook(source)
    compiled = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "temporal_three_way_frame" in compiled
    assert "temporal_frame_folds" in compiled
    assert "run_temporal_frame_cv" in compiled
