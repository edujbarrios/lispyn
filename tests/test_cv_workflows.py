"""Regression tests for repeated and callback-driven Hy-first CV workflows."""

import json
import math
import shutil
import subprocess

import pytest

from lispyn.cv_workflows import (
    repeated_k_fold_indices,
    repeated_k_fold_dataset,
    repeated_k_fold_split,
    run_dataset_cross_validation,
    run_frame_cross_validation,
    save_cv_report,
)


def test_repeated_folds_hold_out_each_sample_once_per_repeat():
    folds = list(repeated_k_fold_indices(41, n_splits=5, n_repeats=3, seed=18))
    assert len(folds) == 15
    assert folds == list(repeated_k_fold_indices(41, n_splits=5, n_repeats=3, seed=18))
    for repeat in range(3):
        validations = []
        for r, fold, train, valid in folds:
            if r != repeat:
                continue
            assert fold in range(5)
            assert set(train).isdisjoint(valid)
            assert set(train) | set(valid) == set(range(41))
            validations.extend(valid)
        assert sorted(validations) == list(range(41))
    assert [valid for r, _, _, valid in folds if r == 0] != [
        valid for r, _, _, valid in folds if r == 1
    ]


def test_repeated_group_folds_do_not_leak_subjects():
    groups = ["subject-" + str(i // 4) for i in range(40)]
    folds = list(repeated_k_fold_indices(
        40, n_splits=5, n_repeats=2, groups=(v for v in groups), seed=17
    ))
    assert len(folds) == 10
    for repeat in range(2):
        seen = []
        for r, _, train, valid in folds:
            if r != repeat:
                continue
            assert {groups[i] for i in train}.isdisjoint(
                {groups[i] for i in valid}
            )
            seen.extend(groups[i] for i in valid)
        assert sorted(seen) == sorted(groups)


def test_repeated_stratification_and_input_validation():
    labels = ["a"] * 15 + ["b"] * 15
    folds = list(repeated_k_fold_indices(
        30, n_splits=5, n_repeats=2, labels=labels, seed=27
    ))
    for _, _, _, valid in folds:
        assert sum(labels[index] == "a" for index in valid) == 3
        assert sum(labels[index] == "b" for index in valid) == 3
    for count in [0, True, -1, 1.4]:
        with pytest.raises(ValueError, match="n_repeats"):
            list(repeated_k_fold_indices(30, n_splits=5, n_repeats=count))
    with pytest.raises(ValueError, match="shuffle=True"):
        list(repeated_k_fold_indices(30, n_repeats=2, shuffle=False))
    with pytest.raises(ValueError, match="seed"):
        list(repeated_k_fold_indices(30, n_repeats=2, seed=True))
    with pytest.raises(ValueError, match="samples per class"):
        list(repeated_k_fold_indices(10, n_splits=3, labels=[0] * 2 + [1] * 8))


def test_dataset_cv_report_and_json_export(tmp_path):
    torch = pytest.importorskip("torch")
    dataset = torch.utils.data.TensorDataset(torch.arange(23))
    seen = []

    def scorer(training, validation, fold, repeat):
        assert training.dataset is dataset
        assert validation.dataset is dataset
        assert set(training.indices).isdisjoint(validation.indices)
        seen.append((repeat, fold, len(validation)))
        return len(validation) / 10

    report = run_dataset_cross_validation(
        dataset, scorer, n_splits=4, n_repeats=2,
        metric="length", output=tmp_path / "cv-report.json",
    )
    assert report["metric"] == "length"
    assert report["folds"] == 8
    assert report["n_splits"] == 4
    assert report["n_repeats"] == 2
    assert report["weighted_mean"] == pytest.approx(
        sum(size * (size / 10) for _, _, size in seen)
        / sum(size for _, _, size in seen)
    )
    assert report["mean"] == pytest.approx(
        sum(size / 10 for _, _, size in seen) / 8
    )
    assert sorted((r, f) for r, f, _ in seen) == [
        (r, f) for r in range(2) for f in range(4)
    ]
    assert json.loads((tmp_path / "cv-report.json").read_text()) == report
    assert all("score" in row and "train_size" in row
               for row in report["fold_results"])
    assert "predictions" not in json.dumps(report)
    assert "dataset" not in json.dumps(report)


def test_group_cv_runner_no_leakage_and_fresh_fold_callback():
    torch = pytest.importorskip("torch")
    dataset = torch.utils.data.TensorDataset(torch.arange(24))
    groups = [i // 3 for i in range(24)]
    callback_runs = []

    def fresh_scorer(training, validation, fold, repeat):
        callback_runs.append((repeat, fold))
        train_groups = {groups[i] for i in training.indices}
        valid_groups = {groups[i] for i in validation.indices}
        assert train_groups.isdisjoint(valid_groups)
        return 1.0

    report = run_dataset_cross_validation(
        dataset, fresh_scorer, n_splits=4, n_repeats=2, groups=groups,
    )
    assert len(callback_runs) == 8
    assert report["mean"] == 1.0
    assert report["std"] == 0.0
    assert report["weighted_mean"] == 1.0


def test_frame_cv_runner_supports_grouping_and_strata(tmp_path):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        "id": list(range(30)),
        "class": [0] * 15 + [1] * 15,
        "session": [i // 3 for i in range(30)],
    })
    original = frame.copy(deep=True)
    seen = []

    def scorer(training, validation, fold, repeat):
        assert set(training.session).isdisjoint(validation.session)
        seen.extend(validation.id.tolist())
        return 0.8 + 0.01 * fold

    report = run_frame_cross_validation(
        frame, scorer, n_splits=5, n_repeats=2, group="session",
        metric="accuracy",
        output=tmp_path / "frame-report.json",
    )
    assert len(seen) == 60
    assert sorted(seen) == sorted(frame.id.tolist() * 2)
    assert frame.equals(original)
    assert report["folds"] == 10
    assert json.loads((tmp_path / "frame-report.json").read_text()) == report

    folds = list(repeated_k_fold_split(
        frame, n_splits=5, n_repeats=2, stratify="class"
    ))
    assert len(folds) == 10
    assert all(set(validation["class"]) == {0, 1}
               for _, _, _, validation in folds)


def test_repeated_dataset_adapter_and_error_cases(tmp_path):
    torch = pytest.importorskip("torch")
    dataset = torch.utils.data.TensorDataset(torch.arange(20))
    subsets = list(repeated_k_fold_dataset(dataset, n_splits=5, n_repeats=2))
    assert len(subsets) == 10
    assert all(train.dataset is dataset and val.dataset is dataset
               for _, _, train, val in subsets)
    with pytest.raises(TypeError, match="scorer"):
        run_dataset_cross_validation(dataset, None, n_splits=2)
    for bad in [True, float("nan"), "0.5", float("inf")]:
        with pytest.raises(ValueError, match="finite real"):
            run_dataset_cross_validation(dataset, lambda *_: bad, n_splits=2)
    with pytest.raises(ValueError, match=".json"):
        save_cv_report({"x": 1}, tmp_path / "invalid.csv")
    with pytest.raises(ValueError, match="grouped or stratified"):
        list(repeated_k_fold_dataset(dataset, n_splits=2,
             groups=[0] * 10 + [1] * 10, labels=[0] * 20))
    with pytest.raises(ValueError, match="requires labels"):
        list(repeated_k_fold_dataset(dataset, n_splits=2, stratify=True))


def test_hy_macros_compile_and_execute(tmp_path):
    pytest.importorskip("hy")
    pytest.importorskip("torch")
    from lispyn.notebooks import build_notebook

    source = tmp_path / "macro_cv.hy"
    source.write_text(
        '(require lispyn.macros [cross-validate cross-validate-table\n'
        ' repeated-cross-validation-folds repeated-tabular-folds\n'
        ' tabular-train-val-test tabular-cross-validation-folds\n'
        ' save-cross-validation])\n'
        '(import torch)\n'
        '(setv dataset (torch.utils.data.TensorDataset (torch.arange 20)))\n'
        '(setv result (cross-validate dataset\n'
        '  (fn [tr va fold rep] (/ (len va) (len dataset)))\n'
        '  :n-splits 5 :n-repeats 2 :metric "coverage"))\n'
        '(assert (= (get result "folds") 10))\n'
        '(assert (= (len (list (repeated-cross-validation-folds dataset\n'
        '  :n-splits 5 :n-repeats 2))) 10))\n'
        '(print "CV MACROS OK")\n',
        encoding="utf-8",
    )
    notebook = build_notebook(source)
    generated = "\n".join(cell.source for cell in notebook.cells
                          if cell.cell_type == "code")
    assert "run_dataset_cross_validation" in generated
    assert "repeated_k_fold_dataset" in generated
    hy_executable = shutil.which("hy")
    assert hy_executable, "Hy executable missing despite installed hy"
    process = subprocess.run(
        [hy_executable, str(source)], cwd=tmp_path,
        capture_output=True, text=True, timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert "CV MACROS OK" in process.stdout


def test_tabular_hy_macros_compile(tmp_path):
    pytest.importorskip("hy")
    from lispyn.notebooks import build_notebook

    source = tmp_path / "tabular_cv.hy"
    source.write_text(
        '(require lispyn.macros [tabular-train-val-test\n'
        ' tabular-cross-validation-folds repeated-tabular-folds\n'
        ' cross-validate-table save-cross-validation])\n'
        '(setv parts (tabular-train-val-test frame :group "subject"))\n'
        '(setv folds (tabular-cross-validation-folds frame :n-splits 3))\n'
        '(setv repeats (repeated-tabular-folds frame :n-splits 3))\n'
        '(setv summary (cross-validate-table frame scorer :n-splits 3))\n'
        '(save-cross-validation summary "report.json")\n',
        encoding="utf-8",
    )
    nb = build_notebook(source)
    source_code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "train_validation_test_split" in source_code
    assert "k_fold_split" in source_code
    assert "repeated_k_fold_split" in source_code
    assert "run_frame_cross_validation" in source_code
    assert "save_cv_report" in source_code
