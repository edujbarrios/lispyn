"""Three-way train/val/test and k-fold regression tests, no sklearn."""

import math
import random

import pytest

from lispyn.cross_validation import (
    k_fold_indices, summarize_cv_scores, three_way_indices,
)
from lispyn.data import (
    k_fold_dataset, k_fold_split, split_dataset_three,
    train_validation_test_split,
)


def _assert_three_disjoint(partitions, count):
    sets = [set(partition) for partition in partitions]
    assert all(sets)
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]
    assert set.union(*sets) == set(range(count))


def _assert_folds(folds, count):
    validations = []
    for training, validation in folds:
        train_set, val_set = set(training), set(validation)
        assert train_set and val_set
        assert not train_set & val_set
        assert train_set | val_set == set(range(count))
        validations.extend(validation)
    assert sorted(validations) == list(range(count))


def test_three_way_random_exact_sizes_and_seed_isolation():
    random.seed(11)
    expected = random.random()
    random.seed(11)
    parts = three_way_indices(50, validation_fraction=0.2, test_fraction=0.1, seed=42)
    assert random.random() == expected
    _assert_three_disjoint(parts, 50)
    assert [len(part) for part in parts] == [35, 10, 5]
    assert parts == three_way_indices(50, validation_fraction=0.2,
                                      test_fraction=0.1, seed=42)
    assert parts != three_way_indices(50, validation_fraction=0.2,
                                      test_fraction=0.1, seed=43)
    ordered = three_way_indices(10, shuffle=False)
    _assert_three_disjoint(ordered, 10)
    assert all(part == sorted(part) for part in ordered)


def test_three_way_stratified_has_all_classes_in_each_partition():
    labels = ["cat"] * 20 + ["dog"] * 15 + ["bird"] * 10
    parts = three_way_indices(
        len(labels), labels=labels, validation_fraction=.2,
        test_fraction=.2, seed=17,
    )
    _assert_three_disjoint(parts, len(labels))
    for indices in parts:
        assert set(labels[index] for index in indices) == {"cat", "dog", "bird"}
    assert parts == three_way_indices(len(labels), labels=labels,
                                      validation_fraction=.2, test_fraction=.2,
                                      seed=17)


def test_three_way_grouped_never_leaks_subjects():
    groups = (["a"] * 8 + ["b"] * 7 + ["c"] * 5 +
              ["d"] * 4 + ["e"] * 3 + ["f"] * 3)
    parts = three_way_indices(len(groups), groups=groups, seed=3)
    _assert_three_disjoint(parts, len(groups))
    group_sets = [set(groups[index] for index in part) for part in parts]
    assert all(group_sets)
    assert all(not group_sets[a] & group_sets[b]
               for a in range(3) for b in range(a + 1, 3))


def test_three_way_dataframes_do_not_mutate_and_support_groups_strata():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        "id": list(range(36)),
        "label": ["a"] * 18 + ["b"] * 18,
        "subject": [f"p{i // 3}" for i in range(36)],
    })
    copy = frame.copy(deep=True)
    tr, va, te = train_validation_test_split(frame, stratify="label", seed=11)
    assert sorted(tr.id.tolist() + va.id.tolist() + te.id.tolist()) == list(range(36))
    for sample in (tr, va, te):
        assert set(sample.label) == {"a", "b"}
    assert frame.equals(copy)
    grouped = train_validation_test_split(frame, group="subject", seed=11)
    group_ids = [set(part.subject) for part in grouped]
    assert all(not group_ids[a] & group_ids[b]
               for a in range(3) for b in range(a + 1, 3))
    assert tuple(x.id.tolist() for x in grouped) == tuple(
        x.id.tolist() for x in train_validation_test_split(frame, group="subject", seed=11)
    )


def test_three_way_pytorch_subsets_are_disjoint():
    torch = pytest.importorskip("torch")
    dataset = torch.utils.data.TensorDataset(torch.arange(30))
    labels = [0] * 15 + [1] * 15
    tr, va, te = split_dataset_three(dataset, labels=labels, seed=23)
    _assert_three_disjoint((tr.indices, va.indices, te.indices), len(dataset))
    assert tr.dataset is dataset and va.dataset is dataset and te.dataset is dataset
    assert {labels[i] for i in tr.indices} == {0, 1}
    assert {labels[i] for i in va.indices} == {0, 1}
    assert {labels[i] for i in te.indices} == {0, 1}
    groups = [i // 3 for i in range(30)]
    other = split_dataset_three(dataset, groups=groups, seed=23)
    group_sets = [{groups[i] for i in subset.indices} for subset in other]
    assert all(not group_sets[a] & group_sets[b]
               for a in range(3) for b in range(a + 1, 3))


def test_three_way_validates_fractions_and_scarce_classes():
    for kwargs, error in [
        ({"validation_fraction": 0}, "validation_fraction"),
        ({"test_fraction": 1}, "test_fraction"),
        ({"validation_fraction": .6, "test_fraction": .4}, "less than 1"),
        ({"test_fraction": float("nan")}, "test_fraction"),
        ({"seed": True}, "seed"),
        ({"labels": [0, 0, 1, 1, 1, 1]}, "three samples per class"),
        ({"groups": [0, 0, 1, 1, 1, 1]}, "three distinct groups"),
        ({"groups": [0, 0, 1, 1, 2, 2], "labels": [0, 0, 1, 1, 0, 0]}, "either"),
    ]:
        with pytest.raises(ValueError, match=error):
            three_way_indices(6, **kwargs)
    with pytest.raises(ValueError, match="At least three"):
        three_way_indices(2)


def test_kfold_random_has_complete_oof_coverage():
    folds = list(k_fold_indices(23, n_splits=5, seed=19))
    _assert_folds(folds, 23)
    assert sorted(len(validation) for _, validation in folds) == [4, 4, 5, 5, 5]
    assert folds == list(k_fold_indices(23, n_splits=5, seed=19))
    assert folds != list(k_fold_indices(23, n_splits=5, seed=20))


def test_stratified_kfold_balances_labels_per_fold():
    labels = [0] * 20 + [1] * 10
    folds = list(k_fold_indices(30, n_splits=5, labels=labels, seed=6))
    _assert_folds(folds, 30)
    for _, valid in folds:
        assert sum(labels[i] == 0 for i in valid) == 4
        assert sum(labels[i] == 1 for i in valid) == 2


def test_grouped_kfold_retains_subject_integrity_and_coverage():
    groups = ["one"] * 10 + ["two"] * 7 + ["three"] * 5 + ["four"] * 4 + ["five"] * 3
    folds = list(k_fold_indices(len(groups), n_splits=5, groups=groups, seed=4))
    _assert_folds(folds, len(groups))
    seen = []
    for _, valid in folds:
        seen.extend(set(groups[i] for i in valid))
    assert sorted(seen) == sorted(set(groups))


def test_kfold_dataframe_and_pytorch_adapters():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        "id": list(range(20)),
        "label": [0] * 10 + [1] * 10,
        "person": [i // 2 for i in range(20)],
    })
    frames = list(k_fold_split(frame, n_splits=5, stratify="label", seed=7))
    _assert_folds([(tr.id.tolist(), val.id.tolist()) for tr, val in frames], 20)
    for _, valid in frames:
        assert valid.label.value_counts().to_dict() == {0: 2, 1: 2}
    grouped = list(k_fold_split(frame, n_splits=5, group="person", seed=7))
    assert all(not set(train.person) & set(valid.person) for train, valid in grouped)
    assert list(k_fold_split(frame, n_splits=5, group="person", seed=7))[0][1].equals(grouped[0][1])

    torch = pytest.importorskip("torch")
    dataset = torch.utils.data.TensorDataset(torch.arange(20))
    dataset_folds = list(k_fold_dataset(dataset, n_splits=5, labels=frame.label, seed=7))
    _assert_folds([(tr.indices, val.indices) for tr, val in dataset_folds], 20)
    assert all(tr.dataset is dataset and val.dataset is dataset for tr, val in dataset_folds)


def test_kfold_invalid_configuration_errors():
    with pytest.raises(ValueError, match="at least two"):
        list(k_fold_indices(10, n_splits=1))
    with pytest.raises(ValueError, match="at least two"):
        list(k_fold_indices(10, n_splits=True))
    with pytest.raises(ValueError, match="dataset length"):
        list(k_fold_indices(3, n_splits=4))
    with pytest.raises(ValueError, match="samples per class"):
        list(k_fold_indices(10, n_splits=3, labels=[0] * 2 + [1] * 8))
    with pytest.raises(ValueError, match="distinct groups"):
        list(k_fold_indices(10, n_splits=3, groups=[0] * 5 + [1] * 5))
    with pytest.raises(ValueError, match="match dataset length"):
        list(k_fold_indices(10, n_splits=2, labels=[0]))
    with pytest.raises(ValueError, match="hashable"):
        list(k_fold_indices(3, n_splits=2, groups=[[0], [1], [2]]))


def test_cv_summary_metrics_and_cli(tmp_path, capsys):
    from lispyn.cli import main

    result = summarize_cv_scores([0.7, 0.8, 0.9], metric="accuracy")
    assert result["folds"] == 3
    assert result["mean"] == pytest.approx(0.8)
    assert result["std"] == pytest.approx(math.sqrt(0.02 / 3))
    assert result["min"] == pytest.approx(0.7)
    output = tmp_path / "summary.json"
    assert main(["cv-summary", "0.7", "0.8", "0.9", "--metric", "accuracy",
                 "--output", str(output)]) == 0
    import json
    assert json.loads(output.read_text())["mean"] == pytest.approx(0.8)
    assert '"folds": 3' in capsys.readouterr().out
    with pytest.raises(ValueError, match="At least one"):
        summarize_cv_scores([])
    with pytest.raises(ValueError, match="finite"):
        summarize_cv_scores([1.0, float("nan")])
    assert main(["cv-summary", "1", "--output", str(tmp_path / "bad.csv")]) == 1
    assert "must be .json" in capsys.readouterr().err
