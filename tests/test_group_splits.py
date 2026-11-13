"""Regression tests for stratified tabular and leakage-safe grouped splits."""

import pytest

from lispyn.data import split_dataset, train_validation_split


def test_stratified_dataframe_split_is_seeded_and_balanced():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        "id": list(range(24)),
        "label": ["a"] * 12 + ["b"] * 8 + ["c"] * 4,
    })
    train, valid = train_validation_split(
        frame, stratify="label", validation_fraction=0.25, seed=13
    )
    train_again, valid_again = train_validation_split(
        frame, stratify="label", validation_fraction=0.25, seed=13
    )
    assert train.equals(train_again)
    assert valid.equals(valid_again)
    assert valid.label.value_counts().to_dict() == {"a": 3, "b": 2, "c": 1}
    assert not set(train.id) & set(valid.id)
    assert set(train.id) | set(valid.id) == set(frame.id)
    assert frame.id.tolist() == list(range(24))


def test_grouped_dataframe_split_prevents_subject_leakage():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        "id": list(range(12)),
        "subject": ["s1"] * 3 + ["s2"] * 3 + ["s3"] * 3 + ["s4"] * 3,
    })
    train, valid = train_validation_split(
        frame, group="subject", validation_fraction=0.25, seed=5
    )
    assert len(train) == 9
    assert len(valid) == 3
    assert not set(train.subject) & set(valid.subject)
    assert not set(train.id) & set(valid.id)
    repeat = train_validation_split(frame, group="subject", validation_fraction=0.25, seed=5)
    assert repeat[0].equals(train) and repeat[1].equals(valid)


def test_grouped_dataset_split_keeps_all_samples_for_each_session():
    torch = pytest.importorskip("torch")
    data = torch.utils.data.TensorDataset(torch.arange(12))
    groups = ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3
    tr, va = split_dataset(data, groups=groups, validation_fraction=0.3, seed=7)
    assert len(tr) + len(va) == len(data)
    assert not {groups[i] for i in tr.indices} & {groups[i] for i in va.indices}
    assert set(tr.indices) | set(va.indices) == set(range(len(data)))
    tr2, va2 = split_dataset(data, groups=groups, validation_fraction=0.3, seed=7)
    assert tr.indices == tr2.indices and va.indices == va2.indices


def test_group_aware_split_rejects_unsupported_inputs():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"label": ["a", "a", "b", "b"], "subject": [1, 1, 1, 1]})
    with pytest.raises(ValueError, match="at least two distinct groups"):
        train_validation_split(frame, group="subject")
    with pytest.raises(ValueError, match="either stratify or group"):
        train_validation_split(frame, group="subject", stratify="label")
    with pytest.raises(ValueError, match="Missing split column"):
        train_validation_split(frame, stratify="unknown")
    with pytest.raises(ValueError, match="at least two samples per class"):
        train_validation_split(frame.iloc[:3], stratify="label")
    with pytest.raises(ValueError, match="seed"):
        train_validation_split(frame, seed=True)

    torch = pytest.importorskip("torch")
    data = torch.utils.data.TensorDataset(torch.arange(4))
    with pytest.raises(ValueError, match="Number of groups"):
        split_dataset(data, groups=[1])
    with pytest.raises(ValueError, match="grouped or stratified"):
        split_dataset(data, groups=[1, 1, 2, 2], labels=[0, 0, 1, 1])
    with pytest.raises(ValueError, match="at least two distinct groups"):
        split_dataset(data, groups=[0, 0, 0, 0])
    with pytest.raises(ValueError, match="hashable"):
        split_dataset(data, groups=[[0], [0], [1], [1]])
