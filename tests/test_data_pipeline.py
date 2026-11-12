"""Regression tests for dataset transformations and PyTorch adapters."""

import pytest

from lispyn.data import (
    dataframe_to_dataset,
    drop_missing,
    make_dataloader,
    read_dataset,
    train_validation_split,
)


def test_csv_jsonl_and_missing_values(tmp_path):
    pd = pytest.importorskip("pandas")
    csv = tmp_path / "sample.csv"
    csv.write_text("x,y\n1,2\n3,\n4,5\n")
    frame = read_dataset(csv)
    clean = drop_missing(frame)
    assert clean.shape == (2, 2)
    assert frame.shape == (3, 2), "drop_missing must not mutate the original"

    jsonl = tmp_path / "sample.jsonl"
    jsonl.write_text('{"x":1,"y":2}\n{"x":3,"y":4}\n')
    assert read_dataset(jsonl).shape == (2, 2)
    with pytest.raises(ValueError, match="Unsupported"):
        read_dataset(tmp_path / "sample.xyz")


def test_split_is_seeded_disjoint_and_validates():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"id": list(range(20)), "y": [i * 2 for i in range(20)]})
    train, valid = train_validation_split(frame, validation_fraction=0.25, seed=17)
    train2, valid2 = train_validation_split(frame, validation_fraction=0.25, seed=17)
    assert train.equals(train2)
    assert valid.equals(valid2)
    assert len(train) == 15
    assert len(valid) == 5
    assert not set(train.id) & set(valid.id)
    assert set(train.id) | set(valid.id) == set(range(20))
    assert frame.id.tolist() == list(range(20))
    with pytest.raises(ValueError, match="validation_fraction"):
        train_validation_split(frame, validation_fraction=0)
    with pytest.raises(ValueError, match="two rows"):
        train_validation_split(frame.iloc[:1])


def test_tensor_dataset_validation_and_dataloader():
    pd = pytest.importorskip("pandas")
    torch = pytest.importorskip("torch")
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7], "label": [0, 1, 0, 1]})
    dataset = dataframe_to_dataset(frame, features=["a", "b"], target="label", target_dtype=torch.long)
    assert dataset.tensors[0].shape == (4, 2)
    assert dataset.tensors[1].dtype == torch.int64
    batches = list(make_dataloader(dataset, batch_size=3, shuffle=False))
    assert len(batches) == 2
    assert batches[0][0].shape == (3, 2)
    first = list(make_dataloader(dataset, batch_size=2, shuffle=True, seed=24))
    second = list(make_dataloader(dataset, batch_size=2, shuffle=True, seed=24))
    assert all(torch.equal(a[0], b[0]) for a, b in zip(first, second))
    with pytest.raises(ValueError, match="target leakage"):
        dataframe_to_dataset(frame, features=["a", "label"], target="label")
    with pytest.raises(ValueError, match="Missing"):
        dataframe_to_dataset(frame, features=["unknown"], target="label")
    with pytest.raises(ValueError, match="positive"):
        make_dataloader(dataset, batch_size=0)
