"""Index-based train/validation subsets, including modality label adapters."""

import random

import pytest


def _dataset():
    torch = pytest.importorskip("torch")
    labels = torch.tensor([0] * 10 + [1] * 10)
    inputs = torch.arange(20).float().unsqueeze(1)
    return torch, torch.utils.data.TensorDataset(inputs, labels), labels


def test_random_split_disjoint_seeded_and_global_rng_unchanged():
    _, dataset, _ = _dataset()
    from lispyn.data import split_dataset

    random.seed(19)
    expected = random.random()
    random.seed(19)
    train, valid = split_dataset(dataset, validation_fraction=0.25, seed=42)
    assert random.random() == expected
    assert len(train) == 15
    assert len(valid) == 5
    assert not set(train.indices).intersection(valid.indices)
    assert set(train.indices) | set(valid.indices) == set(range(20))
    again, _ = split_dataset(dataset, validation_fraction=0.25, seed=42)
    assert train.indices == again.indices
    other, _ = split_dataset(dataset, validation_fraction=0.25, seed=43)
    assert train.indices != other.indices


def test_stratified_split_preserves_classes_and_is_seeded():
    _, dataset, labels = _dataset()
    from lispyn.data import split_dataset

    train, valid = split_dataset(
        dataset, labels=labels, validation_fraction=0.2, seed=42
    )
    assert len(train) == 16
    assert len(valid) == 4
    assert sorted(labels[i].item() for i in valid.indices) == [0, 0, 1, 1]
    assert not set(train.indices) & set(valid.indices)
    train2, valid2 = split_dataset(
        dataset, labels=labels, validation_fraction=0.2, seed=42
    )
    assert train.indices == train2.indices
    assert valid.indices == valid2.indices


def test_auto_labels_from_dataset_targets(tmp_path):
    torch = pytest.importorskip("torch")
    from lispyn.media import text_dataset, audio_folder
    from lispyn.data import split_dataset

    source = tmp_path / "items.csv"
    source.write_text("text,label\na,0\nb,0\nc,1\nd,1\n")
    data = text_dataset(source, label_key="label", label_cast=int)
    assert data.targets == [0, 0, 1, 1]
    tr, va = split_dataset(data, stratify=True, seed=7)
    assert len(tr) == 2 and len(va) == 2
    assert sorted(data.targets[i] for i in va.indices) == [0, 1]

    folder = tmp_path / "audio"
    for name in ("no", "yes"):
        (folder / name).mkdir(parents=True)
        for index in range(2):
            (folder / name / f"{index}.wav").write_bytes(b"unused")
    audio = audio_folder(folder, sample_rate=16000, num_frames=16)
    assert audio.targets == [0, 0, 1, 1]
    tr2, va2 = split_dataset(audio, stratify=True, seed=7)
    assert sorted(audio.targets[i] for i in va2.indices) == [0, 1]


def test_errors_and_missing_labels():
    torch, dataset, labels = _dataset()
    from lispyn.data import split_dataset

    with pytest.raises(ValueError, match="validation_fraction"):
        split_dataset(dataset, validation_fraction=1)
    with pytest.raises(ValueError, match="At least two"):
        split_dataset(torch.utils.data.Subset(dataset, [0]))
    with pytest.raises(ValueError, match="dataset.targets"):
        split_dataset(dataset, stratify=True)
    with pytest.raises(ValueError, match="Number of labels"):
        split_dataset(dataset, labels=[0, 1])
    with pytest.raises(ValueError, match="two samples per class"):
        split_dataset(dataset, labels=[0] * 19 + [1])
    with pytest.raises(ValueError, match="hashable"):
        split_dataset(dataset, labels=[[i] for i in range(len(dataset))])
    with pytest.raises(ValueError, match="seed"):
        split_dataset(dataset, seed=True)
