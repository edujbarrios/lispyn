"""Fast CPU tests for multimodal datasets and fine-tuning helpers."""

import json
from unittest.mock import patch

import pytest


def test_text_files_and_collator(tmp_path):
    torch = pytest.importorskip("torch")
    from lispyn.media import text_dataset, token_collate

    (tmp_path / "text.txt").write_text("hello\nworld\n")
    plain = text_dataset(tmp_path / "text.txt")
    assert [x["text"] for x in plain] == ["hello", "world"]

    (tmp_path / "text.jsonl").write_text("\n".join([
        json.dumps({"body": "yes", "class": 1}),
        json.dumps({"body": "no", "class": 0}),
    ]))
    labeled = text_dataset(tmp_path / "text.jsonl", text_key="body", label_key="class")
    assert labeled[0] == {"text": "yes", "label": 1}

    def fake_tokenizer(lines, **options):
        assert options["return_tensors"] == "pt"
        return {"input_ids": torch.tensor([[len(t)] for t in lines])}

    batch = token_collate(fake_tokenizer)([labeled[0], labeled[1]])
    assert batch["input_ids"].shape == (2, 1)
    assert batch["labels"].tolist() == [1, 0]
    with pytest.raises(ValueError):
        text_dataset(tmp_path / "text.jsonl", label_key="missing")


def test_vision_folder(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image
    from lispyn.media import image_folder

    for name in ("cat", "dog"):
        directory = tmp_path / name
        directory.mkdir()
        Image.new("RGB", (5, 6), "blue").save(directory / "sample.png")
    dataset = image_folder(tmp_path)
    x, label = dataset[0]
    assert x.shape == (3, 6, 5)
    assert label == 0
    assert dataset.classes == ["cat", "dog"]


def test_audio_folder_mock_decoder(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    from lispyn.media import audio_folder

    (tmp_path / "yes").mkdir()
    (tmp_path / "no").mkdir()
    (tmp_path / "yes" / "a.wav").write_bytes(b"fake")
    (tmp_path / "no" / "b.wav").write_bytes(b"fake")
    with patch("lispyn.media.load_audio", return_value=(torch.ones(1, 7), 16000)):
        dataset = audio_folder(tmp_path, sample_rate=16000, num_frames=5)
        waveform, label = dataset[0]
    assert dataset.classes == ["no", "yes"]
    assert waveform.shape == (1, 5)
    assert label == 0


def test_audio_decode_mock_torchaudio(tmp_path):
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")
    from lispyn.media import load_audio

    with patch.object(torchaudio, "load", return_value=(torch.ones(2, 10), 16000)) as decode:
        x, sr = load_audio(tmp_path / "x.wav", sample_rate=16000, mono=True)
    assert decode.called
    assert sr == 16000
    assert x.shape == (1, 10)


def test_head_only_and_full_finetune(tmp_path):
    torch = pytest.importorskip("torch")
    from lispyn.training import fit, trainable, evaluate, save_checkpoint, load_checkpoint

    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    trainable(model, "2")
    assert not model[0].weight.requires_grad
    assert model[2].weight.requires_grad
    with pytest.raises(ValueError):
        trainable(model, "not-present")
    trainable(model)
    assert all(p.requires_grad for p in model.parameters())

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(8, 2), torch.tensor([0, 1] * 4)),
        batch_size=4,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = fit(model, loader, optimizer, loss_fn=loss_fn, epochs=2, validation=loader)
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert evaluate(model, loader, loss_fn=loss_fn) >= 0
    checkpoint = save_checkpoint(model, tmp_path / "checkpoints" / "best.pt", optimizer=optimizer, history=history)
    assert checkpoint.is_file()
    assert load_checkpoint(model, checkpoint, optimizer=optimizer) == history


def test_text_model_with_mapping_batches():
    torch = pytest.importorskip("torch")
    from lispyn.training import fit

    class TextClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 2)

        def forward(self, input_ids, labels):
            logits = self.layer(input_ids.float())
            return {"loss": torch.nn.functional.cross_entropy(logits, labels)}

    model = TextClassifier()
    loader = [{"input_ids": torch.ones(2, 3), "labels": torch.tensor([0, 1])}]
    history = fit(model, loader, torch.optim.SGD(model.parameters(), lr=0.01))
    assert len(history["train_loss"]) == 1
