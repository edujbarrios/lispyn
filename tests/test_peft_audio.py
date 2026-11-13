"""LoRA, full fine-tuning and real WAV decoding smoke tests (CPU only)."""

import math
import struct
import wave

import pytest


def test_parameter_counts_and_validation():
    torch = pytest.importorskip("torch")
    from lispyn.peft_utils import lora, parameter_counts

    model = torch.nn.Linear(3, 2)
    stats = parameter_counts(model)
    assert stats["total"] == 8
    assert stats["trainable"] == 8
    model.weight.requires_grad_(False)
    assert parameter_counts(model)["trainable"] == 2

    with pytest.raises(ValueError, match="modules"):
        lora(model, [])
    with pytest.raises(ValueError, match="rank"):
        lora(model, ["weight"], rank=0)
    with pytest.raises(ValueError, match="dropout"):
        lora(model, ["weight"], dropout=1.0)


def test_real_lora_forward_training_and_adapter_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from lispyn.peft_utils import lora, load_adapter, parameter_counts, save_adapter
    from lispyn.training import fit

    config = transformers.BertConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=64, num_labels=2,
    )
    model = transformers.BertForSequenceClassification(config)
    model = lora(model, ["query", "value"], rank=4, task_type="SEQ_CLS")
    counts = parameter_counts(model)
    assert 0 < counts["trainable"] < counts["total"]
    batch = {
        "input_ids": torch.randint(0, 64, (3, 8)),
        "attention_mask": torch.ones(3, 8, dtype=torch.long),
        "labels": torch.tensor([0, 1, 0]),
    }
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3
    )
    history = fit(model, [batch], optimizer, epochs=1)
    assert len(history["train_loss"]) == 1
    assert math.isfinite(history["train_loss"][0])
    saved = save_adapter(model, tmp_path / "adapter")
    assert (saved / "adapter_config.json").exists()
    restored = load_adapter(transformers.BertForSequenceClassification(config), saved)
    assert parameter_counts(restored)["total"] > 0
    assert not any("lora" in name and p.requires_grad
                   for name, p in restored.named_parameters())


def _wav(path, *, rate=8000, frames=800):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(
            b"".join(struct.pack("<h", int(4000 * math.sin(i / 8)))
                     for i in range(frames))
        )


def test_real_wav_decode_resample_and_labeled_audio_folder(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    from lispyn.media import audio_folder, load_audio

    folder = tmp_path / "speech" / "yes"
    folder.mkdir(parents=True)
    path = folder / "clip.wav"
    _wav(path)
    waveform, rate = load_audio(path, sample_rate=16000)
    assert rate == 16000
    assert waveform.shape[0] == 1
    assert waveform.shape[1] == 1600
    dataset = audio_folder(tmp_path / "speech", sample_rate=16000, num_frames=1200)
    x, label = dataset[0]
    assert x.shape == (1, 1200)
    assert label == 0
    assert dataset.classes == ["yes"]


def test_lora_missing_optional_dependency_is_actionable(monkeypatch):
    """No imported PEFT objects are required for this validation path."""
    from lispyn.peft_utils import lora

    with pytest.raises(ValueError, match="module names"):
        lora(object(), "attention")
