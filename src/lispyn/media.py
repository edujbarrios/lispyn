"""Minimal datasets for images, text and audio; native PyTorch throughout.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def image_folder(root, *, transform=None):
    """Native torchvision.datasets.ImageFolder, with tensors by default.

    Directory layout: root/class_name/image.jpg. Supply weights.transforms()
    for pretrained models so resizing and normalization match the weights.
    """
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("Install compatible torch and torchvision versions") from exc
    return datasets.ImageFolder(str(root), transform=transform or transforms.ToTensor())


def text_dataset(path, *, text_key="text", label_key=None, encoding="utf-8"):
    """Read .txt, .csv, .json or .jsonl into a native torch Dataset.

    Each item is {'text': str} plus optional {'label': raw_label}. Tokenization
    is intentionally separate: pass a tokenizer collator to DataLoader.
    """
    from torch.utils.data import Dataset

    path = Path(path)
    suffix = path.suffix.lower()
    with path.open(encoding=encoding, newline="") as stream:
        if suffix == ".txt":
            records = [{text_key: line.rstrip("\r\n")} for line in stream if line.strip()]
        elif suffix == ".csv":
            records = list(csv.DictReader(stream))
        elif suffix == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        elif suffix == ".json":
            records = json.load(stream)
        else:
            raise ValueError("Text dataset must be .txt, .csv, .json or .jsonl")
    if not isinstance(records, list):
        raise ValueError("JSON text dataset must contain a list of objects")
    output = []
    for row in records:
        if not isinstance(row, dict) or text_key not in row:
            raise ValueError(f"Every record must contain a {text_key!r} field")
        item = {"text": str(row[text_key])}
        if label_key is not None:
            if label_key not in row:
                raise ValueError(f"Missing label field: {label_key}")
            item["label"] = row[label_key]
        output.append(item)

    class TextDataset(Dataset):
        def __len__(self):
            return len(output)

        def __getitem__(self, index):
            return output[index]

    return TextDataset()


def token_collate(tokenizer, *, max_length=128, label_dtype=None):
    """Make a collate_fn for Hugging Face-style tokenizers.

    Tokenizer must accept list[str] and return a mapping of torch tensors.
    Labeled datasets should have numeric labels (for text classification).
    """
    import torch

    if max_length < 1:
        raise ValueError("max_length must be positive")

    def collate(rows):
        batch = tokenizer(
            [row["text"] for row in rows], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        if rows and "label" in rows[0]:
            if any("label" not in row for row in rows):
                raise ValueError("Every text row must have a label")
            batch["labels"] = torch.as_tensor(
                [row["label"] for row in rows], dtype=label_dtype or torch.long
            )
        return batch

    return collate


def load_audio(path, *, sample_rate=None, mono=True):
    """Read audio through torchaudio (TorchCodec needed by recent releases)."""
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("Install compatible torch/torchaudio and torchcodec") from exc
    try:
        waveform, original_rate = torchaudio.load(str(path))
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "Audio decoding failed. Recent torchaudio requires matching torchcodec/FFmpeg support"
        ) from exc
    if mono:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate is not None and original_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_rate, sample_rate)
        original_rate = sample_rate
    return waveform, original_rate


def audio_folder(root, *, sample_rate=None, num_frames=None, mono=True, transform=None):
    """Labeled audio directory dataset: root/class_name/*.wav (etc).

    Set sample_rate and num_frames for uniform batches in a DataLoader.
    When num_frames is set, pad or truncate without random cropping.
    """
    from torch.utils.data import Dataset
    import torch.nn.functional as F

    root = Path(root)
    classes = sorted(folder.name for folder in root.iterdir() if folder.is_dir())
    if not classes:
        raise ValueError("Audio dataset needs class subdirectories")
    samples = [
        (path, idx)
        for idx, name in enumerate(classes)
        for path in sorted((root / name).rglob("*"))
        if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS
    ]
    if not samples:
        raise ValueError("No supported audio files found")
    if num_frames is not None and num_frames < 1:
        raise ValueError("num_frames must be positive")

    class AudioFolder(Dataset):
        def __init__(self):
            self.classes = classes
            self.class_to_idx = {name: i for i, name in enumerate(classes)}
            self.samples = samples

        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            path, label = samples[index]
            signal, _ = load_audio(path, sample_rate=sample_rate, mono=mono)
            if num_frames is not None:
                signal = signal[..., :num_frames]
                signal = F.pad(signal, (0, num_frames - signal.shape[-1]))
            if transform:
                signal = transform(signal)
            return signal, label

    return AudioFolder()
