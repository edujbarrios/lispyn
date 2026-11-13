"""Small, opt-in reproducibility records for trusted ML experiments.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
No dataset rows, tokens, model weights or environment variables are recorded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import random


def seed_all(seed: int) -> int:
    """Seed Python, optional NumPy and optional PyTorch random generators.

    Call before *creating* a model to reproduce its initialization. This does
    not guarantee deterministic CUDA kernels, workers or third-party ops.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed % (2 ** 32))
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
    return seed


def environment_info() -> dict:
    """Public package versions only; no hostnames, paths or environment vars."""
    packages = {}
    for name in ("hy", "torch", "torchvision", "torchaudio", "torchcodec",
                 "numpy", "transformers", "peft"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return {"python": platform.python_version(),
            "system": platform.system(),
            "packages": packages}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return path


def summarize_run(directory) -> str:
    """Give an experiment a readable, short command-line report."""
    path = Path(directory).expanduser() / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data.get("config", {})
    lines = [
        f"Lispyn run: {data.get('status', 'unknown')}",
        f"Model: {data.get('model', 'unknown')}",
        f"Device: {data.get('device', 'unknown')}; seed: {config.get('seed')}",
        f"Epochs: {config.get('epochs')}; tuning: {config.get('tuning')}",
    ]
    summary = data.get("summary")
    if summary:
        lines.append(
            f"Best {summary['monitor']}: {summary['value']:.6g} "
            f"(epoch {summary['epoch']})"
        )
    return "\n".join(lines)
