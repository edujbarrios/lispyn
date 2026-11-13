"""Read-only checks for the Python environment that runs Lispyn.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
Package metadata is inspected without importing CUDA, PyTorch, or other heavy modules.
"""

from __future__ import annotations

from importlib import metadata
import platform
from .compiler import find_hy_executable

from . import __version__


_CORE = ("hy", "nbformat", "packaging")
_NOTEBOOK = ("nbclient", "ipykernel")
_OPTIONAL = ("torch", "torchvision", "torchaudio", "torchcodec",
             "matplotlib", "numpy", "pandas", "transformers", "peft")


def inspect_environment() -> dict:
    """Report installation readiness without modifying the environment."""
    packages = {}
    for name in (*_CORE, *_NOTEBOOK, *_OPTIONAL):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    hy2py = bool(find_hy_executable("hy2py"))
    core_ready = all(packages[name] is not None for name in _CORE) and hy2py
    notebook_ready = core_ready and all(
        packages[name] is not None for name in _NOTEBOOK
    )
    return {
        "lispyn": __version__,
        "python": platform.python_version(),
        "hy2py": hy2py,
        "can_build": core_ready,
        "can_execute": notebook_ready,
        "packages": packages,
    }


def format_report(report: dict) -> str:
    """Human-readable installation summary; never prints local paths or secrets."""
    def status(name):
        version = report["packages"][name]
        return f"{name} {version}" if version else f"{name} missing"

    lines = [
        f"Lispyn {report['lispyn']} · Python {report['python']}",
        "Notebook generation: " + ("ready" if report["can_build"] else "not ready"),
        "Notebook execution: " + ("ready" if report["can_execute"] else "not ready"),
        "Core: " + ", ".join(status(name) for name in _CORE),
        "Compiler: hy2py " + ("found" if report["hy2py"] else "missing"),
        "Jupyter: " + ", ".join(status(name) for name in _NOTEBOOK),
        "Optional: " + ", ".join(status(name) for name in _OPTIONAL),
    ]
    if not report["can_build"]:
        lines.append("Install from the repository: python -m pip install -e .")
    elif not report["can_execute"]:
        lines.append(
            'For notebook execution: python -m pip install -e ".[notebook]"'
        )
    return "\n".join(lines)
