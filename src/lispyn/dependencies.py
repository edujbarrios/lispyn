"""Runtime, opt-in third-party dependencies for Hy and generated Jupyter cells.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.

Never installs packages while compiling Hy or generating a notebook.
Explicit ensure-package/use-package macro calls may invoke pip at *runtime*.
"""

from __future__ import annotations

import importlib
from importlib import metadata, util
import re
import subprocess
import sys

from packaging.requirements import InvalidRequirement, Requirement


_MODULE_NAME = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


def _parse(requirement: str, import_name: str | None) -> tuple[Requirement, str]:
    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError("requirement must be a nonempty package name or version constraint")
    try:
        parsed = Requirement(requirement.strip())
    except InvalidRequirement as exc:
        raise ValueError(f"Invalid Python package requirement: {requirement!r}") from exc
    if parsed.url or parsed.marker:
        raise ValueError("URLs and environment markers are not supported; use named packages")
    module = import_name if import_name is not None else parsed.name.replace("-", "_")
    if not isinstance(module, str) or not _MODULE_NAME.fullmatch(module):
        raise ValueError("import_name must be a dotted Python module name")
    return parsed, module


def _available(requirement: Requirement, import_name: str) -> tuple[bool, str | None]:
    try:
        version = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        version = None
    if version is not None and (
        not requirement.specifier or requirement.specifier.contains(version)
    ):
        return True, version
    # Standard-library or externally supplied modules often lack a pip
    # distribution; a version constraint still requires package metadata.
    if version is None and not requirement.specifier:
        try:
            if util.find_spec(import_name) is not None:
                return True, None
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
            pass
    return False, version


def ensure_package(requirement: str, *, import_name: str | None = None,
                   install: bool = True) -> str:
    """Return an installation status; install only if the requirement is absent.

    A requirement can be 'math', 'numpy>=2', or 'scikit-learn' with
    import_name='sklearn'. To avoid unintentional downloads, use install=False.
    Installations use the *current* Python interpreter (including the Jupyter
    kernel) and do not run shell commands. Installation modifies the runtime
    environment; only execute trusted notebooks and named dependencies.
    """
    parsed, module = _parse(requirement, import_name)
    available, version = _available(parsed, module)
    if available:
        message = (
            f"Requirement already installed in this environment: {parsed.name}"
            + (f" ({version})" if version is not None else " (standard library/module)")
        )
        print(message)
        return message

    if not install:
        raise RuntimeError(
            f"Missing requirement {requirement!r}. Install a compatible build "
            "in this environment before executing the notebook."
        )

    # No shell, flags, direct URLs or arbitrary pip command sequences. Python
    # is the same executable used by the notebook kernel.
    command = [
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        str(parsed),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        message = (result.stderr or result.stdout or "pip returned an error").strip()
        raise RuntimeError(f"Failed to install {parsed.name}: {message[-1500:]}")

    importlib.invalidate_caches()
    available, _ = _available(parsed, module)
    if not available:
        raise RuntimeError(
            f"pip finished but {requirement!r} is not yet visible. "
            "Restart the Jupyter kernel and try again."
        )
    message = (
        f"Installed requirement in this environment: {parsed.name}. "
        "A kernel restart may be needed for native extensions."
    )
    print(message)
    return message


def use_package(requirement: str, *, import_name: str | None = None,
                install: bool = True):
    """Ensure the requirement, then import and return a Python module.

    Hy example: (setv math (use-package "math"))
    """
    parsed, module = _parse(requirement, import_name)
    ensure_package(str(parsed), import_name=module, install=install)
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"{module!r} could not be imported. You may need to restart "
            "the kernel after package installation."
        ) from exc



def ensure_packages(*requirements: str, install: bool = True) -> list[str]:
    """Check a list of named packages for a dedicated notebook pip cell.

    Validate every requirement *before* trying to install anything; then
    skip the distributions/modules already available in this kernel.
    """
    if not requirements:
        raise ValueError("pip-install requires at least one named package")
    for requirement in requirements:
        _parse(requirement, None)
    return [ensure_package(requirement, install=install)
            for requirement in requirements]
