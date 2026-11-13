"""Build a browsable collection of Python Jupyter notebooks from Hy examples.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
Compilation of trusted Hy never executes the training program.
"""

from __future__ import annotations

from pathlib import Path

from .compiler import LispynError
from .notebooks import build_notebook, write_notebook


EXAMPLE_SOURCES = (
    "markdown_cells.hy",
    "resnet_finetune_300.hy",
    "one_call_training.hy",
    "curve_designs.hy",
    "plot_experiment.hy",
    "text_lora.hy",
    "audio_features.hy",
)


def generate_gallery(source_dir="examples", output_dir="example_generated_notebooks"):
    """Compile a stable set of unexecuted notebooks into a dedicated folder."""
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    missing = [filename for filename in EXAMPLE_SOURCES
               if not (source_dir / filename).is_file()]
    if missing:
        raise LispynError(
            f"Missing gallery sources in {source_dir}: {', '.join(missing)}"
        )
    if source_dir == output_dir:
        raise LispynError("Gallery output directory cannot be the source directory")

    outputs = []
    for filename in EXAMPLE_SOURCES:
        src = source_dir / filename
        nb = build_notebook(src)
        outputs.append(write_notebook(nb, output_dir / src.with_suffix(".ipynb").name))
    return outputs
