# Generated Jupyter notebooks

These are **Python notebooks compiled from Lispyn `.hy` sources**, including a Markdown-cell demo and a ResNet18 recipe.

Regenerate the gallery with:

```bash
python -m pip install -e .
lispyn gallery
```

The gallery is **not executed** during generation. ResNet18 includes a 10-epoch classifier fine-tuning phase followed by **300 epochs** of full fine-tuning, but there are no fabricated results.

To execute that notebook, place your images under `examples/data/resnet_images/<class_name>/*.jpg` or `.png`, install compatible PyTorch/TorchVision, and run:

```bash
lispyn execute examples/resnet_finetune_300.hy -o example_generated_notebooks/resnet_finetune_300.ipynb
```

The generated notebooks retain Markdown, Python code, and attribution to **Eduardo J. Barrios**. Gallery output is deterministic for reviewable Git diffs.
