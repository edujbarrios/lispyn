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

The ResNet18 recipe now includes runtime `ensure-package` / `use-package` macros for standard-library and third-party Python imports. Merely generating notebooks never calls pip.

The gallery also includes `synthetic_curves.ipynb`, a complete no-model-training Matplotlib example. Its generated PNG is committed at `docs/images/synthetic_training_curves.png` and embedded in the main README.

Synthetic demo: `synthetic_curves.ipynb` generates two committed PNG figures with the `docusaurus-dark` theme and `turquoise-spectrum` palette, one showing train/validation curves and the other comparing three synthetic runs. Both are generated, not fabricated model results.

The gallery also includes `confusion_matrix.ipynb`: a synthetic classification example with a reusable `defconfusion` macro and a turquoise-on-dark matrix figure at `docs/images/synthetic_confusion_matrix.png`. It is compiled without execution or fabricated model predictions.

`early_stopping.ipynb` demonstrates the `train-until` macro with a small CPU PyTorch model, Markdown narration, best/last checkpoint tracking, a run manifest and Docusaurus-dark loss curves. Gallery creation never executes the training loop.
