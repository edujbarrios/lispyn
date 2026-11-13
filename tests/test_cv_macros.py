"""Smoke-test Hy macro compilation for three-way splits and k-folds."""


def test_three_way_and_kfold_macros_compile(tmp_path):
    from lispyn.notebooks import build_notebook

    source = tmp_path / "folds.hy"
    source.write_text(
        "(require lispyn.macros [train-val-test cross-validation-folds])\n"
        "(setv values (train-val-test dataset :test-fraction 0.1))\n"
        "(setv folds (cross-validation-folds dataset :n-splits 3))\n",
        encoding="utf-8",
    )
    notebook = build_notebook(source)
    code = "\n".join(cell.source for cell in notebook.cells
                     if cell.cell_type == "code")
    assert "split_dataset_three" in code
    assert "k_fold_dataset" in code
