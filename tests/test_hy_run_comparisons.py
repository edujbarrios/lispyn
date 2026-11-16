"""Hy-native saved-run comparison reports and reproducible plots."""

import json
from pathlib import Path
import shutil
import subprocess

import nbformat
import pytest

from lispyn.cli import main
from lispyn.notebooks import build_notebook
from lispyn.research_comparison import show_run_comparison, plot_run_comparison


def _run(root, name, values, *, status=None):
    path = root / name
    path.mkdir(parents=True)
    (path / "history.json").write_text(json.dumps(values), encoding="utf-8")
    if status is not None:
        (path / "manifest.json").write_text(
            json.dumps({"status": status}), encoding="utf-8"
        )
    return path


def test_hy_comparison_reports_baseline_gains_and_exports(tmp_path, capsys):
    base = _run(tmp_path, "baseline", {"val_loss": [1.2, 0.8, 0.9]},
                status="completed")
    better = _run(tmp_path, "candidate", {"val_loss": [0.9, 0.6]},
                  status="completed")
    report = show_run_comparison(
        [base, better], metric="loss", baseline="baseline",
        save=tmp_path / "comparison.md",
    )
    assert report["mode"] == "min"
    assert report["baseline"] == "baseline"
    assert report["runs"][0]["best"] == 0.8
    assert report["runs"][1]["best_gain"] == pytest.approx(0.2)
    assert report["runs"][1]["final_gain"] == pytest.approx(0.3)
    assert "comparison" in capsys.readouterr().out.lower()
    assert "candidate" in (tmp_path / "comparison.md").read_text()
    for ext in ("json", "csv"):
        path = tmp_path / f"comparison.{ext}"
        show_run_comparison([base, better], save=path)
        assert path.is_file() and path.stat().st_size > 50
    content = json.loads((tmp_path / "comparison.json").read_text())
    assert content["runs"][1]["run"] == "candidate"


def test_hy_comparison_uses_accuracy_maximization(tmp_path):
    a = _run(tmp_path, "original", {"val_accuracy": [0.4, 0.7, 0.6]})
    b = _run(tmp_path, "improved", {"val_accuracy": [0.5, 0.9]})
    summary = show_run_comparison(
        [a, b], metric="accuracy", baseline="original"
    )
    assert summary["mode"] == "max"
    assert summary["runs"][1]["best_gain"] == pytest.approx(0.2)
    assert summary["runs"][1]["final_gain"] == pytest.approx(0.3)
    assert summary["runs"][1]["best_epoch"] == 2


def test_hy_run_plot_saves_real_figure_and_respects_theme(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = _run(tmp_path, "baseline", {"val_loss": [1.0, 0.85]})
    b = _run(tmp_path, "candidate", {"val_loss": [0.9, 0.7, 0.6]})
    png = tmp_path / "validation.png"
    fig = plot_run_comparison(
        [a, b], metric="loss", theme="paper", title="Saved runs",
        save=png,
    )
    assert png.stat().st_size > 3000
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert [line.get_label() for line in fig.axes[0].lines] == ["baseline", "candidate"]
    assert [len(line.get_ydata()) for line in fig.axes[0].lines] == [2, 3]
    plt.close(fig)


@pytest.mark.parametrize("runs,diagnostic", [
    ("not a list", "list/tuple"),
    ([], "2 and 20"),
    ([123, 456], "path string"),
])
def test_hy_comparison_rejects_bad_inputs(tmp_path, runs, diagnostic):
    with pytest.raises((TypeError, ValueError), match=diagnostic):
        show_run_comparison(runs)


def test_hy_comparison_rejects_ambiguous_incomplete_and_corrupt_runs(tmp_path):
    a = _run(tmp_path / "set1", "baseline", {"val_loss": [0.9]})
    b = _run(tmp_path / "set2", "baseline", {"val_loss": [0.6]})
    with pytest.raises(ValueError, match="distinct"):
        show_run_comparison([a, b])
    with pytest.raises(ValueError, match="distinct"):
        plot_run_comparison([a, a])
    c = _run(tmp_path, "failed", {"val_loss": [0.2]}, status="failed")
    with pytest.raises(ValueError, match="not completed"):
        show_run_comparison([a, c])
    with pytest.raises(ValueError, match="not completed"):
        plot_run_comparison([a, c])
    bad = _run(tmp_path, "broken", {"val_loss": [0.5]}, status="completed")
    (bad / "manifest.json").write_text("garbage", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid manifest"):
        show_run_comparison([a, bad])
    without = _run(tmp_path, "no-validation", {"train_loss": [0.7]})
    with pytest.raises(ValueError, match="No val_loss"):
        show_run_comparison([a, without])


def test_hy_comparison_rejects_symlinked_history(tmp_path):
    a = _run(tmp_path, "one", {"val_loss": [0.9]})
    b = tmp_path / "symlink"
    b.mkdir()
    (b / "history.json").symlink_to(a / "history.json")
    with pytest.raises(ValueError, match="symlink"):
        show_run_comparison([a, b])


def test_hy_macro_compiles_and_executes_directly(tmp_path):
    pytest.importorskip("hy")
    a = _run(tmp_path / "runs", "baseline", {"val_loss": [1.0, 0.8]})
    b = _run(tmp_path / "runs", "candidate", {"val_loss": [0.9, 0.5]})
    source = tmp_path / "comparison.hy"
    source.write_text(
        '(require lispyn.macros [compare-runs plot-runs])\n'
        '(setv report (compare-runs ["runs/baseline" "runs/candidate"]\n'
        '  :baseline "baseline" :save "summary.md"))\n'
        '(assert (> (get (get (get report "runs") 1) "best_gain") 0))\n'
        '(setv figure (plot-runs ["runs/baseline" "runs/candidate"]\n'
        '  :theme "paper" :save "curves.png"))\n'
        '(print "HY RUN COMPARISON OK")\n',
        encoding="utf-8"
    )
    notebook = build_notebook(source)
    nbformat.validate(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "show_run_comparison" in code and "plot_run_comparison" in code
    exe = shutil.which("hy")
    assert exe is not None
    run = subprocess.run(
        [exe, str(source)], cwd=tmp_path,
        capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stderr
    assert "HY RUN COMPARISON OK" in run.stdout
    assert "candidate" in (tmp_path / "summary.md").read_text()
    assert (tmp_path / "curves.png").stat().st_size > 3000


def test_hy_macro_executes_jupyter_markdown_and_plot(tmp_path, capsys):
    pytest.importorskip("nbclient")
    source = tmp_path / "example.hy"
    source.write_text(
        '(require lispyn.macros [md])\n'
        '(md "# Saved run comparison")\n'
        '(require lispyn.macros [compare-runs])\n'
        '(import lispyn.reproducibility [write-json])\n'
        '(write-json "baseline/history.json" {"val_loss" [0.8 0.7]})\n'
        '(write-json "candidate/history.json" {"val_loss" [0.7 0.5]})\n'
        '(compare-runs ["baseline" "candidate"] :baseline "baseline"\n'
        '  :save "comparison.md")\n'
        '(require lispyn.macros [md])\n'
        '(md "## Comparison plot")\n'
        '(require lispyn.macros [plot-runs])\n'
        '(plot-runs ["baseline" "candidate"] :theme "paper"\n'
        '  :save "comparison.png")\n',
        encoding="utf-8",
    )
    assert main(["execute", str(source), "--timeout", "120"]) == 0
    capsys.readouterr()
    notebook = nbformat.read(source.with_suffix(".ipynb"), as_version=4)
    outputs = [
        output for cell in notebook.cells if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    ]
    assert any("baseline" in str(o.get("data", {}).get("text/markdown", ""))
               and "Best gain" in str(o.get("data", {}).get("text/markdown", ""))
               for o in outputs)
    assert any("image/png" in o.get("data", {}) for o in outputs)
    assert (tmp_path / "comparison.md").is_file()
    assert (tmp_path / "comparison.png").stat().st_size > 3000
