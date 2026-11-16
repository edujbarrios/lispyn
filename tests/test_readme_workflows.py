"""The AI notebook README stays concise and demonstrates real Hy execution modes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_links_to_real_example_and_complete_catalog():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    source = (ROOT / "examples" / "hello.hy").read_text(encoding="utf-8")
    catalog = (ROOT / "supported_macros" / "README.md").read_text(encoding="utf-8")
    assert "examples/hello.hy" in readme
    assert "supported_macros/README.md" in readme
    assert "53 macros" in readme and "53 macros" in catalog
    assert '(print "Lispyn checks passed!")' in source
    assert len(readme) < 5600


def test_ai_notebook_examples_are_verbatim_source_fragments():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    training = (ROOT / "examples/quickstart.hy").read_text(encoding="utf-8")
    validation = (ROOT / "examples/validate_inputs.hy").read_text(encoding="utf-8")
    class_balance = (ROOT / "examples/label_balance.hy").read_text(encoding="utf-8")
    snippets = [
        "(setv [training validation held-out]\n"
        "  (train-val-test dataset :validation-fraction 0.25\n"
        "    :test-fraction 0.25 :seed seed))",
        "(require lispyn.macros [assert-finite assert-shape])\n"
        "(import numpy :as np)\n"
        "(import pandas :as pd)\n\n"
        "(setv features (np.array [[1.0 2.0] [3.0 4.0]]))\n"
        "(assert-shape features [2 2] :name \"features\")\n"
        "(assert-finite features :name \"features\")",
        "(require lispyn.macros [report-run narrate metric-table])\n"
        "(import lispyn.training [evaluate-metrics])\n"
        "(report-run run :title \"PyTorch baseline · recorded metrics\")",
        "(require lispyn.macros [label-balance plot-label-balance])\n"
        "(setv labels [\"cat\" \"cat\" \"cat\" \"dog\" \"dog\" \"bird\"])\n"
        "(setv report (label-balance labels :title \"Training label distribution\"))",
    ]
    assert snippets[0] in readme and snippets[0] in training
    assert snippets[1] in readme and snippets[1] in validation
    assert snippets[2] in readme and snippets[2] in training
    assert snippets[3] in readme and snippets[3] in class_balance


def test_readme_explains_three_distinct_artifacts():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Why use Lispyn for AI notebooks?" in readme
    assert "Realización de notebooks de IA" in readme
    assert "Why use it for research?" not in readme
    assert "git clone https://github.com/edujbarrios/lispyn.git" in readme
    assert "lispyn run examples/hello.hy\n" in readme
    assert "lispyn build examples/hello.hy -o hello.ipynb\n" in readme
    assert "lispyn execute examples/hello.hy -o hello-executed.ipynb\n" in readme
    assert "Nothing is executed" in readme
    assert "saves the executed cells" in readme
    assert "PyTorch is installed separately" in readme
