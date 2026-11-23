"""The front-page Hy examples stay succinct, grounded and runnable."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_links_to_examples_and_complete_catalog():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    example = (ROOT / "examples" / "hello.hy").read_text(encoding="utf-8")
    catalog = (ROOT / "supported_macros" / "README.md").read_text(encoding="utf-8")
    assert "examples/hello.hy" in readme
    assert "examples/showcase.hy" in readme
    assert "examples/quickstart.hy" in readme
    assert "supported_macros/README.md" in readme
    assert "89 supported macros" in readme and "89 macros" in catalog
    assert '(print "Lispyn checks passed!")' in example
    assert len(readme) < 8500


def test_readme_snippets_are_verbatim_showcase_source():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    source = (ROOT / "examples" / "showcase.hy").read_text(encoding="utf-8")
    snippets = [
        '(require lispyn.macros [pip-install])\n'
        '(pip-install "matplotlib>=3.8")',
        '(defmodel make-mlp [features hidden]\n'
        '  (torch.nn.Linear features hidden)\n'
        '  (torch.nn.ReLU)\n'
        '  (torch.nn.Linear hidden 1))\n'
        '(setv model (make-mlp 2 8))\n'
        '(defoptimizer make-adam [model rate] torch.optim.Adam :weight-decay 0.01)\n'
        '(setv optimizer (make-adam model 0.05))',
        '(defplot loss-chart [losses] [fig ax]\n'
        '  (.plot ax (range (len losses)) losses :label "Training loss" :marker "o")\n'
        '  (.set_title ax "Training progress")\n'
        '  (.set_xlabel ax "Epoch")\n'
        '  (.set_ylabel ax "MSE")\n'
        '  (.legend ax))\n'
        '(setv figure (loss-chart losses))\n'
        '(.savefig figure "showcase-loss.png")\n'
        '(plt.close figure)',
    ]
    for snippet in snippets:
        assert snippet in source, "The showcased snippet must be runnable"
        assert snippet in readme


def test_readme_explains_artifacts_installation_and_limits():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Why use Lispyn for AI notebooks?" in readme
    assert "Lisp-powered AI notebooks" in readme
    assert "Why use it for research?" not in readme
    assert "git clone https://github.com/edujbarrios/lispyn.git" in readme
    assert "pip install lispyn" in readme
    assert "pip install -r requirements.txt" in readme
    assert "lispyn run examples/hello.hy\n" in readme
    assert "lispyn build examples/hello.hy -o hello.ipynb\n" in readme
    assert "lispyn execute examples/hello.hy -o hello-executed.ipynb\n" in readme
    assert "Nothing is executed" in readme
    assert "save the executed cells" in readme
    assert "PyTorch is installed separately" in readme
    assert "MACRO_AUTHORING.md" in readme
    assert all(name in readme for name in ("PyTorch", "Matplotlib", "pandas", "NumPy"))


def test_human_and_agent_positioning_is_explicit():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "For humans and AI agents" in readme
    assert "AI coding agents" in readme
    assert "AGENTS.md" in readme
    for macro in ("defeval-step", "defpredict-step", "collect-batches",
                  "with-torch-seed", "with-clean-gradients"):
        assert macro in readme
