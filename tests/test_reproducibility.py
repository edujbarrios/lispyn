"""Experiment manifests and seeded workflows, with no external downloads."""

import json
import random

import pytest


def test_seed_all_repeats_random_generators():
    from lispyn.reproducibility import seed_all

    assert seed_all(17) == 17
    a = random.random()
    seed_all(17)
    assert random.random() == a
    with pytest.raises(ValueError, match="nonnegative"):
        seed_all(-1)
    with pytest.raises(ValueError, match="nonnegative"):
        seed_all(True)


def test_environment_does_not_capture_secrets(monkeypatch):
    from lispyn.reproducibility import environment_info

    monkeypatch.setenv("LISPYN_PRIVATE_TEST_KEY", "never-record-this")
    info = environment_info()
    encoded = json.dumps(info)
    assert info["python"]
    assert "packages" in info
    assert "never-record-this" not in encoded


def _data():
    torch = pytest.importorskip("torch")
    x = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    y = torch.tensor([0, 1, 0, 1])
    return torch, torch.utils.data.TensorDataset(x, y)


def test_experiment_files_summary_and_cli(tmp_path, capsys):
    torch, data = _data()
    from lispyn.cli import main
    from lispyn.workflow import train

    directory = tmp_path / "runs" / "example"
    model = torch.nn.Linear(2, 2)
    result = train(
        model, data, validation=data,
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics="accuracy", batch_size=2, epochs=3, seed=42,
        device="cpu", experiment_dir=directory,
    )
    assert result.manifest_path == directory / "manifest.json"
    assert (directory / "best.pt").is_file()
    assert (directory / "history.json").is_file()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["status"] == "completed"
    assert manifest["config"]["seed"] == 42
    assert manifest["config"]["epochs"] == 3
    assert manifest["config"]["tuning"] == "full"
    assert manifest["environment"]["packages"]["torch"]
    assert manifest["summary"]["monitor"] == "val_loss"
    assert manifest["summary"]["value"] == pytest.approx(
        min(result.history["val_loss"])
    )
    assert manifest["summary"]["epoch"] in (1, 2, 3)
    assert json.loads((directory / "history.json").read_text()) == result.history
    assert main(["report", str(directory)]) == 0
    report = capsys.readouterr().out
    assert "Lispyn run: completed" in report
    assert "Best val_loss:" in report


def test_manifest_reports_failed_runs_without_leaking_exception(tmp_path):
    torch, data = _data()
    from lispyn.workflow import train

    def fail(model, batch):
        raise RuntimeError("sensitive-user-dataset-name")

    directory = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="sensitive-user-dataset-name"):
        train(torch.nn.Linear(2, 2), data, step=fail,
              experiment_dir=directory, device="cpu")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure_type"] == "RuntimeError"
    assert "sensitive-user-dataset-name" not in json.dumps(manifest)


def test_manifest_not_generated_without_opt_in():
    torch, data = _data()
    from lispyn.workflow import train

    result = train(torch.nn.Linear(2, 2), data,
                   loss_fn=torch.nn.CrossEntropyLoss(), device="cpu")
    assert result.manifest_path is None
