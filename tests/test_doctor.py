"""Dependency-free tests of the read-only installation diagnostics."""

import json
from importlib import metadata

from lispyn import doctor


def test_doctor_does_not_import_or_install_optional_packages(monkeypatch):
    seen = []

    def fake_version(name):
        seen.append(name)
        if name in ("hy", "nbformat", "packaging", "nbclient", "ipykernel"):
            return "1.0"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(doctor.metadata, "version", fake_version)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/bin/hy2py")
    result = doctor.inspect_environment()
    assert result["can_build"] is True
    assert result["can_execute"] is True
    assert result["packages"]["torch"] is None
    assert result["packages"]["hy"] == "1.0"
    assert "torch" in seen
    assert "Optional: torch missing" in doctor.format_report(result)


def test_doctor_missing_essentials_returns_actionable_instructions(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(doctor.metadata, "version", missing)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    report = doctor.inspect_environment()
    assert not report["can_build"]
    assert not report["can_execute"]
    assert "python -m pip install -e ." in doctor.format_report(report)


def test_doctor_json_cli(capsys):
    from lispyn.cli import main

    assert main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["lispyn"]
    assert "can_build" in report
    assert "can_execute" in report
    assert "hy" in report["packages"]
    assert "matplotlib" in report["packages"]
    assert "python" in report
    assert "packages" in report
    assert "secret" not in report
    assert main(["doctor"]) == 0
    assert "Notebook generation:" in capsys.readouterr().out
