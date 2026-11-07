from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from adversarial_robustness_lab import cli
from adversarial_robustness_lab.models import SmallCNN
from adversarial_robustness_lab.runtime import save_checkpoint


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("ck") / "m.pt"
    rc = cli.main(
        [
            "train",
            "--synthetic",
            "256",
            "--arch",
            "small_cnn",
            "--epochs",
            "2",
            "--batch-size",
            "64",
            "--lr",
            "0.05",
            "--device",
            "cpu",
            "--out",
            str(path),
            "--eval-clean",
            "64",
        ]
    )
    assert rc == 0
    return path


def test_parse_eps() -> None:
    assert cli.parse_eps("8/255") == pytest.approx(8 / 255)
    assert cli.parse_eps("0.5") == 0.5
    for bad in ("x", "1/0", "-1"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_eps(bad)


def test_info(ckpt: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["info", "--json", "-c", str(ckpt), "--device", "cpu"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["checkpoint"]["arch"] == "small_cnn" and "pgd" in info["attacks"]
    assert cli.main(["info"]) == 0
    assert "architectures" in capsys.readouterr().out


def test_evaluate(ckpt: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "e.json"
    rc = cli.main(
        [
            "evaluate",
            "-c",
            str(ckpt),
            "--synthetic",
            "32",
            "--device",
            "cpu",
            "--attacks",
            "fgsm",
            "pgd",
            "--steps",
            "3",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    text = capsys.readouterr().out
    assert "fgsm" in text and "robust" in text
    rows = json.loads(out.read_text())["results"]
    assert [r["attack"] for r in rows] == ["fgsm", "pgd"]


def test_evaluate_quantized_with_bpda(ckpt: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(
        [
            "evaluate",
            "-c",
            str(ckpt),
            "--synthetic",
            "16",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--quantize-bits",
            "3",
            "--bpda",
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["n"] == 16


def test_autoattack(ckpt: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(
        [
            "autoattack",
            "-c",
            str(ckpt),
            "--synthetic",
            "16",
            "--device",
            "cpu",
            "--steps",
            "3",
            "--targets",
            "1",
            "--queries",
            "20",
        ]
    )
    assert rc == 0
    assert "robust" in capsys.readouterr().out
    cli.main(
        [
            "autoattack",
            "-c",
            str(ckpt),
            "--synthetic",
            "8",
            "--device",
            "cpu",
            "--version",
            "fast",
            "--steps",
            "2",
            "--json",
        ]
    )
    assert json.loads(capsys.readouterr().out)["version"] == "fast"


def test_attack(ckpt: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(
        [
            "attack",
            "-c",
            str(ckpt),
            "--synthetic",
            "4",
            "--device",
            "cpu",
            "--attack",
            "square",
            "--queries",
            "20",
            "--n",
            "4",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Linf" in out and "by square" in out
    cli.main(
        ["attack", "-c", str(ckpt), "--synthetic", "2", "--device", "cpu", "--json", "--steps", "2"]
    )
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 2 and {"fooled", "linf", "l2"} <= set(rows[0])


def test_certify(ckpt: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(
        [
            "certify",
            "-c",
            str(ckpt),
            "--synthetic",
            "4",
            "--device",
            "cpu",
            "--n0",
            "10",
            "--n-samples",
            "100",
            "--sigma",
            "0.25",
            "--batch-size",
            "50",
        ]
    )
    assert rc == 0
    assert "certified accuracy" in capsys.readouterr().out
    cli.main(
        [
            "certify",
            "-c",
            str(ckpt),
            "--synthetic",
            "2",
            "--device",
            "cpu",
            "--n0",
            "5",
            "--n-samples",
            "50",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["n"] == 50 and len(payload["radii"]) == 2


def test_sanity(ckpt: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(
        [
            "sanity",
            "-c",
            str(ckpt),
            "--synthetic",
            "16",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--queries",
            "20",
            "--surrogate",
            str(ckpt),
        ]
    )
    assert rc == 0
    assert "checks passed" in capsys.readouterr().out
    rc = cli.main(
        [
            "sanity",
            "-c",
            str(ckpt),
            "--synthetic",
            "16",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--queries",
            "20",
            "--quantize-bits",
            "2",
            "--strict",
            "--json",
        ]
    )
    assert rc == 2


def test_data_command_reports_bad_archive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "cifar-10-python.tar.gz").write_bytes(b"not the archive")
    assert cli.main(["data", "--data-dir", str(tmp_path)]) == 1
    assert "MD5 mismatch" in capsys.readouterr().err


def test_cifar_split_selection(tmp_path: Path) -> None:
    ns = argparse.Namespace(synthetic=0, data_dir=tmp_path, start=0, n=None)
    with pytest.raises(FileNotFoundError):
        cli.test_split(ns)


def test_module_entrypoint() -> None:
    import runpy

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("adversarial_robustness_lab", run_name="__main__", alter_sys=True)
    assert exc.value.code == 2  # argparse: a command is required


def test_checkpoint_helper_roundtrip(tmp_path: Path) -> None:
    save_checkpoint(tmp_path / "x.pt", SmallCNN(), "small_cnn", {})
    assert cli.main(["info", "-c", str(tmp_path / "x.pt"), "--json"]) == 0
