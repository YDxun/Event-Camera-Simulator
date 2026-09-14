import json
from pathlib import Path

from evsim.cli import main
from evsim.config import SimulatorConfig


def test_validate_cli_passes(capsys):
    assert main(["validate", "--seed", "11"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is True
    assert all(check["passed"] for check in result["checks"])


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"sensor": {"unknown": 1}}', encoding="utf-8")
    try:
        SimulatorConfig.load(path)
    except ValueError as exc:
        assert "Unknown keys" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_demo_cli_executes(tmp_path: Path, capsys):
    output_dir = tmp_path / "demo_test"
    assert (
        main(["demo", "-o", str(output_dir), "--seconds", "0.05", "--fps", "200.0"])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["frames_processed"] == 10
    assert (output_dir / "events.npz").is_file()


def test_benchmark_cli_executes(capsys):
    assert main(["benchmark", "--width", "32", "--height", "24", "--frames", "5"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "speedup_loop_over_vectorized" in result
