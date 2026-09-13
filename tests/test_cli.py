import json

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
