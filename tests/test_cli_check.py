from click.testing import CliRunner
import sys, types

# Minimal stub to satisfy v26meme.cli import without heavy deps
sys.modules.setdefault("arch", types.SimpleNamespace(arch_model=lambda *a, **k: None))

import v26meme.cli as cli_module


def _dummy_cfg() -> dict:
    return {
        "system": {"redis_host": "localhost", "redis_port": 6379},
        "execution": {},
        "eil": {},
    }


def test_check_success(monkeypatch):
    """Command exits 0 when config and Redis are ok."""
    monkeypatch.setattr(cli_module, "load_config", lambda: _dummy_cfg())

    class DummyState:
        def __init__(self, host: str, port: int) -> None:
            assert host == "localhost" and port == 6379

    monkeypatch.setattr(cli_module, "StateManager", DummyState)
    runner = CliRunner()
    result = runner.invoke(cli_module.check)
    assert result.exit_code == 0
    assert "environment ok" in result.output.lower()


def test_check_missing_config(monkeypatch):
    """Missing config sections yield non-zero exit."""
    cfg = _dummy_cfg()
    cfg.pop("execution")
    monkeypatch.setattr(cli_module, "load_config", lambda: cfg)
    monkeypatch.setattr(cli_module, "StateManager", lambda *a, **k: None)
    runner = CliRunner()
    result = runner.invoke(cli_module.check)
    assert result.exit_code != 0
    assert "missing config sections" in result.output.lower()


def test_check_redis_failure(monkeypatch):
    """Redis connection failure yields non-zero exit."""
    monkeypatch.setattr(cli_module, "load_config", lambda: _dummy_cfg())

    class FailState:
        def __init__(self, host: str, port: int) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(cli_module, "StateManager", FailState)
    runner = CliRunner()
    result = runner.invoke(cli_module.check)
    assert result.exit_code != 0
    assert "redis connection failed" in result.output.lower()
