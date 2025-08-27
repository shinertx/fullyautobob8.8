import sys
import types
import redis
from click.testing import CliRunner


def test_loop_exits_when_redis_unavailable(monkeypatch):
    fake_arch = types.ModuleType("arch")
    fake_arch.arch_model = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "arch", fake_arch)

    def boom(self):
        raise redis.exceptions.ConnectionError("mock")

    monkeypatch.setattr(redis.Redis, "ping", boom)

    from v26meme.cli import loop

    runner = CliRunner()
    result = runner.invoke(loop)

    assert result.exit_code == 1
    assert "start redis-server" in result.output.lower()

