import json
import pytest

from v26meme.cli import _coverage_gate_ok
from v26meme.core.state import StateManager

class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.kv = {}
    def hkeys(self, key):
        return list(self.hashes.get(key, {}).keys())
    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)
    def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value
    def ping(self): return True
    # simple KV used by reindex flag + panel telemetry
    def get(self, k): return self.kv.get(k)
    def set(self, k, v): self.kv[k] = v

class FakeState(StateManager):
    def __init__(self):
        # type: ignore[attr-defined] - test double
        self.r = FakeRedis()  # type: ignore[assignment]
    def get(self, key):
        if key == 'coverage:raise_threshold':
            return None
        return None

@pytest.fixture
def cfg():
    return {
        'discovery': {
            'defer_until_coverage': True,
            'min_panel_symbols': 2,
            'min_bars_per_symbol': 5,
        },
        'harvester': {
            'min_coverage_for_research': 0.8,
            'high_coverage_threshold': 0.9,
            'panel_target_days': {'1h': 1}
        }
    }


def test_coverage_gate_handles_double_encoded(cfg):
    st = FakeState()
    # Increase actual bars to exceed dynamic target (24) and coverage > threshold
    meta_single = {'coverage': 0.95, 'actual': 30}
    meta_double = {'coverage': 0.90, 'actual': 28}
    st.r.hset('harvest:coverage', 'ex1:1h:BTC-USD', json.dumps(meta_single))
    st.r.hset('harvest:coverage', 'ex1:1h:ETH-USD', json.dumps(json.dumps(meta_double)))
    ok, stats = _coverage_gate_ok(st, cfg, '1h')
    assert ok is True, f"Gate should pass: {stats}"
    assert stats['eligible'] == 2
    assert stats['symbols'] == 2
    assert stats['target_bars'] >= cfg['discovery']['min_bars_per_symbol']


def test_coverage_gate_requires_min_symbols(cfg):
    st = FakeState()
    meta_single = {'coverage': 0.95, 'actual': 30}
    st.r.hset('harvest:coverage', 'ex1:1h:BTC-USD', json.dumps(meta_single))
    ok, stats = _coverage_gate_ok(st, cfg, '1h')
    assert ok is False
    assert 'target_bars' in stats
    assert stats['target_bars'] >= cfg['discovery']['min_bars_per_symbol']
