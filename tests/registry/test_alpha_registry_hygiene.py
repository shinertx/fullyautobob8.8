import json
import pytest
from typing import List, Dict, Any

from v26meme.core.state import StateManager
from v26meme.cli import _alpha_registry_hygiene

class FakeRedis:
    def __init__(self):
        self.store = {}
    def set(self, k,v): self.store[k]=v
    def get(self,k): return self.store.get(k)
    def hset(self,*a,**k): pass
    def hget(self,*a,**k): return None
    def hkeys(self,*a,**k): return []
    def ping(self): return True

class FakeState(StateManager):
    def __init__(self):
        self.r = FakeRedis()
    def set(self, key: str, value: Any): self.r.set(key, json.dumps(value))
    def get(self, key: str):
        v = self.r.get(key)
        return json.loads(v) if v else None
    # mimic real API used by hygiene
    def get_active_alphas(self):
        return self.get('active_alphas') or []
    def set_active_alphas(self, alphas):
        self.set('active_alphas', alphas)

@pytest.fixture
def base_cfg():
    return {"discovery": {
        "promotion_criteria": {"min_trades":75, "min_sortino":1.25, "max_mdd":0.2, "min_sharpe":1.1, "min_win_rate":0.52},
        "promotion_buffer_multiplier": 1.0,
        "max_return_padding_trim": 5,
        "enforce_current_gates_on_start": False,
    }}

@pytest.fixture
def make_alpha():
    def _m(aid: str, n_trades: int, returns: List[float], sharpe=1.5, sortino=1.6, win=0.6, mdd=0.1):
        return {"id": aid, "performance": {"all": {"n_trades": n_trades, "returns": returns, "sharpe": sharpe, "sortino": sortino, "win_rate": win, "mdd": mdd}}}
    return _m

def test_deduplication_preserves_first(base_cfg, make_alpha):
    st = FakeState()
    a1 = make_alpha('abc', 10, [0.01]*10)
    a2 = make_alpha('def', 12, [0.02]*12)
    a1_dup = make_alpha('abc', 10, [0.01]*10)
    st.set('active_alphas', [a1, a2, a1_dup])
    res = _alpha_registry_hygiene(st, base_cfg)
    assert res['dupes_removed'] == 1
    active = st.get('active_alphas')
    assert [a['id'] for a in active] == ['abc','def']


def test_padding_trim(base_cfg, make_alpha):
    st = FakeState()
    # 5 padding zeros allowed trim (<= max_return_padding_trim)
    returns = [0.01]*8 + [0.0]*5
    a = make_alpha('aaa', 8, returns)
    st.set('active_alphas', [a])
    res = _alpha_registry_hygiene(st, base_cfg)
    assert res['trimmed'] == 1
    active = st.get('active_alphas')
    perf = active[0]['performance']['all']
    assert len(perf['returns']) == perf['n_trades'] == 8


def test_padding_not_trimmed_when_exceeds_limit(base_cfg, make_alpha):
    st = FakeState()
    base_cfg['discovery']['max_return_padding_trim'] = 3
    returns = [0.02]*8 + [0.0]*5
    a = make_alpha('bbb', 8, returns)
    st.set('active_alphas', [a])
    res = _alpha_registry_hygiene(st, base_cfg)
    assert res['trimmed'] == 0
    active = st.get('active_alphas')
    assert len(active[0]['performance']['all']['returns']) == 13


def test_enforcement_drops_non_compliant(base_cfg, make_alpha):
    st = FakeState()
    base_cfg['discovery']['enforce_current_gates_on_start'] = True
    good = make_alpha('good', 80, [0.01]*80)
    bad = make_alpha('bad', 20, [0.02]*20)  # fails min_trades
    st.set('active_alphas', [good, bad])
    res = _alpha_registry_hygiene(st, base_cfg)
    assert res['dropped_gates'] == 1
    active = st.get('active_alphas')
    assert [a['id'] for a in active] == ['good']


def test_no_changes_when_clean(base_cfg, make_alpha):
    st = FakeState()
    a = make_alpha('xyz', 10, [0.01]*10)
    st.set('active_alphas', [a])
    res = _alpha_registry_hygiene(st, base_cfg)
    assert res['dupes_removed'] == 0 and res['trimmed'] == 0 and res['dropped_gates'] == 0
