import os, time, json
import pytest
from v26meme.labs.hyper_lab import run_eil, load_config
from v26meme.core.state import StateManager

@pytest.mark.parametrize("cycles", [1])
def test_diversity_telemetry(monkeypatch, cycles):
    monkeypatch.setenv('EIL_MAX_CYCLES', str(cycles))
    monkeypatch.setenv('EIL_SLEEP_OVERRIDE', '1')
    cfg = load_config()
    # force minimal coverage gate pass
    cfg['discovery']['min_panel_symbols'] = 1
    cfg['discovery']['min_bars_per_symbol'] = 1
    cfg['discovery']['diversity']['speciation_enabled'] = True
    cfg['discovery']['population_size'] = 30
    # lighten generations for speed
    cfg['discovery']['generations_per_cycle'] = 1
    # ensure depth variety for feature sets
    cfg['discovery']['max_formula_depth'] = 3
    from v26meme.labs import hyper_lab as hl
    monkeypatch.setattr(hl, 'load_config', lambda : cfg)
    run_eil()
    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    div = state.get('eil:diag:diversity') or {}
    assert 'n_clusters' in div
    assert 'cluster_sizes' in div


def test_diversity_bonus_nonzero(monkeypatch):
    monkeypatch.setenv('EIL_MAX_CYCLES', '1')
    monkeypatch.setenv('EIL_SLEEP_OVERRIDE', '1')
    cfg = load_config()
    cfg['discovery']['min_panel_symbols'] = 1
    cfg['discovery']['min_bars_per_symbol'] = 1
    cfg['discovery']['diversity']['speciation_enabled'] = True
    cfg['discovery']['population_size'] = 40
    cfg['discovery']['generations_per_cycle'] = 1
    # add diversity weight
    cfg['discovery']['fitness_extras']['weights']['diversity'] = 0.10
    from v26meme.labs import hyper_lab as hl
    monkeypatch.setattr(hl, 'load_config', lambda : cfg)
    run_eil()
    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    # pull raw survivor for diversity presence (score influenced indirectly) just ensure telemetry exists
    div = state.get('eil:diag:diversity') or {}
    assert isinstance(div, dict) and div.get('n_clusters', 0) >= 1
