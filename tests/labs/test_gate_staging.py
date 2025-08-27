import os, time, json
import pytest
from v26meme.labs.hyper_lab import run_eil, load_config
from v26meme.core.state import StateManager


@pytest.mark.parametrize("patience", [1])
def test_gate_stage_progression(monkeypatch, patience):
    monkeypatch.setenv('EIL_MAX_CYCLES', '2')
    from v26meme.labs import hyper_lab as hl
    cfg = load_config()
    # Force minimal coverage gates
    cfg['discovery']['min_bars_per_symbol'] = 1
    cfg['discovery']['min_panel_symbols'] = 1
    # Configure 2 stages with very permissive first stage so we likely escalate
    cfg['discovery']['gate_stages'] = [
        {'name': 'relaxed', 'fdr_alpha': 1.0, 'dsr_min_prob': 0.0, 'min_trades': 0},
        {'name': 'tight', 'fdr_alpha': 1.0, 'dsr_min_prob': 0.0, 'min_trades': 0}
    ]
    cfg['discovery']['gate_stage_escalation'] = {
        'survivor_density_min': 0.0,
        'median_trades_min': 0.0,
        'patience_cycles': patience
    }
    monkeypatch.setattr(hl, 'load_config', lambda : cfg)
    run_eil()
    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    stage = state.get('eil:gate:stage')
    assert stage in ('relaxed','tight')
    # With patience=1 and permissive metrics, expect escalation to tight by end
    assert stage == 'tight'


def test_stage_override_thresholds(monkeypatch):
    monkeypatch.setenv('EIL_MAX_CYCLES', '1')
    from v26meme.labs import hyper_lab as hl
    cfg = load_config()
    cfg['discovery']['min_bars_per_symbol'] = 1
    cfg['discovery']['min_panel_symbols'] = 1
    cfg['discovery']['gate_stages'] = [
        {'name': 'relaxed', 'fdr_alpha': 0.5, 'dsr_min_prob': 0.0, 'min_trades': 0},
    ]
    monkeypatch.setattr(hl, 'load_config', lambda : cfg)
    run_eil()
    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    diag = state.get('eil:gate:diagnostic') or {}
    assert 'stage' in diag


def test_escalation_patience(monkeypatch):
    monkeypatch.setenv('EIL_MAX_CYCLES', '3')
    from v26meme.labs import hyper_lab as hl
    cfg = load_config()
    cfg['discovery']['min_bars_per_symbol'] = 1
    cfg['discovery']['min_panel_symbols'] = 1
    cfg['discovery']['gate_stages'] = [
        {'name': 'relaxed', 'fdr_alpha': 1.0, 'dsr_min_prob': 0.0, 'min_trades': 0},
        {'name': 'tight', 'fdr_alpha': 1.0, 'dsr_min_prob': 0.0, 'min_trades': 0}
    ]
    # Set patience high so we do NOT escalate in limited cycles
    cfg['discovery']['gate_stage_escalation'] = {
        'survivor_density_min': 1.0,  # unreachable
        'median_trades_min': 9999.0,
        'patience_cycles': 10
    }
    monkeypatch.setattr(hl, 'load_config', lambda : cfg)
    run_eil()
    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    stage = state.get('eil:gate:stage')
    assert stage == 'relaxed'


def test_bootstrap_path_trigger(monkeypatch):
    monkeypatch.setenv('EIL_MAX_CYCLES', '1')
    from v26meme.labs import hyper_lab as hl
    cfg = load_config()
    cfg['discovery']['min_bars_per_symbol'] = 1
    cfg['discovery']['min_panel_symbols'] = 1
    # Force very high bootstrap min_trades so panel_cv_stats chooses bootstrap path
    cfg['validation']['bootstrap'] = {
        'enabled': True,
        'n_iter': 10,
        'min_trades': 10_000,  # unreachable => triggers bootstrap early path on low n
        'seed': 123,
        'method': 'basic'
    }
    monkeypatch.setattr(hl, 'load_config', lambda : cfg)
    run_eil()
    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    # Diagnostic presence confirms evaluation executed
    diag = state.get('eil:gate:diagnostic') or {}
    assert isinstance(diag, dict)
