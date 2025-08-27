"""
v26meme Dashboard - Paper-mode alpha factory monitoring
PIT-safe: Read-only telemetry from event-sourced Redis state
"""
import streamlit as st
import pandas as pd
import redis, yaml
import json
import time
import plotly.express as px
from typing import Optional, List, Dict, Any

st.set_page_config(page_title="v26meme v4.7.5 Dashboard", layout="wide")

@st.cache_resource
def get_redis() -> Optional[redis.Redis]:
    """Connect to Redis with connection pooling"""
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        return r
    except redis.exceptions.ConnectionError as e:
        st.error(f"Redis connection failed: {e}")
        return None

r = get_redis()
if not r:
    st.stop()

def get_state(key: str) -> Optional[Any]:
    """Fetch and decode Redis state key safely"""
    val = r.get(key)
    if not val:
        return None
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return None

@st.cache_data(ttl=60)
def load_config():
    """Load config from YAML, cached for 60s."""
    try:
        with open("configs/config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("configs/config.yaml not found.")
        return {}

def equity_curve() -> List[Dict[str, Any]]:
    """Fetch equity curve from sorted set"""
    return [json.loads(v) for v in r.zrange('equity_curve', 0, -1)]

@st.cache_data(ttl=10)
def get_llm_history():
    """Fetch LLM proposal and injection history from Redis sorted sets."""
    dfs = []
    
    # Proposals
    proposals_raw = r.zrange('llm:proposer:history', 0, -1, withscores=False)
    if proposals_raw:
        proposals = [json.loads(p) for p in proposals_raw]
        prop_df = pd.DataFrame(proposals).rename(columns={'generated': 'count'})
        prop_df['type'] = 'Proposed'
        dfs.append(prop_df)

    # Injections
    injections_raw = r.zrange('eil:llm_injection:history', 0, -1, withscores=False)
    if injections_raw:
        injections = [json.loads(i) for i in injections_raw]
        inj_df = pd.DataFrame(injections).rename(columns={'injected': 'count'})
        inj_df['type'] = 'Injected'
        dfs.append(inj_df)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs)
    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    return df.sort_values('ts')

st.title("🧠 v26meme v4.7.5 — Autonomous Alpha Factory [PAPER MODE]")
st.caption("Doctrine: $200 → $1M in 30 days | PIT-correct | No magic numbers | Paper-first")

# --- Discovery & Guardrail Status ---
st.subheader("Discovery & Guardrail Status")

cfg = load_config()
# Fetch real-time telemetry
gate_stage = r.get('gate:stage:current') or "default"
gate_cfg = get_state('eil:gate:current_config') or {}
diagnostic = get_state('eil:gate:diagnostic') or {}
coverage = get_state('coverage:panel:1h') or {}
pbo = get_state('eil:pbo:last') or {}
heartbeat = r.get('loop:heartbeat')
last_update_ago = (int(time.time()) - int(heartbeat)) if heartbeat else None

if last_update_ago is not None and last_update_ago > 120:
    st.warning(f"LOOP STALLED: Last heartbeat {last_update_ago}s ago")
else:
    st.success(f"LOOP OK: Last heartbeat {last_update_ago}s ago" if last_update_ago is not None else "LOOP OK")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Discovery Stage", gate_stage.capitalize())
k2.metric("FDR α-level", f"{gate_cfg.get('fdr_alpha', 0.0):.2f}")
k3.metric("DSR Min Prob", f"{gate_cfg.get('dsr_min_prob', 0.0):.2f}")
k4.metric("Survivor Density", f"{(diagnostic.get('survivor_density', 0) * 100):.2f}%")
k5.metric("PBO (Overfit Prob)", f"{(pbo.get('pbo', 0) * 100):.1f}%")

with st.expander("View More Diagnostics"):
    st.json({
        "gate_config": gate_cfg,
        "coverage_1h": coverage,
        "pbo_details": pbo
    })

# --- LLM & EIL Injection Status ---
st.subheader("LLM Strategy Injection")
llm_cfg = cfg.get('llm', {})

l1, l2, l3, l4 = st.columns(4)
l1.metric("LLM Proposer", "✅ Enabled" if llm_cfg.get('enable') else "❌ Disabled")

last_proposal_run = get_state('llm:proposer:last_run') or {}
last_injection_run = get_state('eil:llm_injection:last_run') or {}

if last_proposal_run:
    last_prop_ts = last_proposal_run.get('ts', 0)
    last_prop_ago = int(time.time() - last_prop_ts) if last_prop_ts > 0 else -1
    l2.metric("Last Proposal", f"{last_prop_ago}s ago" if last_prop_ago >= 0 else "Never", help=f"Seeded with: {last_proposal_run.get('seeded_with', 'N/A')}")
    l3.metric("Proposals Generated", last_proposal_run.get('generated', 0))
else:
    l2.metric("Last Proposal", "Never")
    l3.metric("Proposals Generated", 0)

if last_injection_run:
    l4.metric("Proposals Injected (EIL)", last_injection_run.get('injected', 0))
else:
    l4.metric("Proposals Injected (EIL)", 0)

history_df = get_llm_history()
if not history_df.empty:
    st.plotly_chart(
        px.bar(history_df, x='ts', y='count', color='type', title='LLM Proposal & Injection History (Last 1000 Cycles)',
               labels={'ts': 'Time', 'count': 'Formula Count', 'type': 'Action'}, barmode='group'),
        use_container_width=True)

# --- Key metrics ---
st.subheader("Portfolio Status")
portfolio = get_state('portfolio') or {}
active_alphas = get_state('active_alphas') or []
target_weights = get_state('target_weights') or {}
cur_max_order = get_state('risk:current_max_order') or 0
cur_kf = get_state('risk:current_kelly_fraction') or 0.5
daily_stop = get_state('adaptive:daily_stop_pct')
halted = get_state('risk:halted')

c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio Equity", f"${portfolio.get('equity', 200):.2f}")
c2.metric("Cash", f"${portfolio.get('cash', 200):.2f}")
c3.metric("Active Alphas", len(active_alphas))
if halted:
    c4.metric("⚠️ RISK HALTED", "POSITIONS FLATTENING", delta_color="inverse")
else:
    c4.metric("Daily Stop (adaptive)", f"{(daily_stop*100):.2f}%" if daily_stop else "—")

st.caption(f"Risk caps — max order: ${cur_max_order}, Kelly fraction: {cur_kf}")

# Equity curve with drawdown
st.subheader("Portfolio Performance")
eq = equity_curve()
if eq:
    df = pd.DataFrame(eq)
    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    df['drawdown'] = (df['equity'] - df['equity'].cummax()) / df['equity'].cummax()
    
    # Doctrine scoreboard: track $200 → $1M progress
    progress = (df['equity'].iloc[-1] - 200) / (1_000_000 - 200) * 100
    st.progress(min(100, max(0, int(progress))))
    st.caption(f"Progress to $1M: {progress:.2f}%")
    
    st.plotly_chart(px.line(df, x='ts', y='equity', title='Equity Curve'), use_container_width=True)
    st.plotly_chart(px.area(df, x='ts', y='drawdown', title='Drawdown (PIT-tracked)'), use_container_width=True)
else:
    st.info("Awaiting equity logs...")

# Active alphas with performance
st.subheader("Active Alphas (BH-FDR Controlled)")
if active_alphas:
    flat = []
    for a in active_alphas:
        row = {
            'id': a['id'][:8],
            'lane': a.get('lane', 'core'),
            'universe': (a.get('universe') or [None])[0],
            'n_trades': a.get('performance', {}).get('all', {}).get('n_trades', 0),
            'sharpe': a.get('performance', {}).get('all', {}).get('sharpe', 0),
            'sortino': a.get('performance', {}).get('all', {}).get('sortino', 0),
            'mdd': a.get('performance', {}).get('all', {}).get('mdd', 0),
        }
        flat.append(row)
    st.dataframe(pd.DataFrame(flat), use_container_width=True)
else:
    st.info("No promoted alphas yet (check FDR gates)")

# Auto-refresh every 10s
time.sleep(10)
st.rerun()
