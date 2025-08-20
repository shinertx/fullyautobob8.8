"""
v26meme Dashboard - Paper-mode alpha factory monitoring
PIT-safe: Read-only telemetry from event-sourced Redis state
"""
import streamlit as st
import pandas as pd
import redis
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

def equity_curve() -> List[Dict[str, Any]]:
    """Fetch equity curve from sorted set"""
    return [json.loads(v) for v in r.zrange('equity_curve', 0, -1)]

st.title("🧠 v26meme v4.7.5 — Autonomous Alpha Factory [PAPER MODE]")
st.caption("Doctrine: $200 → $1M in 30 days | PIT-correct | No magic numbers | Paper-first")

# Key metrics
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
