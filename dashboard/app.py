import streamlit as st, pandas as pd, redis, json, time, plotly.express as px, os

st.set_page_config(page_title="v26meme v4.7.5 Dashboard", layout="wide")

@st.cache_resource
def get_redis():
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping(); return r
    except redis.exceptions.ConnectionError as e:
        st.error(f"Redis connection failed: {e}"); return None

r = get_redis()
if not r: st.stop()

def get_state(key):
    val = r.get(key)
    return json.loads(val) if val else None

def equity_curve():
    return [json.loads(v) for v in r.zrange('equity_curve', 0, -1)]

st.title("🧠 v26meme v4.7.5 — Autonomous Alpha Factory (Paper, dynamic mapping enabled)")

portfolio = get_state('portfolio') or {}
active_alphas = get_state('active_alphas') or []
target_weights = get_state('target_weights') or {}
cur_max_order = get_state('risk:current_max_order') or 0
cur_kf = get_state('risk:current_kelly_fraction') or 0.5
daily_stop = get_state('adaptive:daily_stop_pct') or None

c1,c2,c3,c4 = st.columns(4)
c1.metric("Portfolio Equity", f"${portfolio.get('equity', 200):.2f}")
c2.metric("Cash", f"${portfolio.get('cash', 200):.2f}")
c3.metric("Active Alphas", len(active_alphas))
c4.metric("Daily Stop (adapt.)", f"{(daily_stop*100):.2f}%" if daily_stop else "—")
st.caption(f"Risk caps — max order: ${cur_max_order}, Kelly fraction: {cur_kf}")

st.subheader("Portfolio Performance")
eq = equity_curve()
if eq:
    df = pd.DataFrame(eq)
    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    df['drawdown'] = (df['equity'] - df['equity'].cummax()) / df['equity'].cummax()
    st.plotly_chart(px.line(df, x='ts', y='equity', title='Equity'), use_container_width=True)
    st.plotly_chart(px.area(df, x='ts', y='drawdown', title='Drawdown'), use_container_width=True)
else:
    st.info("Awaiting equity logs...")

st.subheader("Target Allocation (by Alpha)")
if target_weights:
    dfw = pd.DataFrame(list(target_weights.items()), columns=['Alpha ID','Weight']).sort_values('Weight', ascending=False)
    st.dataframe(dfw, use_container_width=True)
else:
    st.info("No targets yet.")

st.subheader("Active Alphas")
if active_alphas:
    flat = []
    for a in active_alphas:
        row = {'name': a['name'], 'id': a['id'], 'lane': a.get('lane','core'), 'universe': a.get('universe',[None])[0]}
        perf = a.get('performance', {}).get('all', {})
        row.update({k: perf.get(k) for k in ['n_trades','win_rate','sortino','sharpe','mdd']})
        inst = a.get('instrument') or {}
        row.update({'venue': inst.get('venue'), 'market_id': inst.get('market_id')})
        flat.append(row)
    st.dataframe(pd.DataFrame(flat), use_container_width=True)
else:
    st.info("No promoted alphas yet.")

st.subheader("Slippage Calibration (bps, p90 by canonical)")
slip = get_state("slippage:table") or {}
if slip:
    s = pd.DataFrame(sorted(slip.items(), key=lambda kv: kv[1], reverse=True), columns=["canonical","p90_bps"])
    st.dataframe(s, use_container_width=True)
else:
    st.caption("Micro-live disabled or insufficient samples (need ~20 probes per market).")

st.subheader("Harvester / Quality")
keys = r.hkeys("harvest:checkpoint") or []
if keys:
    rows = []
    for k in keys:
        ts = r.hget("harvest:checkpoint", k)
        rows.append({"key": k, "last_ts": pd.to_datetime(int(ts), unit='ms', utc=True)})
    st.dataframe(pd.DataFrame(rows).sort_values("last_ts", ascending=False), use_container_width=True)
else:
    st.caption("No checkpoints yet.")

time.sleep(10)
st.rerun()
