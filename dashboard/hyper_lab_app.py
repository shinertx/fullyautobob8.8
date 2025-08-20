"""Hyper Lab Dashboard (Streamlit)
Shows EIL (Extreme Iteration Layer) survivor candidates and stats.
Read‑only: pulls from Redis keys eil:candidates:* and adaptive knobs.
"""
import streamlit as st
import redis, json, time
import pandas as pd
from datetime import datetime, timezone
from typing import Any, Dict, List

st.set_page_config(page_title="Hyper Lab Survivors", layout="wide")

@st.cache_resource
def get_redis() -> redis.Redis | None:
    try:
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        return r
    except Exception as e:  # pragma: no cover
        st.error(f"Redis connection failed: {e}")
        return None

r = get_redis()
if not r:
    st.stop()

st.title("⚗️ Hyper Lab — EIL Survivors")
st.caption("Autonomous feature/genetic strategy exploration (paper mode).")

# Fetch survivors
keys = list(r.scan_iter(match="eil:candidates:*", count=500))
rows: List[Dict[str, Any]] = []
for k in keys:
    try:
        d = r.get(k)
        if not d:
            continue
        if isinstance(d, (bytes, bytearray)):
            d = d.decode('utf-8', 'ignore')
        if not isinstance(d, str):
            d = str(d)
        obj = json.loads(d)
        formula_raw = obj.get("formula")
        if isinstance(formula_raw, (list, dict)):
            try:
                formula_str = json.dumps(formula_raw, separators=(",",":"))
            except Exception:
                formula_str = str(formula_raw)
        else:
            formula_str = str(formula_raw)
        if len(formula_str) > 160:
            formula_str = formula_str[:157] + "..."
        ts_val = obj.get("ts")
        try:
            ts_int = int(ts_val)
        except Exception:
            ts_int = 0
        age_s = None
        if ts_int:
            now_s = int(datetime.now(timezone.utc).timestamp())
            age_s = max(0, now_s - ts_int)
        rows.append({
            "id": str(obj.get("fid", ""))[:8],
            "score": round(float(obj.get("score", 0.0) or 0.0), 6),
            "p_value": float(obj.get("p_value", 1.0) or 1.0),
            "dsr_prob": float(obj.get("dsr_prob", 0.0) or 0.0),
            "trades": int(obj.get("trades", 0) or 0),
            "ts": ts_int if ts_int else None,
            "age_s": age_s,
            "formula": formula_str,
        })
    except Exception:
        continue

rows.sort(key=lambda x: x["score"], reverse=True)

# Normalize into DataFrame to avoid Arrow mixed-type issues
if rows:
    try:
        df = pd.DataFrame(rows)
        df['formula'] = df['formula'].astype(str)
        df['formula_preview'] = df['formula'].str.slice(0,120)
    except Exception as e:
        df = pd.DataFrame([{"id":"ERR","score":0.0,"ts":None,"age_s":None,"p_value":1.0,"dsr_prob":0.0,"trades":0,"formula_preview":f"df_build_error:{e}"}])
else:
    df = pd.DataFrame(columns=['id','score','p_value','dsr_prob','trades','ts','age_s','formula_preview'])

col1, col2, col3 = st.columns(3)
col1.metric("Survivor Count", len(rows))
pop_size_raw = r.get("adaptive:population_size")
if isinstance(pop_size_raw, (bytes, bytearray)):
    pop_size_raw = pop_size_raw.decode('utf-8','ignore')
if pop_size_raw is None:
    pop_size = "—"
else:
    try:
        pop_size = int(str(pop_size_raw))
    except Exception:
        pop_size = str(pop_size_raw)
col2.metric("Adaptive Pop Size", pop_size)
col3.metric("Redis Keys Scanned", len(keys))

if len(df):
    display_cols = ['id','score','p_value','dsr_prob','trades','age_s','ts','formula_preview']
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    stale = df['age_s'].max() if 'age_s' in df and len(df) else None
    if stale is not None and stale > 1800:
        st.warning(f"Survivors stale: newest update age {stale}s (>1800s)")
else:
    st.info("No survivors yet. Hyper Lab still warming up.")

with st.expander("Top 5 Formulas"):
    for rrow in rows[:5]:
        st.code(rrow.get("formula"), language="text")

st.caption("Auto-refresh every 15s")
_time = st.empty()
for i in range(15, 0, -1):
    _time.markdown(f"Refreshing in {i}s …")
    time.sleep(1)
st.rerun()
