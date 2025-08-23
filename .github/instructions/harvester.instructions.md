---

applyTo: "v26meme/data/**,data\_harvester.py,configs/**"
description: "Event-sourced OHLCV harvester for Coinbase/Kraken (PIT-correct, rate-safe, autonomous)."
------------------------------------------------------------------------------------------------------

# Doctrine (always do this)

* **PIT safety**

  * Write only historical bars; when *reading* for research, **drop the last in-flight bar**.
  * For resampling, use **`label='right', closed='right'`** so timestamps mark the bar’s *end*. ([Pandas][1])
  * Atomic parquet writes (tmp → replace) + sidecar `.quality.json`.

* **ccxt hygiene**

  * On every exchange instance after `load_markets()`:

    ```python
    ex.enableRateLimit = True       # built-in throttle is OFF by default
    ex.timeout = max(getattr(ex, "timeout", 0) or 0, 10_000)  # 10s
    ```

    The built-in rate-limiter is **disabled by default**; enable it and reuse the same instance. `timeout` defaults to \~10s but set explicitly. Don’t pass timeouts in `params`. ([GitHub][2])
  * Use `exchange.timeframes` to check supported TFs; never hardcode TFs outside registry aliases. OHLCV **may have missing periods** depending on venue. ([GitHub][3])

* **Symbol normalization (no guessing)**

  * Always `load_markets()` and resolve from **ccxt unified markets** (`symbol`, `base`, `quote`), not raw venue strings. ([GitHub][4])
  * Use CCXT’s **common currency** mapping + our small `base_aliases` (e.g., **`XBT→BTC`**, **`XDG→DOGE`**) in the resolver; never parse Kraken’s legacy ids (`XXBTZUSD`, …) by hand. ([GitHub][5])

* **Coinbase specifics**

  * Candles **omit intervals with no trades**; allowed granularities are `{60,300,900,3600,21600,86400}`; **max 300 candles** per request; “don’t poll frequently” (prefer trades/books WS in real time). ([Coinbase Developer Docs][6])
  * Policy: accept higher gap ratios on **1m**, else **fallback 1m → 5m** (per config).
  * Use `limit=300` when fetching OHLCV from Coinbase. ([Coinbase Developer Docs][6])

* **Kraken specifics**

  * REST uses a **call counter with decay** by API key/tier; exceeding it returns `EAPI:Rate limit exceeded`. ([docs.kraken.com][7])
  * Trading engine has **separate per-pair rate limits** and decay by tier (`EOrder:Rate limit exceeded`). ([docs.kraken.com][8])

* **Rate limits & backoff**

  * Wrap every REST call with our **TokenBucket** and also respect `exchange.rateLimit` between bursts.
  * On `RateLimitExceeded`, **back off and skip this symbol for the cycle** (don’t block the loop). ([GitHub][2])

* **Plan, availability & resumability**

  * Build plan from: core symbols + screener + **EIL queue (single drain point)**.
  * Keep a `(venue,canonical)` **miss counter**; suppress after threshold and **reset counter** on first success.

* **QA gates (fail closed)**

  * Validate schema (UTC ts, numeric OHLCV), monotonic index, gaps/dupes; record `coverage` and `gap_ratio`.
  * Allow **venue/TF gap overrides** via config (e.g., Coinbase 1m up to 60%). If still degraded and a TF-fallback exists, **refetch at fallback TF**. ([Coinbase Developer Docs][6])

* **Aggregation & retention**

  * Build **1h from 5m** using only closed bars; read **current + previous month** partitions to cover month boundaries; resample with right-closed/right-labeled bins. Retention: short for 1m/5m. ([Pandas][1])

* **Partial harvest**

  * On cold start, process **one TF per loop (rotating)** to unblock research quickly; checkpoints ensure determinism.

---

## Minimal config keys

```yaml
harvester:
  gap_accept_overrides:
    coinbase: { "1m": 0.60, "5m": 0.45 }  # accept sparse minutes on thin listings
  tf_fallback:
    coinbase: { "1m": "5m" }              # if still degraded → 5m
  quotas:
    coinbase: { max_requests_per_min: 45, min_sleep_ms: 220 }
    kraken:   { max_requests_per_min: 45, min_sleep_ms: 250 }
```

---

## Snippets (Copilot should pattern-match)

**Exchange init** (enable limiter + timeout, then markets)

```python
ex = getattr(ccxt, ex_id)()
ex.enableRateLimit = True
ex.timeout = max(getattr(ex, "timeout", 0) or 0, 10_000)
ex.load_markets()
```

(Use `exchange.timeframes` + registry aliases for unsupported TFs.) ([GitHub][3])

**Coinbase OHLCV**

```python
limit = 300 if ex.id == "coinbase" else 1000
ohlcv = ex.fetch_ohlcv(symbol, tf_resolved, since=since_ms, limit=limit)
```

(Granularity must be one of `{60,300,900,3600,21600,86400}`.) ([Coinbase Developer Docs][6])

**Gap override + TF fallback**

```python
cap = cfg["harvester"]["gap_accept_overrides"].get(ex.id, {}).get(tf_resolved, default_cap)
qa = validate_frame(df, tf_ms, max_gap_pct=cap)
if qa["degraded"]:
    fb = cfg["harvester"]["tf_fallback"].get(ex.id, {}).get(tf_resolved)
    if fb:
        tf_resolved, tf_ms = fb, TF_MS[fb]
        ohlcv = ex.fetch_ohlcv(symbol, tf_resolved, since=since_ms, limit=limit)
        df2 = pd.DataFrame(_normalize_ohlcv_rows(ohlcv))
        qa = validate_frame(df2, tf_ms, max_gap_pct=cap)
```

(Needed because Coinbase omits no-trade minutes.) ([Coinbase Developer Docs][6])

**Resolver (no manual symbol parsing)**

```python
# canonical "BASE_QUOTE_SPOT" → venue symbol
sym = get_resolver().resolve(ex, base, quote, "SPOT", canonical_key=canon)
# CCXT + aliases handle XBT→BTC, XDG→DOGE, etc.
```

([GitHub][5])

**Resample 5m → 1h**

```python
df = df.set_index("timestamp").sort_index()
ohlc = df.resample("1H", label="right", closed="right").agg(
    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
).dropna(subset=["open","high","low","close"]).reset_index()
```

([Pandas][1])

---

## Tests to generate

* **Coinbase:** 1m series with missing minutes **passes** under gap override; if still degraded, **fallback 1m→5m** triggers; `limit=300` enforced. ([Coinbase Developer Docs][6])
* **Kraken:** resolver maps `BTC_USD_SPOT` despite legacy codes; REST rate-limit backoff path increments a counter and skips the symbol for the cycle. ([docs.kraken.com][7])
* **Aggregator:** month-boundary case (needs previous+current month) still yields continuous 1h bars with right-labeled bins. ([Pandas][1])
* **ccxt hygiene:** `enableRateLimit=True` and `timeout≥10_000` set on all exchange instances; single instance reused. ([GitHub][2])

[