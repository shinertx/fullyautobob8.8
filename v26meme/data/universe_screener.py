from typing import List, Dict, Any, Tuple
from loguru import logger
import ccxt, time
from v26meme.data.token_bucket import TokenBucket  # NEW

from v26meme.data.usd_fx import USDFX
from v26meme.registry.canonical import make_canonical
from v26meme.registry.resolver import get_resolver  # NEW: dynamic quote policy

def _spread_bps(t: dict) -> float:
    bid, ask = t.get('bid'), t.get('ask')
    if not bid or not ask or bid <= 0 or ask <= 0: return float('inf')
    mid = 0.5 * (bid + ask)
    return 10000.0 * (ask - bid) / mid if mid > 0 else float('inf')

def _impact_bps(ex, venue_symbol: str, notional_usd: float, usd_per_quote: float) -> float:
    """Estimate impact (bps) for a market order using venue symbol (CCXT format)."""
    try:
        ob = ex.fetch_order_book(venue_symbol, limit=50)
        bids = ob.get('bids', []) if isinstance(ob, dict) else []
        asks = ob.get('asks', []) if isinstance(ob, dict) else []
        if not bids or not asks:
            return float('inf')
        # required quote notional in quote currency units
        need_quote = notional_usd / max(1e-12, usd_per_quote)
        filled_quote = 0.0
        filled_base = 0.0
        for price, size, *rest in asks:
            try:
                price = float(price); size = float(size)
            except Exception:
                continue
            cost = price * size
            remain = need_quote - filled_quote
            take = min(cost, remain)
            if take <= 0:
                break
            filled_quote += take
            filled_base += take / max(1e-12, price)
            if filled_quote >= need_quote * 0.999:
                break
        if filled_quote < need_quote * 0.999 or filled_base <= 0:
            return float('inf')
        best_bid = float(bids[0][0]) if bids and bids[0] else 0.0
        best_ask = float(asks[0][0]) if asks and asks[0] else 0.0
        if best_bid <= 0 or best_ask <= 0:
            return float('inf')
        mid = 0.5 * (best_bid + best_ask)
        vwap = filled_quote / max(1e-12, filled_base)
        if mid <= 0:
            return float('inf')
        if best_ask > 0 and vwap > best_ask * 1.25:
            return float('inf')
        return 10000.0 * (vwap - mid) / mid
    except Exception:
        logger.opt(exception=True).warning(f"Impact calc failed unexpectedly for {venue_symbol}")
        return float('inf')

class UniverseScreener:
    def __init__(self, exchanges: List[str], screener_cfg: dict, feeds_cfg: dict | None = None):
        self.exchanges = {}
        for ex in exchanges:
            try:
                obj = getattr(ccxt, ex)()
                # Harden network/rate behavior
                try:
                    obj.enableRateLimit = True
                    if not getattr(obj, 'timeout', None) or obj.timeout < 10_000:
                        obj.timeout = 10_000
                except Exception:
                    pass
                obj.load_markets()
                self.exchanges[ex] = obj
            except Exception as e:
                logger.warning(f"Could not init {ex}: {e}")
        self.cfg = screener_cfg
        self.feeds = feeds_cfg or {}
        of_cfg = self.feeds.get('orderflow', {})
        self.orderflow_enabled = bool(of_cfg.get('enabled', False))
        self.orderflow_top = int(of_cfg.get('top_levels', 5))
        # removed hard-coded allowed_quotes; use registry policy instead
        self.exclude_stable_stable = bool(screener_cfg.get('exclude_stable_stable', False))
        self._stable_set = {"USDT","USDC","DAI","FDUSD","TUSD","PYUSD"}
        self._resolver = get_resolver()  # dynamic access to policy
        quotas = (screener_cfg.get('quotas') or {})
        self._buckets: Dict[str, TokenBucket] = {}
        for ex_id in self.exchanges.keys():
            q = quotas.get(ex_id, {})
            self._buckets[ex_id] = TokenBucket(q.get('max_requests_per_min', 60), q.get('min_sleep_ms', 200))
        # NEW: merge venue alias mappings (commonCurrencies) into resolver base_aliases
        try:
            from v26meme.registry.resolver import add_alias  # type: ignore
            for ex_id, ex in self.exchanges.items():
                ccmap = getattr(ex, 'commonCurrencies', {}) or {}
                for alt, uni in ccmap.items():
                    try:
                        add_alias(uni, alt)
                    except Exception:
                        continue
        except Exception:
            pass

    def get_active_universe(self) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, dict]]]:
        logger.info(f"[screener] SENTINEL enter get_active_universe spread_cap={self.cfg.get('max_spread_bps')} impact_cap={self.cfg.get('max_impact_bps')} order_usd={self.cfg.get('typical_order_usd')} volume_min={self.cfg.get('min_24h_volume_usd')}")
        if not self.exchanges: return [], {}
        tickers_by_venue: Dict[str, Dict[str, dict]] = {}
        for name, ex in self.exchanges.items():
            try:
                b = self._buckets.get(name)
                if b: b.consume(1)
                # rely on global timeout/rateLimit; no per-call params timeout (doctrine compliance)
                tickers_by_venue[name] = ex.fetch_tickers()
            except Exception as e:
                logger.error(f"fetch_tickers failed on {name}: {e}")
                logger.exception(e)

        if not tickers_by_venue:
            logger.error("No tickers fetched; screener empty."); return [], {}

        fx = USDFX(self.cfg.get('stablecoin_parity_warn_bps', 100))
        fx.load_from_tickers(tickers_by_venue)

        candidates = []
        for venue, ex in self.exchanges.items():
            markets = ex.markets or {}
            ticks = tickers_by_venue.get(venue, {})
            for sym, m in markets.items():
                if (m.get("swap") or m.get("future")) and not self.cfg.get('derivatives_enabled', False): continue
                if "/" not in sym: continue
                t = ticks.get(sym, {})
                last = t.get('last') or t.get('close')
                if not last: continue
                try: price = float(last)
                except Exception: continue

                # Early quote + price filters
                quote = m.get("quote")
                # Dynamic allowed quotes per venue from registry policy (falls back to global list)
                policy = getattr(self._resolver, 'policy', None)
                if policy is None:
                    continue
                allowed_quotes = (getattr(policy, 'allowed_quotes_by_venue', {}) or {}).get(venue, getattr(policy, 'allowed_quotes_global', []))
                if quote not in allowed_quotes:
                    continue
                if price < float(self.cfg.get('min_price', 0.0)):
                    continue
                rate = 1.0 if quote == "USD" else fx.to_usd(quote)
                if rate is None: continue

                qv, bv = t.get('quoteVolume'), t.get('baseVolume')
                vol_usd = 0.0
                try:
                    # Per inspector_output.log, Kraken provides reliable quoteVolume.
                    # Coinbase does not provide volume fields in its fetch_tickers response.
                    if qv is not None:
                        vol_usd = float(qv) * rate
                    # Fallback for exchanges that provide baseVolume but not quoteVolume
                    elif bv is not None and price is not None:
                        vol_usd = float(bv) * float(price) * rate
                    # If an instrument has no volume data (e.g., from Coinbase), it cannot be screened on volume.
                    else:
                        # We can't screen by volume, so we can either assign 0 or skip.
                        # Assigning 0 ensures it will be filtered out by the min_volume check below.
                        vol_usd = 0.0
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse volume for {sym} on {venue}")
                    vol_usd = 0.0
                
                if vol_usd < self.cfg['min_24h_volume_usd'] or price < self.cfg['min_price']: continue

                spr = _spread_bps(t)
                if spr == float('inf') or spr > self.cfg['max_spread_bps']: continue

                base, quote = m.get("base"), m.get("quote")
                if self.exclude_stable_stable and base in self._stable_set and quote in self._stable_set:
                    continue
                canonical = make_canonical(base, quote, kind="SPOT")
                inst = {
                    "venue": venue, "type": "spot",
                    # canonical id for downstream joins
                    "market_id": canonical,
                    # preserve raw venue symbol for API calls (order book / trades)
                    "venue_symbol": m.get("symbol",""),
                    "base": base, "quote": quote, "display": canonical,
                    "precision": m.get("precision", {}), "limits": m.get("limits", {}),
                    "spread_bps": spr, "price": price, "volume_24h_usd": vol_usd,
                    "usd_per_quote": rate
                }
                candidates.append((inst, vol_usd))

        if not candidates:
            logger.warning("Screener filters removed all markets."); return [], tickers_by_venue

        logger.info(f"[screener] candidates_pre_impact count={len(candidates)} sample={[c[0]['market_id'] for c in candidates[:8]]}")

        # Increase the pool of candidates to check for impact
        candidate_pool_size = self.cfg.get('max_markets', 100) * 5 
        short = sorted(candidates, key=lambda kv: kv[1], reverse=True)[:candidate_pool_size]
        
        selected_with_vol = []
        impact_attempts = impact_ok = impact_inf = impact_exceed = impact_exceptions = 0
        samples: list[dict] = []

        for (inst, vol) in short:
            ex = self.exchanges[inst['venue']]
            try:
                b = self._buckets.get(inst['venue'])
                if b: b.consume(1)
                imp = _impact_bps(ex, inst['venue_symbol'], self.cfg['typical_order_usd'], inst.get('usd_per_quote', 1.0))
                impact_attempts += 1
                if imp == float('inf'):
                    impact_inf += 1
                elif imp > self.cfg['max_impact_bps']:
                    impact_exceed += 1
                else:
                    impact_ok += 1
                    inst['impact_bps'] = imp
                    selected_with_vol.append((inst, vol))
                    if len(samples) < 5:
                        samples.append({"market": inst['market_id'], "spr": round(inst['spread_bps'], 2), "imp": round(imp, 2)})
            except Exception as e:
                impact_attempts += 1
                impact_exceptions += 1
                logger.warning(f"Impact calc failed for {inst.get('market_id','unknown')} on {inst.get('venue','unknown')}: {e}")
            
            # Use the exchange's rate limit to avoid being throttled
            rate_limit_ms = getattr(ex, 'rateLimit', 100)
            time.sleep(rate_limit_ms / 1000.0)

        if not selected_with_vol:
            logger.warning(f"Impact screening removed all markets. attempts={impact_attempts} inf={impact_inf} exceed={impact_exceed} exceptions={impact_exceptions} max_spread_bps={self.cfg['max_spread_bps']} max_impact_bps={self.cfg['max_impact_bps']} typical_order_usd={self.cfg['typical_order_usd']}")
            if samples:
                logger.info(f"[screener] impact_samples={samples}")
            return [], tickers_by_venue

        logger.info(f"[screener] impact_ok={impact_ok}/{impact_attempts} retained={len(selected_with_vol)} samples={samples}")
        
        # Sort the successfully screened markets by volume and take the top N
        selected_with_vol.sort(key=lambda kv: kv[1], reverse=True)
        final_selection = [k for k, _ in selected_with_vol[:self.cfg['max_markets']]]

        # Attach canonical display name and optional orderflow features
        for inst in final_selection:
            # Ensure display remains canonical
            inst['display'] = make_canonical(inst['base'], inst['quote'], kind="SPOT")
        if self.orderflow_enabled and final_selection:
            try:
                from v26meme.feeds.orderflow import OrderflowSnap
                snap = OrderflowSnap(top_levels=self.orderflow_top)
                # Group instruments by venue for OB snapshot
                by_venue: Dict[str, List[dict]] = {}
                for inst in final_selection:
                    if inst['venue'].lower() != 'synthetic':
                        by_venue.setdefault(inst['venue'], []).append(inst)
                for venue, inst_list in by_venue.items():
                    try:
                        ex = self.exchanges[venue]
                        b = self._buckets.get(venue)
                        if b: b.consume(1)
                    except Exception as e:
                        logger.warning(f"Orderflow features: could not init {venue}: {e}")
                        continue
                    try:
                        features = snap.collect_for_instruments(ex, inst_list)
                    except Exception as e:
                        logger.warning(f"Orderflow features collection failed for {venue}: {e}")
                        features = {}
                    for inst in inst_list:
                        feats = features.get(inst['display']) or {}
                        if 'of_imbalance' in feats:
                            inst['of_imbalance'] = feats['of_imbalance']
                        if 'of_microprice_dev' in feats:
                            inst['of_microprice_dev'] = feats['of_microprice_dev']
                        if 'of_spread_bps' in feats:
                            inst['of_spread_bps'] = feats['of_spread_bps']
            except Exception as e:
                logger.error(f"Orderflow injection error: {e}")
        return final_selection, tickers_by_venue
