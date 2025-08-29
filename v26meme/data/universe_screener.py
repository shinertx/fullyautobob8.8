from typing import List, Dict, Any, Tuple
from loguru import logger
import ccxt, time
from v26meme.data.token_bucket import TokenBucket

from v26meme.data.usd_fx import USDFX
from v26meme.registry.canonical import make_canonical
from v26meme.registry.resolver import get_resolver

def _spread_bps(t: dict) -> float:
    bid, ask = t.get('bid'), t.get('ask')
    if not bid or not ask or bid <= 0 or ask <= 0: return float('inf')
    mid = 0.5 * (bid + ask)
    return 10000.0 * (ask - bid) / mid if mid > 0 else float('inf')

def _impact_bps(ex, venue_symbol: str, notional_usd: float, usd_per_quote: float) -> float:
    """Estimate impact (bps) for a market order using venue symbol (CCXT format)."""
    try:
        # Add explicit timeout to the fetch_order_book call
        ob = ex.fetch_order_book(venue_symbol, limit=50, params={"timeout": 10000})
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
            # logger.warning(f"Impact for {venue_symbol} has high slippage: vwap={vwap} vs best_ask={best_ask}")
            return float('inf')
        return 10000.0 * (vwap - mid) / mid
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logger.warning(f"Impact calc for {venue_symbol} failed with CCXT error: {type(e).__name__} - {e}")
        return float('inf')
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
        self.cfg = screener_cfg or {}
        self.feeds = feeds_cfg or {}
        of_cfg = self.feeds.get('orderflow', {})
        self.orderflow_enabled = bool(of_cfg.get('enabled', False))
        self.orderflow_top = int(of_cfg.get('top_levels', 5))
        # removed hard-coded allowed_quotes; use registry policy instead
        self.exclude_stable_stable = bool(self.cfg.get('exclude_stable_stable', False))
        self._stable_set = {"USDT","USDC","DAI","FDUSD","TUSD","PYUSD"}
        self._resolver = get_resolver()  # dynamic access to policy
        quotas = (self.cfg.get('quotas') or {})
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

    def get_active_universe(self, debug: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, dict]]]:
        if debug:
            logger.remove()
            logger.add(lambda msg: print(msg, end=""), level="INFO")

        # Use .get() for safe access to all config values
        log_ctx = {
            "spread_cap": self.cfg.get('max_spread_bps'),
            "impact_cap": self.cfg.get('max_impact_bps'),
            "order_usd": self.cfg.get('typical_order_usd'),
            "volume_min": self.cfg.get('min_24h_volume_usd')
        }
        logger.info(f"[screener] get_active_universe enter context={log_ctx}")

        if not self.exchanges: return [], {}

        tickers_by_venue: Dict[str, Dict[str, dict]] = {}
        for name, ex in self.exchanges.items():
            try:
                b = self._buckets.get(name)
                if b: b.consume(1)
                tickers_by_venue[name] = ex.fetch_tickers()
            except Exception as e:
                logger.error(f"fetch_tickers failed on {name}: {e}")

        if not tickers_by_venue:
            logger.error("No tickers fetched; screener empty."); return [], {}

        fx = USDFX(self.cfg.get('stablecoin_parity_warn_bps', 100))
        fx.load_from_tickers(tickers_by_venue)

        # Safely get config values with defaults
        min_vol_usd = self.cfg.get('min_24h_volume_usd', 1_000_000)
        min_price = self.cfg.get('min_price', 0.0)
        max_spread = self.cfg.get('max_spread_bps', 100)
        derivatives_enabled = self.cfg.get('derivatives_enabled', False)

        candidates = []
        for venue, ex in self.exchanges.items():
            markets = ex.markets or {}
            ticks = tickers_by_venue.get(venue, {})
            for sym, m in markets.items():
                if (m.get("swap") or m.get("future")) and not derivatives_enabled: continue
                if "/" not in sym: continue
                
                t = ticks.get(sym, {})
                last = t.get('last') or t.get('close')
                if not last: continue
                try:
                    price = float(last)
                except (ValueError, TypeError):
                    continue

                quote = m.get("quote")
                policy = getattr(self._resolver, 'policy', None)
                if policy is None:
                    from v26meme.registry.resolver import DEFAULT_ALLOWED_QUOTES
                    allowed_quotes = DEFAULT_ALLOWED_QUOTES
                else:
                    allowed_quotes = (getattr(policy, 'allowed_quotes_by_venue', {}) or {}).get(venue, getattr(policy, 'allowed_quotes_global', []))

                if quote not in allowed_quotes: continue
                if price < min_price: continue
                
                rate = 1.0 if quote == "USD" else fx.to_usd(quote)
                if rate is None: continue

                qv, bv = t.get('quoteVolume'), t.get('baseVolume')
                vol_usd = 0.0
                try:
                    if qv is not None:
                        vol_usd = float(qv) * rate
                    elif bv is not None and price is not None:
                        vol_usd = float(bv) * price * rate
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse volume for {sym} on {venue}")
                
                if vol_usd < min_vol_usd: continue

                spr = _spread_bps(t)
                if spr > max_spread: continue

                base = m.get("base")
                if self.exclude_stable_stable and base in self._stable_set and quote in self._stable_set:
                    continue
                
                canonical = make_canonical(base, quote, kind="SPOT")
                candidates.append({
                    "canonical": canonical, "display": canonical, "venue": venue, "venue_symbol": sym,
                    "base": base, "quote": quote, "price": price,
                    "spread_bps": spr, "volume_24h_usd": vol_usd, "rate": rate,
                })

        logger.info(f"[screener] candidates_pre_impact count={len(candidates)} sample={[c['canonical'] for c in candidates[:5]]}")

        # Impact Screening (now safe)
        impact_cap = self.cfg.get('max_impact_bps')
        order_usd = self.cfg.get('typical_order_usd')

        if impact_cap is None or order_usd is None:
            logger.warning("Impact screening skipped: `max_impact_bps` or `typical_order_usd` not configured.")
            return candidates, tickers_by_venue

        final_universe = []
        for c in candidates:
            venue, venue_sym, canonical, rate = c['venue'], c['venue_symbol'], c['canonical'], c['rate']
            try:
                ex = self.exchanges[venue]
                b = self._buckets.get(venue)
                if b: b.consume(1)
                imp = _impact_bps(ex, venue_sym, order_usd, rate)
                if imp > impact_cap:
                    if debug: logger.info(f"Reject {canonical} on {venue}: impact={imp:.2f} > {impact_cap:.2f}")
                    continue
                c['impact_bps'] = imp
                final_universe.append(c)
            except Exception as e:
                logger.warning(f"Impact calc failed for {canonical} on {venue}: {e}")
                continue
        
        logger.info(f"Screening complete. Kept {len(final_universe)}/{len(candidates)} symbols.")
        return final_universe, tickers_by_venue

    def _get_orderflow_snapshot(self, ex, venue_symbol: str) -> Dict[str, List]:
        """Fetch top N levels of bids/asks."""
        try:
            # Add explicit timeout to the fetch_order_book call
            ob = ex.fetch_order_book(venue_symbol, limit=self.orderflow_top, params={"timeout": 10000})
            return {
                'bids': ob.get('bids', [])[:self.orderflow_top],
                'asks': ob.get('asks', [])[:self.orderflow_top]
            }
        except Exception:
            return {'bids': [], 'asks': []}
