from typing import List, Dict, Any, Tuple
from loguru import logger
import ccxt, time

from v26meme.data.usd_fx import USDFX
from v26meme.registry.canonical import make_canonical

def _spread_bps(t: dict) -> float:
    bid, ask = t.get('bid'), t.get('ask')
    if not bid or not ask or bid <= 0 or ask <= 0: return float('inf')
    mid = 0.5 * (bid + ask)
    return 10000.0 * (ask - bid) / mid if mid > 0 else float('inf')

def _impact_bps(ex, market_id: str, notional_usd: float, usd_per_quote: float) -> float:
    try:
        ob = ex.fetch_order_book(market_id, limit=50) # Deeper book for better accuracy
        if not ob or not isinstance(ob, dict):
            logger.warning(f"Impact calc for {market_id}: Order book is missing or not a dict.")
            return float('inf')

        asks, bids = ob.get('asks') or [], ob.get('bids') or []
        
        if not asks or not bids:
            logger.warning(f"Impact calc for {market_id}: no asks or bids. Asks: {len(asks)}, Bids: {len(bids)}")
            return float('inf')
        
        # This is the amount of QUOTE currency we need to spend.
        # For a $10 trade in BTC/USD, need_quote is 10 USD.
        need_quote = notional_usd / max(1e-12, usd_per_quote)
        
        filled_quote = 0.0
        filled_base = 0.0

        for i, entry in enumerate(asks):
            if not entry or len(entry) < 2:
                logger.warning(f"Impact calc for {market_id}: invalid ask entry at index {i}: {entry}")
                continue
            try:
                price, size, *_ = entry
                price, size = float(price), float(size)
            except (ValueError, TypeError) as e:
                logger.warning(f"Impact calc for {market_id}: could not parse price/size from ask entry {entry}. Error: {e}")
                continue

            # How much we can spend at this level
            cost_of_level = price * size
            
            # How much we still need to spend
            quote_remaining = need_quote - filled_quote

            # Spend the minimum of what's available and what we still need
            quote_to_spend = min(cost_of_level, quote_remaining)
            
            base_to_get = quote_to_spend / price

            filled_quote += quote_to_spend
            filled_base += base_to_get

            if filled_quote >= need_quote:
                break

        if filled_quote < (need_quote * 0.999): # Allow for small floating point inaccuracies
            logger.warning(f"Impact calc for {market_id}: could not fill order. Needed {need_quote} USD, filled {filled_quote} USD.")
            return float('inf')

        # The actual VWAP is the total quote spent divided by the total base acquired.
        vwap = filled_quote / filled_base if filled_base > 0 else 0
        if vwap == 0:
            logger.warning(f"Impact calc for {market_id}: VWAP is zero.")
            return float('inf')

        if not bids[0] or len(bids[0]) < 1 or not asks[0] or len(asks[0]) < 1:
             logger.warning(f"Impact calc for {market_id}: Missing best bid/ask price.")
             return float('inf')

        try:
            best_bid_price, *_ = bids[0]
            best_ask_price, *_ = asks[0]
            best_bid, best_ask = float(best_bid_price), float(best_ask_price)
        except (ValueError, TypeError) as e:
            logger.warning(f"Impact calc for {market_id}: could not parse best bid/ask. Error: {e}")
            return float('inf')

        # Sanity check: if VWAP is wildly different from best ask, something is wrong.
        if best_ask > 0 and vwap > best_ask * 1.25:
            logger.warning(f"Impact calc for {market_id}: VWAP {vwap} is >25% higher than best ask {best_ask}. Invalidating.")
            return float('inf')

        mid = 0.5 * (best_bid + best_ask)

        return 10000.0 * (vwap - mid) / mid if mid > 0 else float('inf')
    except Exception:
        logger.opt(exception=True).warning(f"Impact calc failed unexpectedly for {market_id}")
        return float('inf')

class UniverseScreener:
    def __init__(self, exchanges: List[str], screener_cfg: dict, feeds_cfg: dict | None = None):
        self.exchanges = {}
        for ex in exchanges:
            try:
                obj = getattr(ccxt, ex)()
                obj.load_markets()
                self.exchanges[ex] = obj
            except Exception as e:
                logger.warning(f"Could not init {ex}: {e}")
        self.cfg = screener_cfg
        self.feeds = feeds_cfg or {}
        of_cfg = self.feeds.get('orderflow', {})
        self.orderflow_enabled = bool(of_cfg.get('enabled', False))
        self.orderflow_top = int(of_cfg.get('top_levels', 5))

    def get_active_universe(self) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, dict]]]:
        logger.info(f"[screener] SENTINEL enter get_active_universe spread_cap={self.cfg.get('max_spread_bps')} impact_cap={self.cfg.get('max_impact_bps')} order_usd={self.cfg.get('typical_order_usd')} volume_min={self.cfg.get('min_24h_volume_usd')}")
        if not self.exchanges: return [], {}
        tickers_by_venue: Dict[str, Dict[str, dict]] = {}
        for name, ex in self.exchanges.items():
            try:
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

                quote = m.get("quote")
                if quote not in ("USD","USDT","USDC","DAI","FDUSD","TUSD","PYUSD"): continue
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
                canonical = make_canonical(base, quote, kind="SPOT")
                inst = {
                    "venue": venue, "type": "spot", "market_id": m.get("symbol",""),
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
                imp = _impact_bps(ex, inst['market_id'], self.cfg['typical_order_usd'], inst.get('usd_per_quote', 1.0))
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
            inst['display'] = f"{inst['base'].upper()}_{inst['quote'].upper()}_SPOT"
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
