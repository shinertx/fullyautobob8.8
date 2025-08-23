from typing import Dict, Any, List
from loguru import logger

class OrderflowSnap:
    """Collect shallow order book derived orderflow features.

    PIT Notes:
      - Uses only current top N levels (no lookahead/stateful accumulation).
      - Never mutates global state; deterministic per call.
    """
    def __init__(self, top_levels: int = 5):
        self.top = int(top_levels)

    def _features_from_ob(self, ob: Dict[str, Any]) -> Dict[str, float]:
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            return {"of_spread_bps": float("inf"), "of_imbalance": 0.0, "of_microprice_dev": 0.0}
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        if best_bid <= 0:  # avoid division by zero
            spread_bps = float("inf")
        else:
            spread_bps = ((best_ask - best_bid) / best_bid) * 10000.0
        tb = sum(float(b[1]) for b in bids[:self.top])
        ta = sum(float(a[1]) for a in asks[:self.top])
        depth = tb + ta
        imb = (tb - ta) / depth if depth > 0 else 0.0
        micro = (best_bid * ta + best_ask * tb) / (ta + tb) if (ta + tb) > 0 else (0.5 * (best_bid + best_ask))
        dev = 0.0
        try:
            mid = 0.5 * (bids[0][0] + asks[0][0])
            dev = (micro - mid) / mid if mid and mid > 0 else 0.0
        except Exception:
            dev = 0.0
        return {
            "of_spread_bps": float(spread_bps),
            "of_depth": float(depth),
            "of_imbalance": float(imb),
            "of_microprice_dev": float(dev)
        }

    def collect_for_instruments(self, exchange, instruments: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Fetch order books for venue symbols and derive features.

        Args:
            exchange: CCXT-like exchange object exposing fetch_order_book(symbol, limit=int).
            instruments: List of instrument dicts each containing at minimum:
                - display: canonical id (used as key in returned mapping)
                - market_id: canonical id (for logging only)
                - venue_symbol: venue native trading symbol (used for API call)
        Returns:
            Mapping canonical display -> feature dict.
        """
        out: Dict[str, Dict[str, float]] = {}
        for inst in instruments:
            venue_symbol_raw = inst.get("venue_symbol") or inst.get("symbol") or inst.get("market_id") or ""
            venue_symbol = str(venue_symbol_raw)
            display_raw = inst.get("display") or inst.get("market_id") or venue_symbol
            display = str(display_raw)
            canonical_raw = inst.get("market_id") or display
            canonical = str(canonical_raw)
            try:
                ob = exchange.fetch_order_book(venue_symbol, limit=self.top)
                out[display] = self._features_from_ob(ob)
            except Exception as e:  # noqa: broad-except
                logger.debug(
                    f"orderflow snapshot failed for canonical={canonical} venue_symbol={venue_symbol}: {e}"
                )
        return out
