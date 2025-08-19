from loguru import logger
from typing import Dict, Any, List
import random, numpy as np

class MicroLive:
    def __init__(self, exchange, notional_usd: float, cycle_budget_trades: int, enabled: bool = False):
        self.ex = exchange
        self.enabled = bool(enabled)
        self.notional = float(notional_usd)
        self.budget = int(cycle_budget_trades)

    def _simulate_fill(self, market_id: str, side: str) -> Dict[str, float]:
        try: ob = self.ex.fetch_order_book(market_id, limit=10)
        except Exception as e:
            logger.debug(f"OB fetch failed for {market_id}: {e}"); return {"ok": 0}
        bids, asks = ob.get('bids') or [], ob.get('asks') or []
        if not bids or not asks: return {"ok": 0}
        best_bid, best_ask = bids[0][0], asks[0][0]; mid = 0.5*(best_bid+best_ask)
        price = best_ask if side == 'buy' else best_bid
        qty = self.notional / max(1e-12, price)
        book = asks if side=='buy' else bids
        filled = 0.0; cost = 0.0
        for p, q in book:
            take = min(q, qty - filled); if_take = take>0
            if not if_take: break
            cost += take * p; filled += take
            if filled >= qty: break
        if filled < qty: return {"ok": 0}
        vwap = cost / filled
        slip_bps = (vwap - mid)/mid*10000.0 if side=='buy' else (mid - vwap)/mid*10000.0
        return {"ok": 1, "vwap": float(vwap), "mid": float(mid), "slip_bps": float(abs(slip_bps))}

    def sample(self, instruments: List[Dict[str, Any]], record_cb):
        if not self.enabled: return
        picks = random.sample(instruments, min(self.budget, len(instruments)))
        for inst in picks:
            side = random.choice(['buy','sell'])
            res = self._simulate_fill(inst['market_id'], side)
            res.update({"side": side, "market_id": inst['market_id'], "display": inst['display']})
            record_cb(res)
