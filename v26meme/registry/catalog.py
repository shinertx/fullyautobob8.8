from __future__ import annotations
from typing import Dict, Tuple, Any
import json, time
from loguru import logger
from v26meme.registry.canonical import make_canonical
from v26meme.registry.resolver import get_resolver

def _norm_base(base: str) -> str:
    pol = get_resolver().policy
    rev = {}
    for p, aliases in (pol.base_aliases or {}).items():
        for a in aliases:
            rev[a.upper()] = p.upper()
    return rev.get(base.upper(), base.upper())

def build_snapshot(exchange, allowed_quotes: Tuple[str, ...], include_derivatives: bool = False) -> Dict[str, Dict[str, Any]]:
    markets = (getattr(exchange, "markets", {}) or {})
    snap: Dict[str, Dict[str, Any]] = {}
    for sym, m in markets.items():
        if not m:
            continue
        is_spot = bool(m.get("spot", False))
        if not include_derivatives and not is_spot:
            continue
        base = (m.get("base") or "").upper()
        quote = (m.get("quote") or "").upper()
        if not base or not quote:
            continue
        if is_spot and allowed_quotes and quote not in allowed_quotes:
            continue
        canon = make_canonical(_norm_base(base), quote, "SPOT" if is_spot else "DERIV")
        snap[canon] = {
            "symbol": m.get("symbol") or sym,
            "base": base,
            "quote": quote,
            "spot": is_spot,
            "active": bool(m.get("active", True)),
        }
    return snap

class CatalogManager:
    def __init__(self, state, registry_cfg: Dict[str, Any] | None = None):
        self.state = state
        self.cfg = registry_cfg or {}
        self.refresh_s = int(self.cfg.get("catalog_refresh_seconds", 900))

    def _allowed_quotes(self, venue_id: str) -> Tuple[str, ...]:
        pol = get_resolver().policy
        aqbv = pol.allowed_quotes_by_venue or {}
        return aqbv.get(venue_id, pol.allowed_quotes_global)

    def _load_prev(self, venue_id: str) -> Dict[str, Dict[str, Any]]:
        raw = self.state.get(f"registry:catalog:{venue_id}") or {}
        return raw.get("items") or {}

    def _save_curr(self, venue_id: str, items: Dict[str, Dict[str, Any]]):
        payload = {"ts": int(time.time()), "items": items}
        self.state.set(f"registry:catalog:{venue_id}", payload)
        self.state.set("registry:catalog:last_refresh_ts", int(time.time()))

    def refresh(self, exchanges: Dict[str, Any], include_derivatives: bool = False):
        for venue_id, ex in (exchanges or {}).items():
            try:
                try:
                    ex.load_markets()
                except Exception:
                    pass
                allowed = self._allowed_quotes(venue_id)
                prev = self._load_prev(venue_id)
                curr = build_snapshot(ex, allowed, include_derivatives=include_derivatives)
                prev_keys = set(prev.keys())
                curr_keys = set(curr.keys())
                added = sorted(k for k in (curr_keys - prev_keys) if curr[k].get("spot", True))
                removed = sorted(k for k in (prev_keys - curr_keys) if (prev.get(k) or {}).get("spot", True))
                if added or removed:
                    logger.info(f"[catalog] {venue_id}: +{len(added)} / -{len(removed)} (spot)")
                for c in added:
                    self.state.r.rpush("eil:harvest:requests", json.dumps({"canonical": c}))
                self._save_curr(venue_id, curr)
            except Exception as e:
                logger.opt(exception=True).error(f"Catalog refresh failed for {venue_id}: {e}")

    def maybe_refresh(self, exchanges: Dict[str, Any], include_derivatives: bool = False):
        last = self.state.get("registry:catalog:last_refresh_ts") or 0
        if (int(time.time()) - int(last)) >= self.refresh_s:
            self.refresh(exchanges, include_derivatives=include_derivatives)
