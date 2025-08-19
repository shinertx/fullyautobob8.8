from typing import Optional, Dict
from v26meme.registry.venues import venue_symbol_map

# Per-process cache: exchange.id -> {canonical -> unified symbol}
_SYMBOL_CACHE: Dict[str, Dict[str, str]] = {}

def make_canonical(base: str, quote: str, kind: str = "SPOT") -> str:
    return f"{base.upper()}_{quote.upper()}_{kind.upper()}"

def parse_canonical(canonical: str):
    base, quote, kind = canonical.split("_")
    return base.upper(), quote.upper(), kind.upper()

def _build_map_for_exchange(exchange) -> Dict[str, str]:
    try:
        if not getattr(exchange, "markets", None):
            exchange.load_markets()
    except Exception:
        return {}
    mapping: Dict[str, str] = {}
    for m in (getattr(exchange, "markets", {}) or {}).values():
        if not m.get("spot", True):  # spot only for now
            continue
        base = (m.get("base") or "").upper()
        quote = (m.get("quote") or "").upper()
        if not base or not quote:
            continue
        canon = make_canonical(base, quote, "SPOT")
        sym = m.get("symbol") or f"{base}/{quote}"
        mapping.setdefault(canon, sym)  # first writer wins (stable)
    return mapping

def venue_symbol_for(exchange, canonical: str) -> Optional[str]:
    base, quote, kind = parse_canonical(canonical)
    if kind != "SPOT":
        return None
    exid = getattr(exchange, "id", "unknown")
    cmap = _SYMBOL_CACHE.get(exid)
    if cmap is None:
        cmap = _build_map_for_exchange(exchange)
        _SYMBOL_CACHE[exid] = cmap
    sym = cmap.get(canonical)
    if sym:
        return sym
    # Static fallback (legacy)
    static = (venue_symbol_map.get(exid, {}) or {}).get(canonical)
    if static:
        return static
    # Last chance direct symbol attempt
    direct = f"{base}/{quote}"
    try:
        if direct in (getattr(exchange, "markets", {}) or {}):
            return direct
    except Exception:
        pass
    return None
