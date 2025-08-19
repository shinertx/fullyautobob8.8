from typing import Dict, Set

def compute_top_gainers_bases(tickers_by_venue: Dict[str, Dict[str, dict]], top_n: int = 10) -> Set[str]:
    """Take CCXT tickers; return set of BASE symbols for top percentage gainers across venues."""
    items = []
    for _venue, ticks in (tickers_by_venue or {}).items():
        for sym, t in (ticks or {}).items():
            if "/" not in sym: continue
            pct = t.get("percentage")
            if pct is None: continue
            try: pct = float(pct)
            except Exception: continue
            base = sym.split("/")[0].upper()
            items.append((base, pct))
    items.sort(key=lambda x: x[1], reverse=True)
    return set([b for b,_ in items[:top_n]])
