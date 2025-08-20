from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence, Optional, Tuple
import time

DEFAULT_ALLOWED_QUOTES: Tuple[str, ...] = (
    "USD", "USDT", "USDC", "DAI", "FDUSD", "TUSD", "PYUSD", "EUR", "GBP"
)
DEFAULT_BASE_ALIASES: Dict[str, Sequence[str]] = {
    "BTC": ["XBT"],
    "IOTA": ["MIOTA"],
}

@dataclass
class ResolverPolicy:
    allowed_quotes_global: Tuple[str, ...] = DEFAULT_ALLOWED_QUOTES
    allowed_quotes_by_venue: Dict[str, Tuple[str, ...]] | None = None
    base_aliases: Dict[str, Sequence[str]] | None = None
    cache_ttl_s: int = 600

    def __post_init__(self):
        if self.base_aliases is None:
            self.base_aliases = DEFAULT_BASE_ALIASES
        if self.allowed_quotes_by_venue is None:
            self.allowed_quotes_by_venue = {}

class SymbolResolver:
    def __init__(self, policy: ResolverPolicy | None = None):
        self.policy = policy or ResolverPolicy()
        self._cache: Dict[Tuple[str, str, str, str], Tuple[float, Optional[str]]] = {}

    def configure(self, policy_dict: Dict[str, object] | None):
        if not policy_dict:
            return
        if not isinstance(policy_dict, dict):
            return
        allowed_g_raw = policy_dict.get("allowed_quotes_global", DEFAULT_ALLOWED_QUOTES)
        if not isinstance(allowed_g_raw, (list, tuple)):
            allowed_g_raw = DEFAULT_ALLOWED_QUOTES
        allowed_g = tuple(str(x).upper() for x in allowed_g_raw)
        allowed_by_cfg = policy_dict.get("allowed_quotes_by_venue") or {}
        allowed_by: Dict[str, Tuple[str, ...]] = {}
        if isinstance(allowed_by_cfg, dict):
            for k, v in allowed_by_cfg.items():
                if isinstance(v, (list, tuple)):
                    allowed_by[str(k)] = tuple(str(x).upper() for x in v)
        base_aliases_cfg = policy_dict.get("base_aliases") or {}
        base_aliases: Dict[str, Sequence[str]] = {}
        if isinstance(base_aliases_cfg, dict):
            for k, v in base_aliases_cfg.items():
                if isinstance(v, (list, tuple)):
                    base_aliases[str(k).upper()] = [str(a).upper() for a in v]
        if not base_aliases:
            base_aliases = DEFAULT_BASE_ALIASES
        ttl_raw = policy_dict.get("cache_ttl_s", 600)
        if isinstance(ttl_raw, (int, float)):
            ttl = int(ttl_raw)
        elif isinstance(ttl_raw, str) and ttl_raw.isdigit():
            ttl = int(ttl_raw)
        else:
            ttl = 600
        self.policy = ResolverPolicy(allowed_g, allowed_by, base_aliases, ttl)
        self._cache.clear()

    def add_alias(self, base: str, alias: str):
        base, alias = base.upper(), alias.upper()
        ba = self.policy.base_aliases or {}
        cur = list(ba.get(base, []))
        if alias not in cur:
            cur.append(alias)
            ba[base] = cur
            self.policy.base_aliases = ba
        self._cache.clear()

    def _allowed_quotes_for(self, venue_id: str) -> Tuple[str, ...]:
        aqbv = self.policy.allowed_quotes_by_venue or {}
        return aqbv.get(venue_id, self.policy.allowed_quotes_global)

    def _base_aliases(self, base: str) -> Tuple[str, ...]:
        ba = self.policy.base_aliases or {}
        base = base.upper()
        al = [a.upper() for a in ba.get(base, [])]
        return (base, *al)

    @staticmethod
    def _exists_markets(exchange, sym: str) -> bool:
        try:
            return sym in (getattr(exchange, "markets", {}) or {})
        except Exception:
            return False

    def resolve(self, exchange, base: str, quote: str, kind: str, canonical_key: Optional[str] = None) -> Optional[str]:
        if kind.upper() != "SPOT":
            return None
        venue_id = getattr(exchange, "id", "")
        key = (venue_id, base.upper(), quote.upper(), kind.upper())
        now = time.time()
        cached = self._cache.get(key)
        if cached and cached[0] > now:
            return cached[1]

        markets = (getattr(exchange, "markets", {}) or {})
        for b in self._base_aliases(base):
            sym = f"{b}/{quote.upper()}"
            if self._exists_markets(exchange, sym) and (markets[sym] or {}).get("spot", True):
                self._cache[key] = (now + self.policy.cache_ttl_s, sym)
                return sym

        from v26meme.registry.venues import venue_symbol_map
        if canonical_key:
            static = (venue_symbol_map.get(venue_id, {}) or {}).get(canonical_key)
            if static and self._exists_markets(exchange, static) and (markets[static] or {}).get("spot", True):
                self._cache[key] = (now + self.policy.cache_ttl_s, static)
                return static

        allowed = self._allowed_quotes_for(venue_id)
        base_names = set(self._base_aliases(base))
        for sym, meta in markets.items():
            if not meta or not meta.get("spot", True):
                continue
            if meta.get("base", "").upper() in base_names and meta.get("quote", "").upper() in allowed:
                self._cache[key] = (now + self.policy.cache_ttl_s, sym)
                return sym

        self._cache[key] = (now + self.policy.cache_ttl_s, None)
        return None

_GLOBAL_RESOLVER = SymbolResolver()

def get_resolver() -> SymbolResolver:
    return _GLOBAL_RESOLVER

def configure(policy_dict: Dict[str, object] | None):
    _GLOBAL_RESOLVER.configure(policy_dict)

def add_alias(base: str, alias: str):
    _GLOBAL_RESOLVER.add_alias(base, alias)
