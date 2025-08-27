import json
import time
from typing import Any, Dict, List, Tuple, Optional, cast, Union

import redis
from loguru import logger


class StateManager:
    """Redis-backed state manager.

    PIT NOTE: Pure persistence layer; contains no time-dependent branching that could
    induce non-determinism beyond wall-clock timestamps explicitly stored (e.g. equity curve).
    """

    def __init__(self, host: str = 'localhost', port: int = 6379) -> None:
        self.r: redis.Redis = redis.Redis(host=host, port=port, decode_responses=True)  # sync client
        try:
            self.r.ping()
        except Exception as e:  # broad: redis lib raises various connection exceptions
            logger.error(f"Redis connection failed: {e}")
            raise

    # --------------------- internal helpers ---------------------
    @staticmethod
    def _json_load(raw: Optional[str]) -> Any:
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return raw

    # --------------------- simple KV ----------------------------
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self.r.set(key, json.dumps(value), ex=ttl)

    def get(self, key: str) -> Any:
        raw = cast(Optional[str], self.r.get(key))
        return self._json_load(raw)

    # --------------------- hash ops -----------------------------
    def hset(self, key: str, field: str, value: Any) -> None:
        self.r.hset(key, field, json.dumps(value))

    def hget(self, key: str, field: str) -> Any:
        raw = cast(Optional[str], self.r.hget(key, field))
        return self._json_load(raw)

    def hkeys_sync(self, key: str) -> List[str]:
        """Synchronously get all keys in a hash, ensuring string output."""
        # The client is sync, but type hints can be ambiguous. This method provides a
        # clear, synchronous interface, ignoring the incorrect Awaitable hint.
        keys_raw = self.r.hkeys(key)
        # The type checker incorrectly infers an Awaitable. We cast it.
        iterable_keys = cast(List[Union[str, bytes]], keys_raw)
        # decode_responses=True should handle this, but as a safeguard:
        return [k.decode('utf-8') if isinstance(k, bytes) else str(k) for k in (iterable_keys or [])]

    # --------------------- batch ops ----------------------------
    def multi_set(self, mapping: Dict[str, Any]) -> None:
        """Pipeline multiple set operations for performance.
        Deterministic; order of keys does not affect semantics.
        """
        pipe = self.r.pipeline()
        for k, v in mapping.items():
            pipe.set(k, json.dumps(v))
        pipe.execute()

    # --------------------- portfolio ----------------------------
    def get_portfolio(self) -> Dict[str, Any]:
        val = self.get('portfolio')
        if not isinstance(val, dict):  # initialize if absent or corrupted
            return {'cash': 200.0, 'equity': 200.0, 'positions': {}}
        return val

    def set_portfolio(self, portfolio: Dict[str, Any]) -> None:
        self.set('portfolio', portfolio)

    # --------------------- equity curve -------------------------
    def log_historical_equity(self, equity: float) -> None:
        ts = int(time.time())
        self.r.zadd('equity_curve', {json.dumps({'ts': ts, 'equity': equity}): ts})

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        raw = self.r.zrange('equity_curve', 0, -1)  # expected list[str] with decode_responses
        items: List[str] = []
        if isinstance(raw, list):  # defensive for type checker
            items = [s for s in raw if isinstance(s, str)]
        out: List[Dict[str, Any]] = []
        for v in items:
            parsed = self._json_load(v)
            if isinstance(parsed, dict):
                out.append(parsed)
        return out

    # --------------------- alpha registry -----------------------
    def get_active_alphas(self) -> List[Dict[str, Any]]:
        val = self.get('active_alphas')
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
        return []

    def set_active_alphas(self, alphas: List[Dict[str, Any]]) -> None:
        self.set('active_alphas', alphas)

    # --------------------- gene stats ---------------------------
    def gene_incr(self, gene: str, fitness: float) -> None:
        # usage count (#promotions leveraging gene)
        self.r.zincrby('gene_usage', 1, gene)
        # accumulate raw fitness total to later average
        self.r.hincrbyfloat('gene_fitness', gene, float(fitness))

    def gene_top(self, min_count: int = 10, top_n: int = 20) -> List[Tuple[str, float]]:
        genes_raw = self.r.hgetall('gene_fitness') or {}
        genes: Dict[str, str] = {}
        if isinstance(genes_raw, dict):  # runtime guard for type checker
            tmp: Dict[str, str] = {}
            for k, v in genes_raw.items():
                if isinstance(k, str) and isinstance(v, (str, int, float)):
                    tmp[k] = str(v)
            genes = tmp
        out: List[Tuple[str, float]] = []
        for g, total_fit_raw in genes.items():
            try:
                total_fit = float(total_fit_raw)
            except (TypeError, ValueError):
                continue
            cnt_raw = self.r.zscore('gene_usage', g)
            cnt_f: float = 0.0
            if isinstance(cnt_raw, (int, float)):
                cnt_f = float(cnt_raw)
            if cnt_f >= float(min_count) and cnt_f > 0:
                out.append((g, total_fit / cnt_f))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_n]

    def set_ensemble_definition(self, definition: Dict[str, Any]) -> None:
        self.set('ensemble_definition', definition)

    def get_ensemble_definition(self) -> Optional[Dict[str, Any]]:
        val = self.get('ensemble_definition')
        if isinstance(val, dict):
            return val
        return None

    # --------------------- heartbeat ----------------------------
    def heartbeat(self) -> None:
        self.r.set('heartbeat_ts', int(time.time()))
