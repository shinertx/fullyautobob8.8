from typing import Optional
from datetime import datetime, timezone

class Checkpoints:
    """
    Store per (exchange, canonical, timeframe) checkpoint (last_ts_fetched) in Redis.
    """
    def __init__(self, state):
        self.state = state

    def _key(self, ex: str, canonical: str, tf: str) -> str:
        return f"harvest:checkpoint:{ex}:{tf}:{canonical}"

    def get(self, ex: str, canonical: str, tf: str) -> Optional[int]:
        v = self.state.get(self._key(ex, canonical, tf))
        return int(v) if v is not None else None

    def set(self, ex: str, canonical: str, tf: str, ts_ms: int):
        self.state.set(self._key(ex, canonical, tf), int(ts_ms))

    @staticmethod
    def epoch_ms(dt: datetime) -> int:
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
