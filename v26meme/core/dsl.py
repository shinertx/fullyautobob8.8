from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict
import time

class Alpha(BaseModel):
    """Strategy / signal contract shared across discovery → validation → allocation → execution.

    PIT correctness: Pure structural container; holds only historical performance stats (no
    forward projections). Performance['all']['returns'] must be realized trade PnLs up to t-1.
    """
    id: str
    name: str
    formula: List[Any]                 # boolean formula tree (nested lists of conditions)
    universe: List[str]                # canonical symbols (e.g., BTC_USD_SPOT)
    timeframe: str
    lane: str = "moonshot"             # core | moonshot (default moonshot for new discoveries)
    performance: Dict[str, Dict[str, Any]] = {}
    created_ts: int = int(time.time())
    meta: Dict[str, Any] = {}          # robustness probes, factor loadings, etc.

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    # Convenience accessors (read-only style)
    def trades(self) -> int:
        return int(self.performance.get('all', {}).get('n_trades', 0))
    def sharpe(self) -> float:
        return float(self.performance.get('all', {}).get('sharpe', 0.0))
    def sortino(self) -> float:
        return float(self.performance.get('all', {}).get('sortino', 0.0))
    def win_rate(self) -> float:
        return float(self.performance.get('all', {}).get('win_rate', 0.0))
    def mdd(self) -> float:
        return float(self.performance.get('all', {}).get('mdd', 0.0))

__all__ = ["Alpha"]
