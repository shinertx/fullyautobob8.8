from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field
import time
import hashlib
import json

class Alpha(BaseModel):
    """Strategy / signal contract shared across discovery → validation → allocation → execution.

    PIT correctness: Pure structural container; holds only historical performance stats (no
    forward projections). Performance['all']['returns'] must be realized trade PnLs up to t-1.
    """
    id: str
    name: str
    formula_raw: str                   # Raw boolean formula string from EIL
    formula: Optional[List[Any]] = None# Compiled boolean formula tree (nested lists)
    universe: List[str]                # canonical symbols (e.g., BTC_USD_SPOT)
    timeframe: str
    lane: str = "moonshot"             # core | moonshot (default moonshot for new discoveries)
    performance: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    created_ts: int = Field(default_factory=lambda: int(time.time()))
    meta: Dict[str, Any] = Field(default_factory=dict)

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

def normalize_survivor_to_alpha(survivor: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a flat EIL survivor dict into a valid Alpha model dict.

    This acts as a stable contract between the research lab and the promotion
    system, creating the nested performance structure and a deterministic ID.
    """
    formula_obj = survivor.get('formula')
    formula_raw = survivor.get('formula_raw')

    # If raw is missing, create from obj. If obj is missing, it's an error state upstream.
    if not formula_raw and formula_obj:
        formula_raw = json.dumps(formula_obj, separators=(',', ':'))
    
    # Use the raw string for ID generation to ensure determinism
    id_str_base = formula_raw if formula_raw else ''

    universe_str = ''.join(sorted(survivor.get('universe', [])))
    timeframe = survivor.get('timeframe', '')
    id_str = f"{id_str_base}:{universe_str}:{timeframe}"
    alpha_id = hashlib.sha256(id_str.encode()).hexdigest()

    fid = survivor.get('fid', 'unknown_fid')

    # Handle both 'trades' and 'n_trades' for backward compatibility
    n_trades = survivor.get('n_trades', survivor.get('trades', 0))

    # Handle nested performance dict if present, otherwise use survivor dict directly
    perf = survivor.get('performance', {}).get('all', survivor)

    return {
        'id': alpha_id,
        'name': f"eil_{fid[:12]}",
        'formula_raw': formula_raw,
        'formula': formula_obj, # Keep the compiled structure
        'universe': survivor.get('universe', []),
        'timeframe': timeframe,
        'lane': survivor.get('lane', 'moonshot'),
        'created_ts': int(time.time()),
        'performance': {
            'all': {
                'sharpe': perf.get('sharpe', 0.0),
                'sortino': perf.get('sortino', 0.0),
                'mdd': perf.get('mdd', 1.0),
                'win_rate': perf.get('win_rate', 0.0),
                'n_trades': n_trades,
                'returns': perf.get('returns', []),
            }
        },
        'meta': {
            'eil_fid': fid,
            'p_value': perf.get('p_value', 1.0),
            'dsr_prob': perf.get('dsr_prob', 0.0),
            'robust_score': perf.get('robust_score', 0.0),
        }
    }

__all__ = ["Alpha"]
