from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class AssetID:
    source: str
    symbol: str
    chain_id: Optional[int] = None
    address: Optional[str] = None
    decimals: Optional[int] = None

@dataclass(frozen=True)
class InstrumentID:
    venue: str
    type: str  # "spot" | "swap" | "future"
    market_id: str
    base: AssetID
    quote: AssetID
    precision: Dict[str, Any]
    limits: Dict[str, Any]
    display: str
