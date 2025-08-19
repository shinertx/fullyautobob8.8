from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict

class Alpha(BaseModel):
    id: str
    name: str
    formula: List[Any]
    universe: List[str]                # canonical symbols (e.g., BTC_USD_SPOT)
    instrument: Optional[Dict[str, Any]] = None
    timeframe: str
    lane: str = "core"                 # core | moonshot
    performance: Dict[str, Dict[str, Any]] = {}
    model_config = ConfigDict(frozen=True)
