# Minimal-safe placeholder, returns 0.0 unless Etherscan configured.
import os
from loguru import logger


class OnChainFlow:
    _warned = False

    def __init__(self, min_notional_eth: float = 100.0, enabled: bool | None = None):
        self.api_key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
        # auto-disable if key missing unless explicitly forced enabled
        if enabled is None:
            self.enabled = bool(self.api_key)
        else:
            self.enabled = bool(enabled and self.api_key)
        self.min_notional = float(min_notional_eth)
        if not self.api_key and not OnChainFlow._warned:
            logger.warning("OnChainFlow disabled (ETHERSCAN_API_KEY missing)")
            OnChainFlow._warned = True

    def whale_flow_rate(self, window_hours: int = 6) -> float:
        if not self.enabled:
            return 0.0
        # Placeholder until implemented; maintain interface
        return 0.0
