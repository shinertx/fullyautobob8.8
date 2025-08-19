# Minimal-safe placeholder, returns 0.0 unless Etherscan configured.
import os
class OnChainFlow:
    def __init__(self, min_notional_eth: float = 100.0):
        self.api_key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
        self.min_notional = float(min_notional_eth)
    def whale_flow_rate(self, window_hours: int = 6) -> float:
        return 0.0
