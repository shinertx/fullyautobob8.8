from loguru import logger
import time
import pandas as pd
import numpy as np

class RiskManager:
    def __init__(self, state, cfg, lakehouse=None):
        self.state = state
        self.cfg = cfg['risk']
        self.exec_cfg = cfg.get('execution', {})
        self.adapt_cfg = cfg.get('adaptive', {}) or {}
        self.lakehouse = lakehouse
        self._ensure_day_anchor()

    def _ensure_day_anchor(self):
        today = time.strftime("%Y-%m-%d")
        if self.state.get("risk:day_anchor") != today:
            self.state.set("risk:day_anchor", today)
            self.state.set("risk:daily_drawdown", 0.0)
            self.state.set("risk:daily_start_equity", self.state.get_portfolio()['equity'])

    def note_error(self):
        self.state.r.incr("risk:consecutive_errors")

    def reset_errors(self):
        self.state.set("risk:consecutive_errors", 0)

    def _daily_stop_dynamic(self, equity: float) -> float:
        # This is a placeholder for a more sophisticated dynamic stop logic
        return self.cfg.get('daily_stop_pct', 0.10)

    def get_dynamic_position_size(self, symbol: str, price: float, equity: float) -> float:
        """
        Calculates position size based on volatility targeting.
        Aims for each trade to have a similar risk contribution to the portfolio.
        """
        vol_lookback = self.cfg.get('volatility_lookback_window', 20)
        target_risk = self.cfg.get('target_risk_per_trade', 0.01) # 1% of equity per trade

        # Fetch historical data to calculate volatility
        # This requires the lakehouse to be available
        if self.lakehouse is None:
            logger.warning("Lakehouse not available to RiskManager, cannot calculate dynamic size.")
            return 0.0
        
        # Assuming we use '1h' timeframe for volatility calculation, this should be configurable
        tf = self.cfg.get('volatility_timeframe', '1h')
        try:
            # Using a helper from cli.py logic to get data
            from v26meme.cli import _lh_get_data
            data = _lh_get_data(self.lakehouse, {'execution': self.exec_cfg}, symbol, tf)
            if data.empty or len(data) < vol_lookback:
                logger.warning(f"Not enough data for {symbol} on {tf} to calculate volatility.")
                return 0.0
        except Exception as e:
            logger.error(f"Could not fetch data for volatility calculation: {e}")
            return 0.0

        # Calculate annualized volatility
        log_returns = pd.Series(np.log(data['close'] / data['close'].shift(1)))
        daily_vol = log_returns.rolling(window=vol_lookback).std().iloc[-1]
        
        if pd.isna(daily_vol) or daily_vol == 0:
            logger.warning(f"Could not compute valid volatility for {symbol}.")
            return 0.0

        # Position size calculation
        risk_per_share = price * daily_vol
        target_risk_usd = equity * target_risk
        
        if risk_per_share == 0:
            return 0.0
            
        position_size_asset = target_risk_usd / risk_per_share
        
        # Get max order size from existing logic
        max_order_size_usd = self._get_current_max_order(equity)
        max_position_size_asset = max_order_size_usd / price
        
        return min(position_size_asset, max_position_size_asset)

    def _get_current_max_order(self, equity: float) -> float:
        """Gets the max order notional based on equity phases."""
        current_max_order = self.cfg.get('max_order_notional_usd', 150)
        phases = self.cfg.get('phases', {}) or {}
        for thresh in sorted([float(k) for k in phases.keys()]):
            if equity >= float(thresh):
                current_max_order = phases[str(int(thresh))].get('max_order_notional_usd', current_max_order)
        return current_max_order

    def publish_risk_metrics(self, equity: float):
        """Calculates and persists key risk metrics for observability."""
        max_order_usd = self._get_current_max_order(equity)
        self.state.set("risk:current_max_order_usd", max_order_usd)
        logger.trace(f"Published risk metric: current_max_order_usd={max_order_usd}")

    def enforce(self, symbol_weights: dict, equity: float) -> dict:
        """Enforces global risk controls like max symbol weight and gross exposure."""
        if self.state.get('risk:halted'):
            self.state.set('risk:halt_reason', 'enforce_call_while_halted')
            return {k: 0.0 for k in symbol_weights}

        # Max symbol weight
        max_w = self.cfg.get('max_symbol_weight', 1.0)
        capped = {k: min(v, max_w) for k, v in symbol_weights.items()}

        # Max gross weight
        max_gross = self.cfg.get('max_gross_weight', 1.0)
        tot = sum(capped.values())
        if tot > max_gross:
            scale = max_gross / tot
            capped = {k: v * scale for k, v in capped.items()}

        # Conserve mode (if in significant drawdown)
        try:
            start_equity = float(self.state.get('risk:daily_start_equity') or equity)
            dd_trigger = self.cfg.get('conserve_mode', {}).get('dd_trigger_pct', 0.15)
            if equity < start_equity * (1.0 - dd_trigger):
                logger.warning("CONSERVE MODE ACTIVATED due to drawdown.")
                scalar = self.cfg.get('conserve_mode', {}).get('gross_weight_scalar', 0.5)
                capped = {k: v * scalar for k, v in capped.items()}
        except (ValueError, TypeError):
            pass # if start equity not set, skip

        # Final normalization
        tot = sum(capped.values())
        return {k: v / tot for k, v in capped.items()} if tot > 0 else {}
