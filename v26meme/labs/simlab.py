"""
A simulation lab for backtesting trading strategies.
This is the core backtesting engine, designed to be fast and robust for research purposes.
"""
import logging
from typing import Dict, Union, List

import pandas as pd

logger = logging.getLogger(__name__)


class SimLab:
    """
    A simulation lab for backtesting trading strategies.
    """
    def __init__(self, *, fees_bps: int = 0, slippage_bps: int = 0, min_hold_bars: int = 0):
        # Store costs as decimal fractions
        self.fee_pct = float(fees_bps) / 10_000.0
        self.slip_pct = float(slippage_bps) / 10_000.0
        self.min_hold_bars = int(min_hold_bars)

    def run_backtest(self, panel: pd.DataFrame, formula: list) -> Dict[str, Dict[str, Union[int, float, List[float]]]]:
        """
        Run a backtest for a given formula on a panel, which can be single-item or multi-item.
        """
        if panel.empty:
            return {"all": {"n_trades": 0, "sharpe": 0.0, "sortino": 0.0, "mdd": 0.0, "returns": []}}

        # --- 1. Evaluate Formula to get Signals ---
        all_signals = []
        is_multi_item = 'item' in panel.index.names
        
        if is_multi_item:
            for item, group in panel.groupby(level='item'):
                if group.empty: continue
                try:
                    signals = self._evaluate_formula(group.copy(), formula)
                    signal_df = pd.DataFrame({'signals': signals}, index=group.index)
                    all_signals.append(signal_df)
                except Exception as e:
                    logger.debug(f"Could not evaluate formula for {item}: {e}")
                    continue
        else:
            try:
                signals = self._evaluate_formula(panel.copy(), formula)
                signal_df = pd.DataFrame({'signals': signals}, index=panel.index)
                all_signals.append(signal_df)
            except Exception as e:
                logger.debug(f"Could not evaluate formula for single item: {e}")

        if not all_signals:
            return {"all": {"n_trades": 0, "sharpe": 0.0, "sortino": 0.0, "mdd": 0.0, "returns": []}}

        # --- 2. Generate Trade Edges ---
        signals_df = pd.concat(all_signals).sort_index()
        panel = panel.join(signals_df, how='left')
        panel['signals'] = panel['signals'].fillna(False)

        if is_multi_item:
            panel['signals'] = panel['signals'].astype(int)
            panel["edges"] = panel.groupby(level="item")["signals"].diff().fillna(0)
        else:
            panel["edges"] = panel["signals"].astype(int).diff().fillna(0)

        # --- 3. Calculate PnL from Edges (then normalize to returns) ---
        trade_df = self._calculate_trade_returns(panel, self.fee_pct, self.slip_pct)
        if trade_df.empty:
            return {"all": {"n_trades": 0, "sharpe": 0.0, "sortino": 0.0, "mdd": 0.0, "returns": []}}
        # Normalize per-trade PnL by absolute execution price to produce unitless returns
        # This improves statistical power across mixed price scales and is PIT-safe.
        if 'exec_price' in trade_df.columns:
            denom = trade_df['exec_price'].abs().replace(0, pd.NA)
            r = (trade_df['pnl'] / denom).astype(float)
        else:
            # Fallback: scale by |close| if exec price unavailable (should not happen)
            denom = trade_df.get('close', pd.Series(index=trade_df.index, dtype=float)).abs().replace(0, pd.NA)
            r = (trade_df['pnl'] / denom).astype(float)
        r = r.dropna()
        r = r.dropna()
        n = int(len(r))
        mean = float(r.mean()) if n > 0 else 0.0
        std = float(r.std(ddof=1)) if n > 1 else 0.0
        neg = r[r < 0]
        downside = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
        sharpe = (mean / std) if std > 0 else 0.0
        sortino = (mean / downside) if downside > 0 else 0.0
        mdd = float(abs(r.cumsum().min())) if n > 0 else 0.0
        # Build per-symbol return lists if multi-item panel
        per_symbol: Dict[str, List[float]] = {}
        if 'item' in trade_df.index.names:
            try:
                # Recompute normalized returns per symbol to avoid double computation
                if 'exec_price' in trade_df.columns:
                    denom_sym = trade_df['exec_price'].abs().replace(0, pd.NA)
                    r_norm = (trade_df['pnl'] / denom_sym).astype(float)
                else:
                    denom_sym = trade_df.get('close', pd.Series(index=trade_df.index, dtype=float)).abs().replace(0, pd.NA)
                    r_norm = (trade_df['pnl'] / denom_sym).astype(float)
                r_norm = r_norm.dropna()
                for sym, grp in r_norm.groupby(level='item'):
                    vals = grp.dropna().astype(float).tolist()
                    if vals:
                        per_symbol[str(sym)] = [float(x) for x in vals]
            except Exception:
                per_symbol = {}

        return {
            "all": {"n_trades": n, "sharpe": sharpe, "sortino": sortino, "mdd": mdd, "returns": list(map(float, r.values))},
            "per_symbol": per_symbol
        }

    def _evaluate_formula(self, df: pd.DataFrame, formula: list) -> pd.Series:
        """
        Recursively evaluate a formula on a single-item DataFrame.
        Handles nested logic, feature-vs-feature, and feature-vs-literal comparisons.
        The formula now follows a more natural [LEFT, OPERATOR, RIGHT] structure.
        """
        if not isinstance(formula, list) or not formula:
            # An empty formula is considered a "pass" (no signal)
            return pd.Series(False, index=df.index)

        # Handle logical operators (recursive)
        if len(formula) == 3 and formula[1] in ["AND", "OR"]:
            op = formula[1]
            left = self._evaluate_formula(df, formula[0])
            right = self._evaluate_formula(df, formula[2])
            return left & right if op == "AND" else left | right

        # Handle comparison operators (base case)
        if len(formula) == 3:
            left_val, op, right_val = formula
            
            # Resolve left and right sides. They can be feature names (str) or literals (int/float).
            s1 = df[left_val] if isinstance(left_val, str) and left_val in df.columns else left_val
            s2 = df[right_val] if isinstance(right_val, str) and right_val in df.columns else right_val

            # If a feature is missing, return a Series of False
            if isinstance(s1, str) or isinstance(s2, str):
                logger.warning(f"Feature not found in evaluation. Left: '{left_val}', Right: '{right_val}'")
                return pd.Series(False, index=df.index)

            # Ensure we are comparing series with series or series with scalar
            if not isinstance(s1, pd.Series) and not isinstance(s2, pd.Series):
                 return pd.Series(False, index=df.index)

            # The result of a comparison is a boolean Series, which is what we want.
            if op == "<": return s1 < s2
            if op == ">": return s1 > s2
            if op == "<=": return s1 <= s2
            if op == ">=": return s1 >= s2
            if op == "==": return s1 == s2
            if op == "!=": return s1 != s2

        raise ValueError(f"Unsupported formula structure: {formula}")

    def _calculate_trade_returns(self, panel: pd.DataFrame, fee_pct: float, slippage_pct: float) -> pd.DataFrame:
        """
        Calculate returns from trade edges.
        A positive edge (+1.0) is a buy, a negative edge (-1.0) is a sell.
        This is a critical PIT-correct function. PnL is calculated based on the NEXT bar's open price.
        """
        trades = panel[panel["edges"] != 0].copy()
        if trades.empty:
            return pd.DataFrame()

        # Optional hysteresis: enforce minimum bars between consecutive edges per item
        if self.min_hold_bars > 0:
            try:
                if 'item' in panel.index.names:
                    bar_ord = panel.groupby(level='item').cumcount()
                else:
                    # Single-item panel
                    bar_ord = pd.Series(range(len(panel)), index=panel.index)
                edge_ord = bar_ord.loc[trades.index]
                if 'item' in trades.index.names:
                    gaps = edge_ord.groupby(level='item').diff().fillna(self.min_hold_bars)
                else:
                    gaps = edge_ord.diff().fillna(self.min_hold_bars)
                trades = trades[gaps >= self.min_hold_bars].copy()
                if trades.empty:
                    return pd.DataFrame()
            except Exception:
                # If any issue computing ordinals, fall back without hysteresis
                pass

        # Get the price for the NEXT bar to calculate PnL (avoids lookahead)
        is_multi_item = 'item' in panel.index.names
        if is_multi_item:
            trades['pnl_price'] = panel.groupby(level='item')['open'].shift(-1)
        else:
            trades['pnl_price'] = panel['open'].shift(-1)
        
        # Apply slippage and fees to the execution price of the CURRENT bar's close
        trades["exec_price"] = trades["close"] * (1 + trades["edges"] * slippage_pct)
        trades["trade_cost"] = trades["exec_price"] * fee_pct
        
        # PnL is the difference between the future price and our execution price
        # For a buy (edge=1), PnL = pnl_price - exec_price
        # For a sell (edge=-1), PnL = exec_price - pnl_price
        # This simplifies to: PnL = edge * (pnl_price - exec_price)
        trades["pnl"] = trades["edges"] * (trades["pnl_price"] - trades["exec_price"]) - trades["trade_cost"]
        
        # Drop the last trade for each item if it can't be closed (no next bar to calculate PnL)
        trades = trades.dropna(subset=['pnl'])

        # Keep exec_price for normalization upstream
        cols = [c for c in ["pnl", "exec_price", "close"] if c in trades.columns]
        return trades.loc[:, cols]


# Example usage (for testing or ad-hoc analysis)
def run_standalone_backtest(formula: list):
    """
    A simple harness to run a backtest with dummy data to verify the logic.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("--- Running Standalone Backtest ---")
    
    # 1. Create a dummy data panel with a clear, predictable signal
    try:
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(pd.date_range('2024-01-01', periods=10, freq='h')), ['BTC_USD_SPOT']],
            names=['timestamp', 'item']
        )
        panel = pd.DataFrame(index=index)
        # This data is crafted to make the example_formula trigger trades
        panel['open'] =  [100, 101, 102, 103, 104, 103, 102, 101, 100, 99]
        panel['close'] = [101, 102, 103, 104, 103, 102, 101, 100, 99, 98]
        panel['volume'] = 10
        logger.info(f"Created dummy panel with data:\n{panel}")

    except Exception as e:
        logger.error(f"Could not create dummy data: {e}")
        return

    # 2. Run backtest
    sim = SimLab()
    
    # Manually trace the logic to show the intermediate steps
    logger.info("\n--- Tracing Backtest Logic ---")
    signals = sim._evaluate_formula(panel, formula)
    logger.info(f"1. Signals (where is close > 102.5?):\n{signals.astype(int).values}")
    
    panel['signals'] = signals
    panel["edges"] = panel.groupby(level="item")["signals"].astype(int).diff().fillna(0)
    logger.info(f"2. Edges (where does signal change?):\n{panel['edges'].values}")
    
    trade_returns = sim._calculate_trade_returns(panel, 0.001, 0.0005)
    logger.info("--- End Trace ---")

    # 3. Report results
    if not trade_returns.empty:
        logger.info(f"\nBacktest complete. SUCCESS. Generated {len(trade_returns)} trades.")
        logger.info(f"Trade returns:\n{trade_returns}")
        pnl = trade_returns['pnl']
        sharpe = pnl.mean() / pnl.std() if pnl.std() != 0 else 0
        logger.info(f"Total PnL: {pnl.sum():.2f}")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    else:
        logger.info("\nBacktest complete. FAILURE. No trades were generated.")
    logger.info("--- Standalone Backtest Finished ---")


if __name__ == '__main__':
    # A simple formula that should clearly generate trades with the dummy data.
    # Signal is TRUE when close > 102.5
    example_formula = ["close", ">", 102.5]
    
    run_standalone_backtest(example_formula)
