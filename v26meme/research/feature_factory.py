import pandas as pd
import numpy as np
from arch import arch_model
from loguru import logger
import math
from typing import Optional, List, Dict

class FeatureFactory:
    """
    Creates a variety of PIT-correct features for the alpha discovery engine.
    
    The factory is designed to be extensible and configuration-driven. New feature 
    categories can be added by creating a new `_add_*_features` method and calling 
    it from the `create` method, driven by the `feature_configs` dictionary.
    
    All features are calculated on a per-item basis to prevent lookahead bias
    across different assets.
    """

    def create(self, panel: pd.DataFrame, feature_configs: Optional[Dict] = None, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Main method to generate all features based on the provided configuration.
        
        Args:
            panel (pd.DataFrame): DataFrame, which can be single-indexed or 
                                  multi-indexed with 'item' and 'timestamp'.
                                  Must contain 'open', 'high', 'low', 'close', 'volume'.
            feature_configs (dict): Configuration for feature parameters (e.g., window sizes).
            symbol (str, optional): If provided, treats the panel as a single-item dataframe 
                                    and assigns this symbol as the 'item' index.

        Returns:
            pd.DataFrame: The input panel with added feature columns and a multi-index.
        """
        if panel.empty:
            return pd.DataFrame()

        logger.debug("Starting feature creation...")
        
        if feature_configs is None:
            feature_configs = {}

        # If a symbol is provided, we are dealing with a single-item dataframe.
        # We need to set a multi-index to make it compatible with the rest of the pipeline.
        if symbol is not None:
            if 'item' not in panel.columns:
                panel['item'] = symbol
            if not isinstance(panel.index, pd.MultiIndex):
                if 'timestamp' not in panel.columns and isinstance(panel.index, pd.DatetimeIndex):
                    panel.reset_index(inplace=True)
                panel = panel.set_index(['item', 'timestamp'])

        # Ensure data is sorted for windowed operations
        panel = panel.sort_index()

        # Group by item to apply feature calculations independently
        features_list = []
        # Check if the index is a MultiIndex and has the 'item' level
        if isinstance(panel.index, pd.MultiIndex) and 'item' in panel.index.names:
            for item, group in panel.groupby(level='item'):
                if group.empty:
                    continue
                
                group = group.copy()
                # Clean data before feature calculation
                group = self._clean(group)
                
                # Add feature sets based on config
                group = self._add_base_features(group)
                if 'zscore_lookback' in feature_configs:
                    group = self._add_zscore_features(group, feature_configs['zscore_lookback'])
                if 'volatility_windows' in feature_configs:
                    group = self._add_volatility_features(group, feature_configs['volatility_windows'])
                if 'parkinson_vol_windows' in feature_configs:
                    group = self._add_parkinson_vol_features(group, feature_configs['parkinson_vol_windows'])
                if 'momentum' in feature_configs:
                    group = self._add_momentum_features(group, feature_configs['momentum'])
                
                group = self._add_time_features(group)
                
                features_list.append(group)
        else:
            # Handle single-indexed dataframe (for debug script)
            group = panel.copy()
            group = self._clean(group)
            group = self._add_base_features(group)
            if 'zscore_lookback' in feature_configs:
                group = self._add_zscore_features(group, feature_configs['zscore_lookback'])
            if 'volatility_windows' in feature_configs:
                group = self._add_volatility_features(group, feature_configs['volatility_windows'])
            if 'parkinson_vol_windows' in feature_configs:
                group = self._add_parkinson_vol_features(group, feature_configs['parkinson_vol_windows'])
            if 'momentum' in feature_configs:
                group = self._add_momentum_features(group, feature_configs['momentum'])
            group = self._add_time_features(group)
            features_list.append(group)

        if not features_list:
            return pd.DataFrame()

        # Combine features and handle potential NaNs
        panel = pd.concat(features_list).sort_index()
        
        # Lag all features to ensure they are available at the time of decision making.
        feature_cols = [col for col in panel.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # The groupby is only necessary for multi-item panels
        if isinstance(panel.index, pd.MultiIndex) and 'item' in panel.index.names:
            panel[feature_cols] = panel.groupby(level='item')[feature_cols].shift(1)
        else:
            panel[feature_cols] = panel[feature_cols].shift(1)
        
        logger.debug(f"Feature creation complete. Panel shape: {panel.shape}")
        return panel

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the input dataframe by handling non-numeric and infinite values."""
        # This method is called on a copy, so we can modify it.
        cols = [c for c in ['open','high','low','close','volume'] if c in df.columns]
        
        # Coerce to numeric, creating NaNs for non-numeric values.
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Replace infinities with NaNs as well.
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop any rows that have NaN in the essential columns.
        df.dropna(subset=cols, inplace=True)
        
        return df

    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds the most basic return feature."""
        df['return_1p'] = df['close'].pct_change()
        return df

    def _add_zscore_features(self, df: pd.DataFrame, lookback: Optional[int]) -> pd.DataFrame:
        """Adds a rolling z-score of the close price."""
        if lookback and lookback > 0:
            logger.debug(f"Adding z-score feature with lookback {lookback}")
            mean = df['close'].rolling(window=lookback).mean()
            std = df['close'].rolling(window=lookback).std()
            # Avoid division by zero
            std.replace(0, np.nan, inplace=True)
            df[f'zscore_{lookback}'] = (df['close'] - mean) / std
        return df

    def _add_volatility_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Adds volatility-related features."""
        if not windows: return df
        logger.debug(f"Adding volatility features with windows {windows}")
        if 'return_1p' not in df.columns:
            df = self._add_base_features(df)  # Dependency
        
        for window in windows:
            if window > 0:
                df[f'volatility_{window}'] = df['return_1p'].rolling(window=window).std() * np.sqrt(window)
        return df

    def _add_parkinson_vol_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Adds Parkinson volatility, an estimator of volatility using high and low prices."""
        if not windows: return df
        logger.debug(f"Adding Parkinson volatility features with windows {windows}")
        for window in windows:
            if window > 0:
                log_hl = np.log(df['high'] / df['low'])
                # The formula for Parkinson volatility
                parkinson_vol = (1 / (4 * np.log(2))) * (log_hl**2)
                df[f'parkinson_vol_{window}'] = parkinson_vol.rolling(window=window).mean().apply(np.sqrt)
        return df

    def _add_momentum_features(self, df: pd.DataFrame, momentum_cfg: Dict) -> pd.DataFrame:
        """Adds momentum and trend-following features."""
        if not momentum_cfg: return df
        logger.debug(f"Adding momentum features with config {momentum_cfg}")
        for window in momentum_cfg.get('sma_windows', []):
            if window > 0:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        
        for window in momentum_cfg.get('ema_windows', []):
            if window > 0:
                df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()

        for window in momentum_cfg.get('roc_windows', []):
            if window > 0:
                df[f'roc_{window}'] = df['close'].pct_change(periods=window)
        
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds time-based cyclical features."""
        ts_index = df.index.get_level_values('timestamp')
        if isinstance(ts_index, pd.DatetimeIndex):
            df['hod_sin'] = np.sin(2 * np.pi * ts_index.hour / 24.0)
            df['hod_cos'] = np.cos(2 * np.pi * ts_index.hour / 24.0)
        
        logp = np.log10(df['close'].clip(lower=1e-9))
        frac = logp - np.floor(logp)
        prox = np.minimum(np.abs(frac), np.abs(1 - frac))
        df['round_proximity'] = -prox
        
        return df
