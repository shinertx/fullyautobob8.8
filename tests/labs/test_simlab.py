import pandas as pd
import numpy as np
from v26meme.labs.simlab import SimLab
from v26meme.research.feature_factory import FeatureFactory

def test_simlab_moving_average_crossover():
    # Create a dataframe with a clear trend change
    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2025-01-01', periods=100, freq='D')),
        'open': np.concatenate([np.linspace(100, 150, 50), np.linspace(150, 100, 50)]),
        'high': np.concatenate([np.linspace(101, 151, 50), np.linspace(151, 101, 50)]),
        'low': np.concatenate([np.linspace(99, 149, 50), np.linspace(149, 99, 50)]),
        'close': np.concatenate([np.linspace(100, 150, 50), np.linspace(150, 100, 50)]),
        'volume': np.ones(100) * 1000
    }
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Create features
    ff = FeatureFactory()
    df_feat = ff.create(df)

    # A moving average crossover formula
    formula = ['close_vs_sma50', '>', 0.0]

    # Initialize the simulator
    simlab = SimLab(fees_bps=0, slippage_bps=0)

    # Run the backtest
    stats = simlab.run_backtest(df_feat, formula).get("all", {})

    # Assertions
    assert stats['n_trades'] > 0
    assert stats['sharpe'] != 0.0

def test_simlab_fees_and_slippage_impact():
    # Create a dataframe
    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2025-01-01', periods=100, freq='D')),
        'open': np.concatenate([np.linspace(100, 150, 50), np.linspace(150, 100, 50)]),
        'high': np.concatenate([np.linspace(101, 151, 50), np.linspace(151, 101, 50)]),
        'low': np.concatenate([np.linspace(99, 149, 50), np.linspace(149, 99, 50)]),
        'close': np.concatenate([np.linspace(100, 150, 50), np.linspace(150, 100, 50)]),
        'volume': np.ones(100) * 1000
    }
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Create features
    ff = FeatureFactory()
    df_feat = ff.create(df)

    # A moving average crossover formula
    formula = ['close_vs_sma50', '>', 0.0]

    # Initialize the simulator with and without costs
    simlab_with_costs = SimLab(fees_bps=10, slippage_bps=5)
    simlab_without_costs = SimLab(fees_bps=0, slippage_bps=0)

    # Run the backtests
    stats_with_costs = simlab_with_costs.run_backtest(df_feat, formula).get("all", {})
    stats_without_costs = simlab_without_costs.run_backtest(df_feat, formula).get("all", {})

    # Assertions
    assert stats_with_costs['sharpe'] < stats_without_costs['sharpe']
