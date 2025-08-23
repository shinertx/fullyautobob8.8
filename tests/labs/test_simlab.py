import pandas as pd
import numpy as np
from v26meme.labs.simlab import SimLab
from v26meme.research.feature_factory import FeatureFactory

def test_simlab_moving_average_crossover():
    # Create a dataframe where SMA crossover is clearly profitable
    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2025-01-01', periods=500, freq='D')),
        'open': np.concatenate([np.linspace(100, 120, 100), np.linspace(120, 150, 100), np.linspace(150, 130, 100), np.linspace(130, 180, 100), np.linspace(180, 160, 100)]),
        'high': np.concatenate([np.linspace(101, 121, 100), np.linspace(121, 151, 100), np.linspace(151, 131, 100), np.linspace(131, 181, 100), np.linspace(181, 161, 100)]),
        'low': np.concatenate([np.linspace(99, 119, 100), np.linspace(119, 149, 100), np.linspace(149, 129, 100), np.linspace(129, 179, 100), np.linspace(179, 159, 100)]),
        'close': np.concatenate([np.linspace(100, 120, 100), np.linspace(120, 150, 100), np.linspace(150, 130, 100), np.linspace(130, 180, 100), np.linspace(180, 160, 100)]),
        'volume': np.ones(500) * 1000
    }
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Create features
    ff = FeatureFactory()
    df_feat = ff.create(df)

    # A moving average crossover formula (using SMA20)
    formula = ['close_vs_sma20', '>', 0.0]

    # Initialize the simulator
    simlab = SimLab(fees_bps=0, slippage_bps=0)

    # Run the backtest
    stats = simlab.run_backtest(df_feat, formula).get("all", {})

    # Assertions
    assert stats['n_trades'] > 0
    assert stats['sharpe'] > 0.5 # Expect a positive Sharpe for a profitable strategy

def test_simlab_fees_and_slippage_impact():
    # Create a dataframe where SMA crossover is clearly profitable
    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2025-01-01', periods=500, freq='D')),
        'open': np.concatenate([np.linspace(100, 120, 100), np.linspace(120, 150, 100), np.linspace(150, 130, 100), np.linspace(130, 180, 100), np.linspace(180, 160, 100)]),
        'high': np.concatenate([np.linspace(101, 121, 100), np.linspace(121, 151, 100), np.linspace(151, 131, 100), np.linspace(131, 181, 100), np.linspace(181, 161, 100)]),
        'low': np.concatenate([np.linspace(99, 119, 100), np.linspace(119, 149, 100), np.linspace(149, 129, 100), np.linspace(129, 179, 100), np.linspace(179, 159, 100)]),
        'close': np.concatenate([np.linspace(100, 120, 100), np.linspace(120, 150, 100), np.linspace(150, 130, 100), np.linspace(130, 180, 100), np.linspace(180, 160, 100)]),
        'volume': np.ones(500) * 1000
    }
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Create features
    ff = FeatureFactory()
    df_feat = ff.create(df)

    # A moving average crossover formula (using SMA20)
    formula = ['close_vs_sma20', '>', 0.0]

    # Initialize the simulator with and without costs
    simlab_with_costs = SimLab(fees_bps=10, slippage_bps=5)
    simlab_without_costs = SimLab(fees_bps=0, slippage_bps=0)

    # Run the backtests
    stats_with_costs = simlab_with_costs.run_backtest(df_feat, formula).get("all", {})
    stats_without_costs = simlab_without_costs.run_backtest(df_feat, formula).get("all", {})

    # Assertions
    assert stats_with_costs['sharpe'] < stats_without_costs['sharpe']
