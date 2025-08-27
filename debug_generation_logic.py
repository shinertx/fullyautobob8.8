"""
Debug script to provide a transparent, step-by-step view of the alpha
generation, backtesting, and validation process for a small batch of formulas.
"""
import sys
import yaml
import pandas as pd
from pathlib import Path
from loguru import logger

# Ensure the script can find the v26meme package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v26meme.data.lakehouse import Lakehouse
from v26meme.research.feature_factory import FeatureFactory
from v26meme.research.generator import GeneticGenerator
from v26meme.labs.simlab import SimLab
from v26meme.research.validation import Validator

def load_config(file="configs/config.yaml"):
    with open(file, "r") as f:
        return yaml.safe_load(f)

def run_debug_generation():
    """
    Generates a small population of alphas and runs them through the
    backtesting and validation process with verbose logging.
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info("--- Starting Generation Logic Debug Script ---")

    # --- 1. Configuration & Setup ---
    cfg = load_config()
    lake = Lakehouse(preferred_exchange=cfg["execution"]["primary_exchange"])
    ff = FeatureFactory()
    sim = SimLab()
    validator = Validator(cfg.get('validation', {}))
    
    # --- 2. Data Loading ---
    symbol = "BTC_USD_SPOT"
    timeframe = "1h"
    logger.info(f"Loading data for {symbol} on timeframe {timeframe}...")
    
    df_raw = lake.get_data(symbol, timeframe)
    if df_raw.empty:
        logger.error(f"No data found for {symbol} {timeframe}. Exiting.")
        return
    
    df_raw = df_raw.tail(1000) # Use a recent 1000-bar subset
    logger.info(f"Loaded {len(df_raw)} bars.")

    # --- 3. Feature Engineering ---
    logger.info("Creating features...")
    df_features = ff.create(df_raw, feature_configs=cfg.get('features', {}), symbol=symbol)
    logger.info(f"Feature DataFrame prepared with {len(df_features)} rows.")

    # --- 4. Alpha Generation ---
    base_features = cfg['discovery'].get("base_features", [])
    generator = GeneticGenerator(base_features, population_size=20, seed=42) # Small population
    generator.initialize_population()
    population = generator.population
    logger.info(f"Generated a debug population of {len(population)} formulas.")

    # --- 5. Step-by-Step Evaluation & Validation ---
    evaluated_formulas = []
    for i, formula in enumerate(population):
        logger.info(f"--- Formula #{i+1}/{len(population)} ---")
        logger.info(f"Logic: {formula}")

        # Backtest
        sim_result = sim.run_backtest(df_features, formula, fee_pct=0.001, slippage_pct=0.0005)
        
        if sim_result is None or sim_result.empty:
            logger.warning("Backtest Result: 0 trades generated.")
            evaluated_formulas.append({'alpha_id': f'debug_{i}', 'formula': formula, 'trades': pd.DataFrame(), 'sharpe': -1.0})
            continue

        sharpe = sim_result['pnl'].mean() / sim_result['pnl'].std() if sim_result['pnl'].std() != 0 else 0
        logger.info(f"Backtest Result: {len(sim_result)} trades, Sharpe Ratio: {sharpe:.2f}")
        
        evaluated_formulas.append({
            'alpha_id': f'debug_{i}',
            'formula': formula,
            'trades': sim_result,
            'sharpe': sharpe,
        })

    logger.info("\n--- Starting Batch Validation ---")
    survivors, rejections = validator.validate_batch(evaluated_formulas)

    logger.info(f"\n--- Validation Complete ---")
    logger.success(f"Found {len(survivors)} survivors.")
    
    total_rejected = sum(len(fids) for fids in rejections.values())
    logger.warning(f"Rejected {total_rejected} formulas.")
    logger.info(f"Rejection Reasons: {rejections}")

if __name__ == "__main__":
    run_debug_generation()
