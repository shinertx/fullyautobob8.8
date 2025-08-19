import time, json, random, hashlib
import click, yaml
import pandas as pd
from loguru import logger
from pathlib import Path

from v26meme.core.state import StateManager
from v26meme.data.lakehouse import Lakehouse
from v26meme.research.feature_factory import FeatureFactory
from v26meme.research.generator import GeneticGenerator
from v26meme.labs.simlab import SimLab

def load_config(file="configs/config.yaml"):
    with open(file, "r") as f: return yaml.safe_load(f)

def _parse_tf_bars(tf: str, days: int) -> int:
    if tf.endswith('h'):
        return int(24//max(1,int(tf[:-1])))*days
    if tf.endswith('m'):
        m = int(tf[:-1]); per_day = 24*60//max(1,m); return per_day*days
    return 24*days

@click.group()
def cli(): pass

@cli.command()
def run():
    cfg = load_config()
    if not cfg.get('eil',{}).get('enabled', True):
        logger.info("EIL disabled; exiting."); return

    random.seed(cfg['system'].get('seed', 1337))
    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    lake = Lakehouse(preferred_exchange=cfg["execution"]["primary_exchange"])
    ff = FeatureFactory()
    sim = SimLab(cfg['execution']['paper_fees_bps'], cfg['execution']['paper_slippage_bps'],
                 slippage_table=(state.get("slippage:table") or {}))

    base_features = ['return_1p','volatility_20p','momentum_10p','rsi_14','close_vs_sma50',
                     'hod_sin','hod_cos','round_proximity','btc_corr_20p','eth_btc_ratio']
    gen = GeneticGenerator(base_features, population_size=cfg['discovery']['population_size'], seed=cfg['system'].get('seed', 1337))
    tf = "1h"
    nbars = _parse_tf_bars(tf, cfg['eil']['fast_window_days'])

    while True:
        try:
            avail = lake.get_available_symbols(tf)
            if not avail: time.sleep(10); continue
            bases = [s for s in avail if s.endswith("_USD_SPOT")]
            if not bases: time.sleep(10); continue
            panel = random.sample(bases, min(cfg['discovery']['panel_symbols'], len(bases)))

            df_cache = {}
            btc = lake.get_data("BTC_USD_SPOT", tf)
            eth = lake.get_data("ETH_USD_SPOT", tf)
            for canon in panel:
                df = lake.get_data(canon, tf).tail(nbars)
                if df.empty: continue
                df_feat = ff.create(df, symbol=canon, cfg=cfg, other_dfs={'BTC_USD_SPOT': btc, 'ETH_USD_SPOT': eth})
                df_cache[canon] = df_feat.dropna()

            if not gen.population: gen.initialize_population()

            survivors = []
            for f in gen.population:
                fid = hashlib.sha256(json.dumps(f).encode()).hexdigest()
                perfs = []
                for canon, dff in df_cache.items():
                    stats = sim.run_backtest(dff, f).get("all", {})
                    if stats and stats.get("n_trades",0)>0:
                        perfs.append(stats.get("avg_return", 0.0))
                if not perfs: continue
                score = sum(perfs)/max(1,len(perfs))
                survivors.append((score, fid, f))

            survivors.sort(key=lambda x: x[0], reverse=True)
            topk = survivors[: cfg['eil']['survivor_top_k']]
            for score, fid, form in topk:
                state.set(f"eil:candidates:{fid}", {"fid": fid, "formula": form, "score": score, "ts": int(time.time())})
            time.sleep(15)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.opt(exception=True).error(f"EIL loop error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    cli()
