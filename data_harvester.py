# Thin shim to run the v4.7.5 harvester once (used for bootstrap)
from v26meme.data.harvester import run_once  # removed invalid 'run'
from v26meme.core.state import StateManager
from typing import Optional, Dict, Any
import sys, yaml

def harvest(cfg: Dict[str, Any], symbols_map: Optional[Dict[str, Dict[str, str]]] = None) -> None:
    """Bootstrap harvest wrapper expected by cli._ensure_lakehouse_bootstrap.

    Parameters
    ----------
    cfg : dict
        Loaded YAML configuration (configs/config.yaml)
    symbols_map : dict | None
        Mapping of canonical symbols → venue symbols (unused here; canonical resolution
        happens inside v26meme.data.harvester). Accepted for interface compatibility.

    PIT Note: Invokes run_once with provided cfg; no forward-looking data generated.
    """
    state = StateManager(cfg["system"]["redis_host"], cfg["system"]["redis_port"])
    run_once(cfg, state)

if __name__ == "__main__":
    # Allow: python data_harvester.py [cycles]
    cycles = 1
    if len(sys.argv) > 1:
        try: cycles = max(1, int(sys.argv[1]))
        except Exception: cycles = 1
    cfg = yaml.safe_load(open('configs/config.yaml'))
    for i in range(cycles):
        harvest(cfg)
