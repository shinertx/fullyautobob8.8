from pathlib import Path
from datetime import datetime, timedelta, timezone

def prune_retention(base_dir: Path, retention_days: dict):
    """
    Remove old month partitions beyond retention per timeframe (1m/5m usually short).
    """
    now = datetime.now(timezone.utc)
    for tf, days in (retention_days or {}).items():
        cutoff = now - timedelta(days=int(days))
        # Example layout: data/{exchange}/{tf}/{YYYY}/{MM}/{canonical}.parquet
        for ex_dir in (base_dir).glob("*"):
            tf_dir = ex_dir / tf
            if not tf_dir.exists(): continue
            for ydir in tf_dir.glob("*"):
                for mdir in ydir.glob("*"):
                    year = int(ydir.name); month = int(mdir.name)
                    dt = datetime(year=year, month=month, day=1, tzinfo=timezone.utc)
                    if dt < cutoff:
                        for f in mdir.glob("*.parquet"): f.unlink(missing_ok=True)
                        for f in mdir.glob("*.quality.json"): f.unlink(missing_ok=True)
                        try: mdir.rmdir()
                        except Exception: pass
                try:
                    if not any(ydir.iterdir()): ydir.rmdir()
                except Exception: pass
