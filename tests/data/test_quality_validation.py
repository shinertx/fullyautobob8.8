import pandas as pd
from datetime import datetime, timezone, timedelta
from v26meme.data.quality import validate_frame

def test_quality_monotonic():
    ts = [datetime(2025,1,1,0,0, tzinfo=timezone.utc)+timedelta(minutes=i) for i in range(5)]
    df = pd.DataFrame({"timestamp": ts, "open":1,"high":1,"low":1,"close":1,"volume":1})
    qa = validate_frame(df, 60_000)
    assert qa["degraded"]==False and qa["gaps"]==0
