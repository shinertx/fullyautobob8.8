# tests/data/test_lakehouse_quality.py
import unittest
import pandas as pd
import numpy as np
from v26meme.data.quality import validate_frame, TF_MS

class TestLakehouseQuality(unittest.TestCase):

    def test_validate_frame_no_open_bar(self):
        """
        Tests that validate_frame correctly identifies a frame with no open bar.
        """
        tf = '1h'
        tf_ms = TF_MS[tf]
        end_ts = pd.Timestamp.now(tz='UTC').floor('h').value // 1_000_000
        
        # Create a complete, valid frame
        timestamps = pd.to_datetime(np.arange(end_ts - 10 * tf_ms, end_ts, tf_ms), unit='ms', utc=True)
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': 100, 'high': 110, 'low': 90, 'close': 105, 'volume': 1000
        })

        result = validate_frame(df, tf_ms)
        self.assertTrue(result['accepted'])
        self.assertFalse(result['degraded'])
        self.assertIn('df', result)
        self.assertEqual(len(result['df']), 10)

    def test_validate_frame_detects_open_bar(self):
        """
        Tests that validate_frame detects and removes an open (incomplete) bar at the end.
        """
        tf = '1h'
        tf_ms = TF_MS[tf]
        # Simulate current time being halfway through an hour
        now_ts = pd.Timestamp.now(tz='UTC').floor('h').value // 1_000_000 + (tf_ms // 2)
        
        # Create a frame where the last bar is incomplete
        timestamps = pd.to_datetime(np.arange(now_ts - 10 * tf_ms, now_ts + tf_ms, tf_ms), unit='ms', utc=True)
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': 100, 'high': 110, 'low': 90, 'close': 105, 'volume': 1000
        })

        # The last timestamp should be considered open
        result = validate_frame(df, tf_ms)
        self.assertTrue(result['accepted'])
        self.assertTrue(result['degraded'])
        self.assertEqual(result['reason'], 'open_bar_removed')
        self.assertIn('df', result)
        # Should have dropped the last bar
        self.assertEqual(len(result['df']), 10)
        self.assertTrue(result['df']['timestamp'].max() < pd.to_datetime(now_ts, unit='ms', utc=True))

    def test_validate_frame_gap_acceptance(self):
        """
        Tests that validate_frame correctly handles gaps within the acceptance threshold.
        """
        tf = '1h'
        tf_ms = TF_MS[tf]
        end_ts = pd.Timestamp.now(tz='UTC').floor('h').value // 1_000_000

        timestamps = pd.to_datetime(np.arange(end_ts - 10 * tf_ms, end_ts, tf_ms), unit='ms', utc=True)
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': 100, 'high': 110, 'low': 90, 'close': 105, 'volume': 1000
        })
        
        # Drop a few rows to create a gap
        df_with_gap = df.drop([3, 4])

        # Expect it to be accepted but marked as degraded
        result = validate_frame(df_with_gap, tf_ms, max_gap_pct=0.3)
        self.assertTrue(result['accepted'])
        self.assertTrue(result['degraded'])
        self.assertAlmostEqual(result['gap_ratio'], 0.2) # 2 missing out of 10 expected
        self.assertEqual(result['reason'], 'has_gaps')

    def test_validate_frame_gap_rejection(self):
        """
        Tests that validate_frame rejects a frame with gaps exceeding the threshold.
        """
        tf = '1h'
        tf_ms = TF_MS[tf]
        end_ts = pd.Timestamp.now(tz='UTC').floor('h').value // 1_000_000

        timestamps = pd.to_datetime(np.arange(end_ts - 10 * tf_ms, end_ts, tf_ms), unit='ms', utc=True)
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': 100, 'high': 110, 'low': 90, 'close': 105, 'volume': 1000
        })
        
        # Drop many rows to create a large gap
        df_with_gap = df.drop([1, 2, 3, 4, 5])

        result = validate_frame(df_with_gap, tf_ms, max_gap_pct=0.4)
        self.assertFalse(result['accepted'])
        self.assertTrue(result['degraded'])
        self.assertAlmostEqual(result['gap_ratio'], 0.5) # 5 missing out of 10 expected
        self.assertEqual(result['reason'], 'gap_ratio_exceeded')

if __name__ == '__main__':
    unittest.main()
