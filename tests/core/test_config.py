# tests/core/test_config.py
import unittest
import yaml
from v26meme.core.config import RootConfig

class TestConfigValidation(unittest.TestCase):

    def test_config_roundtrip(self):
        """
        Tests that the main config.yaml can be loaded and validated by the Pydantic models.
        This catches missing keys or type mismatches.
        """
        with open("configs/config.yaml", "r") as f:
            raw_config = yaml.safe_load(f)

        try:
            RootConfig.model_validate(raw_config)
            validated = True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            validated = False

        self.assertTrue(validated, "config.yaml should be valid according to RootConfig")

if __name__ == '__main__':
    unittest.main()
