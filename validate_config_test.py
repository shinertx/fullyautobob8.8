import yaml
from pydantic import ValidationError
import sys
from pathlib import Path

# Add project root to the Python path to allow importing from v26meme
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from v26meme.core.config import RootConfig

def validate_config():
    """Loads and validates the main config file against the project's RootConfig."""
    try:
        with open('configs/config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)
        
        # This will raise a detailed ValidationError if the structure is wrong
        RootConfig.model_validate(config_data)

        print("\n--- ✅✅✅ CONFIG VALIDATION PASSED ✅✅✅ ---")
        print("The config structure is valid against v26meme.core.config.RootConfig.")

    except ValidationError as e:
        print("\n--- ❌ CONFIG VALIDATION FAILED ❌ ---")
        print("Pydantic validation error (using v26meme.core.config.RootConfig):")
        print(e)
    except Exception as e:
        print(f"\n--- ❌ AN UNEXPECTED ERROR OCCURRED ❌ ---")
        print(f"Error loading or parsing YAML: {e}")

if __name__ == "__main__":
    validate_config()
