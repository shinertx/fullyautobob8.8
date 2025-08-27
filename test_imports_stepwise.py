
print("--- [TRACE] Starting import test ---")

try:
    import os
    print("--- [SUCCESS] import os ---")
    import time
    print("--- [SUCCESS] import time ---")
    import json
    print("--- [SUCCESS] import json ---")
    import hashlib
    print("--- [SUCCESS] import hashlib ---")
    import random
    print("--- [SUCCESS] import random ---")
    import inspect
    print("--- [SUCCESS] import inspect ---")
    import sys
    print("--- [SUCCESS] import sys ---")
    from pathlib import Path
    print("--- [SUCCESS] from pathlib import Path ---")
    from datetime import datetime, timezone
    print("--- [SUCCESS] from datetime import datetime, timezone ---")
    from typing import Dict, Any, List, Tuple
    print("--- [SUCCESS] from typing import Dict, Any, List, Tuple ---")
    from collections.abc import Awaitable as _Awaitable
    print("--- [SUCCESS] from collections.abc import Awaitable as _Awaitable ---")

    import click
    print("--- [SUCCESS] import click ---")
    
    import yaml
    print("--- [SUCCESS] import yaml ---")
    
    from dotenv import load_dotenv
    print("--- [SUCCESS] from dotenv import load_dotenv ---")
    
    from loguru import logger
    print("--- [SUCCESS] from loguru import logger ---")
    
    import pandas as pd
    print("--- [SUCCESS] import pandas as pd ---")
    
    import numpy as np
    print("--- [SUCCESS] import numpy as np ---")
    
    # Now for the project's own modules
    project_root = Path(__file__).parent.resolve()
    sys.path.insert(0, str(project_root))
    print(f"--- [TRACE] sys.path updated with: {project_root} ---")

    from v26meme.core.state import StateManager
    print("--- [SUCCESS] from v26meme.core.state import StateManager ---")

    from v26meme.core.config import RootConfig
    print("--- [SUCCESS] from v26meme.core.config import RootConfig ---")

    print("--- [TRACE] All imports appear to be successful. ---")

except Exception as e:
    print(f"--- [FAIL] An exception occurred during import: {e}")
    import traceback
    traceback.print_exc()

print("--- [TRACE] Import test finished ---")
