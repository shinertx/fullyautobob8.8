import sys
import traceback
from v26meme.cli import loop

# Monkey patch to trace the issue
original_harvest = None
try:
    from v26meme.data import harvester
    original_harvest = harvester.run_once
    
    def traced_harvest(*args, **kwargs):
        print(f"TRACE: harvest called with args={args}, kwargs={kwargs}", flush=True)
        try:
            result = original_harvest(*args, **kwargs)
            print(f"TRACE: harvest returned {result}", flush=True)
            return result
        except Exception as e:
            print(f"TRACE: harvest failed with {e}", flush=True)
            traceback.print_exc()
            raise
    
    harvester.run_once = traced_harvest
except Exception as e:
    print(f"Failed to patch harvester: {e}")

# Now run the loop
try:
    print("TRACE: Starting loop", flush=True)
    loop()
    print("TRACE: Loop exited normally", flush=True)
except SystemExit as e:
    print(f"TRACE: SystemExit with code {e.code}", flush=True)
except KeyboardInterrupt:
    print("TRACE: KeyboardInterrupt", flush=True)
except Exception as e:
    print(f"TRACE: Loop crashed with {e}", flush=True)
    traceback.print_exc()
