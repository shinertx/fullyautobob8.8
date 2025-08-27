
print("--- [TRACE] Attempting to import v26meme.cli ---")
try:
    import v26meme.cli
    print("--- [SUCCESS] v26meme.cli imported successfully ---")
except Exception as e:
    print(f"--- [FAIL] An exception occurred during import: {e}")
    import traceback
    traceback.print_exc()
