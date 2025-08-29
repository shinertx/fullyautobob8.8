import sys
import importlib


def test_cli_import_does_not_require_arch(monkeypatch):
    # Ensure 'arch' is not available to simulate a lean environment
    if 'arch' in sys.modules:
        del sys.modules['arch']

    # Force a clean import of v26meme.cli (clear any prior imports)
    for mod in list(sys.modules.keys()):
        if mod.startswith('v26meme.labs.hyper_lab') or mod.startswith('v26meme.research.feature_factory'):
            del sys.modules[mod]
    if 'v26meme.cli' in sys.modules:
        del sys.modules['v26meme.cli']

    # Import should succeed without pulling in hyper_lab/arch
    m = importlib.import_module('v26meme.cli')
    assert hasattr(m, 'cli')

