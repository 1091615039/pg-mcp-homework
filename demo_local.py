"""Launch the offline demonstration directly from a source checkout."""

import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    import_module("pg_mcp.demo").main()
