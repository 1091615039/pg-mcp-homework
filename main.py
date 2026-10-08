"""Run the same PostgreSQL MCP application as ``python -m pg_mcp``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> None:
    """Start the installed MCP server from a source checkout."""
    from pg_mcp.__main__ import main as run_server

    run_server()


if __name__ == "__main__":
    main()
