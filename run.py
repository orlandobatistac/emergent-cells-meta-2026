"""Standalone Python entry point. No frontend or HTTP server required."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from emergent_cells.cli import main

if __name__ == "__main__":
    main()
