"""Double-clickable entry point (no console window)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_inspector.gui import main

if __name__ == "__main__":
    main()
