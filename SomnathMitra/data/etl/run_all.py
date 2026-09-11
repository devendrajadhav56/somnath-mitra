"""Runs every sync_*.py in this folder. Add a new sync_<source>.py and it's picked up automatically."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    for path in sorted(Path(__file__).parent.glob("sync_*.py")):
        print(f"\n--- running {path.name} ---")
        runpy.run_path(str(path), run_name="__main__")
