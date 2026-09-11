"""
Structural migration runner for the somnath_clean database.

Each file in versions/ named NNN_description.py must expose:
    VERSION = "NNN_description"
    def apply(clean_db): ...

Applied versions are tracked in clean_db._migrations so re-running this
script is a no-op except for migrations that haven't run yet.

Run with: python migrations/runner.py
"""
import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_clean_db  # noqa: E402

VERSIONS_DIR = Path(__file__).parent / "versions"


def load_migrations():
    modules = []
    for path in sorted(VERSIONS_DIR.glob("[0-9]*.py")):
        module_name = f"migrations.versions.{path.stem}"
        modules.append(importlib.import_module(module_name))
    return modules


def run():
    clean_db = get_clean_db()
    applied = {doc["version"] for doc in clean_db["_migrations"].find({}, {"version": 1})}

    for module in load_migrations():
        if module.VERSION in applied:
            print(f"skip  {module.VERSION} (already applied)")
            continue
        print(f"apply {module.VERSION}: {module.__doc__.strip() if module.__doc__ else ''}")
        module.apply(clean_db)
        clean_db["_migrations"].insert_one(
            {
                "version": module.VERSION,
                "description": module.__doc__ or "",
                "applied_at": datetime.now(timezone.utc),
            }
        )
        print(f"  done")


if __name__ == "__main__":
    run()
