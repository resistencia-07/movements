"""Backend entry point for Motion Lab.

The core implementation remains at the project root for compatibility with
existing imports and launch commands. This wrapper gives backend deployments a
stable entry point without changing the runtime's project-root paths.
"""

from pathlib import Path
import runpy
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os

os.chdir(PROJECT_ROOT)

runpy.run_path(str(PROJECT_ROOT / "app.py"), run_name="__main__")
