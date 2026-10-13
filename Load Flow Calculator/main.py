"""Launch Load Flow Learning Lab with ``python main.py`` or a double-click.

Prefer this project's virtual environment when present; otherwise use the
current Python installation. Dependencies are listed in requirements.txt.
"""

import os
from pathlib import Path
import subprocess
import sys


def main():
    project_dir = Path(__file__).resolve().parent
    environment_dir = project_dir / ".venv"
    python_path = environment_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    # Check the environment prefix, since a Unix venv's executable may be a
    # symlink to the same base interpreter used outside that environment.
    if python_path.is_file() and Path(sys.prefix).resolve() != environment_dir.resolve():
        return subprocess.call(
            [str(python_path), str(project_dir / "main.py"), *sys.argv[1:]],
            cwd=project_dir,
        )

    os.chdir(project_dir)
    from loadflow_app.__main__ import main as launch

    launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
