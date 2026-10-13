"""Launch with python -m loadflow_app."""
import sys


def main():
    if sys.version_info < (3, 11):
        raise SystemExit("Load Flow Learning Lab requires Python 3.11 or newer.")
    try:
        from .gui import main as launch
    except ImportError as exc:
        raise SystemExit(f"A required dependency could not be imported: {exc}\nInstall requirements.txt with python -m pip install -r requirements.txt.\nCheck Tkinter with python -m tkinter (Tkinter is installed through Python or your OS, not pip).") from exc
    try:
        launch()
    except Exception as exc:
        from tkinter import TclError
        if isinstance(exc, TclError):
            raise SystemExit(f"Tkinter could not open the app: {exc}\nCheck your Python/Tcl/Tk installation with python -m tkinter.") from exc
        raise


if __name__ == "__main__":
    main()
