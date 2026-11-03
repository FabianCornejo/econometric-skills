"""Shared setup for the worked examples: puts the skill's scripts on the path, keeps console output
from failing on Windows and quiets warnings."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT.parent / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)
warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
