"""Shared setup for the worked examples: puts the skill's scripts on the path and quiets warnings."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT.parent / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
