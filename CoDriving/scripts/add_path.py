import sys
from pathlib import Path

UTILS_PATH = Path(__file__).resolve().parent.parent
if str(UTILS_PATH) not in sys.path:
  sys.path.insert(0, str(UTILS_PATH))
