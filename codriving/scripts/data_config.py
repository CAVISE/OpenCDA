import os
from pathlib import Path

MAIN_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(MAIN_PATH, "data")
EXPIREMENTS_PATH = os.path.join(MAIN_PATH, "experements")
