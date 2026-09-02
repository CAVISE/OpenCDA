"""Pytest configuration for OpenCOOD unit tests."""

import sys
from pathlib import Path


OPENCOOD_ROOT = Path(__file__).resolve().parents[1]
if str(OPENCOOD_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENCOOD_ROOT))
