"""
Pytest configuration for opencda.core.common.communication tests.

This file suppresses known third-party deprecation noise from protobuf descriptors,
so CI output remains signal-focused.
"""

from __future__ import annotations

import warnings



warnings.filterwarnings(
    "ignore",
    message=r"label\(\) is deprecated\..*",
    category=DeprecationWarning,
)
