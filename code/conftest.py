"""
Root conftest: ensures all package roots are on sys.path.
"""

import os
import sys

# Add project root to path so `api`, `models`, `monitoring` are importable
ROOT = os.path.dirname(os.path.abspath(__file__))
for path in [ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)
