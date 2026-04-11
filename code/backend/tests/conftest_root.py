"""
Root conftest: ensures the project root is on sys.path so that
`backend` and `ml` are importable as top-level packages.
"""

import os
import sys

# Project root is two levels up from this file (quantis/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
