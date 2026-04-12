"""
Root conftest: ensures the project root is on sys.path so that
`backend` and `quant_ml` are importable as top-level packages.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
