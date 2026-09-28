"""Pytest configuration and environment setup for differential testing."""

import sys
from pathlib import Path

# Add original ColorVideoVDP and SimplerColorVideoVDP to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
orig_repo_dir = root_dir / "ColorVideoVDP"
simpler_repo_src = root_dir / "SimplerColorVideoVDP" / "src"

if str(orig_repo_dir) not in sys.path:
    sys.path.insert(0, str(orig_repo_dir))

if str(simpler_repo_src) not in sys.path:
    sys.path.insert(0, str(simpler_repo_src))
