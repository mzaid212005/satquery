import sys
import os
from pathlib import Path

# Add project root directory to sys.path so backend, data, and models are discoverable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app

# Vercel ASGI runtime automatically invokes `app`
