from __future__ import annotations

import sys
from pathlib import Path

# Ensure `import agentic_chatbot` works when tests run from the repo root.
_repo_root = Path(__file__).resolve().parents[1]
_parent = _repo_root.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
