from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "generate_fullrisk_grid_300_base.py"


def _install_base_module() -> None:
    spec = importlib.util.spec_from_file_location("generate_fullrisk_grid_300", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base full-risk generator from {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_fullrisk_grid_300"] = module
    spec.loader.exec_module(module)


def main() -> int:
    _install_base_module()
    from generate_fullrisk_grid_300_next_open import main as next_open_main

    return int(next_open_main())


if __name__ == "__main__":
    raise SystemExit(main())
