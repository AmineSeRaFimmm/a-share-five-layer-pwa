from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = ROOT / "data" / "backtest"
METADATA_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_metadata.json"
RESULTS_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_results.csv"
PRIMARY_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_primary_path.csv"
PANEL_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_panel.parquet"
REFRESH_STATUS_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_refresh_status.json"
GENERATOR_FILE = Path(__file__).resolve().parent / "generate_fullrisk_grid_300.py"


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")


def _json_default(value):
    try:
        import numpy as np
        import pandas as pd

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _write_status(payload: dict) -> None:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    REFRESH_STATUS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _existing_official_grid_is_usable() -> tuple[bool, dict]:
    missing = [
        str(path.relative_to(ROOT))
        for path in (METADATA_FILE, RESULTS_FILE, PRIMARY_FILE, PANEL_FILE)
        if not path.exists() or path.stat().st_size <= 0
    ]
    metadata = _load_json(METADATA_FILE)
    if missing:
        return False, {"missing_files": missing, "metadata": metadata}
    if metadata.get("status") != "ready" or not metadata.get("promoted"):
        return False, {"reason": "official metadata is not ready/promoted", "metadata": metadata}
    required = ["last_trade_date", "observed_trade_days", "observed_sectors", "primary_params"]
    absent = [key for key in required if key not in metadata]
    if absent:
        return False, {"reason": "official metadata missing required keys", "missing_keys": absent, "metadata": metadata}
    try:
        observed_days = int(metadata.get("observed_trade_days", 0))
        observed_sectors = int(metadata.get("observed_sectors", 0))
    except Exception:
        observed_days = 0
        observed_sectors = 0
    if observed_days < 250 or observed_sectors < 20:
        return False, {"reason": "official grid coverage is insufficient", "metadata": metadata}
    return True, {"metadata": metadata}


def _run_generator(promote: bool) -> None:
    cmd = [sys.executable, str(GENERATOR_FILE)]
    if promote:
        cmd.append("--promote")
    print("[fullrisk-production] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    try:
        _run_generator(promote=args.promote)
        metadata = _load_json(METADATA_FILE)
        status = {
            "updated_at": _now(),
            "status": "ready",
            "mode": "regenerated",
            "reason": "300-day full-risk grid regenerated successfully",
            "official_metadata": metadata,
        }
        _write_status(status)
        print(json.dumps(status, ensure_ascii=False, indent=2, default=_json_default), flush=True)
        return 0
    except Exception as exc:
        usable, detail = _existing_official_grid_is_usable()
        status = {
            "updated_at": _now(),
            "status": "kept_existing_ready_grid" if usable else "failed_no_usable_grid",
            "mode": "reuse_official_grid_after_refresh_failure" if usable else "failed",
            "reason": str(exc),
            "traceback": traceback.format_exc(limit=8),
            "official_grid_check": detail,
            "strict_note": (
                "The daily snapshot must not be blocked by a transient SW historical source outage. "
                "Existing official full-risk grid is reused only if it is already promoted, ready, and coverage-valid."
            ),
        }
        _write_status(status)
        print(json.dumps(status, ensure_ascii=False, indent=2, default=_json_default), file=sys.stderr, flush=True)
        return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
