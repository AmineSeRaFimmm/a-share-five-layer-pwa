from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("NO_PROXY", "*")

from avix_utils import calculate_and_store_avix, load_avix_history, build_avix_s3_s4_signal_history  # noqa: E402


DATA_DIR = ROOT / "data"
AVIX_DIR = DATA_DIR / "avix"
AVIX_HISTORY_FILE = AVIX_DIR / "avix_history.csv"
AVIX_STATUS_FILE = AVIX_DIR / "update_status.json"


def _json_default(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _normalize_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"trade_date", "avix"}.issubset(df.columns):
        return pd.DataFrame()
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["avix"] = pd.to_numeric(out["avix"], errors="coerce")
    out = out.dropna(subset=["trade_date", "avix"]).sort_values("trade_date")
    return out.drop_duplicates("trade_date", keep="last").reset_index(drop=True)


def _status(status: str, reason: str, latest: dict | None = None, signal_latest: str = "") -> dict:
    return {
        "status": status,
        "reason": reason,
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "latest": latest or {},
        "signal_latest_trade_date": signal_latest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh AVIX cache independently from the after-close review snapshot.")
    parser.add_argument("--skip-signals", action="store_true", help="Only refresh AVIX index cache; do not rebuild S3/S4 signal cache.")
    args = parser.parse_args()

    AVIX_DIR.mkdir(parents=True, exist_ok=True)

    latest_result: dict | None = None
    try:
        latest_result = calculate_and_store_avix()
    except Exception as exc:
        latest_result = {"error": str(exc)}

    hist = _normalize_history_frame(load_avix_history())
    if hist.empty:
        _write_json(AVIX_STATUS_FILE, _status("failed", "AVIX 历史缓存为空", latest_result))
        return 1

    hist.to_csv(AVIX_HISTORY_FILE, index=False, encoding="utf-8-sig")
    latest = hist.iloc[-1].to_dict()
    latest["trade_date"] = pd.Timestamp(latest["trade_date"]).strftime("%Y-%m-%d")

    signal_latest = ""
    signal_reason = ""
    if not args.skip_signals:
        try:
            signal_hist = build_avix_s3_s4_signal_history(hist)
            if not signal_hist.empty and "trade_date" in signal_hist.columns:
                signal_latest_ts = pd.to_datetime(signal_hist["trade_date"], errors="coerce").max()
                if pd.notna(signal_latest_ts):
                    signal_latest = signal_latest_ts.strftime("%Y-%m-%d")
        except Exception as exc:
            signal_reason = f"S3/S4 信号暂未同步：{exc}"

    reason = "AVIX 缓存更新成功"
    status = "ready"
    if latest_result and latest_result.get("error"):
        status = "ready_from_cache"
        reason = f"实时刷新失败，已保留最近缓存：{latest_result['error']}"
    elif signal_reason:
        status = "ready_with_signal_warning"
        reason = signal_reason

    _write_json(AVIX_STATUS_FILE, _status(status, reason, latest, signal_latest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
