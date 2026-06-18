from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def append_board_history(data_dir: Path, df: pd.DataFrame, target_date: pd.Timestamp, updated_at: str) -> None:
    if df.empty or "板块名称" not in df.columns:
        return
    path = data_dir / "sw_board_history.csv"
    out = df.copy()
    trade_date = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    out["snapshot_date"] = trade_date
    out["snapshot_time"] = updated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "数据日期" not in out.columns:
        out["数据日期"] = trade_date
    if path.exists():
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame()
        combined = pd.concat([old, out], ignore_index=True) if not old.empty else out
    else:
        combined = out
    if "snapshot_date" not in combined.columns:
        combined["snapshot_date"] = combined.get("数据日期", "").astype(str).str[:10]
    combined["snapshot_date"] = combined["snapshot_date"].astype(str).str[:10]
    combined = combined.sort_values(["snapshot_date", "板块名称"])
    combined = combined.drop_duplicates(["snapshot_date", "板块名称"], keep="last")
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def install(uds) -> None:
    original_snapshot_payload = uds._snapshot_payload

    def _snapshot_payload(df: pd.DataFrame, target_date: pd.Timestamp, skip_avix: bool = False) -> dict:
        payload = original_snapshot_payload(df, target_date, skip_avix=skip_avix)
        append_board_history(uds.DATA_DIR, df, target_date, str(payload.get("updated_at", "")))
        return payload

    uds._snapshot_payload = _snapshot_payload
