from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


DATA_DIR = Path("./data")
LATEST_FILE = DATA_DIR / "latest_snapshot.json"
STATUS_FILE = DATA_DIR / "update_status.json"
BOARD_HISTORY_FILE = DATA_DIR / "sw_board_history.csv"
HISTORY_DIR = DATA_DIR / "history"
BACKTEST_SUMMARY_FILE = DATA_DIR / "backtest" / "strategy_summary.json"


NUMERIC_COLUMNS = [
    "涨跌幅",
    "综合博弈得分",
    "逃顶风险分",
    "入场共振分",
    "动态水位",
    "趋势加速度",
    "资金流向",
    "金额流向",
    "上涨占比",
    "成分股覆盖数",
    "fund_raw",
    "成交量",
    "成交额",
    "换手率",
    "开盘价",
    "最高价",
    "最低价",
    "收盘价",
    "第1层_真实趋势",
    "第2层_真假资金",
    "第3层_异动干预",
    "第4层_诱多诱空",
    "第5层_博弈反身",
    "第6层_中期确认",
    "第7层_微观结构",
]


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    res = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in res.columns:
            res[col] = pd.to_numeric(res[col], errors="coerce")
    return res


def _snapshot_frame_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    trade_date = str(payload.get("trade_date", ""))[:10]
    updated_at = str(payload.get("updated_at", ""))
    rows = payload.get("sectors", []) or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if trade_date:
        df["snapshot_date"] = trade_date
    if updated_at:
        df["snapshot_time"] = updated_at
    if "数据日期" not in df.columns and trade_date:
        df["数据日期"] = trade_date
    return _coerce_numeric(df)


@st.cache_data(ttl=60, show_spinner=False)
def load_latest_snapshot() -> dict[str, Any]:
    return _load_json_file(LATEST_FILE)


@st.cache_data(ttl=60, show_spinner=False)
def load_update_status() -> dict[str, Any]:
    status = _load_json_file(STATUS_FILE)
    snapshot = _load_json_file(LATEST_FILE)
    trade_date = str(snapshot.get("trade_date", ""))
    target_date = str(status.get("target_trade_date", status.get("target_date", ""))) if status else ""
    if snapshot.get("status") == "ready" and trade_date and (not target_date or target_date == trade_date):
        merged = dict(status)
        merged.setdefault("target_date", trade_date)
        merged["target_trade_date"] = trade_date
        merged["status"] = "ready"
        merged["reason"] = merged.get("reason") or "主缓存已通过校验"
        return merged
    return status


@st.cache_data(ttl=60, show_spinner=False)
def load_snapshot_frame() -> pd.DataFrame:
    snapshot = load_latest_snapshot()
    return _snapshot_frame_from_payload(snapshot)


@st.cache_data(ttl=120, show_spinner=False)
def load_snapshot_history_from_json() -> pd.DataFrame:
    if not HISTORY_DIR.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(HISTORY_DIR.glob("snapshot_*.json")):
        payload = _load_json_file(path)
        frame = _snapshot_frame_from_payload(payload)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    hist = pd.concat(frames, ignore_index=True)
    if "板块名称" not in hist.columns:
        return pd.DataFrame()
    return _coerce_numeric(hist)


@st.cache_data(ttl=120, show_spinner=False)
def load_board_history() -> pd.DataFrame:
    json_hist = load_snapshot_history_from_json()
    csv_hist = pd.DataFrame()
    if BOARD_HISTORY_FILE.exists():
        try:
            csv_hist = pd.read_csv(BOARD_HISTORY_FILE)
        except Exception:
            csv_hist = pd.DataFrame()
    if not csv_hist.empty and "板块名称" in csv_hist.columns:
        csv_hist = _coerce_numeric(csv_hist)
        hist = pd.concat([csv_hist, json_hist], ignore_index=True) if not json_hist.empty else csv_hist
    else:
        hist = json_hist
    if hist.empty or "板块名称" not in hist.columns:
        return pd.DataFrame()

    date_col = next((c for c in ["snapshot_time", "snapshot_date", "数据日期", "行情日期", "日期"] if c in hist.columns), None)
    if date_col is None:
        return pd.DataFrame()

    hist["snapshot_dt"] = pd.to_datetime(hist[date_col].astype(str).str[:19], errors="coerce")
    hist = hist.dropna(subset=["snapshot_dt"])
    if hist.empty:
        return pd.DataFrame()

    hist["snapshot_day"] = hist["snapshot_dt"].dt.strftime("%Y-%m-%d")
    hist = hist.sort_values("snapshot_dt")
    hist = hist.drop_duplicates(["snapshot_day", "板块名称"], keep="last")
    return hist.reset_index(drop=True)


def load_sector_history_from_cache(sector_name: str, lookback_days: int = 120) -> pd.DataFrame:
    hist = load_board_history()
    if hist.empty:
        return hist
    sector = hist[hist["板块名称"].astype(str) == str(sector_name)].copy()
    if sector.empty:
        return sector
    sector = sector.sort_values("snapshot_dt").tail(lookback_days).copy()
    if "日期" not in sector.columns:
        sector["日期"] = sector["snapshot_dt"]
    if "数据日期" not in sector.columns:
        sector["数据日期"] = sector["snapshot_day"]
    return sector.reset_index(drop=True)


@st.cache_data(ttl=120, show_spinner=False)
def load_backtest_summary() -> dict[str, Any]:
    return _load_json_file(BACKTEST_SUMMARY_FILE)
