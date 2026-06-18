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
RECOMMENDATION_STATE_FILE = DATA_DIR / "recommendation_state.json"
BACKTEST_SUMMARY_FILE = DATA_DIR / "backtest" / "strategy_summary.json"
AVIX_HISTORY_FILE = DATA_DIR / "avix" / "avix_history.csv"
AVIX_SIGNAL_FILE = DATA_DIR / "avix_s3_s4_signals.csv"


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


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    res = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in res.columns:
            res[col] = pd.to_numeric(res[col], errors="coerce")
    return res


def _sanitize_snapshot_sell_rows(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return payload
    clarity = dict(payload.get("clarity_signal") or {})
    sell_rows = clarity.get("sell") or []
    if not sell_rows:
        return payload

    state = _load_json_file(RECOMMENDATION_STATE_FILE)
    if not state:
        return payload

    holdings = state.get("holdings") or []
    if holdings:
        return payload

    payload_trade_date = str(payload.get("trade_date", ""))[:10]
    state_sell_date = str(state.get("last_sell_trade_date", ""))[:10]
    last_action = str(state.get("last_action", "")).lower()
    state_last_sell = state.get("last_sell") or []

    # A same-day sell generated from a real previous holding is valid and should be visible.
    if last_action == "sell" and state_sell_date == payload_trade_date and state_last_sell:
        return payload

    # Otherwise the sell rows are stale display residue from a prior state or duplicate run.
    out = dict(payload)
    clarity["sell"] = []
    out["clarity_signal"] = clarity
    return out


def _load_avix_history_cache() -> pd.DataFrame:
    hist = _read_csv(AVIX_HISTORY_FILE)
    if hist.empty or not {"trade_date", "avix"}.issubset(hist.columns):
        return pd.DataFrame()
    hist = hist.copy()
    hist["trade_date"] = pd.to_datetime(hist["trade_date"], errors="coerce")
    hist["avix"] = pd.to_numeric(hist["avix"], errors="coerce")
    hist = hist.dropna(subset=["trade_date", "avix"]).sort_values("trade_date")
    if hist.empty:
        return pd.DataFrame()
    hist = hist.drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    return hist


def _load_avix_signal_cache() -> pd.DataFrame:
    sig = _read_csv(AVIX_SIGNAL_FILE)
    if sig.empty or not {"trade_date", "avix"}.issubset(sig.columns):
        return pd.DataFrame()
    sig = sig.copy()
    sig["trade_date"] = pd.to_datetime(sig["trade_date"], errors="coerce")
    sig["avix"] = pd.to_numeric(sig["avix"], errors="coerce")
    sig = sig.dropna(subset=["trade_date", "avix"]).sort_values("trade_date")
    return sig.reset_index(drop=True)


def _refresh_snapshot_avix_from_cache(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return payload
    hist = _load_avix_history_cache()
    if hist.empty:
        return payload

    out = dict(payload)
    avix_payload = dict(out.get("avix") or {})
    latest = hist.iloc[-1].to_dict()
    latest["trade_date"] = pd.Timestamp(latest["trade_date"]).strftime("%Y-%m-%d")
    if "valuation_time" in latest and pd.notna(latest.get("valuation_time")):
        latest["valuation_time"] = str(latest["valuation_time"])
    latest.setdefault("source", "AVIX_CACHE")
    latest.setdefault("quality", "缓存最新")

    hist_out = hist.copy()
    hist_out["trade_date"] = hist_out["trade_date"].dt.strftime("%Y-%m-%d")
    signal_hist = _load_avix_signal_cache()
    if not signal_hist.empty:
        signal_hist = signal_hist.copy()
        signal_hist["trade_date"] = signal_hist["trade_date"].dt.strftime("%Y-%m-%d")
        avix_payload["signal_history"] = signal_hist.to_dict(orient="records")
        signal_latest = pd.to_datetime(signal_hist["trade_date"], errors="coerce").max()
        index_latest = pd.to_datetime(hist_out["trade_date"], errors="coerce").max()
        if pd.notna(index_latest) and pd.notna(signal_latest) and signal_latest < index_latest:
            avix_payload["signal_note"] = (
                f"AVIX 指数已到 {index_latest.strftime('%Y-%m-%d')}；"
                f"S3/S4 信号因上证指数源滞后暂到 {signal_latest.strftime('%Y-%m-%d')}。"
            )

    avix_payload["latest"] = latest
    avix_payload["history"] = hist_out.to_dict(orient="records")
    avix_payload["cache_note"] = "AVIX优先读取 data/avix/avix_history.csv 最新缓存"
    out["avix"] = avix_payload
    return out


def _sanitize_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    payload = _sanitize_snapshot_sell_rows(payload)
    payload = _refresh_snapshot_avix_from_cache(payload)
    return payload


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
    return _sanitize_snapshot(_load_json_file(LATEST_FILE))


@st.cache_data(ttl=60, show_spinner=False)
def load_update_status() -> dict[str, Any]:
    status = _load_json_file(STATUS_FILE)
    snapshot = load_latest_snapshot()
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
