# utils.py
from __future__ import annotations
import os
import re
import time
import requests
import urllib3
import numpy as np
import pandas as pd
import akshare as ak
import streamlit as st
from pathlib import Path
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_CACHE_FILE = DATA_DIR / "processed_sw_cache.csv"
A_SHARE_SNAPSHOT_CACHE_FILE = DATA_DIR / "a_share_close_snapshot.csv"
PROCESSED_CACHE_TTL_SECONDS = int(os.environ.get("SW_PROCESSED_CACHE_TTL_SECONDS", str(30 * 60)))
SW_CURRENT_URL = "https://www.swsresearch.com/institute-sw/api/index_publish/current/"
EM_A_SHARE_SNAPSHOT_URLS = [
    "https://82.push2.eastmoney.com/api/qt/clist/get",
    "https://push2.eastmoney.com/api/qt/clist/get",
]
SW_CURRENT_SOURCE_LABEL = "申万current"
SW_TREND_SOURCE_LABEL = "akshare trend"
SW_CURRENT_AFTER_CLOSE_HOUR = int(os.environ.get("SW_CURRENT_AFTER_CLOSE_HOUR", "19"))
MIN_A_SHARE_SNAPSHOT_ROWS = int(os.environ.get("MIN_A_SHARE_SNAPSHOT_ROWS", "1000"))
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= 1. 申万一级行业及ETF映射 =================
SW_CODE_MAPPING: Dict[str, str] = {
    "农林牧渔": "801010", "基础化工": "801030", "钢铁": "801040", "有色金属": "801050",
    "电子": "801080", "家用电器": "801110", "食品饮料": "801120", "纺织服饰": "801130",
    "轻工制造": "801140", "医药生物": "801150", "公用事业": "801160", "交通运输": "801170",
    "房地产": "801180", "商贸零售": "801200", "社会服务": "801210", "综合": "801230",
    "建筑材料": "801710", "建筑装饰": "801720", "电力设备": "801730", "机械设备": "801890",
    "国防军工": "801740", "计算机": "801750", "传媒": "801760", "通信": "801770",
    "银行": "801780", "非银金融": "801790", "汽车": "801880", "煤炭": "801950",
    "石油石化": "801960", "环保": "801970", "美容护理": "801980",
}

SW_ETF_MAPPING: Dict[str, str] = {
    "农林牧渔": "农业ETF华夏(516810)",
    "基础化工": "化工ETF鹏华(159870)",
    "钢铁": "钢铁ETF国泰(515210)",
    "有色金属": "有色金属ETF南方(512400)",
    "电子": "电子ETF天弘(159997)",
    "家用电器": "家电ETF国泰(159996)",
    "食品饮料": "食品饮料ETF华夏(515170)",
    "纺织服饰": "暂无精确ETF｜消费ETF汇添富(159928)",
    "轻工制造": "暂无精确ETF｜家居家电ETF华泰柏瑞(512430)",
    "医药生物": "医药ETF易方达(512010)",
    "公用事业": "电力ETF广发(159611)",
    "交通运输": "交通运输ETF华夏(159666)",
    "房地产": "房地产ETF南方(512200)",
    "商贸零售": "暂无精确ETF｜消费ETF汇添富(159928)",
    "社会服务": "旅游ETF富国(159766)",
    "综合": "暂无精确ETF｜中证1000ETF南方(512100)",
    "建筑材料": "建材ETF国泰(159745)",
    "建筑装饰": "基建ETF广发(516970)",
    "电力设备": "暂无精确ETF｜新能源ETF南方(516160)",
    "机械设备": "机械ETF国泰(516960)",
    "国防军工": "军工ETF国泰(512660)",
    "计算机": "计算机ETF天弘(159998)",
    "传媒": "传媒ETF广发(512980)",
    "通信": "通信ETF国泰(515880)",
    "银行": "银行ETF华宝(512800)",
    "非银金融": "证券保险ETF易方达(512070)",
    "汽车": "汽车ETF国泰(516110)",
    "煤炭": "煤炭ETF国泰(515220)",
    "石油石化": "石油ETF鹏华(159697)",
    "环保": "环保ETF广发(512580)",
    "美容护理": "暂无精确ETF｜消费服务ETF工银(516600)",
}


# ================= 2. 核心工具函数 =================

def _clean_code(x) -> Optional[str]:
    if pd.isna(x):
        return None
    digits = re.sub(r"\D", "", str(x).strip())
    return digits[-6:].zfill(6) if digits else None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def winsorize(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    s = series.copy()
    if s.dropna().empty:
        return s
    return s.clip(s.quantile(lower), s.quantile(upper))


def rank_score(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dropna().empty or s.nunique() <= 1:
        return pd.Series([50.0] * len(s), index=s.index)
    return s.rank(pct=True, method="average") * 100


def _linreg_slope(y: np.ndarray) -> float:
    if y is None or len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    try:
        return float(np.polyfit(x, y.astype(float), 1)[0])
    except Exception:
        return 0.0


def _winsorize_zscore(series: pd.Series) -> pd.Series:
    """先截断极值再做 Z-score 标准化，保证跨板块可比性。"""
    s = winsorize(series)
    std = s.std(ddof=0)
    if std < 1e-9:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.mean()) / std


def _calc_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std < 1e-9:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def _read_processed_disk_cache() -> pd.DataFrame:
    """Read the last processed board snapshot from disk to avoid cold-start network stalls."""
    try:
        if not PROCESSED_CACHE_FILE.exists():
            return pd.DataFrame()
        age = time.time() - PROCESSED_CACHE_FILE.stat().st_mtime
        if age > PROCESSED_CACHE_TTL_SECONDS:
            return pd.DataFrame()
        df = pd.read_csv(PROCESSED_CACHE_FILE)
        if "数据日期" not in df.columns:
            return pd.DataFrame()
        if "成分股覆盖数" in df.columns:
            coverage = pd.to_numeric(df["成分股覆盖数"], errors="coerce").fillna(0)
            if not coverage.empty and float(coverage.max()) <= 0:
                return pd.DataFrame()
        target = _latest_completed_sw_trade_date()
        if target is not None:
            expected_prefix = target.strftime("%Y-%m-%d")
            cached_dates = df["数据日期"].astype(str).str[:10].dropna().unique().tolist()
            if expected_prefix not in cached_dates:
                return pd.DataFrame()
        if not df.empty and "板块名称" in df.columns:
            df["对应ETF"] = df["板块名称"].map(SW_ETF_MAPPING).fillna("暂无")
        return df
    except Exception:
        return pd.DataFrame()


def _write_processed_disk_cache(df: pd.DataFrame) -> None:
    try:
        if not df.empty:
            if "成分股覆盖数" in df.columns:
                coverage = pd.to_numeric(df["成分股覆盖数"], errors="coerce").fillna(0)
                if not coverage.empty and float(coverage.max()) <= 0:
                    return
            df.to_csv(PROCESSED_CACHE_FILE, index=False)
    except Exception:
        pass


def _read_a_share_snapshot_cache() -> pd.DataFrame:
    try:
        if not A_SHARE_SNAPSHOT_CACHE_FILE.exists():
            return pd.DataFrame()
        df = pd.read_csv(A_SHARE_SNAPSHOT_CACHE_FILE)
        if not {"trade_date", "代码", "涨跌幅"}.issubset(df.columns):
            return pd.DataFrame()
        target = _latest_completed_sw_trade_date()
        if target is None:
            return pd.DataFrame()
        target_str = target.strftime("%Y-%m-%d")
        df = df[df["trade_date"].astype(str) == target_str].copy()
        if len(df) < MIN_A_SHARE_SNAPSHOT_ROWS:
            return pd.DataFrame()
        df["代码"] = df["代码"].apply(_clean_code)
        df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
        return df[["代码", "涨跌幅"]].dropna().drop_duplicates("代码", keep="last")
    except Exception:
        return pd.DataFrame()


def _write_a_share_snapshot_cache(df: pd.DataFrame) -> None:
    try:
        if df.empty or len(df) < MIN_A_SHARE_SNAPSHOT_ROWS:
            return
        target = _latest_completed_sw_trade_date()
        if target is None:
            return
        out = df[["代码", "涨跌幅"]].copy()
        out["trade_date"] = target.strftime("%Y-%m-%d")
        out[["trade_date", "代码", "涨跌幅"]].to_csv(A_SHARE_SNAPSHOT_CACHE_FILE, index=False)
    except Exception:
        pass


def _normalise_sw_hist_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        col_str = str(col)
        if "日期" in col_str or col_str.lower() == "date":
            rename_map[col] = "日期"
        elif "收盘" in col_str or col_str.lower() == "close":
            rename_map[col] = "收盘"
        elif "开盘" in col_str or col_str.lower() == "open":
            rename_map[col] = "开盘"
        elif "最高" in col_str or col_str.lower() == "high":
            rename_map[col] = "最高"
        elif "最低" in col_str or col_str.lower() == "low":
            rename_map[col] = "最低"
        elif "成交量" in col_str or col_str.lower() == "volume":
            rename_map[col] = "成交量"
        elif "成交额" in col_str or col_str.lower() == "amount":
            rename_map[col] = "成交额"
    df = df.rename(columns=rename_map)
    needed = ["日期", "收盘", "开盘", "最高", "最低", "成交量", "成交额"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].copy()
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    for col in needed[1:]:
        df[col] = safe_numeric(df[col])
    return df.dropna(subset=["日期", "收盘", "成交量", "成交额"]).sort_values("日期").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def _latest_completed_sw_trade_date() -> Optional[pd.Timestamp]:
    try:
        cal = ak.tool_trade_date_hist_sina()
        date_col = next((c for c in cal.columns if "date" in str(c).lower() or "日期" in str(c)), cal.columns[0])
        dates = pd.to_datetime(cal[date_col], errors="coerce").dropna().sort_values().dt.normalize()
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        today = pd.Timestamp(now.date())
        cutoff = dt_time(SW_CURRENT_AFTER_CLOSE_HOUR, 0)
        eligible = dates[dates <= today] if now.time() >= cutoff else dates[dates < today]
        if eligible.empty:
            return None
        return pd.Timestamp(eligible.iloc[-1]).normalize()
    except Exception:
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        today = pd.Timestamp(now.date())
        if now.time() >= dt_time(SW_CURRENT_AFTER_CLOSE_HOUR, 0):
            return today
        return (today - pd.offsets.BDay(1)).normalize()


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _fetch_sw_current_snapshot() -> Dict[str, dict]:
    try:
        resp = requests.get(
            SW_CURRENT_URL,
            params={"page": 1, "page_size": 100, "indextype": "一级行业"},
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.swsresearch.com/",
            },
            timeout=15,
            verify=False,
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        rows = []
        if isinstance(data, dict):
            for key in ("results", "list", "data"):
                if isinstance(data.get(key), list):
                    rows = data[key]
                    break
        elif isinstance(data, list):
            rows = data

        snapshot: Dict[str, dict] = {}
        code_to_name = {v: k for k, v in SW_CODE_MAPPING.items()}
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = str(row.get("swindexcode") or row.get("indexcode") or row.get("code") or "").strip()
            if code not in code_to_name:
                continue
            prev_close = pd.to_numeric(row.get("l3"), errors="coerce")
            open_ = pd.to_numeric(row.get("l4"), errors="coerce")
            high = pd.to_numeric(row.get("l6"), errors="coerce")
            low = pd.to_numeric(row.get("l7"), errors="coerce")
            close = pd.to_numeric(row.get("l8"), errors="coerce")
            amount_million = pd.to_numeric(row.get("l5"), errors="coerce")
            volume_million = pd.to_numeric(row.get("l11"), errors="coerce")
            if pd.isna(prev_close) or pd.isna(close) or pd.isna(amount_million) or pd.isna(volume_million):
                continue
            snapshot[code_to_name[code]] = {
                "prev_close": float(prev_close),
                "开盘": float(open_) if not pd.isna(open_) else float(close),
                "最高": float(high) if not pd.isna(high) else float(close),
                "最低": float(low) if not pd.isna(low) else float(close),
                "收盘": float(close),
                "成交额": float(amount_million) / 100.0,
                "成交量": float(volume_million) / 100.0,
            }
        return snapshot
    except Exception:
        return {}


def _append_sw_current_if_trend_lags(df: pd.DataFrame, sector_name: str, target_date: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, str]:
    if df.empty:
        return df, SW_TREND_SOURCE_LABEL
    target = target_date or _latest_completed_sw_trade_date()
    if target is None:
        return df, SW_TREND_SOURCE_LABEL

    df = _normalise_sw_hist_df(df)
    if df.empty:
        return df, SW_TREND_SOURCE_LABEL
    df["数据来源"] = SW_TREND_SOURCE_LABEL

    last_date = pd.Timestamp(df["日期"].iloc[-1]).normalize()
    if last_date >= target:
        return df, SW_TREND_SOURCE_LABEL

    current = _fetch_sw_current_snapshot().get(sector_name)
    if not current:
        return df, SW_TREND_SOURCE_LABEL

    last_close = float(df["收盘"].iloc[-1])
    prev_close = float(current["prev_close"])
    if abs(prev_close - last_close) / max(abs(last_close), 1e-9) > 0.0005:
        return df, SW_TREND_SOURCE_LABEL

    next_trade = _next_trade_date_after(last_date)
    if next_trade is not None and next_trade != target:
        return df, SW_TREND_SOURCE_LABEL

    patch_row = {
        "日期": target,
        "收盘": current["收盘"],
        "开盘": current["开盘"],
        "最高": current["最高"],
        "最低": current["最低"],
        "成交量": current["成交量"],
        "成交额": current["成交额"],
        "数据来源": SW_CURRENT_SOURCE_LABEL,
    }
    df = pd.concat([df, pd.DataFrame([patch_row])], ignore_index=True)
    return df.sort_values("日期").drop_duplicates("日期", keep="last").reset_index(drop=True), SW_CURRENT_SOURCE_LABEL


@st.cache_data(ttl=3600, show_spinner=False)
def _next_trade_date_after(date_value: pd.Timestamp) -> Optional[pd.Timestamp]:
    try:
        cal = ak.tool_trade_date_hist_sina()
        date_col = next((c for c in cal.columns if "date" in str(c).lower() or "日期" in str(c)), cal.columns[0])
        dates = pd.to_datetime(cal[date_col], errors="coerce").dropna().sort_values().dt.normalize()
        future = dates[dates > pd.Timestamp(date_value).normalize()]
        if future.empty:
            return None
        return pd.Timestamp(future.iloc[0]).normalize()
    except Exception:
        return None


# ================= 3. 数据获取层 =================
@st.cache_data(ttl=3600 * 12, show_spinner=False)
def fetch_sw_constituents_mapping() -> Dict[str, List[str]]:
    """获取申万一级行业全部成分股代码（多接口容错）。"""
    mapping: Dict[str, List[str]] = {}
    for name, code in SW_CODE_MAPPING.items():
        cons_df = None
        try:
            cons_df = ak.index_stock_cons_sw(symbol=code)
        except Exception:
            pass
        if cons_df is None or cons_df.empty:
            try:
                cons_df = ak.index_component_sw(symbol=code)
            except Exception:
                pass
        if cons_df is None or cons_df.empty:
            try:
                cons_df = ak.index_component_sw(symbol=f"sw{code}")
            except Exception:
                pass
        if cons_df is None or cons_df.empty:
            try:
                cons_df = ak.stock_board_industry_cons_em(symbol=name)
            except Exception:
                continue
        if cons_df is not None and not cons_df.empty:
            possible = ["证券代码", "代码", "成分券代码", "stock_code", "code", "成分代码"]
            col_name = next((c for c in possible if c in cons_df.columns), None)
            if col_name:
                codes = cons_df[col_name].apply(_clean_code).dropna().unique().tolist()
                if codes:
                    mapping[name] = codes
    return mapping


@st.cache_data(ttl=60, show_spinner=False)
def fetch_all_a_shares_spot() -> pd.DataFrame:
    """全市场 A 股最新收盘快照涨跌幅；盘后使用时对应当日收盘涨跌幅。"""
    raw_frames = []
    try:
        raw_frames.append(ak.stock_zh_a_spot_em())
    except Exception:
        pass
    try:
        if not raw_frames or raw_frames[-1].empty:
            raw_frames.append(ak.stock_zh_a_spot())
    except Exception:
        pass

    for df in raw_frames:
        if df is None or df.empty:
            continue
        code_col = next((c for c in ["代码", "f12", "symbol", "code"] if c in df.columns), None)
        pct_col = next((c for c in ["涨跌幅", "f3", "pct_change", "change_pct", "changepercent"] if c in df.columns), None)
        if code_col and pct_col:
            res = df[[code_col, pct_col]].copy()
            res.columns = ["代码", "涨跌幅"]
            res["代码"] = res["代码"].apply(_clean_code)
            res["涨跌幅"] = pd.to_numeric(res["涨跌幅"], errors="coerce")
            res = res.dropna().drop_duplicates("代码", keep="last")
            if len(res) >= MIN_A_SHARE_SNAPSHOT_ROWS:
                _write_a_share_snapshot_cache(res)
                return res
    em_snapshot = _fetch_eastmoney_a_share_close_snapshot()
    if not em_snapshot.empty:
        _write_a_share_snapshot_cache(em_snapshot)
        return em_snapshot
    return _read_a_share_snapshot_cache()


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_eastmoney_a_share_close_snapshot() -> pd.DataFrame:
    """东方财富全 A 最新行情快照，用作 akshare 快照为空时的盘后收盘涨跌幅兜底。"""
    base_params = {
        "pz": 100,
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        "fields": "f12,f3",
    }
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}

    def _fetch_page(page: int) -> Tuple[int, List[dict]]:
        for _ in range(8):
            for url in EM_A_SHARE_SNAPSHOT_URLS:
                try:
                    params = dict(base_params)
                    params["pn"] = page
                    resp = requests.get(url, params=params, headers=headers, timeout=10)
                    resp.raise_for_status()
                    payload = resp.json()
                    data = payload.get("data") or {}
                    if not isinstance(data, dict):
                        continue
                    page_rows = data.get("diff") or []
                    if page_rows:
                        return int(data.get("total") or 0), [r for r in page_rows if isinstance(r, dict)]
                except Exception:
                    continue
        return 0, []

    rows: List[dict] = []
    total = 0
    first_total, first_rows = _fetch_page(1)
    if not first_rows:
        return pd.DataFrame()
    total = first_total
    rows.extend(first_rows)

    max_page = int(np.ceil(total / base_params["pz"])) if total else 80
    for page in range(2, min(max_page, 80) + 1):
        _, page_rows = _fetch_page(page)
        if page_rows:
            rows.extend(page_rows)

    min_required = max(MIN_A_SHARE_SNAPSHOT_ROWS, int(total * 0.95)) if total else MIN_A_SHARE_SNAPSHOT_ROWS
    if len(rows) < min_required:
        return pd.DataFrame()

    try:
        df = pd.DataFrame(rows)
        if "f12" not in df.columns or "f3" not in df.columns:
            return pd.DataFrame()
        res = df[["f12", "f3"]].copy()
        res.columns = ["代码", "涨跌幅"]
        res["代码"] = res["代码"].apply(_clean_code)
        res["涨跌幅"] = pd.to_numeric(res["涨跌幅"], errors="coerce")
        res = res.dropna().drop_duplicates("代码", keep="last")
        if len(res) >= min_required:
            return res
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(ttl=3600 * 4, show_spinner=False)
def fetch_historical_baselines() -> Dict[str, dict]:
    """获取申万行业历史基准特征（近 60 日）。"""
    history_dict: Dict[str, dict] = {}
    target_date = _latest_completed_sw_trade_date()
    for name, code in SW_CODE_MAPPING.items():
        try:
            df = ak.index_hist_sw(symbol=code, period="day")
            if df.empty:
                continue
            df, source_label = _append_sw_current_if_trend_lags(df, name, target_date)
            df = df.tail(60)
            if df.empty:
                continue

            close = df["收盘"].dropna()
            volume = df["成交量"].dropna()
            amount = df["成交额"].dropna()
            if len(close) < 20:
                continue

            recent_20 = close.tail(20)
            dyn_pos = (close.iloc[-1] - recent_20.min()) / (recent_20.max() - recent_20.min() + 1e-6) * 100

            ret_series = close.pct_change()
            ret_3 = ret_series.tail(3).mean()
            ret_prev3 = ret_series.tail(6).head(3).mean()
            acceleration = (ret_3 - ret_prev3) * 100

            vol_ma5 = volume.tail(5).mean()
            flow = (volume.iloc[-1] - vol_ma5) / (vol_ma5 + 1e-6)
            amount_ma5 = amount.tail(5).mean()
            amount_flow = (amount.iloc[-1] - amount_ma5) / (amount_ma5 + 1e-6)

            ma20 = close.tail(20).mean()
            ma60 = close.tail(60).mean()
            ret_20 = close.iloc[-1] / (close.iloc[-20] + 1e-9) - 1.0
            up_days_20 = float((close.pct_change().tail(20) > 0).mean())
            mid_confirm_raw = (
                0.45 * (ret_20 / (close.pct_change().tail(20).std(ddof=0) + 1e-6)) +
                0.25 * (close.iloc[-1] / (ma20 + 1e-9) - 1.0) * 100 +
                0.20 * (ma20 / (ma60 + 1e-9) - 1.0) * 100 +
                0.10 * (up_days_20 - 0.5) * 10
            )

            high_3 = df["最高"].tail(3).to_numpy()
            low_3 = df["最低"].tail(3).to_numpy()
            open_3 = df["开盘"].tail(3).to_numpy()
            close_3 = close.tail(3).to_numpy()
            shadows = [
                (h - max(o, c)) / (h - l + 1e-9) if (h - l) > 0 else 0.0
                for h, l, o, c in zip(high_3, low_3, open_3, close_3)
            ]

            history_dict[name] = {
                "past_close":      float(close.iloc[-2]),
                "dyn_pos":         float(dyn_pos),
                "acceleration":    float(acceleration),
                "flow":            float(flow),
                "amount_flow":     float(amount_flow),
                "avg_shadow_ratio": float(np.mean(shadows)),
                "vol_mean":        float(volume.tail(20).mean()),
                "vol_std":         float(volume.tail(20).std(ddof=0)),
                "amount_mean":     float(amount.tail(20).mean()),
                "amount_std":      float(amount.tail(20).std(ddof=0)),
                "price_std":       float(close.pct_change().tail(20).std(ddof=0)),
                "mid_confirm_raw": float(mid_confirm_raw),
                "last_close":      float(close.iloc[-1]),
                "last_open":       float(df["开盘"].iloc[-1]),
                "last_high":       float(df["最高"].iloc[-1]),
                "last_low":        float(df["最低"].iloc[-1]),
                "last_amount":     float(df["成交额"].iloc[-1]),
                "last_volume":     float(volume.iloc[-1]),
                "data_date":        pd.Timestamp(df["日期"].iloc[-1]).strftime("%Y-%m-%d"),
                "data_source":      source_label,
            }
        except Exception:
            continue
    return history_dict


def _calc_market_regime(spot_df: pd.DataFrame) -> Tuple[int, float, float]:
    if spot_df.empty or "涨跌幅" not in spot_df.columns:
        return 0, 1.0, 0.5
    pct = safe_numeric(spot_df["涨跌幅"]).dropna()
    breadth = float((pct > 0).mean())
    if breadth >= 0.65:
        return 1, 1.10, breadth
    if breadth <= 0.35:
        return -1, 0.80, breadth
    return 0, 1.00, breadth


# ================= 4. 三维共振-熵减模型 =================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_sector_histories() -> Dict[str, pd.DataFrame]:
    """
    拉取所有申万一级行业近 60 日日线，列名已标准化。
    列：date, close, open, high, low, volume, amount
    """
    histories: Dict[str, pd.DataFrame] = {}
    base_rename = {
        "日期": "date", "收盘": "close", "开盘": "open",
        "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
    }
    target_date = _latest_completed_sw_trade_date()
    for name, symbol in SW_CODE_MAPPING.items():
        try:
            raw = ak.index_hist_sw(symbol=symbol, period="day")
            if raw.empty:
                continue
            raw, source_label = _append_sw_current_if_trend_lags(raw, name, target_date)
            raw = raw.tail(60)
            raw = raw.rename(columns={k: v for k, v in base_rename.items() if k in raw.columns})
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            for col in ["close", "open", "high", "low", "volume", "amount"]:
                if col in raw.columns:
                    raw[col] = pd.to_numeric(raw[col], errors="coerce")
            raw = raw.dropna(subset=["close", "amount", "volume"]).sort_values("date").reset_index(drop=True)
            if len(raw) >= 20:
                histories[name] = raw
        except Exception:
            continue
    return histories


def _summarize_backtest(bt: pd.DataFrame) -> Dict[str, float]:
    if bt.empty:
        return {}

    bt = bt.copy()
    bt["strategy_nav"] = (1 + bt["strategy_ret"]).cumprod()
    bt["benchmark_nav"] = (1 + bt["benchmark_ret"]).cumprod()

    def _max_drawdown(nav: pd.Series) -> float:
        dd = nav / nav.cummax() - 1.0
        return float(dd.min())

    periods = max(len(bt), 1)
    return {
        "交易日数": float(periods),
        "胜率": float(bt["direction_hit"].mean()),
        "相对胜率": float(bt["relative_hit"].mean()),
        "累计收益": float(bt["strategy_nav"].iloc[-1] - 1.0),
        "基准收益": float(bt["benchmark_nav"].iloc[-1] - 1.0),
        "年化收益": float(bt["strategy_nav"].iloc[-1] ** (252 / periods) - 1.0),
        "最大回撤": _max_drawdown(bt["strategy_nav"]),
        "夏普比率": float((bt["strategy_ret"].mean() / (bt["strategy_ret"].std(ddof=0) + 1e-9)) * np.sqrt(252)),
    }


def _build_strategy_backtest(scored: pd.DataFrame, lookback_days: int, strategy: str) -> pd.DataFrame:
    bt_rows = []
    for dt, day in scored.groupby("date", sort=True):
        day = day.copy()
        breadth = float((day["涨跌幅"] > 0).mean()) if "涨跌幅" in day.columns else 0.5
        bench_ret = float(day["next_ret"].mean())

        if strategy == "top1":
            tradable = day[day["逃顶风险简分"] < 78].copy()
            top_n = 1
            min_score = 0.0
            breadth_floor = 0.0
        elif strategy == "top3_balanced":
            tradable = day[day["逃顶风险简分"] < 72].copy()
            top_n = 3
            min_score = 58.0
            breadth_floor = 0.0
        elif strategy == "top3_regime":
            tradable = day[day["逃顶风险简分"] < 72].copy()
            top_n = 3
            min_score = 58.0
            breadth_floor = 0.38
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        if breadth < breadth_floor:
            picks = pd.DataFrame()
        else:
            picks = (
                tradable[tradable["综合博弈得分"] >= min_score]
                .sort_values("综合博弈得分", ascending=False)
                .head(top_n)
            )

        if picks.empty:
            strategy_ret = 0.0
            names = "空仓"
            top_score = 0.0
            risk = 0.0
        else:
            strategy_ret = float(picks["next_ret"].mean())
            names = " / ".join(picks["板块名称"].astype(str).tolist())
            top_score = float(picks["综合博弈得分"].mean())
            risk = float(picks["逃顶风险简分"].mean())

        bt_rows.append({
            "date": dt,
            "持有板块": names,
            "综合博弈得分": top_score,
            "风险简分": risk,
            "市场广度": breadth * 100,
            "strategy_ret": strategy_ret,
            "benchmark_ret": bench_ret,
            "direction_hit": strategy_ret > 0,
            "relative_hit": strategy_ret > bench_ret,
        })

    bt = pd.DataFrame(bt_rows).sort_values("date").tail(lookback_days).reset_index(drop=True)
    if bt.empty:
        return bt
    bt["strategy_nav"] = (1 + bt["strategy_ret"]).cumprod()
    bt["benchmark_nav"] = (1 + bt["benchmark_ret"]).cumprod()
    return bt


@st.cache_data(ttl=3600 * 12, show_spinner=False)
def _build_walk_forward_scores(lookback_days: int = 520) -> pd.DataFrame:
    """构建无未来函数的逐日截面得分池。"""
    histories: Dict[str, pd.DataFrame] = {}
    base_rename = {
        "日期": "date", "收盘": "close", "开盘": "open",
        "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
    }
    for name, symbol in SW_CODE_MAPPING.items():
        try:
            raw = ak.index_hist_sw(symbol=symbol, period="day").tail(lookback_days + 80)
            if raw.empty:
                continue
            raw = raw.rename(columns={k: v for k, v in base_rename.items() if k in raw.columns})
            needed = ["date", "close", "open", "high", "low", "volume", "amount"]
            if not all(c in raw.columns for c in needed):
                continue
            raw = raw[needed].copy()
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            for col in needed[1:]:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
            raw = raw.dropna().sort_values("date").reset_index(drop=True)
            if len(raw) >= 90:
                histories[name] = raw
        except Exception:
            continue

    rows = []
    for name, hist in histories.items():
        close = hist["close"].to_numpy(dtype=float)
        open_ = hist["open"].to_numpy(dtype=float)
        high = hist["high"].to_numpy(dtype=float)
        low = hist["low"].to_numpy(dtype=float)
        amount = hist["amount"].to_numpy(dtype=float)
        n = len(hist)
        for i in range(60, n - 1):
            prev_close = close[i - 1]
            pct_today = (close[i] / (prev_close + 1e-9) - 1.0) * 100.0
            ret_window = np.diff(close[i - 20:i + 1]) / (close[i - 20:i] + 1e-9)
            price_std = max(float(np.std(ret_window, ddof=0)), 1e-6)
            recent_20 = close[i - 19:i + 1]
            dyn_pos = (close[i] - recent_20.min()) / (recent_20.max() - recent_20.min() + 1e-9) * 100.0
            last6 = np.diff(close[i - 6:i + 1]) / (close[i - 6:i] + 1e-9) * 100.0
            accel = float(last6[-3:].mean() - last6[:3].mean())

            amount_mean = max(float(np.mean(amount[i - 20:i])), 1e-6)
            abnormal = amount[i] / amount_mean
            amplitude = (high[i] - low[i]) / (open_[i] + 1e-9)
            buy_power = (close[i] - low[i]) / (high[i] - low[i] + 1e-9)
            upper_shadow = (high[i] - max(open_[i], close[i])) / (high[i] - low[i] + 1e-9)
            trap = abnormal - (pct_today / (price_std * 100.0)) + upper_shadow * 5.0
            efficiency = amount[i] / (abs(pct_today) + 0.1)

            ma20 = float(np.mean(close[i - 19:i + 1]))
            ma60 = float(np.mean(close[i - 59:i + 1]))
            ret20 = close[i] / (close[i - 20] + 1e-9) - 1.0
            up_days = float(np.mean(np.diff(close[i - 19:i + 1]) > 0))
            mid_confirm = (
                0.45 * (ret20 / (price_std + 1e-6)) +
                0.25 * (close[i] / (ma20 + 1e-9) - 1.0) * 100 +
                0.20 * (ma20 / (ma60 + 1e-9) - 1.0) * 100 +
                0.10 * (up_days - 0.5) * 10
            )

            rows.append({
                "date": hist["date"].iloc[i],
                "板块名称": name,
                "next_ret": close[i + 1] / (close[i] + 1e-9) - 1.0,
                "涨跌幅": pct_today,
                "trend_raw": (pct_today * 0.6 + (dyn_pos - 50) / 50 * 0.2 + accel * 0.2) / (price_std * 100.0),
                "fund_raw": amplitude * 0.4 + buy_power * 0.4 + abnormal * 0.2,
                "abnormal_raw": abnormal,
                "trap_raw": trap,
                "efficiency_raw": efficiency,
                "mid_confirm_raw": mid_confirm,
                "动态水位": dyn_pos,
                "上影线诱多率": upper_shadow * 100.0,
            })

    panel = pd.DataFrame(rows)
    if panel.empty:
        return pd.DataFrame()

    scored_days = []
    raw_cols = ["trend_raw", "fund_raw", "abnormal_raw", "trap_raw", "efficiency_raw", "mid_confirm_raw"]
    for dt, day in panel.groupby("date", sort=True):
        if len(day) < 20:
            continue
        day = day.copy()
        for col in raw_cols:
            day[col] = winsorize(pd.to_numeric(day[col], errors="coerce"))
        day["第1层_真实趋势"] = rank_score(day["trend_raw"])
        day["第2层_真假资金"] = rank_score(day["fund_raw"])
        day["第3层_异动干预"] = rank_score(day["abnormal_raw"])
        day["第4层_诱多诱空"] = 100 - rank_score(day["trap_raw"])
        day["第5层_博弈反身"] = rank_score(day["efficiency_raw"])
        day["第6层_中期确认"] = rank_score(day["mid_confirm_raw"])
        day["综合博弈得分"] = (
            day["第1层_真实趋势"] * 0.14 +
            day["第2层_真假资金"] * 0.18 +
            day["第3层_异动干预"] * 0.14 +
            day["第4层_诱多诱空"] * 0.26 +
            day["第5层_博弈反身"] * 0.18 +
            day["第6层_中期确认"] * 0.10
        )
        day["逃顶风险简分"] = (day["动态水位"] * 0.45 + (100 - day["第4层_诱多诱空"]) * 0.55).clip(0, 100)
        scored_days.append(day)

    scored = pd.concat(scored_days, ignore_index=True) if scored_days else pd.DataFrame()
    if scored.empty:
        return pd.DataFrame()
    return scored


@st.cache_data(ttl=3600 * 12, show_spinner=False)
def run_walk_forward_backtest(
    lookback_days: int = 520,
    strategy: str = "top3_balanced",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """用免费申万行业日线做滚动验证：今日收盘打分，下一交易日验证。"""
    scored = _build_walk_forward_scores(lookback_days)
    if scored.empty:
        return pd.DataFrame(), {}

    bt = _build_strategy_backtest(scored, lookback_days, strategy)
    return bt, _summarize_backtest(bt)


@st.cache_data(ttl=3600 * 12, show_spinner=False)
def compare_walk_forward_strategies(lookback_days: int = 520) -> pd.DataFrame:
    scored = _build_walk_forward_scores(lookback_days)
    if scored.empty:
        return pd.DataFrame()

    rows = []
    strategy_names = {
        "top1": "Top1 单押",
        "top3_balanced": "Top3 稳健等权",
        "top3_regime": "Top3 + 广度过滤",
    }
    for strategy, label in strategy_names.items():
        bt = _build_strategy_backtest(scored, lookback_days, strategy)
        summary = _summarize_backtest(bt)
        if not summary:
            continue
        rows.append({
            "策略": label,
            "方向胜率": summary["胜率"],
            "相对胜率": summary["相对胜率"],
            "累计收益": summary["累计收益"],
            "等权基准": summary["基准收益"],
            "年化收益": summary["年化收益"],
            "最大回撤": summary["最大回撤"],
            "夏普比率": summary["夏普比率"],
            "交易日数": summary["交易日数"],
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600 * 12, show_spinner=False)
def compare_walk_forward_windows(
    windows: Tuple[int, ...] = (120, 240, 360, 520),
    strategy: str = "top3_balanced",
) -> pd.DataFrame:
    max_window = max(windows) if windows else 520
    scored = _build_walk_forward_scores(max_window)
    if scored.empty:
        return pd.DataFrame()

    rows = []
    for window in windows:
        bt = _build_strategy_backtest(scored, int(window), strategy)
        summary = _summarize_backtest(bt)
        if not summary:
            continue
        rows.append({
            "窗口": f"{int(window)}日",
            "方向胜率": summary["胜率"],
            "相对胜率": summary["相对胜率"],
            "累计收益": summary["累计收益"],
            "等权基准": summary["基准收益"],
            "年化收益": summary["年化收益"],
            "最大回撤": summary["最大回撤"],
            "夏普比率": summary["夏普比率"],
        })
    return pd.DataFrame(rows)


def _build_3d_scores(all_hist: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    三维共振-熵减模型核心计算。
    返回 DataFrame：板块名称, 逃顶风险分, 入场共振分, 三维共振信号, 动态水位_3d, 市场环境, market_heat
    """
    if not all_hist:
        return pd.DataFrame()

    results = []
    turnover_list = []

    for name, raw in all_hist.items():
        close  = raw["close"]
        high   = raw["high"]
        low    = raw["low"]
        open_  = raw["open"]
        amount = raw["amount"]
        volume = raw["volume"]

        pct_chg      = close.pct_change() * 100
        amp          = (high - low) / (close + 1e-9)
        upper_shadow = (high - np.maximum(open_, close)) / (high - low + 1e-9)

        amount_ma20  = amount.rolling(20).mean()
        vol_ma20     = volume.rolling(20).mean()
        pct_std10    = pct_chg.rolling(10).std(ddof=0)
        price_std20  = pct_chg.rolling(20).std(ddof=0)

        mf_raw      = (amount / (amount_ma20 + 1e-9)) * (1 - amp)
        pe_raw      = (pct_chg * amount_ma20) / (amount + 1e-9)
        ema3        = close.ewm(span=3, adjust=False).mean()
        ema8        = close.ewm(span=8, adjust=False).mean()
        ema20       = close.ewm(span=20, adjust=False).mean()
        vd_raw      = (ema3 - ema8) - (ema8 - ema20)
        ah_raw      = upper_shadow + (volume / (vol_ma20 + 1e-9) - pct_chg / 100)
        cs_raw      = (1.0 / (pct_std10 + 1e-6)).clip(upper=50)
        min60       = close.rolling(60, min_periods=10).min()
        max60       = close.rolling(60, min_periods=10).max()
        gr_raw      = (close - min60) / (max60 - min60 + 1e-9)
        entropy_raw = price_std20 * (1 - cs_raw / 50)

        def _fused(s: pd.Series, w: float = 0.65) -> float:
            t5  = s.tail(5).dropna()
            t20 = s.tail(20).dropna()
            v5  = float(t5.iloc[-1])  if not t5.empty  else 0.0
            v20 = float(t20.mean())   if not t20.empty else 0.0
            return w * v5 + (1 - w) * v20

        def _last(s: pd.Series) -> float:
            t = s.tail(20).dropna()
            return float(t.iloc[-1]) if not t.empty else 0.0

        latest_turnover = float(amount.iloc[-1]) if not amount.empty else 1.0

        results.append({
            "板块名称":    name,
            "mf":          _fused(mf_raw),
            "pe":          _fused(pe_raw),
            "vd":          _fused(vd_raw),
            "ah":          _fused(ah_raw),
            "cs":          _last(cs_raw),
            "gr":          _last(gr_raw),
            "entropy":     _last(entropy_raw),
            "动态水位_3d": float(gr_raw.iloc[-1] * 100) if not gr_raw.empty else 50.0,
        })
        turnover_list.append({"板块名称": name, "latest_turnover": latest_turnover})

    if not results:
        return pd.DataFrame()

    score_df    = pd.DataFrame(results)
    turnover_df = pd.DataFrame(turnover_list)

    for col in ["mf", "pe", "vd", "ah", "cs", "gr", "entropy"]:
        score_df[col] = _winsorize_zscore(score_df[col])

    score_df    = score_df.merge(turnover_df, on="板块名称", how="left")
    total_t     = score_df["latest_turnover"].sum()
    weights     = score_df["latest_turnover"] / (total_t + 1e-9)
    market_heat = float(np.average(score_df["mf"], weights=weights))
    is_bear     = market_heat < -0.8

    score_df["逃顶风险分_raw"] = (
        0.25 * score_df["gr"]      +
        0.20 * score_df["ah"]      +
        0.20 * score_df["vd"]      +
        0.15 * score_df["entropy"] -
        0.10 * score_df["mf"]      -
        0.10 * score_df["pe"]
    )
    score_df["入场共振分_raw"] = (
        0.30 * score_df["mf"]      +
        0.25 * score_df["cs"]      -
        0.25 * score_df["gr"]      -
        0.20 * score_df["entropy"]
    )
    if is_bear:
        score_df["入场共振分_raw"] *= 0.3

    score_df["逃顶风险分"] = (_calc_zscore(score_df["逃顶风险分_raw"]) * 50 + 50).clip(0, 100).round(1)
    score_df["入场共振分"] = (_calc_zscore(score_df["入场共振分_raw"]) * 50 + 50).clip(0, 100).round(1)

    def _get_signal(row: pd.Series) -> str:
        if row["逃顶风险分"] >= 85 or row["vd"] > 1.5:
            return "🔴 终极逃顶（高风险）"
        if row["逃顶风险分"] >= 70:
            return "🟠 高位能量耗尽（大幅减仓）"
        if row["入场共振分"] >= 75 and not is_bear:
            return "🟢 低位共振（可重仓潜伏）"
        if row["逃顶风险分"] >= 55:
            return "🟡 预警（观察减持）"
        return "⚪ 安全持仓"

    score_df["三维共振信号"] = score_df.apply(_get_signal, axis=1)
    score_df["市场环境"]     = "熊市" if is_bear else ("牛市" if market_heat > 0.8 else "震荡市")
    score_df["market_heat"]  = round(market_heat, 3)

    keep = ["板块名称", "逃顶风险分", "入场共振分", "三维共振信号", "动态水位_3d", "市场环境", "market_heat"]
    return score_df[keep].reset_index(drop=True)


# ================= 5. 板块详情页专用：历史 K 线 + 逐日打分 =================

@st.cache_data(ttl=3600 * 4, show_spinner=False)
def fetch_sector_history(sector_name: str, lookback_days: int = 120) -> pd.DataFrame:
    """获取指定申万一级行业真实历史 K 线，并逐日计算六层博弈得分。"""
    code = SW_CODE_MAPPING.get(sector_name)
    if not code:
        return pd.DataFrame()

    try:
        df_raw = ak.index_hist_sw(symbol=code, period="day")
        if df_raw.empty:
            return pd.DataFrame()
        df_raw, source_label = _append_sw_current_if_trend_lags(df_raw, sector_name)
        df_raw = df_raw.tail(lookback_days + 60)
    except Exception:
        return pd.DataFrame()

    rename_map: Dict[str, str] = {}
    for col in df_raw.columns:
        if "日期" in col or col.lower() == "date":
            rename_map[col] = "日期"
        elif col in ("收盘", "close"):
            rename_map[col] = "收盘价"
        elif col in ("开盘", "open"):
            rename_map[col] = "开盘价"
        elif col in ("最高", "high"):
            rename_map[col] = "最高价"
        elif col in ("最低", "low"):
            rename_map[col] = "最低价"
        elif col in ("成交量", "volume"):
            rename_map[col] = "成交量"
        elif col in ("成交额", "amount"):
            rename_map[col] = "成交额"
    df_raw = df_raw.rename(columns=rename_map)

    needed = ["日期", "收盘价", "开盘价", "最高价", "最低价", "成交量", "成交额"]
    for c in needed:
        if c not in df_raw.columns:
            return pd.DataFrame()

    cols = needed + (["数据来源"] if "数据来源" in df_raw.columns else [])
    df = df_raw[cols].copy()
    df["日期"] = pd.to_datetime(df["日期"])
    for c in needed[1:]:
        df[c] = safe_numeric(df[c])
    df = df.dropna(subset=["收盘价", "成交量", "成交额"]).sort_values("日期").reset_index(drop=True)
    if "数据来源" not in df.columns:
        df["数据来源"] = source_label
    df["数据日期"] = df["日期"].dt.strftime("%Y-%m-%d") + "（" + df["数据来源"].astype(str) + "）"
    if len(df) < 25:
        return pd.DataFrame()

    closes  = df["收盘价"].to_numpy(dtype=float)
    volumes = df["成交量"].to_numpy(dtype=float)
    amounts = df["成交额"].to_numpy(dtype=float)
    highs   = df["最高价"].to_numpy(dtype=float)
    lows    = df["最低价"].to_numpy(dtype=float)
    opens   = df["开盘价"].to_numpy(dtype=float)
    n = len(df)

    pct_chg      = np.full(n, np.nan)
    dyn_pos_arr  = np.full(n, np.nan)
    accel_arr    = np.full(n, np.nan)
    shadow_arr   = np.full(n, np.nan)
    price_std_arr = np.full(n, np.nan)
    vol_mean_arr = np.full(n, np.nan)
    vol_std_arr  = np.full(n, np.nan)
    amount_ma20  = np.full(n, np.nan)
    mid_confirm_raw_arr = np.full(n, np.nan)

    for i in range(1, n):
        pct_chg[i] = (closes[i] / closes[i - 1] - 1.0) * 100.0 if closes[i - 1] != 0 else 0.0
        win20 = max(0, i - 19)
        win6  = max(0, i - 5)
        win3  = max(0, i - 2)

        c20 = closes[win20: i + 1]
        dyn_pos_arr[i] = (closes[i] - c20.min()) / (c20.max() - c20.min() + 1e-6) * 100.0

        if i >= 6:
            rets = np.diff(closes[win6: i + 1]) / (closes[win6: i] + 1e-9) * 100.0
            if len(rets) >= 6:
                accel_arr[i] = rets[-3:].mean() - rets[-6:-3].mean()

        h3 = highs[win3: i + 1];  l3 = lows[win3: i + 1]
        o3 = opens[win3: i + 1];  c3 = closes[win3: i + 1]
        if len(h3) >= 1:
            shadow_arr[i] = float(np.mean(
                [(h - max(o, c)) / (h - l + 1e-9) for h, l, o, c in zip(h3, l3, o3, c3)]
            ))

        c20v = closes[win20: i]; v20 = volumes[win20: i]; a20 = amounts[win20: i]
        if len(c20v) >= 2:
            price_std_arr[i] = max(float(np.std(np.diff(c20v) / (c20v[:-1] + 1e-9), ddof=0)), 1e-6)
        if len(v20) >= 2:
            vol_mean_arr[i] = float(np.mean(v20))
            vol_std_arr[i]  = max(float(np.std(v20, ddof=0)), 1e-6)
        if len(a20) > 0:
            amount_ma20[i] = float(np.mean(a20))
        if i >= 20:
            c20_confirm = closes[i - 19: i + 1]
            c60_confirm = closes[max(0, i - 59): i + 1]
            ret20 = closes[i] / (closes[i - 20] + 1e-9) - 1.0
            vol20 = np.std(np.diff(c20_confirm) / (c20_confirm[:-1] + 1e-9), ddof=0)
            ma20 = np.mean(c20_confirm)
            ma60 = np.mean(c60_confirm)
            up_days = np.mean(np.diff(c20_confirm) > 0)
            mid_confirm_raw_arr[i] = (
                0.45 * (ret20 / (vol20 + 1e-6)) +
                0.25 * (closes[i] / (ma20 + 1e-9) - 1.0) * 100 +
                0.20 * (ma20 / (ma60 + 1e-9) - 1.0) * 100 +
                0.10 * (up_days - 0.5) * 10
            )

    ps = np.where(np.isnan(price_std_arr), 0.01, price_std_arr)
    vm = np.where(np.isnan(vol_mean_arr),  volumes, vol_mean_arr)
    vs = np.where(np.isnan(vol_std_arr),   1.0,    vol_std_arr)
    am = np.where(np.isnan(amount_ma20),   amounts, amount_ma20)
    dy = np.where(np.isnan(dyn_pos_arr),   50.0,   dyn_pos_arr)
    ac = np.where(np.isnan(accel_arr),     0.0,    accel_arr)
    sh = np.where(np.isnan(shadow_arr),    0.0,    shadow_arr)

    df["涨跌幅"]      = pct_chg
    df["动态水位"]    = dy
    df["趋势加速度"]  = ac
    df["上影线诱多率"] = sh * 100.0

    amp_arr      = (highs - lows) / (opens + 1e-9)
    buy_pow      = (closes - lows) / (highs - lows + 1e-9)
    fund_raw_arr = (
        pd.Series(amp_arr).rank(pct=True).values * 0.4 +
        pd.Series(buy_pow).rank(pct=True).values * 0.4 +
        pd.Series(amounts).rank(pct=True).values * 0.2
    ) * 2.0 + 0.5

    df["trend_raw"]      = (pct_chg * 0.6 + (dy - 50) / 50 * 0.2 + ac * 0.2) / (ps * 100.0)
    df["fund_raw"]       = fund_raw_arr
    df["abnormal_raw"]   = (volumes - vm) / vs
    df["trap_raw"]       = df["abnormal_raw"] - (pct_chg / (ps * 100.0)) + sh * 5.0
    df["efficiency_raw"] = amounts / (np.abs(pct_chg) + 0.1)
    df["mid_confirm_raw"] = np.where(np.isnan(mid_confirm_raw_arr), 0.0, mid_confirm_raw_arr)

    for col in ["trend_raw", "fund_raw", "abnormal_raw", "trap_raw", "efficiency_raw", "mid_confirm_raw"]:
        df[col] = winsorize(pd.to_numeric(df[col], errors="coerce"))

    df["第1层_真实趋势"] = rank_score(df["trend_raw"])
    df["第2层_真假资金"] = rank_score(df["fund_raw"])
    df["第3层_异动干预"] = rank_score(df["abnormal_raw"])
    df["第4层_诱多诱空"] = 100 - rank_score(df["trap_raw"])
    df["第5层_博弈反身"] = rank_score(df["efficiency_raw"])
    df["第6层_中期确认"] = rank_score(df["mid_confirm_raw"])

    df["综合博弈得分"] = (
        df["第1层_真实趋势"] * 0.14 +
        df["第2层_真假资金"] * 0.18 +
        df["第3层_异动干预"] * 0.14 +
        df["第4层_诱多诱空"] * 0.26 +
        df["第5层_博弈反身"] * 0.18 +
        df["第6层_中期确认"] * 0.10
    ).round(1)

    out_cols = [
        "日期", "数据日期", "数据来源", "收盘价", "开盘价", "最高价", "最低价", "成交量", "成交额",
        "涨跌幅", "动态水位", "趋势加速度", "上影线诱多率",
        "第1层_真实趋势", "第2层_真假资金", "第3层_异动干预",
        "第4层_诱多诱空", "第5层_博弈反身", "第6层_中期确认", "综合博弈得分",
        "trend_raw", "fund_raw", "abnormal_raw", "trap_raw", "efficiency_raw",
        "mid_confirm_raw",
    ]
    return df.tail(lookback_days).reset_index(drop=True)[out_cols]


# ================= 6. 成分股实时行情 =================

@st.cache_data(ttl=120, show_spinner=False)
def fetch_sector_top_stocks(
    sector_name: str,
    sw_mapping: Dict[str, List[str]],
    spot_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """获取板块龙头/权重股实时行情（优先东财接口，降级申万快照）。"""
    rows = []
    try:
        cons_em  = ak.stock_board_industry_cons_em(symbol=sector_name)
        if not cons_em.empty:
            code_col   = next((c for c in ["代码", "股票代码"] if c in cons_em.columns), None)
            name_col   = next((c for c in ["名称", "股票名称"] if c in cons_em.columns), None)
            pct_col    = next((c for c in ["涨跌幅", "最新涨跌幅"] if c in cons_em.columns), None)
            inflow_col = next((c for c in ["主力净流入", "主力净额"] if c in cons_em.columns), None)
            if code_col and pct_col:
                sub = cons_em[[code_col, name_col or code_col, pct_col]].copy()
                sub.columns = ["代码", "名称", "涨跌幅"]
                sub["主力净流入"] = (
                    pd.to_numeric(cons_em[inflow_col], errors="coerce") / 1e8
                    if inflow_col else np.nan
                )
                sub["涨跌幅"] = pd.to_numeric(sub["涨跌幅"], errors="coerce")
                rows = (
                    sub.dropna(subset=["涨跌幅"])
                    .sort_values("涨跌幅", ascending=False)
                    .head(top_n)
                    .to_dict("records")
                )
    except Exception:
        pass

    if not rows and sector_name in sw_mapping and not spot_df.empty:
        pct_map = spot_df.set_index("代码")["涨跌幅"].to_dict()
        data = sorted(
            [(c, pct_map[c]) for c in sw_mapping[sector_name] if c in pct_map],
            key=lambda x: x[1], reverse=True,
        )
        for c, p in data[:top_n]:
            rows.append({"代码": c, "名称": c, "涨跌幅": p, "主力净流入": np.nan})

    if not rows:
        return pd.DataFrame()

    res = pd.DataFrame(rows)

    def _tag(p: float, inflow) -> str:
        if pd.isna(p):      return "数据缺失"
        if p > 5:           return "强势突破"
        if p > 2:           return "温和放量" if pd.isna(inflow) or inflow > 0 else "放量滞涨"
        if p > 0:           return "横盘偏强"
        if p > -2:          return "缩量回踩"
        return "弱势下跌"

    res["技术形态"]      = res.apply(lambda r: _tag(r["涨跌幅"], r["主力净流入"]), axis=1)
    res["今日涨幅"]      = res["涨跌幅"].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
    res["主力净流入(亿)"] = res["主力净流入"].apply(lambda x: f"{x:+.2f}" if not pd.isna(x) else "—")
    return res[["代码", "名称", "今日涨幅", "主力净流入(亿)", "技术形态"]].reset_index(drop=True)


# ================= 7. 主引擎：全截面博弈 + 三维共振 + 宏微观一票否决 =================

@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_processed_sw_data() -> pd.DataFrame:
    """
    返回全截面 DataFrame，核心字段：
    综合博弈得分、六层得分、信号状态（活力模型）
    逃顶风险分、入场共振分、三维共振信号（三维共振-熵减模型）
    终极信号、交易指引（宏微观双模共振·风险一票否决制）
    fund_raw（供轮动分析页使用，不清理）
    """
    cached_df = _read_processed_disk_cache()
    if not cached_df.empty:
        return cached_df

    hist_bases    = fetch_historical_baselines()
    sw_mapping    = fetch_sw_constituents_mapping()
    a_shares_spot = fetch_all_a_shares_spot()

    if not hist_bases:
        return pd.DataFrame()

    stock_pct_map = (
        a_shares_spot.set_index("代码")["涨跌幅"].to_dict()
        if not a_shares_spot.empty else {}
    )

    # ── 构建截面行情 ──
    rows = []
    for name, base in hist_bases.items():
        p_close  = base.get("past_close", 0.0)
        l_close  = base.get("last_close", 0.0)
        if p_close == 0:
            continue
        pct_today = (l_close / p_close - 1.0) * 100.0

        l_open   = base.get("last_open",   l_close)
        l_high   = base.get("last_high",   l_close)
        l_low    = base.get("last_low",    l_close)
        amt_today = base.get("last_amount", 0.0)

        rising_ratio, median_pct, micro_factor, micro_tag = 0.0, 0.0, 1.0, "结构正常"
        coverage_count = 0
        if name in sw_mapping and stock_pct_map:
            pcts = np.array([stock_pct_map[c] for c in sw_mapping[name] if c in stock_pct_map])
            coverage_count = int(len(pcts))
            if len(pcts) > 5:
                rising_ratio = float((pcts > 0).mean() * 100)
                median_pct   = float(np.median(pcts))
                if pct_today > 0.5 and (rising_ratio < 40 or (pct_today - median_pct) > 1.2):
                    micro_factor, micro_tag = 0.6, "权重掩护(失真)"
                elif pct_today > 1.0 and rising_ratio > 75:
                    micro_factor, micro_tag = 1.15, "全员共振(极强)"

        rows.append({
            "板块名称":    name,
            "对应ETF":     SW_ETF_MAPPING.get(name, "暂无"),
            "数据日期":    f'{base.get("data_date", "")}（{base.get("data_source", SW_TREND_SOURCE_LABEL)}）',
            "开盘价": l_open, "最高价": l_high, "最低价": l_low, "收盘价": l_close,
            "成交额":      amt_today,
            "涨跌幅":      pct_today,
            "中位涨幅":    median_pct,
            "上涨占比":    rising_ratio,
            "成分股覆盖数":  coverage_count,
            "微观标签":    micro_tag,
            "动态水位":    base.get("dyn_pos",         50.0),
            "趋势加速度":  base.get("acceleration",    0.0),
            "资金流向":    base.get("flow",             0.0),
            "金额流向":    base.get("amount_flow",      base.get("flow", 0.0)),
            "上影线诱多率": base.get("avg_shadow_ratio", 0.0) * 100.0,
            "vol_mean":    base.get("vol_mean",         1.0),
            "price_std":   base.get("price_std",        0.01),
            "amount_mean": base.get("amount_mean",      max(amt_today, 1.0)),
            "amount_std":  base.get("amount_std",       1.0),
            "mid_confirm_raw": base.get("mid_confirm_raw", 0.0),
            "micro_factor": micro_factor,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ── 活力合成 ──
    df["amplitude"] = (df["最高价"] - df["最低价"]) / (df["开盘价"] + 1e-9)
    df["buy_power"] = (df["收盘价"] - df["最低价"]) / (df["最高价"] - df["最低价"] + 1e-9)
    df["fund_raw"]  = (
        df["amplitude"].rank(pct=True) * 0.4 +
        df["buy_power"].rank(pct=True) * 0.4 +
        df["成交额"].rank(pct=True)   * 0.2
    ) * 2.0 + 0.5

    df["trend_raw"]      = (
        df["涨跌幅"] * 0.6 +
        (df["动态水位"] - 50) / 50 * 0.2 +
        df["趋势加速度"] * 0.2
    ) / (df["price_std"] * 100.0)
    # 成交额只能与成交额均值比较；旧版用成交额/成交量均值，量纲不一致，会扭曲异动和诱多分。
    df["abnormal_raw"]   = df["成交额"] / (df["amount_mean"] + 1e-9)
    df["trap_raw"]       = (
        df["abnormal_raw"]
        - (df["涨跌幅"] / (df["price_std"] * 100.0))
        + (df["上影线诱多率"] / 10.0)
    )
    df["efficiency_raw"] = df["成交额"] / (df["涨跌幅"].abs() + 0.1)

    for col in ["trend_raw", "fund_raw", "abnormal_raw", "trap_raw", "efficiency_raw", "mid_confirm_raw"]:
        df[col] = winsorize(pd.to_numeric(df[col], errors="coerce"))

    df["第1层_真实趋势"] = rank_score(df["trend_raw"])
    df["第2层_真假资金"] = rank_score(df["fund_raw"])
    df["第3层_异动干预"] = rank_score(df["abnormal_raw"])
    df["第4层_诱多诱空"] = 100 - rank_score(df["trap_raw"])
    df["第5层_博弈反身"] = rank_score(df["efficiency_raw"])
    df["第6层_中期确认"] = rank_score(df["mid_confirm_raw"])

    regime_factor = _calc_market_regime(df)[1]
    df["综合博弈得分"] = (
        (
            df["第1层_真实趋势"] * 0.14 +
            df["第2层_真假资金"] * 0.18 +
            df["第3层_异动干预"] * 0.14 +
            df["第4层_诱多诱空"] * 0.26 +
            df["第5层_博弈反身"] * 0.18 +
            df["第6层_中期确认"] * 0.10
        ) * regime_factor * df["micro_factor"]
    ).clip(0, 100).round(1)

    # ── 六层信号状态（微观日内） ──
    def _sig(r: pd.Series) -> str:
        if r["微观标签"] == "权重掩护(失真)":
            return "指数失真(防骗)"
        if r["动态水位"] > 80 and r["第4层_诱多诱空"] < 40:
            return "高位派发(快跑)"
        if (r["综合博弈得分"] >= 70 and r["第4层_诱多诱空"] >= 60
                and r["动态水位"] < 70 and r["趋势加速度"] > 0 and r["资金流向"] > 0):
            return "真·主升初期"
        if r["综合博弈得分"] >= 65 and r["第1层_真实趋势"] >= 60:
            return "主升候选"
        if r["上影线诱多率"] > 40 and r["第4层_诱多诱空"] < 50:
            return "画线骗炮(规避)"
        return "震荡观察"

    df["信号状态"] = df.apply(_sig, axis=1)

    # ── 三维共振-熵减模型 ──
    all_hist = fetch_all_sector_histories()
    score_3d = _build_3d_scores(all_hist)

    if not score_3d.empty:
        df = df.merge(score_3d, on="板块名称", how="left")
        df["逃顶风险分"]  = df["逃顶风险分"].fillna(50.0)
        df["入场共振分"]  = df["入场共振分"].fillna(50.0)
        df["三维共振信号"] = df["三维共振信号"].fillna("⚪ 安全持仓")
        df["市场环境"]    = df["市场环境"].fillna("震荡市")
        df["market_heat"] = df["market_heat"].fillna(0.0)
        df["动态水位_3d"] = df["动态水位_3d"].fillna(df["动态水位"])
    else:
        df["逃顶风险分"]  = (df["动态水位"] / 100.0 * 50 + (100 - df["第4层_诱多诱空"]) / 100 * 50).round(1)
        df["入场共振分"]  = ((df["综合博弈得分"] + df["第2层_真假资金"]) / 2).round(1)
        df["三维共振信号"] = df["逃顶风险分"].apply(
            lambda x: "🔴 终极逃顶（高风险）" if x >= 85
            else ("🟠 高位能量耗尽（大幅减仓）" if x >= 70 else "⚪ 安全持仓")
        )
        df["市场环境"]    = "数据不足"
        df["market_heat"] = 0.0
        df["动态水位_3d"] = df["动态水位"]

    # ================= 🌟 宏微观双模共振交易引擎（风险一票否决制）=================

    # 纯粹微观日内战术信号（保留在内部用于决策，不暴露到最终列）
    def _micro_sig(r: pd.Series) -> str:
        if r.get("微观标签", "") == "权重掩护(失真)":
            return "指数失真(防骗)"
        if r.get("动态水位", 50) > 80 and r.get("第4层_诱多诱空", 50) < 40:
            return "高位派发(快跑)"
        if r.get("上影线诱多率", 0) > 40 and r.get("第4层_诱多诱空", 50) < 50:
            return "画线骗炮(规避)"
        if (r.get("综合博弈得分", 0) >= 70 and r.get("第4层_诱多诱空", 0) >= 60
                and r.get("动态水位", 50) < 70
                and r.get("趋势加速度", 0) > 0 and r.get("资金流向", 0) > 0):
            return "真·主升初期"
        if r.get("综合博弈得分", 0) >= 65 and r.get("第1层_真实趋势", 0) >= 60:
            return "主升候选"
        return "平淡震荡"

    df["微观日内状态"] = df.apply(_micro_sig, axis=1)

    # 瀑布流四道防线决策引擎
    def _ultimate_decision(row: pd.Series) -> pd.Series:
        macro   = row.get("三维共振信号", "数据不足")
        micro   = row.get("微观日内状态", "平淡震荡")
        dyn_pos = row.get("动态水位_3d", 50.0)

        # 🛑 第一顺位：微观致命风险（无视宏观）
        if micro == "高位派发(快跑)":
            return pd.Series(["🚨 强制清仓",   "【微观否决】高位放量风险显著，优先降风险，避免继续追高。"])
        if micro == "指数失真(防骗)":
            return pd.Series(["🦊 诱多陷阱",   "【微观否决】指数强于成分股广度，追高胜率下降，等待确认。"])

        # 🛑 第二顺位：宏观致命风险（无视日内大涨）
        if "终极逃顶" in macro:
            if micro in ["真·主升初期", "主升候选"]:
                return pd.Series(["💣 鱼尾诱多", "【宏观否决】大周期风险已抬升，日内强势更适合减仓观察。"])
            return pd.Series(["🛑 崩盘前夕",   "【宏观否决】中期风险显著抬升，优先控制仓位和回撤。"])
        if "能量耗尽" in macro:
            return pd.Series(["⚠️ 强弩之末",   "大周期上涨动能走弱，逢高降低暴露，等待结构修复。"])

        # 🛡️ 第三顺位：防守与洗盘判定
        if micro == "画线骗炮(规避)":
            if "低位共振" in macro or dyn_pos < 50:
                return pd.Series(["🎣 暴力洗盘", "低位资金结构仍在，若后续缩量企稳，可分批观察低吸机会。"])
            return pd.Series(["🛡️ 战术减仓",   "中高位上影线抛压加重，适合降低仓位并等待确认。"])

        # 🚀 第四顺位：进攻与持仓
        if "低位共振" in macro:
            if micro in ["真·主升初期", "主升候选"]:
                return pd.Series(["🚀 满仓做多", "低位共振叠加日内强势，属于高优先级候选，适合按计划提高关注。"])
            return pd.Series(["🌱 底仓潜伏",   "低位资金结构较好，但日内确认不足，适合分批观察。"])
        if "安全持仓" in macro:
            if micro in ["真·主升初期", "主升候选"]:
                return pd.Series(["📈 顺势加仓", "宏微观风险可控且日内动能较强，适合顺势跟踪。"])
            return pd.Series(["☕ 稳健持有",   "无明显系统性风险，板块处于正常蓄势震荡。"])
        if "预警" in macro:
            if micro in ["真·主升初期", "主升候选"]:
                return pd.Series(["🔥 冲顶阶段", "趋势虽强但处于危险水位，收紧止损位，【只卖不买】。"])
            return pd.Series(["⚖️ 控仓观望",   "多空高位激烈博弈，方向不明，保持低仓位观望。"])

        return pd.Series(["👀 混沌无序",       "无明显多空博弈信号，建议观望。"])

    df[["终极信号", "交易指引"]] = df.apply(_ultimate_decision, axis=1)

    # ── 清理内部中间列（保留 fund_raw 供轮动分析页使用）──
    drop_cols = [c for c in [
        "amplitude", "buy_power", "vol_mean", "amount_mean", "amount_std", "price_std",
        "micro_factor", "trend_raw", "abnormal_raw",
        "trap_raw", "efficiency_raw", "mid_confirm_raw", "微观日内状态",
    ] if c in df.columns]
    df = df.drop(columns=drop_cols)

    result = df.sort_values("综合博弈得分", ascending=False).reset_index(drop=True)
    _write_processed_disk_cache(result)
    return result
