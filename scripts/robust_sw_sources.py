from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import akshare as ak
import numpy as np
import pandas as pd
import requests

import utils

SW_CONSTITUENTS_CACHE_FILE = utils.DATA_DIR / "sw_constituents_mapping.json"
ROBUST_HTTP_TIMEOUT = int(os.environ.get("ROBUST_HTTP_TIMEOUT", "20"))
ROBUST_RETRY_TIMES = int(os.environ.get("ROBUST_RETRY_TIMES", "4"))
ROBUST_RETRY_SLEEP = float(os.environ.get("ROBUST_RETRY_SLEEP", "1.2"))
SW_PREV_CLOSE_MAX_DRIFT = float(os.environ.get("SW_PREV_CLOSE_MAX_DRIFT", "0.005"))
MIN_CONST_SECTORS = int(os.environ.get("MIN_CONST_SECTORS", "25"))
MIN_CONST_CODES = int(os.environ.get("MIN_CONST_CODES", "1000"))


def _log(message: str) -> None:
    print(f"[robust-sw] {message}", flush=True)


def _retry(fn: Callable[[], Any], label: str, times: int = ROBUST_RETRY_TIMES) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, times + 1):
        try:
            value = fn()
            if value is not None:
                if isinstance(value, pd.DataFrame) and value.empty:
                    raise ValueError("empty dataframe")
                return value
        except Exception as exc:  # data-source boundary
            last_exc = exc
        if attempt < times:
            time.sleep(ROBUST_RETRY_SLEEP * attempt)
    if last_exc is not None:
        _log(f"{label} failed after {times} attempts: {last_exc}")
    else:
        _log(f"{label} failed after {times} attempts")
    return None


def _normalise_code(value: Any) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return digits[:6] if len(digits) >= 6 else ""


def _extract_codes(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    preferred = [
        "证券代码", "股票代码", "代码", "成分券代码", "stock_code", "code", "成分代码",
        "证券代码Constituent Code", "SECURITY_CODE", "成份券代码",
    ]
    col = next((c for c in preferred if c in df.columns), None)
    if col is None:
        for c in df.columns:
            sample = df[c].dropna().astype(str).head(20).tolist()
            if sample and sum(bool(re.search(r"\d{6}", x)) for x in sample) >= max(3, len(sample) // 2):
                col = c
                break
    if col is None:
        return []
    return sorted({code for code in (_normalise_code(x) for x in df[col].tolist()) if code})


def _read_constituents_cache() -> Dict[str, List[str]]:
    try:
        if not SW_CONSTITUENTS_CACHE_FILE.exists():
            return {}
        payload = json.loads(SW_CONSTITUENTS_CACHE_FILE.read_text(encoding="utf-8"))
        mapping = payload.get("mapping", payload) if isinstance(payload, dict) else {}
        out: Dict[str, List[str]] = {}
        if isinstance(mapping, dict):
            for name, codes in mapping.items():
                if isinstance(codes, list):
                    clean = sorted({code for code in (_normalise_code(c) for c in codes) if code})
                    if clean:
                        out[str(name)] = clean
        return out
    except Exception as exc:
        _log(f"read constituents cache failed: {exc}")
        return {}


def _write_constituents_cache(mapping: Dict[str, List[str]]) -> None:
    try:
        total_codes = sum(len(v) for v in mapping.values())
        if len(mapping) < MIN_CONST_SECTORS or total_codes < MIN_CONST_CODES:
            return
        payload = {
            "updated_at": pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S"),
            "sector_count": len(mapping),
            "code_count": total_codes,
            "mapping": mapping,
        }
        SW_CONSTITUENTS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        SW_CONSTITUENTS_CACHE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        _log(f"write constituents cache failed: {exc}")


def _build_neutral_constituents_fallback() -> Dict[str, List[str]]:
    """
    Last-resort production fallback when SW constituent sources are unavailable.

    Each SW sector receives one valid A-share code from the current/cached A-share
    snapshot. This makes coverage non-zero, but keeps len(pcts) <= 5 inside
    utils.get_processed_sw_data(), so micro_factor remains neutral at 1.0 and
    micro breadth labels are effectively disabled instead of being faked.
    """
    try:
        spot = _retry(utils.fetch_all_a_shares_spot, "neutral constituent fallback A-share spot", times=2)
        if not isinstance(spot, pd.DataFrame) or spot.empty:
            spot = _retry(utils._fetch_eastmoney_a_share_close_snapshot, "neutral constituent fallback A-share snapshot", times=2)
        if not isinstance(spot, pd.DataFrame) or spot.empty:
            spot = utils._read_a_share_snapshot_cache()
        if not isinstance(spot, pd.DataFrame) or spot.empty or "代码" not in spot.columns:
            return {}
        codes = sorted({code for code in (_normalise_code(c) for c in spot["代码"].tolist()) if code})
        if not codes:
            return {}
        fallback = {
            name: [codes[idx % len(codes)]]
            for idx, name in enumerate(utils.SW_CODE_MAPPING.keys())
        }
        _log(
            "constituents using NEUTRAL FALLBACK: "
            f"sectors={len(fallback)}, total_placeholder_codes={sum(len(v) for v in fallback.values())}; "
            "micro breadth disabled, micro_factor remains neutral"
        )
        return fallback
    except Exception as exc:
        _log(f"neutral constituents fallback failed: {exc}")
        return {}


def fetch_sw_constituents_mapping_latest() -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for name, code in utils.SW_CODE_MAPPING.items():
        candidates: List[Tuple[str, Callable[[], pd.DataFrame]]] = []
        if hasattr(ak, "index_stock_cons_sw"):
            candidates.append((f"index_stock_cons_sw({code})", lambda code=code: ak.index_stock_cons_sw(symbol=code)))
        candidates.extend([
            (f"index_component_sw({code})", lambda code=code: ak.index_component_sw(symbol=code)),
            (f"stock_board_industry_cons_em({name})", lambda name=name: ak.stock_board_industry_cons_em(symbol=name)),
        ])
        for label, fn in candidates:
            df = _retry(fn, f"constituents {name} {label}", times=2)
            codes = _extract_codes(df) if isinstance(df, pd.DataFrame) else []
            if len(codes) >= 5:
                mapping[name] = codes
                break

    total_codes = sum(len(v) for v in mapping.values())
    if len(mapping) >= MIN_CONST_SECTORS and total_codes >= MIN_CONST_CODES:
        _write_constituents_cache(mapping)
        _log(f"constituents ready: sectors={len(mapping)}, codes={total_codes}")
        return mapping

    cached = _read_constituents_cache()
    cached_codes = sum(len(v) for v in cached.values())
    if len(cached) >= MIN_CONST_SECTORS and cached_codes >= MIN_CONST_CODES:
        _log(f"constituents live partial sectors={len(mapping)}, using static cache sectors={len(cached)}, codes={cached_codes}")
        return cached

    neutral = _build_neutral_constituents_fallback()
    neutral_codes = sum(len(v) for v in neutral.values())
    if len(neutral) >= len(utils.SW_CODE_MAPPING) and neutral_codes > 0:
        _log(
            f"constituents live/cache unavailable: live sectors={len(mapping)}, codes={total_codes}; "
            f"cache sectors={len(cached)}, codes={cached_codes}; using neutral fallback"
        )
        return neutral

    _log(f"constituents insufficient: live sectors={len(mapping)}, codes={total_codes}; cache sectors={len(cached)}, codes={cached_codes}")
    return mapping


def _request_sw_current() -> Dict[str, dict]:
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.swsresearch.com/",
        "Accept": "application/json,text/plain,*/*",
    }

    def _once() -> Dict[str, dict]:
        resp = session.get(
            utils.SW_CURRENT_URL,
            params={"page": 1, "page_size": 100, "indextype": "一级行业"},
            headers=headers,
            timeout=ROBUST_HTTP_TIMEOUT,
            verify=False,
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        rows: List[dict] = []
        if isinstance(data, dict):
            for key in ("results", "list", "data", "records", "rows"):
                if isinstance(data.get(key), list):
                    rows = data[key]
                    break
        elif isinstance(data, list):
            rows = data

        code_to_name = {v: k for k, v in utils.SW_CODE_MAPPING.items()}
        snapshot: Dict[str, dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = _normalise_code(row.get("swindexcode") or row.get("indexcode") or row.get("code") or row.get("index_code"))
            name = code_to_name.get(code)
            if not name:
                continue
            prev_close = pd.to_numeric(row.get("l3"), errors="coerce")
            open_ = pd.to_numeric(row.get("l4"), errors="coerce")
            amount_million = pd.to_numeric(row.get("l5"), errors="coerce")
            high = pd.to_numeric(row.get("l6"), errors="coerce")
            low = pd.to_numeric(row.get("l7"), errors="coerce")
            close = pd.to_numeric(row.get("l8"), errors="coerce")
            volume_million = pd.to_numeric(row.get("l11"), errors="coerce")
            if pd.isna(prev_close) or pd.isna(close) or pd.isna(amount_million) or pd.isna(volume_million):
                continue
            snapshot[name] = {
                "prev_close": float(prev_close),
                "开盘": float(open_) if not pd.isna(open_) else float(close),
                "最高": float(high) if not pd.isna(high) else float(close),
                "最低": float(low) if not pd.isna(low) else float(close),
                "收盘": float(close),
                "成交额": float(amount_million) / 100.0,
                "成交量": float(volume_million) / 100.0,
            }
        if len(snapshot) < 30:
            raise ValueError(f"SW current returned only {len(snapshot)} sectors")
        return snapshot

    result = _retry(_once, "SW current snapshot")
    return result if isinstance(result, dict) else {}


def _append_current_to_hist(
    df: pd.DataFrame,
    sector_name: str,
    target_date: Optional[pd.Timestamp],
    current_snapshot: Dict[str, dict],
) -> Tuple[pd.DataFrame, str]:
    if df is None or df.empty:
        return pd.DataFrame(), utils.SW_TREND_SOURCE_LABEL
    target = target_date or utils._latest_completed_sw_trade_date()
    if target is None:
        return df, utils.SW_TREND_SOURCE_LABEL
    norm = utils._normalise_sw_hist_df(df)
    if norm.empty:
        return norm, utils.SW_TREND_SOURCE_LABEL
    norm["数据来源"] = utils.SW_TREND_SOURCE_LABEL
    last_date = pd.Timestamp(norm["日期"].iloc[-1]).normalize()
    target = pd.Timestamp(target).normalize()
    if last_date >= target:
        return norm, utils.SW_TREND_SOURCE_LABEL

    current = current_snapshot.get(sector_name)
    if not current:
        return norm, utils.SW_TREND_SOURCE_LABEL
    next_trade = utils._next_trade_date_after(last_date)
    if next_trade is not None and pd.Timestamp(next_trade).normalize() != target:
        return norm, utils.SW_TREND_SOURCE_LABEL

    last_close = float(norm["收盘"].iloc[-1])
    prev_close = float(current["prev_close"])
    drift = abs(prev_close - last_close) / max(abs(last_close), 1e-9)
    if drift > SW_PREV_CLOSE_MAX_DRIFT:
        _log(f"reject SW current {sector_name}: prev_close drift={drift:.4%}")
        return norm, utils.SW_TREND_SOURCE_LABEL

    patch_row = {
        "日期": target,
        "收盘": current["收盘"],
        "开盘": current["开盘"],
        "最高": current["最高"],
        "最低": current["最低"],
        "成交量": current["成交量"],
        "成交额": current["成交额"],
        "数据来源": utils.SW_CURRENT_SOURCE_LABEL,
    }
    out = pd.concat([norm, pd.DataFrame([patch_row])], ignore_index=True)
    return out.sort_values("日期").drop_duplicates("日期", keep="last").reset_index(drop=True), utils.SW_CURRENT_SOURCE_LABEL


def fetch_historical_baselines_latest() -> Dict[str, dict]:
    history_dict: Dict[str, dict] = {}
    target_date = utils._latest_completed_sw_trade_date()
    current_snapshot = _request_sw_current() if target_date is not None else {}
    patched = 0
    missing: List[str] = []

    for name, code in utils.SW_CODE_MAPPING.items():
        try:
            raw = _retry(lambda code=code: ak.index_hist_sw(symbol=code, period="day"), f"index_hist_sw {name}")
            if not isinstance(raw, pd.DataFrame) or raw.empty:
                missing.append(name)
                continue
            df, source_label = _append_current_to_hist(raw, name, target_date, current_snapshot)
            if df.empty:
                missing.append(name)
                continue
            if source_label == utils.SW_CURRENT_SOURCE_LABEL:
                patched += 1
            df = df.tail(60)
            close = df["收盘"].dropna()
            volume = df["成交量"].dropna()
            amount = df["成交额"].dropna()
            if len(close) < 20:
                missing.append(name)
                continue

            recent_20 = close.tail(20)
            dyn_pos = (close.iloc[-1] - recent_20.min()) / (recent_20.max() - recent_20.min() + 1e-6) * 100
            ret_series = close.pct_change()
            acceleration = (ret_series.tail(3).mean() - ret_series.tail(6).head(3).mean()) * 100
            vol_ma5 = volume.tail(5).mean()
            amount_ma5 = amount.tail(5).mean()
            flow = (volume.iloc[-1] - vol_ma5) / (vol_ma5 + 1e-6)
            amount_flow = (amount.iloc[-1] - amount_ma5) / (amount_ma5 + 1e-6)
            ma20 = close.tail(20).mean()
            ma60 = close.tail(60).mean()
            ret_20 = close.iloc[-1] / (close.iloc[-20] + 1e-9) - 1.0
            up_days_20 = float((close.pct_change().tail(20) > 0).mean())
            mid_confirm_raw = (
                0.45 * (ret_20 / (close.pct_change().tail(20).std(ddof=0) + 1e-6))
                + 0.25 * (close.iloc[-1] / (ma20 + 1e-9) - 1.0) * 100
                + 0.20 * (ma20 / (ma60 + 1e-9) - 1.0) * 100
                + 0.10 * (up_days_20 - 0.5) * 10
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
                "past_close": float(close.iloc[-2]),
                "dyn_pos": float(dyn_pos),
                "acceleration": float(acceleration),
                "flow": float(flow),
                "amount_flow": float(amount_flow),
                "avg_shadow_ratio": float(np.mean(shadows)),
                "vol_mean": float(volume.tail(20).mean()),
                "vol_std": float(volume.tail(20).std(ddof=0)),
                "amount_mean": float(amount.tail(20).mean()),
                "amount_std": float(amount.tail(20).std(ddof=0)),
                "price_std": float(close.pct_change().tail(20).std(ddof=0)),
                "mid_confirm_raw": float(mid_confirm_raw),
                "last_close": float(close.iloc[-1]),
                "last_open": float(df["开盘"].iloc[-1]),
                "last_high": float(df["最高"].iloc[-1]),
                "last_low": float(df["最低"].iloc[-1]),
                "last_amount": float(df["成交额"].iloc[-1]),
                "last_volume": float(volume.iloc[-1]),
                "data_date": pd.Timestamp(df["日期"].iloc[-1]).strftime("%Y-%m-%d"),
                "data_source": source_label,
            }
        except Exception as exc:
            missing.append(name)
            _log(f"baseline {name} failed: {exc}")

    target_str = target_date.strftime("%Y-%m-%d") if target_date is not None else "unknown"
    latest_dates = sorted({v.get("data_date", "") for v in history_dict.values()})
    _log(f"baselines ready: target={target_str}, sectors={len(history_dict)}, current_patched={patched}, latest_dates={latest_dates}, missing={missing}")
    return history_dict


def fetch_all_a_shares_spot_latest() -> pd.DataFrame:
    df = _retry(utils.fetch_all_a_shares_spot, "A-share spot original", times=2)
    if isinstance(df, pd.DataFrame) and len(df) >= utils.MIN_A_SHARE_SNAPSHOT_ROWS:
        _log(f"A-share spot ready from original: rows={len(df)}")
        return df
    em = _retry(utils._fetch_eastmoney_a_share_close_snapshot, "Eastmoney A-share snapshot")
    if isinstance(em, pd.DataFrame) and len(em) >= utils.MIN_A_SHARE_SNAPSHOT_ROWS:
        utils._write_a_share_snapshot_cache(em)
        _log(f"A-share spot ready from Eastmoney: rows={len(em)}")
        return em
    cached = utils._read_a_share_snapshot_cache()
    if isinstance(cached, pd.DataFrame) and len(cached) >= utils.MIN_A_SHARE_SNAPSHOT_ROWS:
        _log(f"A-share spot using same-target cache: rows={len(cached)}")
        return cached
    _log("A-share spot unavailable")
    return pd.DataFrame()


def install() -> None:
    original_spot = utils.fetch_all_a_shares_spot

    def _spot_wrapper() -> pd.DataFrame:
        df = _retry(original_spot, "A-share spot original", times=2)
        if isinstance(df, pd.DataFrame) and len(df) >= utils.MIN_A_SHARE_SNAPSHOT_ROWS:
            _log(f"A-share spot ready from original: rows={len(df)}")
            return df
        em = _retry(utils._fetch_eastmoney_a_share_close_snapshot, "Eastmoney A-share snapshot")
        if isinstance(em, pd.DataFrame) and len(em) >= utils.MIN_A_SHARE_SNAPSHOT_ROWS:
            utils._write_a_share_snapshot_cache(em)
            _log(f"A-share spot ready from Eastmoney: rows={len(em)}")
            return em
        cached = utils._read_a_share_snapshot_cache()
        if isinstance(cached, pd.DataFrame) and len(cached) >= utils.MIN_A_SHARE_SNAPSHOT_ROWS:
            _log(f"A-share spot using same-target cache: rows={len(cached)}")
            return cached
        _log("A-share spot unavailable")
        return pd.DataFrame()

    utils.fetch_historical_baselines = fetch_historical_baselines_latest  # type: ignore[assignment]
    utils.fetch_sw_constituents_mapping = fetch_sw_constituents_mapping_latest  # type: ignore[assignment]
    utils.fetch_all_a_shares_spot = _spot_wrapper  # type: ignore[assignment]
    _log("installed production latest-trade-date data-source overrides")
