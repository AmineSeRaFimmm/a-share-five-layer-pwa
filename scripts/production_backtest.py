from __future__ import annotations

import numpy as np
import pandas as pd
import akshare as ak

from utils import SW_CODE_MAPPING, rank_score, winsorize


PRODUCTION_SCORE_VERSION = "production_score_v2_regime_micro_neutral"
RAW_COLS = ["trend_raw", "fund_raw", "abnormal_raw", "trap_raw", "efficiency_raw", "mid_confirm_raw"]


def _regime_factor(day: pd.DataFrame) -> tuple[float, float]:
    breadth = float((pd.to_numeric(day["涨跌幅"], errors="coerce") > 0).mean()) if "涨跌幅" in day.columns else 0.5
    if breadth >= 0.65:
        return 1.10, breadth
    if breadth <= 0.35:
        return 0.80, breadth
    return 1.00, breadth


def build_walk_forward_scores(lookback_days: int = 520) -> pd.DataFrame:
    histories: dict[str, pd.DataFrame] = {}
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
            trap = abnormal - (pct_today / (price_std * 100.0)) + upper_shadow * 10.0
            efficiency = amount[i] / (abs(pct_today) + 0.1)

            ma20 = float(np.mean(close[i - 19:i + 1]))
            ma60 = float(np.mean(close[i - 59:i + 1]))
            ret20 = close[i] / (close[i - 20] + 1e-9) - 1.0
            up_days = float(np.mean(np.diff(close[i - 19:i + 1]) > 0))
            mid_confirm = (
                0.45 * (ret20 / (price_std + 1e-6))
                + 0.25 * (close[i] / (ma20 + 1e-9) - 1.0) * 100
                + 0.20 * (ma20 / (ma60 + 1e-9) - 1.0) * 100
                + 0.10 * (up_days - 0.5) * 10
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
    for dt, day in panel.groupby("date", sort=True):
        if len(day) < 20:
            continue
        day = day.copy()
        for col in RAW_COLS:
            day[col] = winsorize(pd.to_numeric(day[col], errors="coerce"))
        day["第1层_真实趋势"] = rank_score(day["trend_raw"])
        day["第2层_真假资金"] = rank_score(day["fund_raw"])
        day["第3层_异动干预"] = rank_score(day["abnormal_raw"])
        day["第4层_诱多诱空"] = 100 - rank_score(day["trap_raw"])
        day["第5层_博弈反身"] = rank_score(day["efficiency_raw"])
        day["第6层_中期确认"] = rank_score(day["mid_confirm_raw"])
        raw_score = (
            day["第1层_真实趋势"] * 0.14
            + day["第2层_真假资金"] * 0.18
            + day["第3层_异动干预"] * 0.14
            + day["第4层_诱多诱空"] * 0.26
            + day["第5层_博弈反身"] * 0.18
            + day["第6层_中期确认"] * 0.10
        )
        factor, breadth = _regime_factor(day)
        day["market_regime_factor"] = factor
        day["historical_micro_factor"] = 1.0
        day["score_basis"] = PRODUCTION_SCORE_VERSION
        day["综合博弈得分"] = (raw_score * factor * day["historical_micro_factor"]).clip(0, 100).round(1)
        day["逃顶风险简分"] = (day["动态水位"] * 0.45 + (100 - day["第4层_诱多诱空"]) * 0.55).clip(0, 100)
        day["市场广度"] = breadth * 100
        scored_days.append(day)

    scored = pd.concat(scored_days, ignore_index=True) if scored_days else pd.DataFrame()
    if scored.empty:
        return pd.DataFrame()
    return scored
