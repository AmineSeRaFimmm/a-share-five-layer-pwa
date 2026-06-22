from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import (  # noqa: E402
    SW_CODE_MAPPING,
    _build_3d_scores,
    _calc_market_regime,
    rank_score,
    winsorize,
)

DATA_DIR = ROOT / "data"
BACKTEST_DIR = DATA_DIR / "backtest"

# Official production files that the model backtest page reads.
RESULTS_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_results.csv"
TOP20_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_top20.csv"
DEDUP_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_dedup_top.csv"
METADATA_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_metadata.json"
PANEL_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_panel.parquet"
PRIMARY_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_primary_path.csv"
STRATEGY_COMPARISON_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_strategy_comparison.csv"
WINDOW_ROBUSTNESS_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_window_robustness.csv"
RECENT_SIGNALS_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_recent_signals.csv"

# Shadow files generated daily until the new generator is reconciled with the trusted baseline.
CANDIDATE_RESULTS_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_results_candidate.csv"
CANDIDATE_TOP20_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_top20_candidate.csv"
CANDIDATE_DEDUP_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_dedup_top_candidate.csv"
CANDIDATE_METADATA_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_metadata_candidate.json"
CANDIDATE_PANEL_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_panel_candidate.parquet"
CANDIDATE_PRIMARY_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_primary_path_candidate.csv"
CANDIDATE_STRATEGY_COMPARISON_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_strategy_comparison_candidate.csv"
CANDIDATE_WINDOW_ROBUSTNESS_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_window_robustness_candidate.csv"
CANDIDATE_RECENT_SIGNALS_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_recent_signals_candidate.csv"
COMPARE_REPORT_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_compare_report.json"

LOOKBACK_DAYS = 300
WARMUP_DAYS = 90
MIN_SECTORS_PER_DAY = 20
PRIMARY_BUY_BREADTH = 0.70
PRIMARY_SELL_BREADTH = 0.35
PRIMARY_MIN_SCORE = 54
PRIMARY_MAX_RISK = 45

BUY_BREADTHS = [0.55, 0.60, 0.65, 0.70]
SELL_BREADTHS = [0.35, 0.40, 0.45, 0.50]
SCORE_THRESHOLDS = [54, 56, 58, 60, 62, 65]
RISK_THRESHOLDS = [45, 50, 55, 65]
WINDOWS = [120, 180, 240, 300]

COMPARE_METRICS = [
    "累计收益",
    "年化收益",
    "最大回撤",
    "交易次数",
    "交易胜率",
    "日胜率",
    "相对胜率",
    "profit_factor",
    "最长连续亏损",
    "持仓占比",
]

# Shadow mode tolerance only classifies the reconciliation report; it does not promote or overwrite files.
MAX_RETURN_DIFF_FOR_MATCH = 0.02
MAX_DRAWDOWN_DIFF_FOR_MATCH = 0.02
MAX_WINRATE_DIFF_FOR_MATCH = 0.03


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


SW_FETCH_TIMEOUT_SECONDS = _env_float("SW_FETCH_TIMEOUT_SECONDS", 15.0)
SW_FETCH_RETRIES = max(1, _env_int("SW_FETCH_RETRIES", 3))
SW_FETCH_RETRY_SLEEP_SECONDS = max(0.0, _env_float("SW_FETCH_RETRY_SLEEP_SECONDS", 2.0))


def _log_sw_fetch(message: str) -> None:
    print(f"[sw-history] {message}", file=sys.stderr, flush=True)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _normalize_sw_history(raw: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for col in raw.columns:
        c = str(col)
        lower = c.lower()
        if "日期" in c or lower == "date":
            rename[col] = "date"
        elif c in ("收盘", "close"):
            rename[col] = "close"
        elif c in ("开盘", "open"):
            rename[col] = "open"
        elif c in ("最高", "high"):
            rename[col] = "high"
        elif c in ("最低", "low"):
            rename[col] = "low"
        elif c in ("成交量", "volume"):
            rename[col] = "volume"
        elif c in ("成交额", "amount"):
            rename[col] = "amount"
    df = raw.rename(columns=rename)
    needed = ["date", "open", "high", "low", "close", "volume", "amount"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for col in needed[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=needed).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _normalize_hs300(raw: pd.DataFrame) -> pd.DataFrame:
    rename: dict[object, str] = {}
    for col in raw.columns:
        c = str(col)
        lower = c.lower()
        if "日期" in c or lower == "date":
            rename[col] = "date"
        elif c in ("收盘", "close"):
            rename[col] = "close"
    df = raw.rename(columns=rename)
    if not {"date", "close"}.issubset(df.columns):
        return pd.DataFrame()
    out = df[["date", "close"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _fetch_hs300_history() -> pd.DataFrame:
    errors: list[str] = []
    try:
        raw = ak.index_zh_a_hist(symbol="000300", period="daily")
        hist = _normalize_hs300(raw)
        if not hist.empty:
            return hist
    except Exception as exc:
        errors.append(f"index_zh_a_hist: {exc}")
    try:
        raw = ak.stock_zh_index_daily(symbol="sh000300")
        hist = _normalize_hs300(raw)
        if not hist.empty:
            return hist
    except Exception as exc:
        errors.append(f"stock_zh_index_daily: {exc}")
    raise RuntimeError("沪深300指数历史数据获取失败：" + "；".join(errors))


def _fetch_sw_history_raw(symbol: str) -> pd.DataFrame:
    original_get = requests.get

    def get_with_default_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", SW_FETCH_TIMEOUT_SECONDS)
        return original_get(*args, **kwargs)

    requests.get = get_with_default_timeout
    try:
        return ak.index_hist_sw(symbol=symbol, period="day")
    finally:
        requests.get = original_get


def _fetch_histories(extra_days: int = 180) -> Dict[str, pd.DataFrame]:
    histories: Dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    fetch_days = LOOKBACK_DAYS + WARMUP_DAYS + extra_days
    min_rows = LOOKBACK_DAYS + WARMUP_DAYS
    for name, symbol in SW_CODE_MAPPING.items():
        last_error = ""
        for attempt in range(1, SW_FETCH_RETRIES + 1):
            try:
                raw = _fetch_sw_history_raw(symbol)
                hist = _normalize_sw_history(raw).tail(fetch_days).reset_index(drop=True)
                if len(hist) >= min_rows:
                    histories[name] = hist
                    break
                raw_shape = getattr(raw, "shape", "unknown")
                raw_columns = list(raw.columns) if hasattr(raw, "columns") else []
                last_error = f"normalized rows {len(hist)} < {min_rows}; raw_shape={raw_shape}; columns={raw_columns}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < SW_FETCH_RETRIES:
                _log_sw_fetch(f"{name}({symbol}) attempt {attempt}/{SW_FETCH_RETRIES} failed: {last_error}; retrying")
                if SW_FETCH_RETRY_SLEEP_SECONDS > 0:
                    time.sleep(SW_FETCH_RETRY_SLEEP_SECONDS)
        else:
            failures.append(f"{name}({symbol}): {last_error}")
    if failures:
        _log_sw_fetch(
            f"fetched {len(histories)}/{len(SW_CODE_MAPPING)} sectors; "
            f"timeout={SW_FETCH_TIMEOUT_SECONDS}s retries={SW_FETCH_RETRIES}"
        )
        for failure in failures[:10]:
            _log_sw_fetch(f"failure: {failure}")
        if len(failures) > 10:
            _log_sw_fetch(f"... {len(failures) - 10} more failures")
    return histories


def _candidate_dates(histories: Dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    dates = sorted(set().union(*[set(h["date"]) for h in histories.values()])) if histories else []
    valid: list[pd.Timestamp] = []
    for dt in dates:
        count = 0
        for hist in histories.values():
            idx = hist.index[hist["date"] == dt]
            if len(idx) == 0:
                continue
            i = int(idx[0])
            if i >= WARMUP_DAYS and i < len(hist) - 1:
                count += 1
        if count >= MIN_SECTORS_PER_DAY:
            valid.append(pd.Timestamp(dt))
    return valid[-LOOKBACK_DAYS:]


def _daily_panel_from_fullrisk(histories: Dict[str, pd.DataFrame], dt: pd.Timestamp) -> pd.DataFrame:
    all_hist: Dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    for name, hist in histories.items():
        idx = hist.index[hist["date"] == dt]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        if i < WARMUP_DAYS or i >= len(hist) - 1:
            continue
        cut = hist.iloc[: i + 1].copy().reset_index(drop=True)
        if len(cut) < WARMUP_DAYS:
            continue
        all_hist[name] = cut
        prev_close = float(hist["close"].iloc[i - 1])
        close = float(hist["close"].iloc[i])
        open_ = float(hist["open"].iloc[i])
        high = float(hist["high"].iloc[i])
        low = float(hist["low"].iloc[i])
        amount = float(hist["amount"].iloc[i])
        volume = float(hist["volume"].iloc[i])
        pct_today = (close / (prev_close + 1e-9) - 1.0) * 100.0
        win20 = cut.tail(20)
        close20 = win20["close"].to_numpy(dtype=float)
        dyn_pos = (close - float(np.min(close20))) / (float(np.max(close20)) - float(np.min(close20)) + 1e-9) * 100.0
        if len(close20) >= 7:
            rets6 = np.diff(close20[-7:]) / (close20[-7:-1] + 1e-9) * 100.0
            acceleration = float(np.mean(rets6[-3:]) - np.mean(rets6[:3]))
        else:
            acceleration = 0.0
        price_std = max(float(np.std(np.diff(close20) / (close20[:-1] + 1e-9), ddof=0)), 1e-6) if len(close20) >= 3 else 0.01
        amount_mean = max(float(cut["amount"].iloc[max(0, i - 20):i].mean()), 1e-6)
        abnormal = amount / amount_mean
        amplitude = (high - low) / (open_ + 1e-9)
        buy_power = (close - low) / (high - low + 1e-9)
        upper_shadow = (high - max(open_, close)) / (high - low + 1e-9)
        trap = abnormal - (pct_today / (price_std * 100.0)) + upper_shadow * 5.0
        efficiency = amount / (abs(pct_today) + 0.1)
        if len(cut) >= 61:
            ma20 = float(cut["close"].tail(20).mean())
            ma60 = float(cut["close"].tail(60).mean())
            ret20 = close / (float(cut["close"].iloc[-21]) + 1e-9) - 1.0
            close20_tail = cut["close"].tail(20).to_numpy(dtype=float)
            vol20 = max(float(np.std(np.diff(close20_tail) / (close20_tail[:-1] + 1e-9), ddof=0)), 1e-6)
            up_days = float(np.mean(np.diff(close20_tail) > 0))
            mid_confirm_raw = (
                0.45 * (ret20 / (vol20 + 1e-6))
                + 0.25 * (close / (ma20 + 1e-9) - 1.0) * 100
                + 0.20 * (ma20 / (ma60 + 1e-9) - 1.0) * 100
                + 0.10 * (up_days - 0.5) * 10
            )
        else:
            mid_confirm_raw = 0.0
        next_ret = float(hist["close"].iloc[i + 1] / (close + 1e-9) - 1.0)
        rows.append({
            "date": dt,
            "板块名称": name,
            "next_ret": next_ret,
            "涨跌幅": pct_today,
            "trend_raw": (pct_today * 0.6 + (dyn_pos - 50) / 50 * 0.2 + acceleration * 0.2) / (price_std * 100.0),
            "fund_raw": amplitude * 0.4 + buy_power * 0.4 + abnormal * 0.2,
            "abnormal_raw": abnormal,
            "trap_raw": trap,
            "efficiency_raw": efficiency,
            "mid_confirm_raw": mid_confirm_raw,
            "动态水位": dyn_pos,
            "趋势加速度": acceleration,
            "成交额": amount,
            "成交量": volume,
            "上影线诱多率": upper_shadow * 100.0,
        })

    if len(rows) < MIN_SECTORS_PER_DAY:
        return pd.DataFrame()
    risk = _build_3d_scores(all_hist)
    if risk.empty or not {"板块名称", "逃顶风险分", "入场共振分"}.issubset(risk.columns):
        raise RuntimeError(f"{dt:%Y-%m-%d} 完整风险分生成失败；严格模式不允许退回简分")

    day = pd.DataFrame(rows)
    raw_cols = ["trend_raw", "fund_raw", "abnormal_raw", "trap_raw", "efficiency_raw", "mid_confirm_raw"]
    for col in raw_cols:
        day[col] = winsorize(pd.to_numeric(day[col], errors="coerce"))
    day["第1层_真实趋势"] = rank_score(day["trend_raw"])
    day["第2层_真假资金"] = rank_score(day["fund_raw"])
    day["第3层_异动干预"] = rank_score(day["abnormal_raw"])
    day["第4层_诱多诱空"] = 100 - rank_score(day["trap_raw"])
    day["第5层_博弈反身"] = rank_score(day["efficiency_raw"])
    day["第6层_中期确认"] = rank_score(day["mid_confirm_raw"])
    regime_factor = _calc_market_regime(day)[1]
    day["综合博弈得分"] = (
        (
            day["第1层_真实趋势"] * 0.14
            + day["第2层_真假资金"] * 0.18
            + day["第3层_异动干预"] * 0.14
            + day["第4层_诱多诱空"] * 0.26
            + day["第5层_博弈反身"] * 0.18
            + day["第6层_中期确认"] * 0.10
        ) * regime_factor
    ).clip(0, 100).round(1)
    out = day.merge(risk, on="板块名称", how="inner")
    if len(out) < MIN_SECTORS_PER_DAY:
        raise RuntimeError(f"{dt:%Y-%m-%d} 完整风险分覆盖不足：{len(out)}")
    return out


def build_fullrisk_panel() -> pd.DataFrame:
    histories = _fetch_histories()
    if len(histories) < MIN_SECTORS_PER_DAY:
        raise RuntimeError(f"申万历史数据覆盖不足：{len(histories)}")
    dates = _candidate_dates(histories)
    if len(dates) < LOOKBACK_DAYS:
        raise RuntimeError(f"可回测交易日不足：{len(dates)} < {LOOKBACK_DAYS}")
    panels = []
    for dt in dates:
        panels.append(_daily_panel_from_fullrisk(histories, dt))
    panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    if panel.empty:
        raise RuntimeError("完整风险分逐日面板为空")
    observed_days = panel["date"].nunique()
    if observed_days != LOOKBACK_DAYS:
        raise RuntimeError(f"完整风险分面板交易日数量异常：{observed_days} != {LOOKBACK_DAYS}")
    return panel.sort_values(["date", "综合博弈得分"], ascending=[True, False]).reset_index(drop=True)


def _trade_returns(bt: pd.DataFrame) -> pd.Series:
    if bt.empty or "trade_id" not in bt.columns:
        return pd.Series(dtype=float)
    values = []
    for _, part in bt.dropna(subset=["trade_id"]).groupby("trade_id"):
        nav = float((1.0 + part["strategy_ret"]).prod())
        values.append(nav - 1.0)
    return pd.Series(values, dtype=float)


def _max_losing_streak(returns: Iterable[float]) -> int:
    best = 0
    cur = 0
    for value in returns:
        if value < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    return float((nav / nav.cummax() - 1.0).min())


def _summarize_path(bt: pd.DataFrame, buy_breadth: float, sell_breadth: float, min_score: float, max_risk: float) -> dict:
    if bt.empty:
        return {}
    trade_ret = _trade_returns(bt)
    wins = trade_ret[trade_ret > 0]
    losses = trade_ret[trade_ret < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if not losses.empty and abs(float(losses.sum())) > 1e-12 else np.nan
    return {
        "买入广度": buy_breadth,
        "卖出广度": sell_breadth,
        "综合分阈值": min_score,
        "风险分阈值": max_risk,
        "累计收益": float(bt["strategy_nav"].iloc[-1] - 1.0),
        "年化收益": float(bt["strategy_nav"].iloc[-1] ** (252 / max(len(bt), 1)) - 1.0),
        "最大回撤": _max_drawdown(bt["strategy_nav"]),
        "交易次数": int(len(trade_ret)),
        "交易胜率": float((trade_ret > 0).mean()) if not trade_ret.empty else 0.0,
        "日胜率": float((bt.loc[bt["is_holding"], "strategy_ret"] > 0).mean()) if bt["is_holding"].any() else 0.0,
        "相对胜率": float(bt["relative_hit"].mean()),
        "profit_factor": profit_factor,
        "最长连续亏损": _max_losing_streak(trade_ret.tolist()),
        "持仓占比": float(bt["is_holding"].mean()),
    }


def _attach_hs300(bt: pd.DataFrame, hs300: pd.DataFrame) -> pd.DataFrame:
    out = bt.copy()
    hs = hs300.copy()
    hs["hs300_ret"] = hs["close"].shift(-1) / (hs["close"] + 1e-9) - 1.0
    out = out.merge(hs[["date", "hs300_ret"]], on="date", how="left")
    if out["hs300_ret"].isna().any():
        missing = out.loc[out["hs300_ret"].isna(), "date"].dt.strftime("%Y-%m-%d").head(5).tolist()
        raise RuntimeError(f"沪深300指数收益缺失，示例日期：{missing}")
    out["hs300_nav"] = (1.0 + out["hs300_ret"]).cumprod()
    return out


def run_top1_fullrisk_backtest(panel: pd.DataFrame, buy_breadth: float, sell_breadth: float, min_score: float, max_risk: float, hs300: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    rows = []
    current_position = ""
    current_trade_id = 0
    next_trade_id = 0
    for dt, day in panel.groupby("date", sort=True):
        day = day.copy()
        breadth = float((day["涨跌幅"] > 0).mean())
        benchmark_ret = float(day["next_ret"].mean())
        new_position = current_position
        action = "持有" if current_position else "空仓"
        if breadth < sell_breadth:
            new_position = ""
            action = "卖出" if current_position else "空仓"
        elif breadth >= buy_breadth:
            picks = (
                day[(day["综合博弈得分"] >= min_score) & (day["逃顶风险分"] < max_risk)]
                .sort_values("综合博弈得分", ascending=False)
                .head(1)
            )
            if not picks.empty:
                picked = str(picks.iloc[0]["板块名称"])
                if not current_position:
                    action = "买入"
                elif picked != current_position:
                    action = "换仓"
                else:
                    action = "持有"
                new_position = picked
        if new_position != current_position:
            if new_position:
                next_trade_id += 1
                current_trade_id = next_trade_id
            else:
                current_trade_id = 0
        current_position = new_position
        held = day[day["板块名称"].astype(str) == current_position] if current_position else pd.DataFrame()
        if held.empty:
            ret = 0.0
            score = 0.0
            risk = 0.0
            resonance = 0.0
            trade_id = np.nan
            name = "空仓"
        else:
            row = held.iloc[0]
            ret = float(row["next_ret"])
            score = float(row["综合博弈得分"])
            risk = float(row["逃顶风险分"])
            resonance = float(row.get("入场共振分", 0.0))
            trade_id = current_trade_id
            name = current_position
        rows.append({
            "date": dt,
            "持有板块": name,
            "动作": action,
            "trade_id": trade_id,
            "strategy_ret": ret,
            "benchmark_ret": benchmark_ret,
            "综合博弈得分": score,
            "逃顶风险分": risk,
            "入场共振分": resonance,
            "市场广度": breadth,
            "direction_hit": ret > 0,
            "relative_hit": ret > benchmark_ret,
            "is_holding": bool(current_position),
        })
    bt = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if bt.empty:
        return bt, {}
    bt["strategy_nav"] = (1.0 + bt["strategy_ret"]).cumprod()
    bt["benchmark_nav"] = (1.0 + bt["benchmark_ret"]).cumprod()
    if hs300 is not None:
        bt = _attach_hs300(bt, hs300)
    summary = _summarize_path(bt, buy_breadth, sell_breadth, min_score, max_risk)
    return bt, summary


def _grid_results(panel: pd.DataFrame, hs300: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    primary_bt = pd.DataFrame()
    for buy in BUY_BREADTHS:
        for sell in SELL_BREADTHS:
            if sell >= buy:
                continue
            for score in SCORE_THRESHOLDS:
                for risk in RISK_THRESHOLDS:
                    bt, summary = run_top1_fullrisk_backtest(panel, buy, sell, score, risk)
                    if not summary:
                        continue
                    rows.append(summary)
                    if buy == PRIMARY_BUY_BREADTH and sell == PRIMARY_SELL_BREADTH and score == PRIMARY_MIN_SCORE and risk == PRIMARY_MAX_RISK:
                        primary_bt = bt
    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("完整风险分参数网格结果为空")
    results = results.sort_values(["累计收益", "最大回撤", "交易胜率"], ascending=[False, False, False]).reset_index(drop=True)
    if primary_bt.empty:
        raise RuntimeError("主策略 70/35/54/45 路径为空")
    primary_bt = _attach_hs300(primary_bt, hs300)
    return results, primary_bt


def _dedup_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["累计收益", "年化收益", "最大回撤", "交易次数", "交易胜率", "日胜率", "相对胜率", "profit_factor", "最长连续亏损", "持仓占比"]
    param_cols = ["买入广度", "卖出广度", "综合分阈值", "风险分阈值"]
    rows = []
    for _, part in results.groupby(metric_cols, dropna=False, sort=False):
        first = part.iloc[0].to_dict()
        first["组合数"] = int(len(part))
        for col in param_cols:
            vals = part[col].dropna().unique().tolist()
            if len(vals) > 1:
                vals = sorted(vals)
                first[col] = ",".join(str(int(v)) if float(v).is_integer() else str(v) for v in vals)
        rows.append(first)
    dedup = pd.DataFrame(rows)
    return dedup.sort_values(["累计收益", "最大回撤", "交易胜率"], ascending=[False, False, False]).reset_index(drop=True)


def _primary_row(results: pd.DataFrame) -> dict:
    row = results[
        (results["买入广度"].round(4) == PRIMARY_BUY_BREADTH)
        & (results["卖出广度"].round(4) == PRIMARY_SELL_BREADTH)
        & (results["综合分阈值"].round(4) == PRIMARY_MIN_SCORE)
        & (results["风险分阈值"].round(4) == PRIMARY_MAX_RISK)
    ]
    if row.empty:
        raise RuntimeError("找不到主策略 70/35/54/45 结果行")
    return row.iloc[0].to_dict()


def _load_official_primary() -> dict | None:
    if not RESULTS_FILE.exists():
        return None
    try:
        official = pd.read_csv(RESULTS_FILE)
        for col in ["买入广度", "卖出广度", "综合分阈值", "风险分阈值", *COMPARE_METRICS]:
            if col in official.columns:
                official[col] = pd.to_numeric(official[col], errors="coerce")
        return _primary_row(official)
    except Exception:
        return None


def _compare_primary(candidate: dict, official: dict | None) -> dict:
    if official is None:
        return {
            "status": "no_official_baseline",
            "message": "未找到正式基准表，candidate 已生成但未自动提升为正式表。",
            "official_primary": None,
            "candidate_primary": candidate,
            "metric_diffs": {},
        }
    diffs: dict[str, dict] = {}
    for metric in COMPARE_METRICS:
        old = official.get(metric)
        new = candidate.get(metric)
        try:
            old_f = float(old)
            new_f = float(new)
            diff = new_f - old_f
        except Exception:
            old_f = old
            new_f = new
            diff = None
        diffs[metric] = {"official": old_f, "candidate": new_f, "diff": diff}

    return_diff = abs(float(diffs["累计收益"]["diff"])) if diffs.get("累计收益", {}).get("diff") is not None else np.inf
    dd_diff = abs(float(diffs["最大回撤"]["diff"])) if diffs.get("最大回撤", {}).get("diff") is not None else np.inf
    wr_diff = abs(float(diffs["交易胜率"]["diff"])) if diffs.get("交易胜率", {}).get("diff") is not None else np.inf
    status = "match_within_tolerance" if (
        return_diff <= MAX_RETURN_DIFF_FOR_MATCH
        and dd_diff <= MAX_DRAWDOWN_DIFF_FOR_MATCH
        and wr_diff <= MAX_WINRATE_DIFF_FOR_MATCH
    ) else "mismatch_requires_review"
    return {
        "status": status,
        "message": "candidate 与正式基准表差异在容忍范围内。" if status == "match_within_tolerance" else "candidate 与正式基准表差异较大，需人工复核后再 promote。",
        "official_primary": official,
        "candidate_primary": candidate,
        "metric_diffs": diffs,
        "tolerance": {
            "max_return_diff": MAX_RETURN_DIFF_FOR_MATCH,
            "max_drawdown_diff": MAX_DRAWDOWN_DIFF_FOR_MATCH,
            "max_winrate_diff": MAX_WINRATE_DIFF_FOR_MATCH,
        },
    }


def _strategy_comparison(results: pd.DataFrame) -> pd.DataFrame:
    items = [
        ("主策略：70/35/54/45", 0.70, 0.35, 54, 45),
        ("低广度确认：60/45/58/55", 0.60, 0.45, 58, 55),
        ("高分确认：70/35/60/45", 0.70, 0.35, 60, 45),
        ("低风险优先：70/35/54/50", 0.70, 0.35, 54, 50),
    ]
    rows = []
    for label, buy, sell, score, risk in items:
        row = results[
            (results["买入广度"].round(4) == buy)
            & (results["卖出广度"].round(4) == sell)
            & (results["综合分阈值"].round(4) == score)
            & (results["风险分阈值"].round(4) == risk)
        ]
        if row.empty:
            continue
        item = row.iloc[0].to_dict()
        item["策略"] = label
        rows.append(item)
    if not rows:
        return pd.DataFrame()
    cols = ["策略", "买入广度", "卖出广度", "综合分阈值", "风险分阈值", *COMPARE_METRICS]
    return pd.DataFrame(rows)[[c for c in cols if c in pd.DataFrame(rows).columns]]


def _window_robustness(panel: pd.DataFrame, hs300: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        sub = panel.sort_values("date").groupby("date", sort=True).filter(lambda x: True)
        dates = sorted(sub["date"].dropna().unique())[-window:]
        sub = sub[sub["date"].isin(dates)].copy()
        bt, summary = run_top1_fullrisk_backtest(sub, PRIMARY_BUY_BREADTH, PRIMARY_SELL_BREADTH, PRIMARY_MIN_SCORE, PRIMARY_MAX_RISK, hs300=hs300)
        if not summary:
            continue
        summary["窗口"] = f"{window}日"
        summary["沪深300收益"] = float(bt["hs300_nav"].iloc[-1] - 1.0) if "hs300_nav" in bt.columns else np.nan
        summary["行业等权收益"] = float(bt["benchmark_nav"].iloc[-1] - 1.0) if "benchmark_nav" in bt.columns else np.nan
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    cols = ["窗口", "累计收益", "行业等权收益", "沪深300收益", "年化收益", "最大回撤", "交易次数", "交易胜率", "日胜率", "相对胜率", "profit_factor", "持仓占比"]
    return pd.DataFrame(rows)[[c for c in cols if c in pd.DataFrame(rows).columns]]


def _recent_signals(primary_bt: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    out = primary_bt.copy().sort_values("date", ascending=False).head(n)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["strategy_ret", "benchmark_ret", "hs300_ret", "市场广度"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    cols = ["date", "持有板块", "动作", "综合博弈得分", "逃顶风险分", "入场共振分", "市场广度", "strategy_ret", "benchmark_ret", "hs300_ret"]
    return out[[c for c in cols if c in out.columns]]


def _write_outputs(results: pd.DataFrame, dedup: pd.DataFrame, primary_bt: pd.DataFrame, panel: pd.DataFrame, metadata: dict, strategy_cmp: pd.DataFrame, window_robust: pd.DataFrame, recent_signals: pd.DataFrame, promote: bool) -> None:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(CANDIDATE_RESULTS_FILE, index=False, encoding="utf-8-sig")
    results.head(20).to_csv(CANDIDATE_TOP20_FILE, index=False, encoding="utf-8-sig")
    dedup.head(20).to_csv(CANDIDATE_DEDUP_FILE, index=False, encoding="utf-8-sig")
    primary_bt.to_csv(CANDIDATE_PRIMARY_FILE, index=False, encoding="utf-8-sig")
    strategy_cmp.to_csv(CANDIDATE_STRATEGY_COMPARISON_FILE, index=False, encoding="utf-8-sig")
    window_robust.to_csv(CANDIDATE_WINDOW_ROBUSTNESS_FILE, index=False, encoding="utf-8-sig")
    recent_signals.to_csv(CANDIDATE_RECENT_SIGNALS_FILE, index=False, encoding="utf-8-sig")
    _write_json(CANDIDATE_METADATA_FILE, metadata)
    try:
        panel.to_parquet(CANDIDATE_PANEL_FILE, index=False)
    except Exception:
        panel.to_csv(CANDIDATE_PANEL_FILE.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    if promote:
        results.to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")
        results.head(20).to_csv(TOP20_FILE, index=False, encoding="utf-8-sig")
        dedup.head(20).to_csv(DEDUP_FILE, index=False, encoding="utf-8-sig")
        primary_bt.to_csv(PRIMARY_FILE, index=False, encoding="utf-8-sig")
        strategy_cmp.to_csv(STRATEGY_COMPARISON_FILE, index=False, encoding="utf-8-sig")
        window_robust.to_csv(WINDOW_ROBUSTNESS_FILE, index=False, encoding="utf-8-sig")
        recent_signals.to_csv(RECENT_SIGNALS_FILE, index=False, encoding="utf-8-sig")
        _write_json(METADATA_FILE, {**metadata, "promoted": True})
        try:
            panel.to_parquet(PANEL_FILE, index=False)
        except Exception:
            panel.to_csv(PANEL_FILE.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def generate_fullrisk_grid_300(promote: bool = False) -> dict:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_fullrisk_panel()
    hs300 = _fetch_hs300_history()
    results, primary_bt = _grid_results(panel, hs300)
    dedup = _dedup_results(results)
    strategy_cmp = _strategy_comparison(results)
    window_robust = _window_robustness(panel, hs300)
    recent_signals = _recent_signals(primary_bt)
    candidate_primary = _primary_row(results)
    official_primary = _load_official_primary()
    compare = _compare_primary(candidate_primary, official_primary)

    metadata = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ready",
        "update_mode": "promote" if promote else "shadow_compare",
        "promoted": bool(promote),
        "script": "scripts/generate_fullrisk_grid_300.py",
        "method": "daily_recomputed_full_3d_risk_no_simple_fallback",
        "lookback_days": LOOKBACK_DAYS,
        "warmup_days": WARMUP_DAYS,
        "min_sectors_per_day": MIN_SECTORS_PER_DAY,
        "first_trade_date": panel["date"].min().strftime("%Y-%m-%d"),
        "last_trade_date": panel["date"].max().strftime("%Y-%m-%d"),
        "observed_trade_days": int(panel["date"].nunique()),
        "observed_sectors": int(panel["板块名称"].nunique()),
        "grid_size": int(len(results)),
        "primary_params": {
            "buy_breadth": PRIMARY_BUY_BREADTH,
            "sell_breadth": PRIMARY_SELL_BREADTH,
            "min_score": PRIMARY_MIN_SCORE,
            "max_risk": PRIMARY_MAX_RISK,
        },
        "primary_summary": candidate_primary,
        "compare_status": compare["status"],
        "outputs": {
            "candidate_results": str(CANDIDATE_RESULTS_FILE.relative_to(ROOT)),
            "candidate_top20": str(CANDIDATE_TOP20_FILE.relative_to(ROOT)),
            "candidate_dedup_top": str(CANDIDATE_DEDUP_FILE.relative_to(ROOT)),
            "candidate_primary_path": str(CANDIDATE_PRIMARY_FILE.relative_to(ROOT)),
            "candidate_strategy_comparison": str(CANDIDATE_STRATEGY_COMPARISON_FILE.relative_to(ROOT)),
            "candidate_window_robustness": str(CANDIDATE_WINDOW_ROBUSTNESS_FILE.relative_to(ROOT)),
            "candidate_recent_signals": str(CANDIDATE_RECENT_SIGNALS_FILE.relative_to(ROOT)),
            "compare_report": str(COMPARE_REPORT_FILE.relative_to(ROOT)),
            "official_results": str(RESULTS_FILE.relative_to(ROOT)),
        },
        "strict_checks": [
            "完整风险分由 _build_3d_scores 逐日使用当日及以前历史生成",
            "不允许退回逃顶风险简分",
            "next_ret 仅用于下一交易日验证，不进入当日评分",
            "主策略参数固定为 70/35/54/45",
            "当前版本暂不加入 ETF 成本",
            "正式净值曲线包含策略、行业等权和沪深300指数三条线",
            "默认影子对账模式不覆盖正式表，只有 --promote 才覆盖",
        ],
    }
    _write_outputs(results, dedup, primary_bt, panel, metadata, strategy_cmp, window_robust, recent_signals, promote=promote)
    _write_json(COMPARE_REPORT_FILE, {**metadata, "comparison": compare})
    return {**metadata, "comparison": compare}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true", help="将 candidate 结果提升并覆盖正式 300 日完整风险分表。默认只做影子对账。")
    args = parser.parse_args()
    metadata = generate_fullrisk_grid_300(promote=args.promote)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
