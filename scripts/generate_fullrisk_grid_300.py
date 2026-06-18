from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd

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
RESULTS_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_results.csv"
TOP20_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_top20.csv"
DEDUP_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_dedup_top.csv"
METADATA_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_metadata.json"
PANEL_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_panel.parquet"
PRIMARY_FILE = BACKTEST_DIR / "top1_fullrisk_grid_300_primary_path.csv"

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
    df = df.dropna(subset=needed).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def _fetch_histories(extra_days: int = 180) -> Dict[str, pd.DataFrame]:
    histories: Dict[str, pd.DataFrame] = {}
    fetch_days = LOOKBACK_DAYS + WARMUP_DAYS + extra_days
    for name, symbol in SW_CODE_MAPPING.items():
        try:
            raw = ak.index_hist_sw(symbol=symbol, period="day")
            hist = _normalize_sw_history(raw).tail(fetch_days).reset_index(drop=True)
            if len(hist) >= LOOKBACK_DAYS + WARMUP_DAYS:
                histories[name] = hist
        except Exception:
            continue
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
            vol20 = max(float(np.std(np.diff(cut["close"].tail(20).to_numpy(dtype=float)) / (cut["close"].tail(20).to_numpy(dtype=float)[:-1] + 1e-9), ddof=0)), 1e-6)
            up_days = float(np.mean(np.diff(cut["close"].tail(20).to_numpy(dtype=float)) > 0))
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


def run_top1_fullrisk_backtest(panel: pd.DataFrame, buy_breadth: float, sell_breadth: float, min_score: float, max_risk: float) -> tuple[pd.DataFrame, dict]:
    rows = []
    current_position = ""
    current_trade_id = 0
    next_trade_id = 0
    for dt, day in panel.groupby("date", sort=True):
        day = day.copy()
        breadth = float((day["涨跌幅"] > 0).mean())
        benchmark_ret = float(day["next_ret"].mean())
        new_position = current_position
        if breadth < sell_breadth:
            new_position = ""
        elif breadth >= buy_breadth:
            picks = (
                day[(day["综合博弈得分"] >= min_score) & (day["逃顶风险分"] < max_risk)]
                .sort_values("综合博弈得分", ascending=False)
                .head(1)
            )
            if not picks.empty:
                new_position = str(picks.iloc[0]["板块名称"])
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
            trade_id = np.nan
            name = "空仓"
        else:
            row = held.iloc[0]
            ret = float(row["next_ret"])
            score = float(row["综合博弈得分"])
            risk = float(row["逃顶风险分"])
            trade_id = current_trade_id
            name = current_position
        rows.append({
            "date": dt,
            "持有板块": name,
            "trade_id": trade_id,
            "strategy_ret": ret,
            "benchmark_ret": benchmark_ret,
            "综合博弈得分": score,
            "逃顶风险分": risk,
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
    trade_ret = _trade_returns(bt)
    wins = trade_ret[trade_ret > 0]
    losses = trade_ret[trade_ret < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if not losses.empty and abs(float(losses.sum())) > 1e-12 else np.nan
    summary = {
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
    return bt, summary


def _grid_results(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def generate_fullrisk_grid_300(write_panel: bool = True) -> dict:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_fullrisk_panel()
    results, primary_bt = _grid_results(panel)
    dedup = _dedup_results(results)

    results.to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")
    results.head(20).to_csv(TOP20_FILE, index=False, encoding="utf-8-sig")
    dedup.head(20).to_csv(DEDUP_FILE, index=False, encoding="utf-8-sig")
    primary_bt.to_csv(PRIMARY_FILE, index=False, encoding="utf-8-sig")
    if write_panel:
        try:
            panel.to_parquet(PANEL_FILE, index=False)
        except Exception:
            panel.to_csv(PANEL_FILE.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    primary = results[
        (results["买入广度"].round(4) == PRIMARY_BUY_BREADTH)
        & (results["卖出广度"].round(4) == PRIMARY_SELL_BREADTH)
        & (results["综合分阈值"].round(4) == PRIMARY_MIN_SCORE)
        & (results["风险分阈值"].round(4) == PRIMARY_MAX_RISK)
    ].iloc[0].to_dict()

    metadata = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ready",
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
        "primary_summary": primary,
        "outputs": {
            "results": str(RESULTS_FILE.relative_to(ROOT)),
            "top20": str(TOP20_FILE.relative_to(ROOT)),
            "dedup_top": str(DEDUP_FILE.relative_to(ROOT)),
            "primary_path": str(PRIMARY_FILE.relative_to(ROOT)),
        },
        "strict_checks": [
            "完整风险分由 _build_3d_scores 逐日使用当日及以前历史生成",
            "不允许退回逃顶风险简分",
            "next_ret 仅用于下一交易日验证，不进入当日评分",
            "主策略参数固定为 70/35/54/45",
            "当前版本暂不加入 ETF 成本",
        ],
    }
    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return metadata


def main() -> int:
    metadata = generate_fullrisk_grid_300(write_panel=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
