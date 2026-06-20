from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "generate_fullrisk_grid_300_base.py"


def _load_base_module():
    existing = sys.modules.get("generate_fullrisk_grid_300")
    if existing is not None and hasattr(existing, "_fetch_histories") and hasattr(existing, "build_fullrisk_panel"):
        return existing
    spec = importlib.util.spec_from_file_location("generate_fullrisk_grid_300", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base full-risk generator from {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_fullrisk_grid_300"] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_module()


def _attach_execution_prices(panel: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    price_rows: list[dict] = []
    for name, hist in histories.items():
        if hist.empty or not {"date", "open", "close"}.issubset(hist.columns):
            continue
        hist = hist.sort_values("date").reset_index(drop=True)
        for i in range(len(hist) - 1):
            close = float(hist.at[i, "close"])
            next_open = float(hist.at[i + 1, "open"])
            next_close = float(hist.at[i + 1, "close"])
            if close <= 0 or next_open <= 0 or next_close <= 0:
                continue
            price_rows.append({
                "date": pd.Timestamp(hist.at[i, "date"]),
                "板块名称": name,
                "execution_trade_date": pd.Timestamp(hist.at[i + 1, "date"]),
                "overnight_ret": next_open / close - 1.0,
                "execution_intraday_ret": next_close / next_open - 1.0,
            })
    prices = pd.DataFrame(price_rows)
    if prices.empty:
        raise RuntimeError("next-open execution prices are empty")
    out = panel.merge(prices, on=["date", "板块名称"], how="left")
    required = ["execution_trade_date", "overnight_ret", "execution_intraday_ret"]
    if out[required].isna().any().any():
        missing = out[out[required].isna().any(axis=1)][["date", "板块名称"]].head(5).to_dict(orient="records")
        raise RuntimeError(f"missing next-open execution fields: {missing}")
    return out


def build_fullrisk_panel() -> pd.DataFrame:
    histories = base._fetch_histories()
    if len(histories) < base.MIN_SECTORS_PER_DAY:
        raise RuntimeError(f"not enough SW histories: {len(histories)}")
    original_fetch = base._fetch_histories
    base._fetch_histories = lambda: histories
    try:
        panel = base.build_fullrisk_panel()
    finally:
        base._fetch_histories = original_fetch
    return _attach_execution_prices(panel, histories)


def _trade_returns(bt: pd.DataFrame) -> pd.Series:
    closed = pd.to_numeric(bt.get("closed_trade_return", pd.Series(dtype=float)), errors="coerce").dropna()
    return closed.reset_index(drop=True)


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


def _sector_row(day: pd.DataFrame, sector: str) -> pd.Series | None:
    rows = day[day["板块名称"].astype(str) == sector] if sector else pd.DataFrame()
    return None if rows.empty else rows.iloc[0]


def _segment_return(day: pd.DataFrame, old_position: str, target_position: str) -> tuple[float, float, float]:
    overnight_ret = 0.0
    intraday_ret = 0.0
    old_row = _sector_row(day, old_position)
    new_row = _sector_row(day, target_position)
    if old_row is not None:
        overnight_ret = float(old_row["overnight_ret"])
    if new_row is not None:
        intraday_ret = float(new_row["execution_intraday_ret"])
    return (1.0 + overnight_ret) * (1.0 + intraday_ret) - 1.0, overnight_ret, intraday_ret


def run_top1_fullrisk_backtest(panel: pd.DataFrame, buy_breadth: float, sell_breadth: float, min_score: float, max_risk: float, hs300: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    current_position = ""
    current_trade_id = 0
    next_trade_id = 0
    active_trade_nav = 1.0

    for dt, day in panel.groupby("date", sort=True):
        day = day.copy()
        breadth = float((day["涨跌幅"] > 0).mean())
        benchmark_ret = float(day["next_ret"].mean())
        old_position = current_position
        target_position = current_position
        action = "持有" if current_position else "空仓"
        picked_row = None
        closed_trade_return = np.nan
        closed_trade_id = np.nan

        if breadth < sell_breadth:
            target_position = ""
            action = "卖出" if current_position else "空仓"
        elif breadth >= buy_breadth:
            picks = day[(day["综合博弈得分"] >= min_score) & (day["逃顶风险分"] < max_risk)].sort_values("综合博弈得分", ascending=False).head(1)
            if not picks.empty:
                picked_row = picks.iloc[0]
                picked = str(picked_row["板块名称"])
                action = "买入" if not current_position else ("换仓" if picked != current_position else "持有")
                target_position = picked

        ret, overnight_ret, intraday_ret = _segment_return(day, old_position, target_position)
        if old_position and target_position == old_position:
            active_trade_nav *= 1.0 + ret
        elif old_position and not target_position:
            active_trade_nav *= 1.0 + overnight_ret
            closed_trade_return = active_trade_nav - 1.0
            closed_trade_id = current_trade_id
            current_trade_id = 0
            active_trade_nav = 1.0
        elif old_position and target_position and target_position != old_position:
            active_trade_nav *= 1.0 + overnight_ret
            closed_trade_return = active_trade_nav - 1.0
            closed_trade_id = current_trade_id
            next_trade_id += 1
            current_trade_id = next_trade_id
            active_trade_nav = 1.0 + intraday_ret
        elif not old_position and target_position:
            next_trade_id += 1
            current_trade_id = next_trade_id
            active_trade_nav = 1.0 + intraday_ret
        else:
            current_trade_id = 0
            active_trade_nav = 1.0

        current_position = target_position
        target_row = _sector_row(day, current_position) if current_position else None
        signal_row = target_row or picked_row or _sector_row(day, old_position)
        if signal_row is None:
            score = risk = resonance = 0.0
            signal_sector = ""
            execution_trade_date = pd.NaT
        else:
            score = float(signal_row.get("综合博弈得分", 0.0))
            risk = float(signal_row.get("逃顶风险分", 0.0))
            resonance = float(signal_row.get("入场共振分", 0.0))
            signal_sector = str(signal_row.get("板块名称", ""))
            execution_trade_date = signal_row.get("execution_trade_date", pd.NaT)

        trade_id = closed_trade_id if np.isfinite(closed_trade_id) else (current_trade_id if current_position else np.nan)
        rows.append({
            "date": dt,
            "execution_trade_date": execution_trade_date,
            "动作前持仓": old_position or "空仓",
            "持有板块": current_position or "空仓",
            "信号板块": signal_sector,
            "动作": action,
            "trade_id": trade_id,
            "closed_trade_return": closed_trade_return,
            "strategy_ret": ret,
            "overnight_ret": overnight_ret,
            "execution_intraday_ret": intraday_ret,
            "benchmark_ret": benchmark_ret,
            "综合博弈得分": score,
            "逃顶风险分": risk,
            "入场共振分": resonance,
            "市场广度": breadth,
            "direction_hit": ret > 0,
            "relative_hit": ret > benchmark_ret,
            "is_holding": bool(old_position or current_position),
        })

    bt = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if bt.empty:
        return bt, {}
    if current_trade_id and np.isfinite(active_trade_nav) and abs(active_trade_nav - 1.0) > 1e-12:
        if pd.isna(bt.at[bt.index[-1], "closed_trade_return"]):
            bt.at[bt.index[-1], "closed_trade_return"] = active_trade_nav - 1.0
    bt["strategy_nav"] = (1.0 + bt["strategy_ret"]).cumprod()
    bt["benchmark_nav"] = (1.0 + bt["benchmark_ret"]).cumprod()
    if hs300 is not None:
        bt = base._attach_hs300(bt, hs300)
    return bt, _summarize_path(bt, buy_breadth, sell_breadth, min_score, max_risk)


def _grid_results(panel: pd.DataFrame, hs300: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    primary_bt = pd.DataFrame()
    for buy in base.BUY_BREADTHS:
        for sell in base.SELL_BREADTHS:
            if sell >= buy:
                continue
            for score in base.SCORE_THRESHOLDS:
                for risk in base.RISK_THRESHOLDS:
                    bt, summary = run_top1_fullrisk_backtest(panel, buy, sell, score, risk)
                    if not summary:
                        continue
                    rows.append(summary)
                    if buy == base.PRIMARY_BUY_BREADTH and sell == base.PRIMARY_SELL_BREADTH and score == base.PRIMARY_MIN_SCORE and risk == base.PRIMARY_MAX_RISK:
                        primary_bt = bt
    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("empty grid results")
    results = results.sort_values(["累计收益", "最大回撤", "交易胜率"], ascending=[False, False, False]).reset_index(drop=True)
    if primary_bt.empty:
        raise RuntimeError("empty primary path")
    return results, base._attach_hs300(primary_bt, hs300)


def _window_robustness(panel: pd.DataFrame, hs300: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for window in base.WINDOWS:
        dates = sorted(panel["date"].dropna().unique())[-window:]
        sub = panel[panel["date"].isin(dates)].copy()
        bt, summary = run_top1_fullrisk_backtest(sub, base.PRIMARY_BUY_BREADTH, base.PRIMARY_SELL_BREADTH, base.PRIMARY_MIN_SCORE, base.PRIMARY_MAX_RISK, hs300=hs300)
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
    if "execution_trade_date" in out.columns:
        out["execution_trade_date"] = pd.to_datetime(out["execution_trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    cols = ["date", "execution_trade_date", "动作前持仓", "持有板块", "信号板块", "动作", "综合博弈得分", "逃顶风险分", "入场共振分", "市场广度", "overnight_ret", "execution_intraday_ret", "strategy_ret", "benchmark_ret", "hs300_ret"]
    return out[[c for c in cols if c in out.columns]]


def generate_fullrisk_grid_300(promote: bool = False) -> dict:
    base.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_fullrisk_panel()
    hs300 = base._fetch_hs300_history()
    results, primary_bt = _grid_results(panel, hs300)
    dedup = base._dedup_results(results)
    strategy_cmp = base._strategy_comparison(results)
    window_robust = _window_robustness(panel, hs300)
    recent_signals = _recent_signals(primary_bt)
    candidate_primary = base._primary_row(results)
    official_primary = base._load_official_primary()
    compare = base._compare_primary(candidate_primary, official_primary)
    metadata = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ready",
        "update_mode": "promote" if promote else "shadow_compare",
        "promoted": bool(promote),
        "script": "scripts/generate_fullrisk_grid_300_next_open.py",
        "method": "full_3d_risk_next_open_execution_no_simple_fallback",
        "execution_assumption": "signal after close; buy/sell/switch at next trading day's open",
        "lookback_days": base.LOOKBACK_DAYS,
        "warmup_days": base.WARMUP_DAYS,
        "min_sectors_per_day": base.MIN_SECTORS_PER_DAY,
        "first_trade_date": panel["date"].min().strftime("%Y-%m-%d"),
        "last_trade_date": panel["date"].max().strftime("%Y-%m-%d"),
        "observed_trade_days": int(panel["date"].nunique()),
        "observed_sectors": int(panel["板块名称"].nunique()),
        "grid_size": int(len(results)),
        "primary_params": {"buy_breadth": base.PRIMARY_BUY_BREADTH, "sell_breadth": base.PRIMARY_SELL_BREADTH, "min_score": base.PRIMARY_MIN_SCORE, "max_risk": base.PRIMARY_MAX_RISK},
        "primary_summary": candidate_primary,
        "compare_status": compare["status"],
        "outputs": {
            "candidate_results": str(base.CANDIDATE_RESULTS_FILE.relative_to(base.ROOT)),
            "candidate_top20": str(base.CANDIDATE_TOP20_FILE.relative_to(base.ROOT)),
            "candidate_dedup_top": str(base.CANDIDATE_DEDUP_FILE.relative_to(base.ROOT)),
            "candidate_primary_path": str(base.CANDIDATE_PRIMARY_FILE.relative_to(base.ROOT)),
            "candidate_strategy_comparison": str(base.CANDIDATE_STRATEGY_COMPARISON_FILE.relative_to(base.ROOT)),
            "candidate_window_robustness": str(base.CANDIDATE_WINDOW_ROBUSTNESS_FILE.relative_to(base.ROOT)),
            "candidate_recent_signals": str(base.CANDIDATE_RECENT_SIGNALS_FILE.relative_to(base.ROOT)),
            "compare_report": str(base.COMPARE_REPORT_FILE.relative_to(base.ROOT)),
            "official_results": str(base.RESULTS_FILE.relative_to(base.ROOT)),
        },
        "strict_checks": [
            "full risk score is still generated by the original daily _build_3d_scores path",
            "no fallback to simplified risk score",
            "T close signal executes only at T+1 open",
            "old holding takes close-to-next-open overnight return before sell/switch",
            "new holding takes next-open-to-next-close intraday return after buy/switch",
            "primary strategy parameters remain 70/35/54/45",
            "ETF transaction cost is still intentionally excluded",
        ],
    }
    base._write_outputs(results, dedup, primary_bt, panel, metadata, strategy_cmp, window_robust, recent_signals, promote=promote)
    base._write_json(base.COMPARE_REPORT_FILE, {**metadata, "comparison": compare})
    return {**metadata, "comparison": compare}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    metadata = generate_fullrisk_grid_300(promote=args.promote)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=base._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
