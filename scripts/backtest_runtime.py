from __future__ import annotations

from typing import Dict

import akshare as ak
import numpy as np
import pandas as pd

from utils import (
    FINAL_BUY_BREADTH_FLOOR,
    FINAL_MAX_RISK,
    FINAL_MIN_SCORE,
    FINAL_SELL_BREADTH_FLOOR,
    SW_CODE_MAPPING,
    rank_score,
    winsorize,
)

BACKTEST_TRANSACTION_COST_RATE = 0.0001
BACKTEST_SCORE_BASIS = "historical_live_aligned_v1"
BACKTEST_SCORE_BASIS_NOTE = (
    "回测链路独立于今日复盘实时链路；历史得分复用实时六层权重，并加入市场广度 regime_factor "
    "和历史截面微观结构代理因子。历史数据无法还原每日成分股级上涨占比，micro_factor 使用行业截面代理，"
    "因此不会改动今日复盘页的实时成分股级计算。"
)


def _regime_factor_from_breadth(breadth: float) -> float:
    if breadth >= 0.65:
        return 1.10
    if breadth <= 0.35:
        return 0.80
    return 1.00


def _historical_micro_factor(row: pd.Series, breadth: float, median_pct: float) -> float:
    pct_today = float(row.get("涨跌幅", 0.0) or 0.0)
    if pct_today > 0.5 and (breadth < 0.40 or (pct_today - median_pct) > 1.20):
        return 0.60
    if pct_today > 1.0 and breadth > 0.75:
        return 1.15
    return 1.00


def _max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def _trade_returns_from_rows(bt: pd.DataFrame) -> pd.Series:
    if bt.empty or "trade_id" not in bt.columns or not bt["trade_id"].notna().any():
        return pd.Series(dtype=float)
    components: dict[int, list[float]] = {}
    for _, row in bt.iterrows():
        trade_id = row.get("trade_id")
        if pd.notna(trade_id) and float(trade_id) > 0:
            components.setdefault(int(trade_id), []).append(float(row.get("trade_component_ret", row.get("net_strategy_ret", 0.0)) or 0.0))
        exit_trade_id = row.get("exit_trade_id")
        exit_cost = float(row.get("exit_cost", 0.0) or 0.0)
        if pd.notna(exit_trade_id) and float(exit_trade_id) > 0 and exit_cost > 0:
            components.setdefault(int(exit_trade_id), []).append(-exit_cost)
    returns = []
    for vals in components.values():
        nav = 1.0
        for val in vals:
            nav *= 1.0 + float(val)
        returns.append(nav - 1.0)
    return pd.Series(returns, dtype=float)


def _summarize_backtest(bt: pd.DataFrame) -> Dict[str, object]:
    if bt.empty:
        return {}

    bt = bt.copy()
    if "net_strategy_ret" not in bt.columns:
        bt["net_strategy_ret"] = bt["strategy_ret"]
    if "transaction_cost" not in bt.columns:
        bt["transaction_cost"] = 0.0
    if "is_holding" not in bt.columns:
        bt["is_holding"] = bt["strategy_ret"].abs() > 0

    bt["strategy_nav"] = (1.0 + pd.to_numeric(bt["strategy_ret"], errors="coerce").fillna(0.0)).cumprod()
    bt["net_strategy_nav"] = (1.0 + pd.to_numeric(bt["net_strategy_ret"], errors="coerce").fillna(0.0)).cumprod()
    bt["benchmark_nav"] = (1.0 + pd.to_numeric(bt["benchmark_ret"], errors="coerce").fillna(0.0)).cumprod()

    periods = max(len(bt), 1)
    holding_mask = bt["is_holding"].astype(bool)
    holding_returns = bt.loc[holding_mask, "strategy_ret"]
    trade_returns = _trade_returns_from_rows(bt)
    has_trade_metrics = not trade_returns.empty

    wins = trade_returns[trade_returns > 0] if has_trade_metrics else pd.Series(dtype=float)
    losses = trade_returns[trade_returns < 0] if has_trade_metrics else pd.Series(dtype=float)
    avg_profit = float(wins.mean()) if not wins.empty else None
    avg_loss = float(losses.mean()) if not losses.empty else None
    payoff = float(avg_profit / abs(avg_loss)) if avg_profit is not None and avg_loss is not None and avg_loss < 0 else None

    net_ret = pd.to_numeric(bt["net_strategy_ret"], errors="coerce").fillna(0.0)
    gross_ret = pd.to_numeric(bt["strategy_ret"], errors="coerce").fillna(0.0)
    return {
        "交易日数": float(periods),
        "胜率": float(bt["direction_hit"].mean()),
        "相对胜率": float(bt["relative_hit"].mean()),
        "累计收益": float(bt["strategy_nav"].iloc[-1] - 1.0),
        "基准收益": float(bt["benchmark_nav"].iloc[-1] - 1.0),
        "年化收益": float(bt["strategy_nav"].iloc[-1] ** (252 / periods) - 1.0),
        "最大回撤": _max_drawdown(bt["strategy_nav"]),
        "夏普比率": float((gross_ret.mean() / (gross_ret.std(ddof=0) + 1e-9)) * np.sqrt(252)),
        "持仓日胜率": float((holding_returns > 0).mean()) if not holding_returns.empty else 0.0,
        "交易胜率": float((trade_returns > 0).mean()) if has_trade_metrics else None,
        "平均盈利": avg_profit,
        "平均亏损": avg_loss,
        "盈亏比": payoff,
        "持仓暴露率": float(holding_mask.mean()),
        "交易次数": float(len(trade_returns)) if has_trade_metrics else None,
        "成本后收益": float(bt["net_strategy_nav"].iloc[-1] - 1.0),
        "成本后年化": float(bt["net_strategy_nav"].iloc[-1] ** (252 / periods) - 1.0),
        "成本后最大回撤": _max_drawdown(bt["net_strategy_nav"]),
        "单边成本假设": BACKTEST_TRANSACTION_COST_RATE,
        "总交易成本": float(bt["transaction_cost"].sum()),
    }


def _build_top1_breadth_backtest(scored: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    rows = []
    current_position = ""
    current_trade_id = 0
    next_trade_id = 0

    for dt, day in scored.groupby("date", sort=True):
        day = day.copy()
        breadth = float((day["涨跌幅"] > 0).mean()) if "涨跌幅" in day.columns else 0.5
        bench_ret = float(day["next_ret"].mean())
        risk_col = "逃顶风险分" if "逃顶风险分" in day.columns else "逃顶风险简分"
        new_position = current_position
        action = "hold" if current_position else "flat"

        if breadth < FINAL_SELL_BREADTH_FLOOR:
            new_position = ""
            action = "sell" if current_position else "flat"
        elif breadth >= FINAL_BUY_BREADTH_FLOOR:
            picks = (
                day[
                    (pd.to_numeric(day["综合博弈得分"], errors="coerce") >= FINAL_MIN_SCORE)
                    & (pd.to_numeric(day[risk_col], errors="coerce") < FINAL_MAX_RISK)
                ]
                .sort_values("综合博弈得分", ascending=False)
                .head(1)
            )
            if not picks.empty:
                new_position = str(picks.iloc[0]["板块名称"])
                action = "buy" if not current_position else ("switch" if current_position != new_position else "hold")

        exit_trade_id = np.nan
        exit_cost = 0.0
        entry_cost = 0.0
        if current_position != new_position:
            if current_position:
                exit_trade_id = current_trade_id
                exit_cost = BACKTEST_TRANSACTION_COST_RATE
            if new_position:
                next_trade_id += 1
                current_trade_id = next_trade_id
                entry_cost = BACKTEST_TRANSACTION_COST_RATE
            else:
                current_trade_id = 0

        current_position = new_position
        held = day[day["板块名称"].astype(str) == current_position] if current_position else pd.DataFrame()
        if held.empty:
            strategy_ret = 0.0
            names = "空仓"
            top_score = 0.0
            risk = 0.0
            regime_factor = _regime_factor_from_breadth(breadth)
            micro_factor = 1.0
            is_holding = False
            trade_id = np.nan
        else:
            row = held.iloc[0]
            strategy_ret = float(row["next_ret"])
            names = current_position
            top_score = float(row["综合博弈得分"])
            risk = float(row[risk_col])
            regime_factor = float(row.get("regime_factor", _regime_factor_from_breadth(breadth)) or 1.0)
            micro_factor = float(row.get("micro_factor", 1.0) or 1.0)
            is_holding = True
            trade_id = current_trade_id if current_trade_id else np.nan

        net_strategy_ret = (1.0 + strategy_ret) * (1.0 - entry_cost) * (1.0 - exit_cost) - 1.0
        trade_component_ret = (1.0 + strategy_ret) * (1.0 - entry_cost) - 1.0 if is_holding else 0.0
        rows.append({
            "date": dt,
            "持有板块": names,
            "综合博弈得分": top_score,
            "风险分": risk,
            "市场广度": breadth * 100,
            "regime_factor": regime_factor,
            "micro_factor": micro_factor,
            "strategy_ret": strategy_ret,
            "net_strategy_ret": net_strategy_ret,
            "benchmark_ret": bench_ret,
            "transaction_cost": entry_cost + exit_cost,
            "entry_cost": entry_cost,
            "exit_cost": exit_cost,
            "trade_component_ret": trade_component_ret,
            "trade_id": trade_id,
            "exit_trade_id": exit_trade_id,
            "is_holding": is_holding,
            "action": action,
            "direction_hit": strategy_ret > 0,
            "relative_hit": strategy_ret > bench_ret,
        })

    bt = pd.DataFrame(rows).sort_values("date").tail(lookback_days).reset_index(drop=True)
    if bt.empty:
        return bt
    bt["strategy_nav"] = (1.0 + bt["strategy_ret"]).cumprod()
    bt["net_strategy_nav"] = (1.0 + bt["net_strategy_ret"]).cumprod()
    bt["benchmark_nav"] = (1.0 + bt["benchmark_ret"]).cumprod()
    return bt


def _weights(names: list[str]) -> dict[str, float]:
    if not names:
        return {}
    w = 1.0 / len(names)
    return {name: w for name in names}


def _turnover(old_names: list[str], new_names: list[str]) -> float:
    old_w = _weights(old_names)
    new_w = _weights(new_names)
    names = set(old_w) | set(new_w)
    return float(sum(abs(new_w.get(name, 0.0) - old_w.get(name, 0.0)) for name in names))


def _build_rebalanced_backtest(scored: pd.DataFrame, lookback_days: int, strategy: str) -> pd.DataFrame:
    rows = []
    prev_names: list[str] = []
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
            picks = tradable[tradable["综合博弈得分"] >= min_score].sort_values("综合博弈得分", ascending=False).head(top_n)

        names_list = picks["板块名称"].astype(str).tolist() if not picks.empty else []
        gross_ret = float(picks["next_ret"].mean()) if names_list else 0.0
        cost = _turnover(prev_names, names_list) * BACKTEST_TRANSACTION_COST_RATE
        net_ret = (1.0 + gross_ret) * (1.0 - cost) - 1.0
        prev_names = names_list
        top_score = float(picks["综合博弈得分"].mean()) if names_list else 0.0
        risk_col = "逃顶风险简分"
        risk = float(picks[risk_col].mean()) if names_list else 0.0
        rows.append({
            "date": dt,
            "持有板块": " / ".join(names_list) if names_list else "空仓",
            "综合博弈得分": top_score,
            "风险简分": risk,
            "市场广度": breadth * 100,
            "strategy_ret": gross_ret,
            "net_strategy_ret": net_ret,
            "benchmark_ret": bench_ret,
            "transaction_cost": cost,
            "is_holding": bool(names_list),
            "direction_hit": gross_ret > 0,
            "relative_hit": gross_ret > bench_ret,
        })

    bt = pd.DataFrame(rows).sort_values("date").tail(lookback_days).reset_index(drop=True)
    if bt.empty:
        return bt
    bt["strategy_nav"] = (1.0 + bt["strategy_ret"]).cumprod()
    bt["net_strategy_nav"] = (1.0 + bt["net_strategy_ret"]).cumprod()
    bt["benchmark_nav"] = (1.0 + bt["benchmark_ret"]).cumprod()
    return bt


def _build_strategy_backtest(scored: pd.DataFrame, lookback_days: int, strategy: str) -> pd.DataFrame:
    if strategy == "top1_breadth_final":
        return _build_top1_breadth_backtest(scored, lookback_days)
    return _build_rebalanced_backtest(scored, lookback_days, strategy)


def _build_walk_forward_scores(lookback_days: int = 520) -> pd.DataFrame:
    """Build historical scores without touching the live snapshot scoring path."""
    histories: Dict[str, pd.DataFrame] = {}
    base_rename = {
        "日期": "date", "收盘": "close", "开盘": "open",
        "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
    }
    for name, symbol in SW_CODE_MAPPING.items():
        try:
            raw = ak.index_hist_sw(symbol=symbol, period="day").tail(lookback_days + 90)
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
                "趋势加速度": accel,
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
        breadth = float((day["涨跌幅"] > 0).mean())
        median_pct = float(day["涨跌幅"].median())
        regime_factor = _regime_factor_from_breadth(breadth)
        for col in raw_cols:
            day[col] = winsorize(pd.to_numeric(day[col], errors="coerce"))
        day["第1层_真实趋势"] = rank_score(day["trend_raw"])
        day["第2层_真假资金"] = rank_score(day["fund_raw"])
        day["第3层_异动干预"] = rank_score(day["abnormal_raw"])
        day["第4层_诱多诱空"] = 100 - rank_score(day["trap_raw"])
        day["第5层_博弈反身"] = rank_score(day["efficiency_raw"])
        day["第6层_中期确认"] = rank_score(day["mid_confirm_raw"])
        base_score = (
            day["第1层_真实趋势"] * 0.14
            + day["第2层_真假资金"] * 0.18
            + day["第3层_异动干预"] * 0.14
            + day["第4层_诱多诱空"] * 0.26
            + day["第5层_博弈反身"] * 0.18
            + day["第6层_中期确认"] * 0.10
        )
        day["regime_factor"] = regime_factor
        day["micro_factor"] = day.apply(lambda row: _historical_micro_factor(row, breadth, median_pct), axis=1)
        day["综合博弈得分"] = (base_score * day["regime_factor"] * day["micro_factor"]).clip(0, 100).round(1)
        day["逃顶风险简分"] = (day["动态水位"] * 0.45 + (100 - day["第4层_诱多诱空"]) * 0.55).clip(0, 100).round(1)
        day["逃顶风险分"] = day["逃顶风险简分"]
        day["入场共振分"] = ((day["综合博弈得分"] + day["第2层_真假资金"]) / 2).clip(0, 100).round(1)
        scored_days.append(day)

    scored = pd.concat(scored_days, ignore_index=True) if scored_days else pd.DataFrame()
    if scored.empty:
        return pd.DataFrame()
    return scored.reset_index(drop=True)
