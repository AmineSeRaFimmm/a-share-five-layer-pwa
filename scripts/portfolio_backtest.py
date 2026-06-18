from __future__ import annotations

import numpy as np
import pandas as pd

from production_backtest import (
    BUY_BREADTH_FLOOR,
    MAX_RISK,
    MIN_SCORE,
    PRIMARY_STRATEGY_LABEL,
    SELL_BREADTH_FLOOR,
    TRANSACTION_COST_RATE,
)

ACCUMULATE_STRATEGY_LABEL = "Top1 + 广度过滤 + 保留旧仓"


def build_top1_accumulate_backtest(scored: pd.DataFrame, lookback_days: int = 360) -> pd.DataFrame:
    """保留旧仓方法：新Top1出现时加入组合，不主动卖旧仓；广度跌破卖出线时全清。

    回测口径：固定资金、组合内持仓等权；新标的加入时按新等权组合计算次日收益。
    成本口径：按组合换手扣真实ETF单边费率；交易级胜率用每个ETF持仓段收益统计。
    """
    if scored.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    active_positions: dict[str, int] = {}
    next_trade_id = 0

    for dt, day in scored.groupby("date", sort=True):
        day = day.copy()
        breadth = float((day["涨跌幅"] > 0).mean())
        bench_ret = float(day["next_ret"].mean())
        action = "flat"
        trade_ids: list[int] = []
        exit_trade_ids: list[int] = []
        entry_cost = 0.0
        exit_cost = 0.0
        turnover = 0.0

        old_names = list(active_positions.keys())
        if breadth < SELL_BREADTH_FLOOR:
            if active_positions:
                exit_trade_ids = list(active_positions.values())
                turnover = 1.0
                exit_cost = TRANSACTION_COST_RATE * turnover
                action = "sell"
                active_positions = {}
        else:
            candidate = pd.DataFrame()
            if breadth >= BUY_BREADTH_FLOOR:
                candidate = (
                    day[(day["综合博弈得分"] >= MIN_SCORE) & (day["逃顶风险简分"] < MAX_RISK)]
                    .sort_values("综合博弈得分", ascending=False)
                    .head(1)
                )
            if not candidate.empty:
                new_name = str(candidate.iloc[0]["板块名称"])
                if new_name not in active_positions:
                    old_n = len(active_positions)
                    new_n = old_n + 1
                    next_trade_id += 1
                    active_positions[new_name] = next_trade_id
                    # 固定资金等权组合：从N只扩到N+1只，组合换手为新仓买入 1/(N+1)
                    # 加上旧仓从 1/N 降到 1/(N+1) 的减仓部分，合计 2/(N+1)。
                    # 空仓首次买入时换手为100%。
                    turnover = 1.0 if old_n == 0 else 2.0 / new_n
                    entry_cost = TRANSACTION_COST_RATE * turnover
                    action = "buy" if old_n == 0 else "add"
                else:
                    action = "hold"
            elif active_positions:
                action = "hold"

        holding_names = list(active_positions.keys())
        if holding_names:
            selected = day[day["板块名称"].astype(str).isin(holding_names)].copy()
        else:
            selected = pd.DataFrame()

        if selected.empty:
            strategy_ret = 0.0
            names = " / ".join(old_names) if action == "sell" and old_names else "空仓"
            top_score = 0.0
            risk = 0.0
            is_holding = False
        else:
            strategy_ret = float(selected["next_ret"].mean())
            names = " / ".join(selected["板块名称"].astype(str).tolist())
            top_score = float(selected["综合博弈得分"].mean())
            risk = float(selected["逃顶风险简分"].mean())
            is_holding = True
            trade_ids = [active_positions[str(name)] for name in selected["板块名称"].astype(str).tolist() if str(name) in active_positions]

        portfolio_size = max(len(holding_names), 1)
        # 交易段胜率按每个ETF自身收益统计，买入成本只归入新加入ETF，清仓成本归入被卖ETF。
        trade_components = []
        if is_holding and not selected.empty:
            for _, row in selected.iterrows():
                name = str(row["板块名称"])
                tid = active_positions.get(name, 0)
                component_ret = float(row["next_ret"])
                if action in {"buy", "add"} and tid == next_trade_id:
                    component_ret = (1.0 + component_ret) * (1.0 - TRANSACTION_COST_RATE) - 1.0
                trade_components.append({"trade_id": tid, "ret": component_ret})
        exit_components = [{"trade_id": tid, "ret": -TRANSACTION_COST_RATE} for tid in exit_trade_ids]

        net_strategy_ret = (1.0 + strategy_ret) * (1.0 - entry_cost) * (1.0 - exit_cost) - 1.0
        rows.append({
            "date": dt,
            "策略": ACCUMULATE_STRATEGY_LABEL,
            "动作": action,
            "持有板块": names,
            "综合博弈得分": top_score,
            "风险简分": risk,
            "市场广度": breadth * 100,
            "trade_id": trade_ids[0] if len(trade_ids) == 1 else 0,
            "trade_ids": trade_ids,
            "exit_trade_id": exit_trade_ids[0] if len(exit_trade_ids) == 1 else 0,
            "exit_trade_ids": exit_trade_ids,
            "is_holding": is_holding,
            "position_count": len(holding_names),
            "transaction_cost": entry_cost + exit_cost,
            "turnover": turnover,
            "trade_components": trade_components,
            "exit_components": exit_components,
            "strategy_ret": strategy_ret,
            "net_strategy_ret": net_strategy_ret,
            "benchmark_ret": bench_ret,
            "direction_hit": strategy_ret > 0,
            "relative_hit": strategy_ret > bench_ret,
        })

    bt = pd.DataFrame(rows).sort_values("date").tail(lookback_days).reset_index(drop=True)
    if bt.empty:
        return bt
    bt["strategy_nav"] = (1 + bt["strategy_ret"]).cumprod()
    bt["net_strategy_nav"] = (1 + bt["net_strategy_ret"]).cumprod()
    bt["benchmark_nav"] = (1 + bt["benchmark_ret"]).cumprod()
    return bt


def trade_returns_from_components(bt: pd.DataFrame) -> pd.Series:
    components: dict[int, list[float]] = {}
    for col in ["trade_components", "exit_components"]:
        if col not in bt.columns:
            continue
        for value in bt[col]:
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                tid = int(item.get("trade_id") or 0)
                if tid <= 0:
                    continue
                components.setdefault(tid, []).append(float(item.get("ret") or 0.0))
    if not components:
        return pd.Series(dtype=float)
    return pd.Series({tid: float(np.prod([1.0 + r for r in returns]) - 1.0) for tid, returns in components.items()}).dropna()


def summarize_accumulate_backtest(bt: pd.DataFrame) -> dict[str, float]:
    from production_backtest import summarize_backtest

    summary = summarize_backtest(bt)
    trades = trade_returns_from_components(bt)
    if trades.empty:
        return summary
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    summary.update({
        "交易胜率": float((trades > 0).mean()),
        "平均盈利": avg_win,
        "平均亏损": avg_loss,
        "盈亏比": float(avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0,
        "交易次数": float(len(trades)),
        "平均持仓数": float(pd.to_numeric(bt.get("position_count", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()),
        "最大持仓数": float(pd.to_numeric(bt.get("position_count", pd.Series(dtype=float)), errors="coerce").fillna(0).max()),
    })
    return summary
