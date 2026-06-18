from __future__ import annotations

import numpy as np
import pandas as pd
import akshare as ak

from utils import SW_CODE_MAPPING, rank_score, winsorize


PRODUCTION_SCORE_VERSION = "production_score_v2_regime_micro_neutral"
PRIMARY_STRATEGY_KEY = "top1_breadth"
PRIMARY_STRATEGY_LABEL = "Top1 + 广度过滤"
BUY_BREADTH_FLOOR = 0.60
SELL_BREADTH_FLOOR = 0.45
MIN_SCORE = 58.0
MAX_RISK = 55.0
TRANSACTION_COST_RATE = 0.0001
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


def build_top1_breadth_backtest(scored: pd.DataFrame, lookback_days: int = 360) -> pd.DataFrame:
    """今日推荐同口径：Top1 + 买入广度60% + 卖出广度45% + 综合/风险阈值。"""
    if scored.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    current_position = ""
    current_trade_id = 0
    next_trade_id = 0
    for dt, day in scored.groupby("date", sort=True):
        day = day.copy()
        breadth = float((day["涨跌幅"] > 0).mean())
        bench_ret = float(day["next_ret"].mean())
        action = "flat"
        display_position = ""
        selected = pd.DataFrame()
        trade_id = 0
        exit_trade_id = 0
        entry_cost = 0.0
        exit_cost = 0.0

        if breadth < SELL_BREADTH_FLOOR:
            sold_position = current_position
            sold_trade_id = current_trade_id
            current_position = ""
            current_trade_id = 0
            action = "sell" if sold_position else "flat"
            display_position = sold_position
            exit_trade_id = sold_trade_id if sold_position else 0
            exit_cost = TRANSACTION_COST_RATE if sold_position else 0.0
        else:
            if breadth >= BUY_BREADTH_FLOOR:
                candidate = (
                    day[(day["综合博弈得分"] >= MIN_SCORE) & (day["逃顶风险简分"] < MAX_RISK)]
                    .sort_values("综合博弈得分", ascending=False)
                    .head(1)
                )
                if not candidate.empty:
                    new_position = str(candidate.iloc[0]["板块名称"])
                    if current_position != new_position:
                        if current_position:
                            exit_trade_id = current_trade_id
                            exit_cost = TRANSACTION_COST_RATE
                        next_trade_id += 1
                        current_trade_id = next_trade_id
                        entry_cost = TRANSACTION_COST_RATE
                        action = "buy"
                    else:
                        action = "hold"
                    current_position = new_position
            if current_position:
                selected = day[day["板块名称"].astype(str) == current_position].head(1)
                trade_id = current_trade_id
                if action == "flat":
                    action = "hold"

        if selected.empty:
            strategy_ret = 0.0
            names = display_position or "空仓"
            top_score = 0.0
            risk = 0.0
            is_holding = False
        else:
            strategy_ret = float(selected["next_ret"].iloc[0])
            names = str(selected["板块名称"].iloc[0])
            top_score = float(selected["综合博弈得分"].iloc[0])
            risk = float(selected["逃顶风险简分"].iloc[0])
            is_holding = True

        trade_component_ret = (1.0 + strategy_ret) * (1.0 - entry_cost) - 1.0 if trade_id else 0.0
        net_strategy_ret = (1.0 + strategy_ret) * (1.0 - entry_cost) * (1.0 - exit_cost) - 1.0
        rows.append({
            "date": dt,
            "策略": PRIMARY_STRATEGY_LABEL,
            "动作": action,
            "持有板块": names,
            "综合博弈得分": top_score,
            "风险简分": risk,
            "市场广度": breadth * 100,
            "trade_id": trade_id,
            "exit_trade_id": exit_trade_id,
            "is_holding": is_holding,
            "entry_cost": entry_cost,
            "exit_cost": exit_cost,
            "transaction_cost": entry_cost + exit_cost,
            "trade_component_ret": trade_component_ret,
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


def build_reference_backtest(scored: pd.DataFrame, lookback_days: int, strategy: str) -> pd.DataFrame:
    rows = []
    for dt, day in scored.groupby("date", sort=True):
        day = day.copy()
        breadth = float((day["涨跌幅"] > 0).mean())
        bench_ret = float(day["next_ret"].mean())
        if strategy == "top1":
            picks = day.sort_values("综合博弈得分", ascending=False).head(1)
        elif strategy == "top3_balanced":
            picks = day[day["逃顶风险简分"] < 72].sort_values("综合博弈得分", ascending=False).head(3)
        elif strategy == "top3_regime":
            picks = pd.DataFrame() if breadth < 0.38 else day[day["逃顶风险简分"] < 72].sort_values("综合博弈得分", ascending=False).head(3)
        else:
            raise ValueError(f"unknown strategy: {strategy}")
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
        rows.append({
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
    bt = pd.DataFrame(rows).sort_values("date").tail(lookback_days).reset_index(drop=True)
    if bt.empty:
        return bt
    bt["strategy_nav"] = (1 + bt["strategy_ret"]).cumprod()
    bt["benchmark_nav"] = (1 + bt["benchmark_ret"]).cumprod()
    return bt


def _trade_returns(bt: pd.DataFrame, ret_col: str = "trade_component_ret") -> pd.Series:
    if "trade_id" not in bt.columns:
        return pd.Series(dtype=float)
    components: dict[int, list[float]] = {}
    active = bt[pd.to_numeric(bt["trade_id"], errors="coerce").fillna(0) > 0].copy()
    if ret_col in active.columns:
        for _, row in active.iterrows():
            tid = int(row["trade_id"])
            components.setdefault(tid, []).append(float(pd.to_numeric(row[ret_col], errors="coerce") or 0.0))
    if "exit_trade_id" in bt.columns and "exit_cost" in bt.columns:
        exits = bt[pd.to_numeric(bt["exit_trade_id"], errors="coerce").fillna(0) > 0].copy()
        for _, row in exits.iterrows():
            tid = int(row["exit_trade_id"])
            exit_cost = float(pd.to_numeric(row["exit_cost"], errors="coerce") or 0.0)
            components.setdefault(tid, []).append(-exit_cost)
    if not components:
        return pd.Series(dtype=float)
    return pd.Series({tid: float(np.prod([1.0 + r for r in returns]) - 1.0) for tid, returns in components.items()}).dropna()


def summarize_backtest(bt: pd.DataFrame) -> dict[str, float]:
    if bt.empty:
        return {}
    bt = bt.copy()
    if "strategy_nav" not in bt.columns:
        bt["strategy_nav"] = (1 + bt["strategy_ret"]).cumprod()
    if "net_strategy_ret" not in bt.columns:
        bt["net_strategy_ret"] = bt["strategy_ret"]
    if "net_strategy_nav" not in bt.columns:
        bt["net_strategy_nav"] = (1 + bt["net_strategy_ret"]).cumprod()
    if "benchmark_nav" not in bt.columns:
        bt["benchmark_nav"] = (1 + bt["benchmark_ret"]).cumprod()

    def _max_drawdown(nav: pd.Series) -> float:
        dd = nav / nav.cummax() - 1.0
        return float(dd.min())

    periods = max(len(bt), 1)
    holding_mask = bt.get("is_holding", pd.Series(False, index=bt.index)).astype(bool)
    holding_days = int(holding_mask.sum())
    holding_win_rate = float((bt.loc[holding_mask, "strategy_ret"] > 0).mean()) if holding_days else 0.0
    exposure = float(holding_days / periods) if periods else 0.0

    trades = _trade_returns(bt)
    trade_count = int(len(trades))
    trade_win_rate = float((trades > 0).mean()) if trade_count else 0.0
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    profit_loss_ratio = float(avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0

    return {
        "交易日数": float(periods),
        "胜率": float(bt["direction_hit"].mean()),
        "相对胜率": float(bt["relative_hit"].mean()),
        "累计收益": float(bt["strategy_nav"].iloc[-1] - 1.0),
        "基准收益": float(bt["benchmark_nav"].iloc[-1] - 1.0),
        "年化收益": float(bt["strategy_nav"].iloc[-1] ** (252 / periods) - 1.0),
        "最大回撤": _max_drawdown(bt["strategy_nav"]),
        "夏普比率": float((bt["strategy_ret"].mean() / (bt["strategy_ret"].std(ddof=0) + 1e-9)) * np.sqrt(252)),
        "持仓日胜率": holding_win_rate,
        "交易胜率": trade_win_rate,
        "平均盈利": avg_win,
        "平均亏损": avg_loss,
        "盈亏比": profit_loss_ratio,
        "持仓暴露率": exposure,
        "交易次数": float(trade_count),
        "成本后收益": float(bt["net_strategy_nav"].iloc[-1] - 1.0),
        "单边成本假设": TRANSACTION_COST_RATE,
    }
