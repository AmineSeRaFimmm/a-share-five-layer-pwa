from __future__ import annotations

from datetime import datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import update_daily_snapshot as uds

AFTER_CLOSE_TIME = dt_time(19, 0)


def _target_trade_date(now: datetime, calendar: pd.Series) -> tuple[pd.Timestamp | None, str]:
    today = pd.Timestamp(now.date()).normalize()
    today_is_trade_day = uds._is_trade_day(today, calendar)

    if today_is_trade_day and now.time() >= AFTER_CLOSE_TIME:
        return today, "after_close"

    eligible = calendar[calendar < today] if today_is_trade_day else calendar[calendar <= today]
    if eligible.empty:
        return None, "交易日历为空"
    return pd.Timestamp(eligible.iloc[-1]).normalize(), "last_completed_trade_day"


def _recover_previous_recommendation(target_date: pd.Timestamp, history_file: Path) -> pd.DataFrame:
    position_rows: list[dict] = []
    for path in sorted(uds.HISTORY_DIR.glob("snapshot_*.json")):
        try:
            payload = uds._read_json(path)
            trade_date = pd.Timestamp(payload.get("trade_date")).normalize()
        except Exception:
            continue
        if trade_date >= target_date:
            continue
        clarity = payload.get("clarity_signal") or {}
        sells = clarity.get("sell") or []
        buys = clarity.get("buy") or []
        if sells:
            position_rows = []
        elif buys:
            position_rows = buys

    if position_rows:
        return uds._state_rows_to_frame(position_rows)

    if not history_file.exists():
        return pd.DataFrame()
    try:
        hist = pd.read_csv(history_file)
    except Exception:
        return pd.DataFrame()
    required = {"snapshot_time", "板块名称", "涨跌幅", "综合博弈得分", "逃顶风险分"}
    if hist.empty or not required.issubset(hist.columns):
        return pd.DataFrame()
    hist = hist.copy()
    if "数据日期" in hist.columns:
        hist["_trade_date"] = pd.to_datetime(hist["数据日期"].astype(str).str[:10], errors="coerce")
        hist = hist[hist["_trade_date"].dt.normalize() < target_date]
    hist["slot"] = hist["snapshot_time"].astype(str).str[:16]
    recovered = pd.DataFrame()
    for slot in sorted(hist["slot"].dropna().unique().tolist()):
        prev = hist[hist["slot"] == slot].copy()
        prev_breadth = float((pd.to_numeric(prev["涨跌幅"], errors="coerce") > 0).mean())
        if prev_breadth < uds.SELL_BREADTH_FLOOR:
            recovered = pd.DataFrame()
            continue
        if prev_breadth < uds.BUY_BREADTH_FLOOR:
            continue
        cand = (
            prev[
                (pd.to_numeric(prev["综合博弈得分"], errors="coerce") >= uds.MIN_SCORE)
                & (pd.to_numeric(prev["逃顶风险分"], errors="coerce") < uds.MAX_RISK)
            ]
            .sort_values("综合博弈得分", ascending=False)
            .head(1)
            .copy()
        )
        if not cand.empty:
            recovered = cand
    return recovered


def _load_recommendation_holding(target_date: pd.Timestamp, history_file: Path) -> pd.DataFrame:
    state = uds._read_json(uds.RECOMMENDATION_STATE_FILE)
    if isinstance(state, dict) and state:
        if state.get("last_sell_trade_date") == target_date.strftime("%Y-%m-%d"):
            return pd.DataFrame()
        holdings = state.get("holdings", [])
        if holdings:
            return uds._state_rows_to_frame(holdings)
        if state.get("as_of_trade_date"):
            return pd.DataFrame()
    return _recover_previous_recommendation(target_date, history_file)


def _write_recommendation_state(
    target_date: pd.Timestamp,
    holdings: pd.DataFrame,
    breadth_ratio: float,
    last_sell: pd.DataFrame | None = None,
    action: str = "hold",
) -> None:
    previous_state = uds._read_json(uds.RECOMMENDATION_STATE_FILE)
    rows = holdings.to_dict(orient="records") if not holdings.empty else []
    sell_rows = last_sell.to_dict(orient="records") if last_sell is not None and not last_sell.empty else []
    trade_date = target_date.strftime("%Y-%m-%d")
    prior_buy_date = previous_state.get("last_buy_trade_date", "") if isinstance(previous_state, dict) else ""
    prior_sell_date = previous_state.get("last_sell_trade_date", "") if isinstance(previous_state, dict) else ""
    uds._write_json(uds.RECOMMENDATION_STATE_FILE, {
        "as_of_trade_date": trade_date,
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "market_breadth": breadth_ratio,
        "position_state": "holding" if rows else "flat",
        "last_action": action,
        "holdings": rows,
        "last_buy_trade_date": trade_date if action == "buy" else prior_buy_date,
        "last_sell_trade_date": trade_date if action == "sell" else prior_sell_date,
        "last_sell": sell_rows if action == "sell" else [],
    })


def _build_recommendations(df: pd.DataFrame, history_file: Path, target_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    breadth_ratio = float((df["涨跌幅"] > 0).mean())
    today_buy = pd.DataFrame()
    if breadth_ratio >= uds.BUY_BREADTH_FLOOR:
        today_buy = (
            df[(df["综合博弈得分"] >= uds.MIN_SCORE) & (df["逃顶风险分"] < uds.MAX_RISK)]
            .sort_values("综合博弈得分", ascending=False)
            .head(1)
            .copy()
        )

    previous_hold = _load_recommendation_holding(target_date, history_file)
    sell_rows = []
    if breadth_ratio < uds.SELL_BREADTH_FLOOR:
        if not previous_hold.empty:
            for _, hold in previous_hold.iterrows():
                name = str(hold.get("板块名称", ""))
                if not name:
                    continue
                match = df[df["板块名称"].astype(str) == name]
                item = match.iloc[0].to_dict() if not match.empty else {"板块名称": name}
                item["卖出原因"] = f"市场广度 < {uds.SELL_BREADTH_FLOOR:.0%}"
                if "对应ETF" not in item or not item.get("对应ETF"):
                    item["对应ETF"] = hold.get("对应ETF", "")
                sell_rows.append(item)
        today_sell = pd.DataFrame(sell_rows)
        _write_recommendation_state(target_date, pd.DataFrame(), breadth_ratio, today_sell, action="sell" if sell_rows else "flat")
    elif not today_buy.empty:
        _write_recommendation_state(target_date, today_buy, breadth_ratio, action="buy")
    else:
        _write_recommendation_state(target_date, previous_hold, breadth_ratio, action="hold" if not previous_hold.empty else "flat")
    return today_buy, pd.DataFrame(sell_rows)


def main() -> int:
    uds._target_trade_date = _target_trade_date
    uds._recover_previous_recommendation = _recover_previous_recommendation
    uds._load_recommendation_holding = _load_recommendation_holding
    uds._write_recommendation_state = _write_recommendation_state
    uds._build_recommendations = _build_recommendations
    return uds.main()


if __name__ == "__main__":
    raise SystemExit(main())
