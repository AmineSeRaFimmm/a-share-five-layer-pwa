from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
LATEST_FILE = DATA_DIR / "latest_snapshot.json"
RECOMMENDATION_STATE_FILE = DATA_DIR / "recommendation_state.json"
LIVE_HISTORY_FILE = DATA_DIR / "live_recommendation_history.csv"
LEDGER_VERSION = "live_state_v2"

DEFAULT_RULES = {
    "buy_breadth_floor": 0.70,
    "sell_breadth_floor": 0.35,
    "min_score": 54.0,
    "max_risk": 45.0,
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        return json.loads(text) if text else {}
    except Exception:
        return {}


def _read_existing_ledger() -> pd.DataFrame:
    if not LIVE_HISTORY_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(LIVE_HISTORY_FILE)
    except Exception:
        return pd.DataFrame()
    if df.empty or "账本版本" not in df.columns:
        return pd.DataFrame()
    return df[df["账本版本"].astype(str) == LEDGER_VERSION].copy()


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _as_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _normalize_breadth(value: object) -> float:
    val = _as_float(value, 0.0)
    return val / 100.0 if val > 1.0 else val


def _find_sector_row(payload: dict, sector_name: str) -> dict:
    if not sector_name:
        return {}
    for row in payload.get("sectors", []) or []:
        if _as_text(row.get("板块名称")) == sector_name:
            return row if isinstance(row, dict) else {}
    return {}


def _metric_row(row: dict, payload: dict, fallback_sector: str = "") -> dict:
    sector = _as_text(row.get("板块名称")) or fallback_sector
    if not row and sector:
        row = _find_sector_row(payload, sector)
    return {
        "信号板块": sector,
        "对应ETF": _as_text(row.get("对应ETF")),
        "涨跌幅": _as_float(row.get("涨跌幅"), np.nan),
        "综合博弈得分": _as_float(row.get("综合博弈得分"), np.nan),
        "逃顶风险分": _as_float(row.get("逃顶风险分"), np.nan),
        "入场共振分": _as_float(row.get("入场共振分"), np.nan),
    }


def _previous_after_position(existing: pd.DataFrame, trade_date: str) -> str:
    if existing.empty or "trade_date" not in existing.columns or "动作后持仓" not in existing.columns:
        return ""
    prior = existing.copy()
    prior["_date"] = pd.to_datetime(prior["trade_date"], errors="coerce")
    target = pd.to_datetime(trade_date, errors="coerce")
    prior = prior[pd.notna(prior["_date"])]
    if pd.notna(target):
        prior = prior[prior["_date"] < target]
    prior = prior.sort_values("_date")
    if prior.empty:
        return ""
    value = _as_text(prior.iloc[-1].get("动作后持仓"))
    return "" if value == "空仓" else value


def _first_record(rows: list[dict]) -> dict:
    return rows[0] if rows and isinstance(rows[0], dict) else {}


def build_live_recommendation_history() -> pd.DataFrame:
    payload = _read_json(LATEST_FILE)
    state = _read_json(RECOMMENDATION_STATE_FILE)
    existing = _read_existing_ledger()

    trade_date = _as_text(state.get("as_of_trade_date"))[:10] or _as_text(payload.get("trade_date"))[:10]
    if not trade_date:
        return existing

    updated_at = _as_text(state.get("updated_at")) or _as_text(payload.get("updated_at"))
    clarity = payload.get("clarity_signal") or {}
    rules = clarity.get("rules") or DEFAULT_RULES
    market_breadth = _normalize_breadth(state.get("market_breadth", payload.get("market_breadth")))
    action_state = _as_text(state.get("last_action")) or "flat"
    holdings = state.get("holdings") if isinstance(state.get("holdings"), list) else []
    last_sell = state.get("last_sell") if isinstance(state.get("last_sell"), list) else []
    holding_row = _first_record(holdings)
    sell_row = _first_record(last_sell)
    prior_position = _previous_after_position(existing, trade_date)

    recommendation_action = action_state if action_state in {"buy", "sell", "hold", "flat"} else "flat"
    before = prior_position or "空仓"
    after = "空仓"
    buy_sector = ""
    sell_sector = ""
    signal_sector = ""
    metric_source: dict = {}
    reason = "无持仓且无买入信号"

    if recommendation_action == "buy":
        buy_sector = _as_text(holding_row.get("板块名称"))
        signal_sector = buy_sector
        after = buy_sector or "空仓"
        position_action = "买入" if before == "空仓" else ("持有" if before == after else "换仓")
        reason = f"市场广度 >= {_as_float(rules.get('buy_breadth_floor'), DEFAULT_RULES['buy_breadth_floor']):.0%}，综合分/风险满足阈值"
        metric_source = holding_row or _find_sector_row(payload, signal_sector)
    elif recommendation_action == "sell":
        sell_sector = _as_text(sell_row.get("板块名称")) or prior_position
        signal_sector = sell_sector
        before = sell_sector or before
        after = "空仓"
        position_action = "卖出" if signal_sector else "空仓"
        reason = _as_text(sell_row.get("卖出原因")) or f"市场广度 < {_as_float(rules.get('sell_breadth_floor'), DEFAULT_RULES['sell_breadth_floor']):.0%}"
        metric_source = sell_row or _find_sector_row(payload, signal_sector)
    elif recommendation_action == "hold":
        hold_sector = _as_text(holding_row.get("板块名称")) or prior_position
        signal_sector = hold_sector
        before = prior_position or hold_sector or "空仓"
        after = hold_sector or "空仓"
        position_action = "持有" if hold_sector else "空仓"
        reason = "无新买入/卖出信号，延续上一交易日持仓"
        metric_source = holding_row or _find_sector_row(payload, signal_sector)
    else:
        recommendation_action = "flat"
        before = prior_position or "空仓"
        after = "空仓"
        position_action = "空仓"
        reason = "无持仓且无买入信号"
        metric_source = {}

    metric = _metric_row(metric_source, payload, signal_sector)
    row = {
        "trade_date": trade_date,
        "updated_at": updated_at,
        "source": "recommendation_state+latest_snapshot",
        "账本版本": LEDGER_VERSION,
        "推荐动作": recommendation_action,
        "仓位动作": position_action,
        "动作前持仓": before,
        "动作后持仓": after,
        "信号板块": metric["信号板块"],
        "买入板块": buy_sector,
        "卖出板块": sell_sector,
        "动作原因": reason,
        "市场广度": market_breadth,
        "综合博弈得分": metric["综合博弈得分"],
        "逃顶风险分": metric["逃顶风险分"],
        "入场共振分": metric["入场共振分"],
        "涨跌幅": metric["涨跌幅"],
        "对应ETF": metric["对应ETF"],
        "规则买入广度": _as_float(rules.get("buy_breadth_floor"), DEFAULT_RULES["buy_breadth_floor"]),
        "规则卖出广度": _as_float(rules.get("sell_breadth_floor"), DEFAULT_RULES["sell_breadth_floor"]),
        "规则最低综合分": _as_float(rules.get("min_score"), DEFAULT_RULES["min_score"]),
        "规则最高风险分": _as_float(rules.get("max_risk"), DEFAULT_RULES["max_risk"]),
    }

    if not existing.empty and "trade_date" in existing.columns:
        existing = existing[existing["trade_date"].astype(str).str[:10] != trade_date].copy()
    out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    out = out.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    return out


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ledger = build_live_recommendation_history()
    ledger.to_csv(LIVE_HISTORY_FILE, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(ledger)} live recommendation rows to {LIVE_HISTORY_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
