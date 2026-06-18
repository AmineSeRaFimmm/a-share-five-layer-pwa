from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
HISTORY_DIR = DATA_DIR / "history"
LATEST_FILE = DATA_DIR / "latest_snapshot.json"
LIVE_HISTORY_FILE = DATA_DIR / "live_recommendation_history.csv"

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


def _payloads_by_trade_date() -> list[dict]:
    by_date: dict[str, dict] = {}
    paths = sorted(HISTORY_DIR.glob("snapshot_*.json"))
    if LATEST_FILE.exists():
        paths.append(LATEST_FILE)

    for path in paths:
        payload = _read_json(path)
        trade_date = _as_text(payload.get("trade_date"))[:10]
        if not trade_date:
            continue
        payload["_source_file"] = str(path.relative_to(ROOT))
        current = by_date.get(trade_date)
        current_updated = _as_text(current.get("updated_at")) if current else ""
        payload_updated = _as_text(payload.get("updated_at"))
        if current is None or payload_updated >= current_updated:
            by_date[trade_date] = payload
    return [by_date[k] for k in sorted(by_date)]


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


def build_live_recommendation_history() -> pd.DataFrame:
    rows: list[dict] = []
    current_position = ""

    for payload in _payloads_by_trade_date():
        trade_date = _as_text(payload.get("trade_date"))[:10]
        updated_at = _as_text(payload.get("updated_at"))
        clarity = payload.get("clarity_signal") or {}
        buys = clarity.get("buy") or []
        sells = clarity.get("sell") or []
        rules = clarity.get("rules") or DEFAULT_RULES
        market_breadth = _normalize_breadth(payload.get("market_breadth"))
        before = current_position or "空仓"

        buy_row = buys[0] if buys and isinstance(buys[0], dict) else {}
        sell_row = sells[0] if sells and isinstance(sells[0], dict) else {}

        action = "空仓"
        recommendation_action = "flat"
        after = "空仓"
        buy_sector = ""
        sell_sector = ""
        reason = "无持仓且无买入信号"
        metric_source: dict = {}
        signal_sector = ""

        if sells:
            recommendation_action = "sell"
            sell_sector = _as_text(sell_row.get("板块名称")) or (current_position if current_position else _as_text(sell_row.get("信号板块")))
            signal_sector = sell_sector
            action = "卖出"
            after = "空仓"
            reason = _as_text(sell_row.get("卖出原因")) or f"市场广度 < {_as_float(rules.get('sell_breadth_floor'), DEFAULT_RULES['sell_breadth_floor']):.0%}"
            metric_source = sell_row or _find_sector_row(payload, sell_sector)
            current_position = ""
        elif buys:
            recommendation_action = "buy"
            buy_sector = _as_text(buy_row.get("板块名称"))
            signal_sector = buy_sector
            if not current_position:
                action = "买入"
            elif current_position == buy_sector:
                action = "持有"
            else:
                action = "换仓"
            after = buy_sector or "空仓"
            reason = f"市场广度 >= {_as_float(rules.get('buy_breadth_floor'), DEFAULT_RULES['buy_breadth_floor']):.0%}，综合分/风险满足阈值"
            metric_source = buy_row
            current_position = buy_sector
        elif current_position:
            recommendation_action = "hold"
            signal_sector = current_position
            action = "持有"
            after = current_position
            reason = "无新买入/卖出信号，延续上一交易日持仓"
            metric_source = _find_sector_row(payload, current_position)
        else:
            metric_source = {}

        metric = _metric_row(metric_source, payload, signal_sector)
        rows.append({
            "trade_date": trade_date,
            "updated_at": updated_at,
            "source": _as_text(payload.get("_source_file")),
            "推荐动作": recommendation_action,
            "仓位动作": action,
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
        })

    out = pd.DataFrame(rows)
    if not out.empty:
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
