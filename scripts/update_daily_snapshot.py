from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("SW_PROCESSED_CACHE_TTL_SECONDS", "0")

from avix_utils import calculate_and_store_avix, load_avix_history, build_avix_s3_s4_signal_history  # noqa: E402
from ui_theme import clean_signal  # noqa: E402
from utils import (  # noqa: E402
    SW_CODE_MAPPING,
    fetch_historical_baselines,
    get_processed_sw_data,
    _build_strategy_backtest,
    _build_walk_forward_scores,
    _summarize_backtest,
)

DATA_DIR = ROOT / "data"
LATEST_FILE = DATA_DIR / "latest_snapshot.json"
STATUS_FILE = DATA_DIR / "update_status.json"
PROCESSED_CACHE_FILE = DATA_DIR / "processed_sw_cache.csv"
HISTORY_DIR = DATA_DIR / "history"
AVIX_DIR = DATA_DIR / "avix"
BACKTEST_DIR = DATA_DIR / "backtest"
RECOMMENDATION_STATE_FILE = DATA_DIR / "recommendation_state.json"

SAMPLE_SECTORS = ["银行", "电子", "煤炭", "基础化工", "机械设备", "食品饮料", "医药生物", "电力设备"]

BUY_BREADTH_FLOOR = 0.70
SELL_BREADTH_FLOOR = 0.35
MIN_SCORE = 54.0
MAX_RISK = 45.0


def _json_default(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _trade_calendar() -> pd.Series:
    cal = ak.tool_trade_date_hist_sina()
    date_col = next((c for c in cal.columns if "date" in str(c).lower() or "日期" in str(c)), cal.columns[0])
    return pd.to_datetime(cal[date_col], errors="coerce").dropna().sort_values().dt.normalize()


def _is_trade_day(day: pd.Timestamp, calendar: pd.Series) -> bool:
    day = pd.Timestamp(day).normalize()
    return bool((calendar == day).any())


def _target_trade_date(now: datetime, calendar: pd.Series) -> tuple[pd.Timestamp | None, str]:
    today = pd.Timestamp(now.date())
    if dt_time(18, 0) <= now.time() <= dt_time(23, 59):
        if not _is_trade_day(today, calendar):
            return None, "今天不是 A 股交易日"
        return today, "evening"

    eligible = calendar[calendar <= today]
    if eligible.empty:
        return None, "交易日历为空"
    return pd.Timestamp(eligible.iloc[-1]).normalize(), "morning_or_manual"


def _status(target: str | None, status: str, reason: str, latest_source_dates: dict | None = None) -> dict:
    return {
        "target_date": target,
        "target_trade_date": target,
        "last_attempt_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "reason": reason,
        "latest_source_dates": latest_source_dates or {},
    }


def _validate_ready_dates(target_date: pd.Timestamp) -> tuple[bool, dict[str, str]]:
    bases = fetch_historical_baselines()
    latest_dates = {name: str((bases.get(name) or {}).get("data_date", "")) for name in SAMPLE_SECTORS}
    target_str = target_date.strftime("%Y-%m-%d")
    ok = all(latest_dates.get(name) == target_str for name in SAMPLE_SECTORS)
    return ok, latest_dates


def _validate_dataframe(df: pd.DataFrame, target_date: pd.Timestamp) -> list[str]:
    errors: list[str] = []
    required = ["板块名称", "对应ETF", "涨跌幅", "综合博弈得分", "逃顶风险分"]
    if len(df) < 30:
        errors.append(f"申万一级行业数量不足：{len(df)}")
    for col in required:
        if col not in df.columns:
            errors.append(f"缺少字段：{col}")
    if "数据日期" in df.columns:
        target_str = target_date.strftime("%Y-%m-%d")
        dates = df["数据日期"].astype(str).str[:10].dropna().unique().tolist()
        if dates != [target_str]:
            errors.append(f"trade_date 不一致：{dates} != {target_str}")
    else:
        errors.append("缺少字段：数据日期")
    for col in ["综合博弈得分", "逃顶风险分"]:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").isna().mean() > 0.05:
            errors.append(f"{col} 大面积为空")
    if "涨跌幅" in df.columns:
        breadth = float((pd.to_numeric(df["涨跌幅"], errors="coerce") > 0).mean())
        if not 0 <= breadth <= 1:
            errors.append("市场广度不在合法区间")
    if "成分股覆盖数" in df.columns and pd.to_numeric(df["成分股覆盖数"], errors="coerce").fillna(0).max() <= 0:
        errors.append("成分股覆盖数全部为 0")
    return errors


def _load_valid_processed_cache(target_date: pd.Timestamp) -> pd.DataFrame:
    if not PROCESSED_CACHE_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PROCESSED_CACHE_FILE)
    except Exception:
        return pd.DataFrame()
    return df if not _validate_dataframe(df, target_date) else pd.DataFrame()


def _state_rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows or [])
    for col in ["涨跌幅", "综合博弈得分", "逃顶风险分"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _recover_previous_recommendation(target_date: pd.Timestamp, history_file: Path) -> pd.DataFrame:
    prior_rows: list[dict] = []
    for path in sorted(HISTORY_DIR.glob("snapshot_*.json")):
        try:
            payload = _read_json(path)
            trade_date = pd.Timestamp(payload.get("trade_date")).normalize()
        except Exception:
            continue
        if trade_date >= target_date:
            continue
        buys = ((payload.get("clarity_signal") or {}).get("buy") or [])
        if buys:
            prior_rows = buys

    if prior_rows:
        return _state_rows_to_frame(prior_rows)

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
        if prev_breadth < BUY_BREADTH_FLOOR:
            continue
        cand = (
            prev[
                (pd.to_numeric(prev["综合博弈得分"], errors="coerce") >= MIN_SCORE)
                & (pd.to_numeric(prev["逃顶风险分"], errors="coerce") < MAX_RISK)
            ]
            .sort_values("综合博弈得分", ascending=False)
            .head(1)
            .copy()
        )
        if not cand.empty:
            recovered = cand
    return recovered


def _load_recommendation_holding(target_date: pd.Timestamp, history_file: Path) -> pd.DataFrame:
    state = _read_json(RECOMMENDATION_STATE_FILE)
    if isinstance(state, dict) and state.get("last_sell_trade_date") == target_date.strftime("%Y-%m-%d"):
        return pd.DataFrame()
    holdings = state.get("holdings", []) if isinstance(state, dict) else []
    if holdings:
        return _state_rows_to_frame(holdings)
    return _recover_previous_recommendation(target_date, history_file)


def _write_recommendation_state(
    target_date: pd.Timestamp,
    holdings: pd.DataFrame,
    breadth_ratio: float,
    last_sell: pd.DataFrame | None = None,
) -> None:
    rows = holdings.to_dict(orient="records") if not holdings.empty else []
    sell_rows = last_sell.to_dict(orient="records") if last_sell is not None and not last_sell.empty else []
    _write_json(RECOMMENDATION_STATE_FILE, {
        "as_of_trade_date": target_date.strftime("%Y-%m-%d"),
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "market_breadth": breadth_ratio,
        "holdings": rows,
        "last_sell_trade_date": target_date.strftime("%Y-%m-%d") if sell_rows else "",
        "last_sell": sell_rows,
    })


def _build_recommendations(df: pd.DataFrame, history_file: Path, target_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    breadth_ratio = float((df["涨跌幅"] > 0).mean())
    today_buy = pd.DataFrame()
    if breadth_ratio >= BUY_BREADTH_FLOOR:
        today_buy = (
            df[(df["综合博弈得分"] >= MIN_SCORE) & (df["逃顶风险分"] < MAX_RISK)]
            .sort_values("综合博弈得分", ascending=False)
            .head(1)
            .copy()
        )

    previous_hold = _load_recommendation_holding(target_date, history_file)
    sell_rows = []
    if breadth_ratio < SELL_BREADTH_FLOOR:
        if previous_hold.empty:
            state = _read_json(RECOMMENDATION_STATE_FILE)
            if state.get("last_sell_trade_date") == target_date.strftime("%Y-%m-%d"):
                sell_rows = state.get("last_sell", []) or []
        else:
            for _, hold in previous_hold.iterrows():
                name = str(hold.get("板块名称", ""))
                if not name:
                    continue
                match = df[df["板块名称"].astype(str) == name]
                item = match.iloc[0].to_dict() if not match.empty else {"板块名称": name}
                item["卖出原因"] = f"市场广度 < {SELL_BREADTH_FLOOR:.0%}"
                if "对应ETF" not in item or not item.get("对应ETF"):
                    item["对应ETF"] = hold.get("对应ETF", "")
                sell_rows.append(item)
        today_sell = pd.DataFrame(sell_rows)
        _write_recommendation_state(target_date, pd.DataFrame(), breadth_ratio, today_sell)
    elif not today_buy.empty:
        _write_recommendation_state(target_date, today_buy, breadth_ratio)
    else:
        _write_recommendation_state(target_date, previous_hold, breadth_ratio)
    return today_buy, pd.DataFrame(sell_rows)


def _load_cached_avix() -> tuple[dict, pd.DataFrame]:
    try:
        df = load_avix_history()
        if not df.empty and {"trade_date", "avix"}.issubset(df.columns):
            df = df.copy()
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
            df = df.dropna(subset=["trade_date"]).sort_values("trade_date")
            latest = df.iloc[-1].to_dict()
            latest["quality"] = str(latest.get("quality", "历史缓存"))
            return latest, df
    except Exception:
        pass

    for path in [DATA_DIR / "avix_300_close_mid.csv", DATA_DIR / "avix_300_hist_clean.csv"]:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if df.empty or "avix" not in df.columns:
                continue
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
                df = df.dropna(subset=["trade_date"]).sort_values("trade_date")
            latest = df.iloc[-1].to_dict()
            latest["quality"] = str(latest.get("quality", "历史缓存"))
            return latest, df
        except Exception:
            continue
    return {}, pd.DataFrame()


def _avix_payload_from_history(latest: dict | None, avix_hist: pd.DataFrame) -> dict:
    signal_hist = build_avix_s3_s4_signal_history(avix_hist)
    if signal_hist.empty:
        raise ValueError("AVIX S3/S4 信号历史为空")
    return {
        "latest": latest or {},
        "history": avix_hist.to_dict(orient="records") if not avix_hist.empty else [],
        "signal_history": signal_hist.to_dict(orient="records") if not signal_hist.empty else [],
    }


def _refresh_avix_payload(skip_avix: bool = False) -> dict:
    if skip_avix:
        latest, avix_hist = _load_cached_avix()
        payload = _avix_payload_from_history(latest, avix_hist)
        payload["note"] = "本次跳过 AVIX 实时刷新，使用历史缓存"
        return payload

    latest = calculate_and_store_avix()
    avix_hist = load_avix_history()
    return _avix_payload_from_history(latest, avix_hist)


def _build_backtest_payload(lookback_days: int = 360) -> dict:
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    try:
        scored = _build_walk_forward_scores(520)
        if scored.empty:
            return {
                "generated_at": generated_at,
                "lookback_days": lookback_days,
                "status": "empty",
                "error": "滚动评分结果为空",
            }

        bt = _build_strategy_backtest(scored, lookback_days, "top1_breadth_final")
        summary = _summarize_backtest(bt)

        rows = []
        strategy_names = {
            "top1_breadth_final": "Top1 + 广度过滤（70/35/54/45）",
            "top1": "Top1 单押",
            "top3_balanced": "Top3 稳健等权",
            "top3_regime": "Top3 + 广度过滤",
        }
        for strategy, label in strategy_names.items():
            s_bt = _build_strategy_backtest(scored, lookback_days, strategy)
            s_summary = _summarize_backtest(s_bt)
            if not s_summary:
                continue
            rows.append({
                "策略": label,
                "方向胜率": s_summary["胜率"],
                "相对胜率": s_summary["相对胜率"],
                "累计收益": s_summary["累计收益"],
                "等权基准": s_summary["基准收益"],
                "年化收益": s_summary["年化收益"],
                "最大回撤": s_summary["最大回撤"],
                "夏普比率": s_summary["夏普比率"],
                "交易日数": s_summary["交易日数"],
            })
        cmp_df = pd.DataFrame(rows)

        robust_rows = []
        for window in (120, 240, 360, 520):
            w_bt = _build_strategy_backtest(scored, int(window), "top1_breadth_final")
            w_summary = _summarize_backtest(w_bt)
            if not w_summary:
                continue
            robust_rows.append({
                "窗口": f"{int(window)}日",
                "方向胜率": w_summary["胜率"],
                "相对胜率": w_summary["相对胜率"],
                "累计收益": w_summary["累计收益"],
                "等权基准": w_summary["基准收益"],
                "年化收益": w_summary["年化收益"],
                "最大回撤": w_summary["最大回撤"],
                "夏普比率": w_summary["夏普比率"],
            })
        robust_df = pd.DataFrame(robust_rows)

        if bt.empty or not summary:
            return {
                "generated_at": generated_at,
                "lookback_days": lookback_days,
                "status": "empty",
                "error": "回测结果为空",
            }

        recent = bt.tail(30).copy()
        if "date" in recent.columns:
            recent["date"] = pd.to_datetime(recent["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "strategy_ret" in recent.columns:
            recent["模型次日收益"] = pd.to_numeric(recent["strategy_ret"], errors="coerce") * 100
        if "benchmark_ret" in recent.columns:
            recent["等权次日收益"] = pd.to_numeric(recent["benchmark_ret"], errors="coerce") * 100

        curve_cols = [c for c in ["date", "strategy_nav", "benchmark_nav"] if c in bt.columns]
        curve = bt[curve_cols].tail(260).copy()
        if "date" in curve.columns:
            curve["date"] = pd.to_datetime(curve["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        signal_cols = [
            "date",
            "持有板块",
            "综合博弈得分",
            "风险分",
            "风险简分",
            "模型次日收益",
            "等权次日收益",
        ]
        return {
            "generated_at": generated_at,
            "lookback_days": lookback_days,
            "status": "ready",
            "summary": summary,
            "recent_curve": curve.to_dict(orient="records"),
            "strategy_comparison": cmp_df.to_dict(orient="records") if not cmp_df.empty else [],
            "window_robustness": robust_df.to_dict(orient="records") if not robust_df.empty else [],
            "recent_signals": recent[[c for c in signal_cols if c in recent.columns]].to_dict(orient="records"),
        }
    except Exception as exc:
        return {
            "generated_at": generated_at,
            "lookback_days": lookback_days,
            "status": "error",
            "error": str(exc),
        }


def _latest_strategy_payload(signal_history: list[dict], strategy: str) -> dict:
    if not signal_history:
        return {"latest": {}, "recent_buy": [], "recent_sell": []}
    df = pd.DataFrame(signal_history).copy()
    if df.empty:
        return {"latest": {}, "recent_buy": [], "recent_sell": []}
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.sort_values("trade_date")
    buy_col = f"{strategy}_buy"
    sell_col = f"{strategy}_sell"
    sell_reason_col = f"{strategy}_sell_reason"
    cols = [
        "trade_date", "execution_trade_date", "execution_sse_open", "avix",
        "sse_open", "sse_close", "sse_ret1", "sse_ret10", "sse_ma5",
        f"{strategy}_signal", buy_col, sell_col, sell_reason_col,
    ]
    cols = [c for c in cols if c in df.columns]
    latest = df.iloc[-1][cols].to_dict() if cols else df.iloc[-1].to_dict()
    recent_buy = df[df[buy_col].astype(bool)].tail(20)[cols].to_dict(orient="records") if buy_col in df.columns else []
    recent_sell = df[df[sell_col].astype(bool)].tail(20)[cols].to_dict(orient="records") if sell_col in df.columns else []
    return {
        "latest": latest,
        "recent_buy": recent_buy,
        "recent_sell": recent_sell,
    }


def _snapshot_payload(df: pd.DataFrame, target_date: pd.Timestamp, skip_avix: bool = False) -> dict:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    df = df.copy()
    if "终极信号" in df.columns:
        df["信号"] = df["终极信号"].map(clean_signal)
    else:
        df["信号"] = "未分类"

    breadth_pct = float((df["涨跌幅"] > 0).mean() * 100)
    risk_pool = df[df["信号"].str.contains("强制清仓|崩盘|鱼尾|诱多|强弩|战术减仓", na=False)]
    opp_pool = df[df["信号"].str.contains("满仓|底仓|顺势", na=False)]
    today_buy, today_sell = _build_recommendations(df, DATA_DIR / "sw_board_history.csv", target_date)

    avix_payload = _refresh_avix_payload(skip_avix=skip_avix)
    AVIX_DIR.mkdir(parents=True, exist_ok=True)
    avix_history = avix_payload.get("history", []) or []
    if avix_history:
        pd.DataFrame(avix_history).to_csv(AVIX_DIR / "avix_history.csv", index=False)

    signal_history = avix_payload.get("signal_history", []) or []

    sectors = df.to_dict(orient="records")
    return {
        "trade_date": target_date.strftime("%Y-%m-%d"),
        "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ready",
        "market_breadth": breadth_pct,
        "coverage_mean": float(df["成分股覆盖数"].mean()) if "成分股覆盖数" in df.columns else 0.0,
        "market_environment": str(df["市场环境"].iloc[0]) if "市场环境" in df.columns else "震荡市",
        "market_heat": float(df["market_heat"].iloc[0]) if "market_heat" in df.columns else 0.0,
        "top_risk": "无" if risk_pool.empty else str(risk_pool.sort_values("逃顶风险分", ascending=False).iloc[0]["板块名称"]),
        "top_opportunity": "等待" if opp_pool.empty else str(opp_pool.sort_values("综合博弈得分", ascending=False).iloc[0]["板块名称"]),
        "clarity_signal": {
            "buy": today_buy.to_dict(orient="records"),
            "sell": today_sell.to_dict(orient="records"),
            "rules": {
                "buy_breadth_floor": BUY_BREADTH_FLOOR,
                "sell_breadth_floor": SELL_BREADTH_FLOOR,
                "min_score": MIN_SCORE,
                "max_risk": MAX_RISK,
            },
        },
        "s3_signal": _latest_strategy_payload(signal_history, "s3"),
        "s4_signal": _latest_strategy_payload(signal_history, "s4"),
        "s3_s4_signal": _latest_strategy_payload(signal_history, "s3_s4"),
        "avix": avix_payload,
        "sectors": sectors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-date", help="YYYY-MM-DD；默认按北京时间和交易日历推导")
    parser.add_argument("--force", action="store_true", help="忽略已更新状态，强制重算")
    parser.add_argument("--skip-avix", action="store_true", help="本地快速验证使用：跳过 AVIX 外部刷新，仅使用已有缓存")
    parser.add_argument("--skip-backtest", action="store_true", help="本地快速验证使用：跳过回测刷新，复用已有回测缓存")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    calendar = _trade_calendar()
    if args.target_date:
        target_date = pd.Timestamp(args.target_date).normalize()
        if not _is_trade_day(target_date, calendar):
            payload = _status(args.target_date, "skipped", "目标日期不是 A 股交易日")
            _write_json(STATUS_FILE, payload)
            return 0
    else:
        target_date, reason = _target_trade_date(now, calendar)
        if target_date is None:
            payload = _status(None, "skipped", reason)
            _write_json(STATUS_FILE, payload)
            return 0

    target_str = target_date.strftime("%Y-%m-%d")
    previous = _read_json(LATEST_FILE)
    if not args.force and previous.get("trade_date") == target_str and previous.get("status") == "ready":
        _write_json(STATUS_FILE, _status(target_str, "skipped", "该交易日已经成功更新"))
        return 0

    ready, latest_dates = _validate_ready_dates(target_date)
    if not ready:
        _write_json(STATUS_FILE, _status(target_str, "waiting_partial", "申万样本行业数据尚未全部更新", latest_dates))
        return 0

    df = _load_valid_processed_cache(target_date)
    if df.empty:
        df = get_processed_sw_data()
    errors = _validate_dataframe(df, target_date)
    if errors:
        _write_json(STATUS_FILE, _status(target_str, "failed_validation", "；".join(errors), latest_dates))
        return 1

    payload = _snapshot_payload(df, target_date, skip_avix=args.skip_avix)
    if args.skip_backtest:
        backtest_payload = _read_json(BACKTEST_DIR / "strategy_summary.json") or {
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
            "lookback_days": 360,
            "status": "skipped",
            "error": "本次跳过回测刷新，且不存在历史回测缓存",
        }
    else:
        backtest_payload = _build_backtest_payload()
    _write_json(HISTORY_DIR / f"snapshot_{target_str}.json", payload)
    _write_json(LATEST_FILE, payload)
    _write_json(BACKTEST_DIR / "strategy_summary.json", backtest_payload)
    _write_json(STATUS_FILE, _status(target_str, "ready", "更新成功", latest_dates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
