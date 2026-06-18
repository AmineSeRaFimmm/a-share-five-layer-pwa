from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from snapshot_store import load_backtest_summary
from ui_theme import inject_theme, metric_card, page_header


st.set_page_config(page_title="模型回测", layout="wide")
inject_theme()

page_header("模型回测", "收盘信号，下一交易日验证", "Walk-forward")

payload = load_backtest_summary()
summary = payload.get("summary", {}) or {}
bt = pd.DataFrame(payload.get("recent_curve", []) or [])
cmp_df = pd.DataFrame(payload.get("strategy_comparison", []) or [])
robust_df = pd.DataFrame(payload.get("window_robustness", []) or [])
recent = pd.DataFrame(payload.get("recent_signals", []) or [])
generated_at = payload.get("generated_at", "-")
lookback_days = payload.get("lookback_days", "-")
score_basis = payload.get("score_basis", "-")
score_basis_note = payload.get("score_basis_note", "")
transaction_cost_rate = payload.get("transaction_cost_rate", summary.get("单边成本假设", None))

FULLRISK_RESULTS_PATH = Path("data/backtest/top1_fullrisk_grid_300_results.csv")
FULLRISK_TOP_PATH = Path("data/backtest/top1_fullrisk_grid_300_dedup_top.csv")
FULLRISK_METADATA_PATH = Path("data/backtest/top1_fullrisk_grid_300_metadata.json")
fullrisk_results = pd.read_csv(FULLRISK_RESULTS_PATH) if FULLRISK_RESULTS_PATH.exists() else pd.DataFrame()
fullrisk_top = pd.read_csv(FULLRISK_TOP_PATH) if FULLRISK_TOP_PATH.exists() else pd.DataFrame()
try:
    fullrisk_meta = json.loads(FULLRISK_METADATA_PATH.read_text(encoding="utf-8")) if FULLRISK_METADATA_PATH.exists() else {}
except Exception:
    fullrisk_meta = {}


def pct(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value) * 100:+.2f}%"
    except Exception:
        return "-"


def pct_plain(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "-"


def number(value: object, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def count_value(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.0f}"
    except Exception:
        return "-"


def bp_text(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value) * 10000:.1f}bp"
    except Exception:
        return "-"


def _to_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "买入广度", "卖出广度", "综合分阈值", "风险分阈值", "累计收益", "年化收益", "最大回撤",
        "交易次数", "交易胜率", "日胜率", "相对胜率", "profit_factor", "最长连续亏损", "持仓占比", "组合数",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


fullrisk_results = _to_numeric_columns(fullrisk_results)
fullrisk_top = _to_numeric_columns(fullrisk_top)


def _pick_fullrisk_primary() -> pd.Series | None:
    for df in [fullrisk_results, fullrisk_top]:
        if df.empty:
            continue
        required = {"买入广度", "卖出广度", "综合分阈值", "风险分阈值"}
        if not required.issubset(df.columns):
            continue
        mask = (
            (df["买入广度"].round(4) == 0.7000)
            & (df["卖出广度"].round(4) == 0.3500)
            & (df["综合分阈值"].round(4) == 54.0000)
            & (df["风险分阈值"].round(4) == 45.0000)
        )
        match = df[mask].copy()
        if not match.empty:
            return match.iloc[0]
    return None


primary_fullrisk = _pick_fullrisk_primary()
if primary_fullrisk is None:
    st.error("未找到 300 日完整风险分主策略行：买入70% / 卖出35% / 综合分54 / 风险45。")
    st.stop()

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown(
    f"""
<div class="metric-grid">
  {metric_card("累计收益", pct(primary_fullrisk.get('累计收益')), "300日完整风险分")}
  {metric_card("年化收益", pct(primary_fullrisk.get('年化收益')), "300日折算")}
  {metric_card("最大回撤", pct(primary_fullrisk.get('最大回撤')), "回测路径内")}
  {metric_card("盈亏因子", number(primary_fullrisk.get('profit_factor')), "profit factor")}
</div>
<div class="metric-grid" style="margin-top:12px;">
  {metric_card("交易胜率", pct_plain(primary_fullrisk.get('交易胜率')), "单笔交易")}
  {metric_card("日胜率", pct_plain(primary_fullrisk.get('日胜率')), "持仓日/信号日")}
  {metric_card("相对胜率", pct_plain(primary_fullrisk.get('相对胜率')), "跑赢行业等权")}
  {metric_card("交易次数", count_value(primary_fullrisk.get('交易次数')), "300日窗口")}
</div>
<div class="metric-grid" style="margin-top:12px;">
  {metric_card("持仓占比", pct_plain(primary_fullrisk.get('持仓占比')), "300日暴露率")}
  {metric_card("最长连亏", count_value(primary_fullrisk.get('最长连续亏损')), "连续亏损交易")}
  {metric_card("买入/卖出广度", f"{float(primary_fullrisk.get('买入广度')):.0%} / {float(primary_fullrisk.get('卖出广度')):.0%}", "广度过滤")}
  {metric_card("分数/风险阈值", f"{count_value(primary_fullrisk.get('综合分阈值'))} / {count_value(primary_fullrisk.get('风险分阈值'))}", "综合分 ≥ / 风险 <")}
</div>
""",
    unsafe_allow_html=True,
)

meta_generated = fullrisk_meta.get("generated_at", "-")
meta_range = f"{fullrisk_meta.get('first_trade_date', '-')} → {fullrisk_meta.get('last_trade_date', '-')}"
meta_days = fullrisk_meta.get("observed_trade_days", "-")
meta_method = fullrisk_meta.get("method", "static_csv_pending_regeneration")
st.caption(
    "主卡来源：data/backtest/top1_fullrisk_grid_300_results.csv · "
    f"窗口：300日 · 样本：{meta_range} · 交易日：{meta_days} · "
    "口径：完整风险分参数网格 · "
    "主策略：Top1 + 广度过滤（买入70% / 卖出35% / 综合分>=54 / 风险<45）"
)
if fullrisk_meta:
    st.success(f"完整风险分表已由脚本每日可复算生成：{meta_generated} · {meta_method}")
    checks = fullrisk_meta.get("strict_checks") or []
    if checks:
        st.caption("严格检查：" + "；".join(str(x) for x in checks))
else:
    st.warning("当前 300 日完整风险分表尚未由新脚本生成 metadata。下一次 GitHub Actions 成功运行后会写入生成时间、样本区间和严格检查说明。")
st.warning("该主卡使用 300 日完整风险分表作为基准回测。当前版本按你的要求暂不加入 ETF 成本，因此主卡不展示成本后收益。")

if not fullrisk_top.empty:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="section-title">完整风险分 300 日参数排名</div>', unsafe_allow_html=True)
    st.caption("主卡同源。逐日完整风险分参数网格结果，已按完全相同收益路径去重。排序：收益优先，其次回撤，其次胜率。")
    show = fullrisk_top.head(10).copy()
    st.dataframe(
        show.style.format({
            "买入广度": "{:.0%}",
            "卖出广度": "{:.0%}",
            "综合分阈值": "{:.0f}",
            "风险分阈值": "{:.0f}",
            "累计收益": "{:+.2%}",
            "年化收益": "{:+.2%}",
            "最大回撤": "{:+.2%}",
            "交易胜率": "{:.1%}",
            "日胜率": "{:.1%}",
            "相对胜率": "{:.1%}",
            "profit_factor": "{:.2f}",
            "持仓占比": "{:.1%}",
        }),
        hide_index=True,
        width="stretch",
    )
    st.markdown("</div>", unsafe_allow_html=True)

if not summary or bt.empty:
    st.info("辅助 360 日回测缓存暂不可用；主卡仍使用 300 日完整风险分表。")
    st.stop()

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown('<div class="panel"><div class="section-title">辅助净值曲线：360日 live-aligned 回测</div>', unsafe_allow_html=True)
if "date" in bt.columns:
    bt["date"] = pd.to_datetime(bt["date"], errors="coerce")
fig = go.Figure()
fig.add_trace(go.Scatter(x=bt["date"], y=bt["strategy_nav"], name="辅助策略毛净值", line=dict(color="#1f5eff", width=2.4)))
if "net_strategy_nav" in bt.columns:
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["net_strategy_nav"], name="辅助策略成本后净值", line=dict(color="#12b76a", width=2.2)))
fig.add_trace(go.Scatter(x=bt["date"], y=bt["benchmark_nav"], name="行业等权", line=dict(color="#667085", width=1.7)))
fig.update_layout(
    height=430,
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=8, b=0),
    yaxis_title="净值",
    xaxis=dict(gridcolor="#eef2f6"),
    yaxis=dict(gridcolor="#eef2f6"),
)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
st.caption(
    f"辅助缓存生成：{generated_at} · 窗口：{lookback_days}日 · "
    f"口径：{score_basis} · 单边成本：{bp_text(transaction_cost_rate)}"
)
if score_basis_note:
    st.info(score_basis_note)
st.markdown("</div>", unsafe_allow_html=True)

left, right = st.columns(2, gap="medium")

with left:
    st.markdown('<div class="panel"><div class="section-title">辅助策略对比：360日 live-aligned 回测</div>', unsafe_allow_html=True)
    if cmp_df.empty:
        st.caption("暂无策略对比缓存。")
    else:
        st.dataframe(
            cmp_df.style.format({
                "方向胜率": "{:.1%}",
                "相对胜率": "{:.1%}",
                "持仓日胜率": "{:.1%}",
                "交易胜率": "{:.1%}",
                "平均盈利": "{:+.2%}",
                "平均亏损": "{:+.2%}",
                "盈亏比": "{:.2f}",
                "持仓暴露率": "{:.1%}",
                "交易次数": "{:.0f}",
                "成本后收益": "{:+.2%}",
                "累计收益": "{:+.2%}",
                "等权基准": "{:+.2%}",
                "年化收益": "{:+.2%}",
                "最大回撤": "{:+.2%}",
                "夏普比率": "{:.2f}",
                "交易日数": "{:.0f}",
            }, na_rep="-"),
            hide_index=True,
            width="stretch",
        )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><div class="section-title">辅助窗口稳健性：live-aligned 回测</div>', unsafe_allow_html=True)
    if robust_df.empty:
        st.caption("暂无窗口稳健性缓存。")
    else:
        st.dataframe(
            robust_df.style.format({
                "方向胜率": "{:.1%}",
                "相对胜率": "{:.1%}",
                "累计收益": "{:+.2%}",
                "成本后收益": "{:+.2%}",
                "等权基准": "{:+.2%}",
                "年化收益": "{:+.2%}",
                "最大回撤": "{:+.2%}",
                "夏普比率": "{:.2f}",
            }),
            hide_index=True,
            width="stretch",
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown('<div class="panel"><div class="section-title">辅助最近信号：360日 live-aligned 回测</div>', unsafe_allow_html=True)
if recent.empty:
    st.caption("暂无最近信号缓存。")
else:
    if "date" in recent.columns:
        recent["date"] = pd.to_datetime(recent["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    show_cols = [
        "date", "持有板块", "action", "综合博弈得分", "风险分", "风险简分",
        "市场广度", "regime_factor", "micro_factor", "transaction_cost",
        "模型次日收益", "成本后次日收益", "等权次日收益",
    ]
    st.dataframe(
        recent[[c for c in show_cols if c in recent.columns]].style.format({
            "综合博弈得分": "{:.1f}",
            "风险分": "{:.1f}",
            "风险简分": "{:.1f}",
            "市场广度": "{:.1f}",
            "regime_factor": "{:.2f}",
            "micro_factor": "{:.2f}",
            "transaction_cost": "{:.4%}",
            "模型次日收益": "{:+.2f}%",
            "成本后次日收益": "{:+.2f}%",
            "等权次日收益": "{:+.2f}%",
        }),
        hide_index=True,
        width="stretch",
    )
st.markdown("</div>", unsafe_allow_html=True)
