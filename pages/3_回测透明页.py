from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from snapshot_store import load_backtest_summary
from ui_theme import inject_theme, metric_card, page_header


st.set_page_config(page_title="模型回测", layout="wide")
inject_theme()

page_header("模型回测", "今日推荐同口径：Top1 + 广度过滤，收盘信号下一交易日验证", "Walk-forward")

payload = load_backtest_summary()
if not payload:
    st.error("暂无本地回测缓存。请先由云端任务生成 data/backtest/strategy_summary.json。")
    st.stop()

summary = payload.get("summary", {}) or {}
bt = pd.DataFrame(payload.get("recent_curve", []) or [])
cmp_df = pd.DataFrame(payload.get("strategy_comparison", []) or [])
robust_df = pd.DataFrame(payload.get("window_robustness", []) or [])
recent = pd.DataFrame(payload.get("recent_signals", []) or [])
generated_at = payload.get("generated_at", "-")
lookback_days = payload.get("lookback_days", "-")
score_basis = payload.get("score_basis", "legacy_or_unknown")
score_basis_note = payload.get("score_basis_note", "")
primary_strategy = payload.get("primary_strategy", "legacy_or_unknown")
primary_label = payload.get("primary_strategy_label", "旧缓存策略")
strategy_rules = payload.get("strategy_rules", {}) or {}

if not summary or bt.empty:
    st.error(f"回测缓存不完整：{payload.get('error', '缺少 summary 或 recent_curve')}")
    st.stop()

if primary_strategy != "top1_breadth":
    st.warning("当前回测缓存不是 Top1 + 广度过滤新口径。请重新运行每日快照任务以刷新 data/backtest/strategy_summary.json。")


def pct(value: float) -> str:
    try:
        return f"{float(value) * 100:+.2f}%"
    except Exception:
        return "-"


def pct_plain(value: float | None) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "-"


def pct_signed(value: float | None) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:+.2f}%"
    except Exception:
        return "-"


def number_plain(value: float | None) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "-"


cost_rate = summary.get("单边成本假设", 0.001)
try:
    cost_note = f"单边成本 {float(cost_rate) * 100:.2f}%"
except Exception:
    cost_note = "含交易成本"

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown(
    f"""
<div class="metric-grid">
  {metric_card("持仓日胜率", pct_plain(summary.get("持仓日胜率")), "仅统计实际持仓日")}
  {metric_card("交易胜率", pct_plain(summary.get("交易胜率")), "按单笔交易复利收益")}
  {metric_card("平均盈利", pct_signed(summary.get("平均盈利")), "盈利交易均值")}
  {metric_card("平均亏损", pct_signed(summary.get("平均亏损")), "亏损交易均值")}
</div>
<div class="metric-grid" style="margin-top:12px;">
  {metric_card("盈亏比", number_plain(summary.get("盈亏比")), "平均盈利 / 平均亏损绝对值")}
  {metric_card("持仓暴露率", pct_plain(summary.get("持仓暴露率")), "持仓日 / 总交易日")}
  {metric_card("交易次数", f"{int(float(summary.get('交易次数', 0)))}", f"{lookback_days}日窗口")}
  {metric_card("成本后收益", pct(summary.get("成本后收益", 0)), cost_note)}
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown(
    f"""
<div class="metric-grid">
  {metric_card("方向胜率", f"{float(summary.get('胜率', 0)) * 100:.1f}%", "全交易日收益 > 0")}
  {metric_card("相对胜率", f"{float(summary.get('相对胜率', 0)) * 100:.1f}%", "跑赢行业等权")}
  {metric_card("累计收益", pct(summary.get("累计收益", 0)), f"基准 {pct(summary.get('基准收益', 0))}")}
  {metric_card("最大回撤", pct(summary.get("最大回撤", 0)), f"夏普 {float(summary.get('夏普比率', 0)):.2f}")}
</div>
""",
    unsafe_allow_html=True,
)

st.caption(f"缓存生成：{generated_at} · 主策略：{primary_label} · 评分口径：{score_basis} · 数据源：data/backtest/strategy_summary.json")
if score_basis_note:
    st.caption(f"口径说明：{score_basis_note}")

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown(f'<div class="panel"><div class="section-title">净值曲线｜{primary_label}</div>', unsafe_allow_html=True)
if "date" in bt.columns:
    bt["date"] = pd.to_datetime(bt["date"], errors="coerce")
fig = go.Figure()
fig.add_trace(go.Scatter(x=bt["date"], y=bt["strategy_nav"], name=primary_label, line=dict(color="#1f5eff", width=2.4)))
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
st.markdown("</div>", unsafe_allow_html=True)

left, right = st.columns(2, gap="medium")

with left:
    st.markdown('<div class="panel"><div class="section-title">策略对比</div>', unsafe_allow_html=True)
    if cmp_df.empty:
        st.caption("暂无策略对比缓存。")
    else:
        st.dataframe(
            cmp_df.style.format({
                "方向胜率": "{:.1%}",
                "相对胜率": "{:.1%}",
                "累计收益": "{:+.2%}",
                "等权基准": "{:+.2%}",
                "年化收益": "{:+.2%}",
                "最大回撤": "{:+.2%}",
                "夏普比率": "{:.2f}",
                "交易日数": "{:.0f}",
            }),
            hide_index=True,
            width="stretch",
        )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(f'<div class="panel"><div class="section-title">窗口稳健性｜{primary_label}</div>', unsafe_allow_html=True)
    if robust_df.empty:
        st.caption("暂无窗口稳健性缓存。")
    else:
        st.dataframe(
            robust_df.style.format({
                "方向胜率": "{:.1%}",
                "相对胜率": "{:.1%}",
                "累计收益": "{:+.2%}",
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
st.markdown(f'<div class="panel"><div class="section-title">最近信号｜{primary_label}</div>', unsafe_allow_html=True)
if recent.empty:
    st.caption("暂无最近信号缓存。")
else:
    if "date" in recent.columns:
        recent["date"] = pd.to_datetime(recent["date"], errors="coerce")
        recent = recent.sort_values("date", ascending=False)
        recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
    show_cols = ["date", "动作", "持有板块", "综合博弈得分", "风险简分", "市场广度", "模型次日收益", "等权次日收益"]
    st.dataframe(
        recent[[c for c in show_cols if c in recent.columns]].style.format({
            "综合博弈得分": "{:.1f}",
            "风险简分": "{:.1f}",
            "市场广度": "{:.1f}",
            "模型次日收益": "{:+.2f}%",
            "等权次日收益": "{:+.2f}%",
        }),
        hide_index=True,
        width="stretch",
    )
st.markdown("</div>", unsafe_allow_html=True)
