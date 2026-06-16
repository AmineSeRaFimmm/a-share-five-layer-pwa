from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from snapshot_store import load_latest_snapshot, load_snapshot_frame, load_update_status
from ui_theme import clean_signal, inject_theme, metric_card, page_header, signal_badge


st.set_page_config(page_title="今日复盘", page_icon=" ", layout="wide")
inject_theme()
components.html('<script src="/app/static/pwa.js"></script>', height=0, width=0)

def records_to_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records or [])
    for col in [
        "涨跌幅",
        "综合博弈得分",
        "逃顶风险分",
        "入场共振分",
        "动态水位",
        "趋势加速度",
        "资金流向",
        "上涨占比",
        "成分股覆盖数",
        "fund_raw",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


snapshot = load_latest_snapshot()
status = load_update_status()

if not snapshot:
    st.error("暂无本地快照。请先运行 scripts/update_daily_snapshot.py 生成 data/latest_snapshot.json。")
    if status:
        st.caption(f"最近一次尝试：{status.get('last_attempt_at', '-')} · {status.get('status', '-')} · {status.get('reason', '-')}")
    st.stop()

df = load_snapshot_frame()
if df.empty:
    st.error("快照中没有板块数据。")
    st.stop()

df = df.copy()
if "终极信号" in df.columns:
    df["信号"] = df["终极信号"].map(clean_signal)
else:
    df["信号"] = "未分类"

trade_date = str(snapshot.get("trade_date", ""))
updated_at = str(snapshot.get("updated_at", ""))
target_trade_date = str(status.get("target_trade_date", "")) if status else ""
status_code = str(status.get("status", snapshot.get("status", "")))
status_reason = str(status.get("reason", ""))
is_current_ready = status_code == "ready" and (not target_trade_date or target_trade_date == trade_date)

env_val = str(snapshot.get("market_environment", "震荡市"))
heat_val = float(snapshot.get("market_heat", 0.0) or 0.0)
breadth = float(snapshot.get("market_breadth", 0.0) or 0.0)
coverage_mean = float(snapshot.get("coverage_mean", 0.0) or 0.0)
top_risk = str(snapshot.get("top_risk", "无"))
top_opp = str(snapshot.get("top_opportunity", "等待"))

risk_pool = df[df["信号"].str.contains("强制清仓|崩盘|鱼尾|诱多|强弩|战术减仓", na=False)]
opp_pool = df[df["信号"].str.contains("满仓|底仓|顺势", na=False)]
n_opp = int(len(opp_pool))
n_risk = int(len(risk_pool))

page_header(
    "今日复盘",
    "板块强弱、风险分层与轮动候选",
    f"{env_val} · Heat {heat_val:.3f}",
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

status_label = "已更新" if is_current_ready else "今日数据尚未更新"
status_note = f"目标 {target_trade_date or '-'} · {status_reason}" if not is_current_ready else "主缓存 ready"
st.markdown(
    f"""
<div class="metric-grid">
  {metric_card("数据日期", trade_date or "-", status_label)}
  {metric_card("更新时间", updated_at[-8:-3] if len(updated_at) >= 16 else "-", updated_at)}
  {metric_card("最近尝试", str(status.get("last_attempt_at", "-"))[-8:-3] if status else "-", status_note)}
  {metric_card("等待数据源", status_reason if not is_current_ready else "无", status_code)}
</div>
""",
    unsafe_allow_html=True,
)

if not is_current_ready:
    st.warning("今日数据尚未更新，当前页面显示最近一次已通过校验的交易日快照。")

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown(
    f"""
<div class="metric-grid">
  {metric_card("上涨广度", f"{breadth:.1f}%", "板块收盘快照")}
  {metric_card("进攻候选", f"{n_opp}", top_opp)}
  {metric_card("风险警报", f"{n_risk}", top_risk)}
  {metric_card("成分覆盖", f"{coverage_mean:.1f}", "平均匹配股数")}
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

main_col, side_col = st.columns([2.15, 1], gap="medium")

with main_col:
    st.markdown('<div class="panel"><div class="section-title">综合排名</div>', unsafe_allow_html=True)
    show_cols = [
        "板块名称", "数据日期", "信号", "涨跌幅", "综合博弈得分", "逃顶风险分", "入场共振分",
        "动态水位", "趋势加速度", "资金流向", "上涨占比", "对应ETF",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(
        df[show_cols],
        height=620,
        hide_index=True,
        width="stretch",
        column_config={
            "板块名称": st.column_config.TextColumn("板块", width="small"),
            "数据日期": st.column_config.TextColumn("日期", width="small"),
            "信号": st.column_config.TextColumn("信号", width="medium"),
            "涨跌幅": st.column_config.NumberColumn("涨跌幅", format="%+.2f%%"),
            "综合博弈得分": st.column_config.ProgressColumn("综合分", min_value=0, max_value=100, format="%.1f"),
            "逃顶风险分": st.column_config.ProgressColumn("风险", min_value=0, max_value=100, format="%.1f"),
            "入场共振分": st.column_config.ProgressColumn("共振", min_value=0, max_value=100, format="%.1f"),
            "动态水位": st.column_config.ProgressColumn("水位", min_value=0, max_value=100, format="%.1f"),
            "趋势加速度": st.column_config.NumberColumn("加速度", format="%+.3f"),
            "资金流向": st.column_config.NumberColumn("资金", format="%+.3f"),
            "上涨占比": st.column_config.ProgressColumn("广度", min_value=0, max_value=100, format="%.1f"),
            "对应ETF": st.column_config.TextColumn("ETF", width="medium"),
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.markdown('<div class="panel"><div class="section-title">风险与机会</div>', unsafe_allow_html=True)
    tab_risk, tab_opp = st.tabs(["风险", "机会"])
    with tab_risk:
        risk_show = risk_pool.sort_values("逃顶风险分", ascending=False).head(10) if not risk_pool.empty else pd.DataFrame()
        if risk_show.empty:
            st.caption("暂无风险警报")
        for _, row in risk_show.iterrows():
            st.markdown(
                f"""
<div class="signal-row">
  <div>
    <div class="signal-name">{row["板块名称"]}</div>
    <div class="signal-meta">风险 {row["逃顶风险分"]:.1f} · 水位 {row.get("动态水位", 0):.1f} · {row.get("对应ETF", "")}</div>
  </div>
  {signal_badge(row.get("终极信号", ""))}
</div>
""",
                unsafe_allow_html=True,
            )
    with tab_opp:
        opp_show = opp_pool.sort_values("综合博弈得分", ascending=False).head(10) if not opp_pool.empty else pd.DataFrame()
        if opp_show.empty:
            st.caption("暂无进攻候选")
        for _, row in opp_show.iterrows():
            st.markdown(
                f"""
<div class="signal-row">
  <div>
    <div class="signal-name">{row["板块名称"]}</div>
    <div class="signal-meta">综合 {row["综合博弈得分"]:.1f} · 共振 {row.get("入场共振分", 0):.1f} · {row.get("对应ETF", "")}</div>
  </div>
  {signal_badge(row.get("终极信号", ""))}
</div>
""",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown('<div class="panel"><div class="section-title">今日推荐</div>', unsafe_allow_html=True)

clarity = snapshot.get("clarity_signal", {}) or {}
rules = clarity.get("rules", {}) or {}
today_buy = records_to_df(clarity.get("buy", []))
today_sell = records_to_df(clarity.get("sell", []))
buy_floor = float(rules.get("buy_breadth_floor", 0.60)) * 100
sell_floor = float(rules.get("sell_breadth_floor", 0.45)) * 100
min_score = float(rules.get("min_score", 58.0))
max_risk = float(rules.get("max_risk", 55.0))

st.markdown(
    f"""
<div class="metric-grid">
  {metric_card("策略", "澄势精选", "Top1 动量")}
  {metric_card("今日广度", f"{breadth:.1f}%", "买入通过" if breadth >= buy_floor else "买入未通过")}
  {metric_card("买入候选", f"{len(today_buy)}", f"综合分 >= {min_score:.0f} · 风险 < {max_risk:.0f}")}
  {metric_card("卖出触发", f"{len(today_sell)}", f"广度 < {sell_floor:.0f}%")}
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
buy_col, sell_col = st.columns([1.25, 1], gap="medium")

with buy_col:
    st.markdown('<div class="section-title">买入候选</div>', unsafe_allow_html=True)
    if today_buy.empty:
        st.markdown('<div class="empty-state">暂无推荐</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="recommend-grid">', unsafe_allow_html=True)
        for rank, (_, row) in enumerate(today_buy.iterrows(), start=1):
            st.markdown(
                f"""
<div class="recommend-card">
  <div class="recommend-kicker">BUY #{rank} · 建议权重 100%</div>
  <div class="recommend-title">{row["板块名称"]}</div>
  {signal_badge(row.get("终极信号", ""))}
  <div class="recommend-meta">
    <div class="recommend-stat"><div class="recommend-stat-label">综合</div><div class="recommend-stat-value">{row["综合博弈得分"]:.1f}</div></div>
    <div class="recommend-stat"><div class="recommend-stat-label">风险</div><div class="recommend-stat-value">{row["逃顶风险分"]:.1f}</div></div>
    <div class="recommend-stat"><div class="recommend-stat-label">涨跌</div><div class="recommend-stat-value">{row["涨跌幅"]:+.2f}%</div></div>
  </div>
  <div class="signal-meta" style="margin-top:10px;">{row.get("对应ETF", "")}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

with sell_col:
    st.markdown('<div class="section-title">卖出触发</div>', unsafe_allow_html=True)
    if today_sell.empty:
        st.markdown('<div class="empty-state">暂无</div>', unsafe_allow_html=True)
    else:
        for _, row in today_sell.iterrows():
            st.markdown(
                f"""
<div class="signal-row">
  <div>
    <div class="signal-name">{row["板块名称"]}</div>
    <div class="signal-meta">{row.get("卖出原因", "")}</div>
    <div class="signal-meta">风险 {float(row.get("逃顶风险分", 0) or 0):.1f} · 综合 {float(row.get("综合博弈得分", 0) or 0):.1f} · {row.get("对应ETF", "")}</div>
  </div>
  {signal_badge(row.get("终极信号", row.get("信号", "")))}
</div>
""",
                unsafe_allow_html=True,
            )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown('<div class="panel"><div class="section-title">AVIX 指数</div>', unsafe_allow_html=True)
avix = snapshot.get("avix", {}) or {}
latest_avix = avix.get("latest") or {}
avix_hist = pd.DataFrame(avix.get("history", []))
if latest_avix:
    latest_value = float(latest_avix.get("avix", 0.0) or 0.0)
    latest_quality = str(latest_avix.get("quality", "Close Mid"))
    latest_time = str(latest_avix.get("valuation_time", trade_date))
    st.markdown(
        f"""
<div class="metric-grid">
  {metric_card("AVIX", f"{latest_value:.2f}", latest_quality)}
  {metric_card("估值时间", latest_time[-8:-3] if len(latest_time) >= 16 else "-", latest_time)}
  {metric_card("近月期限", str(latest_avix.get("near_dte", "-")), str(latest_avix.get("near_expiry", "")))}
  {metric_card("次月期限", str(latest_avix.get("next_dte", "-")), str(latest_avix.get("next_expiry", "")))}
</div>
""",
        unsafe_allow_html=True,
    )
elif avix.get("error"):
    st.caption(f"AVIX 暂不可用：{avix.get('error')}")
else:
    st.caption("暂无 AVIX 快照")

if not avix_hist.empty and {"trade_date", "avix"}.issubset(avix_hist.columns):
    avix_hist = avix_hist.copy()
    avix_hist["trade_date"] = pd.to_datetime(avix_hist["trade_date"], errors="coerce")
    avix_hist["label"] = avix_hist["trade_date"].dt.strftime("%Y-%m-%d")
    avix_hist["avix"] = pd.to_numeric(avix_hist["avix"], errors="coerce")
    avix_hist = avix_hist.dropna(subset=["label", "avix"]).tail(120)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=avix_hist["label"],
            y=avix_hist["avix"],
            mode="lines",
            name="AVIX",
            line=dict(color="#1f5eff", width=2.4),
            hovertemplate="%{x}<br>AVIX %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=360,
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        xaxis=dict(type="category", nticks=8, gridcolor="#eef2f7"),
        yaxis=dict(title="波动率点数", gridcolor="#eef2f7", zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)
