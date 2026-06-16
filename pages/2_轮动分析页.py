from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from snapshot_store import load_snapshot_frame
from ui_theme import clean_signal, inject_theme, metric_card, page_header, signal_badge


st.set_page_config(page_title="资金轮动诊断", layout="wide")
inject_theme()

page_header("资金轮动诊断", "资金强度、轮动路径与板块生态", "Rotation")

with st.spinner("正在加载板块快照..."):
    df = load_snapshot_frame()

if df.empty:
    st.error("暂无本地板块快照。请先由云端任务生成 data/latest_snapshot.json。")
    st.stop()

if "fund_raw" not in df.columns:
    st.error("缺少 fund_raw 字段。")
    st.stop()

df = df.copy()
df["信号"] = df["终极信号"].map(clean_signal) if "终极信号" in df.columns else "未分类"
df["fund_rank"] = df["fund_raw"].rank(ascending=False, method="min").astype(int)
df["fund_pct"] = df["fund_raw"].rank(ascending=True, pct=True) * 100

fund_mean = float(df["fund_raw"].mean())
spread = float(df["fund_raw"].quantile(.8) - df["fund_raw"].quantile(.2))
breadth = float((df["涨跌幅"] > 0).mean() * 100)
env_val = df["市场环境"].iloc[0] if "市场环境" in df.columns else "震荡市"
heat_val = float(df["market_heat"].iloc[0]) if "market_heat" in df.columns else 0.0

if breadth >= 50 and fund_mean > 1.0:
    regime = "放量扩散"
elif spread > 1.2 and breadth < 50:
    regime = "极端抱团"
elif breadth < 35 and fund_mean < .8:
    regime = "缩量普跌"
else:
    regime = "混沌震荡"

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown(
    f"""
<div class="metric-grid">
  {metric_card("市场状态", regime, f"{env_val} · Heat {heat_val:.3f}")}
  {metric_card("上涨广度", f"{breadth:.1f}%", "板块上涨比例")}
  {metric_card("资金均值", f"{fund_mean:.2f}", "fund_raw")}
  {metric_card("资金极差", f"{spread:.2f}", "80%-20%")}
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

selected_name = st.selectbox(
    "目标板块",
    options=df.sort_values("fund_rank")["板块名称"].tolist(),
    index=0,
)
row = df[df["板块名称"] == selected_name].iloc[0]

left, right = st.columns([1.25, 1], gap="medium")

with left:
    st.markdown('<div class="panel"><div class="section-title">目标板块</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="metric-grid">
  {metric_card("板块", selected_name, str(row.get("对应ETF", "")))}
  {metric_card("资金排名", f"{int(row['fund_rank'])}/{len(df)}", f"fund_raw {row['fund_raw']:.2f}")}
  {metric_card("综合分", f"{row.get('综合博弈得分', 0):.1f}", f"风险 {row.get('逃顶风险分', 0):.1f}")}
  {metric_card("涨跌幅", f"{row.get('涨跌幅', 0):+.2f}%", f"广度 {row.get('上涨占比', 0):.1f}%")}
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(signal_badge(row.get("终极信号", "未分类")), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><div class="section-title">资金极值</div>', unsafe_allow_html=True)
    top5 = df.nlargest(5, "fund_raw")[["板块名称", "fund_raw", "涨跌幅", "信号", "综合博弈得分"]]
    bot5 = df.nsmallest(5, "fund_raw")[["板块名称", "fund_raw", "涨跌幅", "信号", "综合博弈得分"]]
    tab_top, tab_bottom = st.tabs(["资金最强", "资金最弱"])
    with tab_top:
        st.dataframe(top5.style.format({"fund_raw": "{:.2f}", "涨跌幅": "{:+.2f}%", "综合博弈得分": "{:.1f}"}), hide_index=True, width="stretch")
    with tab_bottom:
        st.dataframe(bot5.style.format({"fund_raw": "{:.2f}", "涨跌幅": "{:+.2f}%", "综合博弈得分": "{:.1f}"}), hide_index=True, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

HISTORY_FILE = Path("./data/sw_board_history.csv")
hist_panel = pd.DataFrame()
delta_today = pd.Series(dtype=float)

st.markdown('<div class="panel"><div class="section-title">资金路径</div>', unsafe_allow_html=True)
if HISTORY_FILE.exists():
    try:
        hist = pd.read_csv(HISTORY_FILE)
        if "snapshot_time" in hist.columns and "fund_raw" in hist.columns:
            hist["snapshot_day"] = pd.to_datetime(hist["snapshot_time"], errors="coerce").dt.strftime("%Y-%m-%d")
            hist = hist.dropna(subset=["snapshot_day"])
            daily = hist.sort_values("snapshot_time").groupby(["snapshot_day", "板块名称"]).tail(1)
            hist_panel = daily.pivot(index="snapshot_day", columns="板块名称", values="fund_raw").fillna(0)
            if len(hist_panel) >= 2:
                delta_today = hist_panel.iloc[-1] - hist_panel.iloc[-2]
                current_leader = hist_panel.iloc[-1].idxmax()
                trail = [hist_panel.loc[d].idxmax() for d in hist_panel.index[-5:]]
                st.markdown(
                    f"""
<div class="metric-grid">
  {metric_card("当前主线", current_leader, "fund_raw 最高")}
  {metric_card("目标增量", f"{delta_today.get(selected_name, 0):+.2f}", selected_name)}
  {metric_card("流入第一", str(delta_today.nlargest(1).index[0]), f"{delta_today.nlargest(1).iloc[0]:+.2f}")}
  {metric_card("流出第一", str(delta_today.nsmallest(1).index[0]), f"{delta_today.nsmallest(1).iloc[0]:+.2f}")}
</div>
""",
                    unsafe_allow_html=True,
                )
                st.caption("近五日主线：" + " / ".join(trail))

                fig = go.Figure()
                for name, color in [(selected_name, "#1f5eff"), (current_leader, "#0f766e")]:
                    if name in hist_panel.columns:
                        fig.add_trace(go.Scatter(x=hist_panel.index, y=hist_panel[name], name=name, mode="lines+markers", line=dict(color=color, width=2)))
                fig.update_layout(height=300, template="plotly_white", hovermode="x unified", margin=dict(l=0, r=0, t=8, b=0), yaxis_title="fund_raw")
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            else:
                st.caption("历史天数不足。")
        else:
            st.caption("历史文件缺少 fund_raw。")
    except Exception:
        st.caption("历史文件解析失败。")
else:
    st.caption("暂无历史快照。")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

col_buy, col_sell = st.columns(2, gap="medium")
with col_buy:
    st.markdown('<div class="panel"><div class="section-title">轮动候选</div>', unsafe_allow_html=True)
    buy_candidates = df[df["信号"].str.contains("满仓|底仓|顺势", na=False)].sort_values("fund_raw", ascending=False)
    cols = ["板块名称", "信号", "综合博弈得分", "入场共振分", "fund_raw", "涨跌幅", "对应ETF"]
    st.dataframe(
        buy_candidates[[c for c in cols if c in buy_candidates.columns]].style.format({
            "综合博弈得分": "{:.1f}",
            "入场共振分": "{:.1f}",
            "fund_raw": "{:.2f}",
            "涨跌幅": "{:+.2f}%",
        }),
        hide_index=True,
        width="stretch",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_sell:
    st.markdown('<div class="panel"><div class="section-title">风险池</div>', unsafe_allow_html=True)
    sell_alerts = df[df["信号"].str.contains("强制清仓|崩盘|鱼尾|诱多|强弩", na=False)].sort_values("逃顶风险分", ascending=False)
    cols2 = ["板块名称", "信号", "逃顶风险分", "动态水位", "涨跌幅", "对应ETF"]
    st.dataframe(
        sell_alerts[[c for c in cols2 if c in sell_alerts.columns]].style.format({
            "逃顶风险分": "{:.1f}",
            "动态水位": "{:.1f}",
            "涨跌幅": "{:+.2f}%",
        }),
        hide_index=True,
        width="stretch",
    )
    st.markdown("</div>", unsafe_allow_html=True)
