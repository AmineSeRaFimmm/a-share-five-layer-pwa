from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from snapshot_store import load_sector_history_from_cache, load_snapshot_frame
from ui_theme import clean_signal, inject_theme, metric_card, page_header


st.set_page_config(page_title="板块详情", layout="wide")
inject_theme()

with st.spinner("正在加载板块快照..."):
    df_current = load_snapshot_frame()

if df_current.empty:
    st.error("暂无本地板块快照。请先由云端任务生成 data/latest_snapshot.json。")
    st.stop()

df_current = df_current.copy()
df_current["信号"] = df_current["终极信号"].map(clean_signal) if "终极信号" in df_current.columns else "未分类"

st.markdown(
    """
<style>
  div[data-baseweb="select"] > div {
    border-radius: 14px;
    border-color: #dce9f2;
    background: #f7fbfe;
    box-shadow: none;
  }
  div[data-baseweb="select"] > div:hover {
    border-color: #bed4e4;
  }
  div[data-testid="stSegmentedControl"] label {
    border-radius: 999px;
    border-color: #dce9f2;
    color: #5f7285;
    background: #f7fbfe;
  }
  div[data-testid="stSegmentedControl"] label[aria-checked="true"],
  div[data-testid="stSegmentedControl"] label[data-baseweb="radio"] input:checked + div,
  div[data-testid="stSegmentedControl"] [aria-pressed="true"],
  div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
  div[data-testid="stSegmentedControl"] [data-selected="true"] {
    background: #eef8ff !important;
    color: #1d7ff2 !important;
    border-color: #9ccff5 !important;
  }
  div[data-testid="stSegmentedControl"] svg {
    color: #1d7ff2 !important;
  }
</style>
""",
    unsafe_allow_html=True,
)

page_header("板块详情", "单板块结构、价格序列与因子拆解", "Research")
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

sector_options = df_current["板块名称"].tolist()
if "detail_sector" not in st.session_state or st.session_state.detail_sector not in sector_options:
    st.session_state.detail_sector = sector_options[0]
if "detail_window" not in st.session_state:
    st.session_state.detail_window = "近 3 个月"

days_map = {"近 1 个月": 22, "近 3 个月": 60, "近半年": 120}
selected_sector = st.session_state.detail_sector
window_label = st.session_state.detail_window
lookback_days = days_map[window_label]
sector_data = df_current[df_current["板块名称"] == selected_sector].iloc[0]
show_volume = True

with st.spinner("正在加载本地历史序列..."):
    hist_df = load_sector_history_from_cache(selected_sector, lookback_days)

has_history = not hist_df.empty and len(hist_df) >= 5

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown(
    f"""
<div class="metric-grid">
  {metric_card("综合分", f"{sector_data.get('综合博弈得分', 0):.1f}", "六层合成")}
  {metric_card("风险分", f"{sector_data.get('逃顶风险分', 50):.1f}", "逃顶风险")}
  {metric_card("共振分", f"{sector_data.get('入场共振分', 50):.1f}", "入场结构")}
  {metric_card("涨跌幅", f"{sector_data.get('涨跌幅', 0):+.2f}%", f"广度 {sector_data.get('上涨占比', 0):.1f}%")}
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
left, right = st.columns([2.2, 1], gap="medium")

with left:
    st.markdown('<div class="panel"><div class="section-title">价格与得分</div>', unsafe_allow_html=True)
    if has_history:
        has_ohlc = all(c in hist_df.columns for c in ["开盘价", "最高价", "最低价", "收盘价"])
        rows_fig = 3 if show_volume and "成交量" in hist_df.columns else 2
        row_heights = [0.58, 0.25, 0.17] if rows_fig == 3 else [0.65, 0.35]
        fig = make_subplots(rows=rows_fig, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=row_heights)
        h = hist_df.copy()
        if has_ohlc:
            fig.add_trace(
                go.Candlestick(
                    x=h["日期"],
                    open=h["开盘价"],
                    high=h["最高价"],
                    low=h["最低价"],
                    close=h["收盘价"],
                    name="K线",
                    increasing_line_color="#c2410c",
                    decreasing_line_color="#0f766e",
                ),
                row=1,
                col=1,
            )
            h["MA5"] = h["收盘价"].rolling(5).mean()
            h["MA20"] = h["收盘价"].rolling(20).mean()
            fig.add_trace(go.Scatter(x=h["日期"], y=h["MA5"], name="MA5", line=dict(color="#1f5eff", width=1.3)), row=1, col=1)
            fig.add_trace(go.Scatter(x=h["日期"], y=h["MA20"], name="MA20", line=dict(color="#667085", width=1.3)), row=1, col=1)
        else:
            fig.add_trace(
                go.Scatter(
                    x=h["日期"],
                    y=h["涨跌幅"] if "涨跌幅" in h.columns else h["综合博弈得分"],
                    name="本地历史",
                    mode="lines+markers",
                    line=dict(color="#1f5eff", width=2),
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=h["日期"],
                y=h["综合博弈得分"],
                name="综合分",
                mode="lines",
                line=dict(color="#1f5eff", width=2.2),
                fill="tozeroy",
                fillcolor="rgba(31,94,255,.08)",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#d0d5dd", row=2, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="#e5e7eb", row=2, col=1)
        if show_volume and "成交量" in h.columns and has_ohlc:
            colors = ["#c2410c" if c >= o else "#0f766e" for c, o in zip(h["收盘价"], h["开盘价"])]
            fig.add_trace(go.Bar(x=h["日期"], y=h["成交量"], name="成交量", marker_color=colors, opacity=.65), row=3, col=1)
        fig.update_layout(
            height=630 if show_volume else 520,
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=8, b=0),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        st.caption("历史数据不足。")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><div class="section-title">当前判断</div>', unsafe_allow_html=True)
    st.selectbox("板块", sector_options, key="detail_sector", label_visibility="collapsed")
    st.segmented_control(
        "周期",
        ["近 1 个月", "近 3 个月", "近半年"],
        key="detail_window",
        label_visibility="collapsed",
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(
            [
                ["市场环境", sector_data.get("市场环境", "")],
                ["微观标签", sector_data.get("微观标签", "")],
                ["动态水位", f"{sector_data.get('动态水位', 0):.1f}%"],
                ["趋势加速度", f"{sector_data.get('趋势加速度', 0):+.3f}"],
                ["资金流向", f"{sector_data.get('资金流向', 0):+.3f}"],
                ["上影线", f"{sector_data.get('上影线诱多率', 0):.1f}%"],
            ],
            columns=["指标", "值"],
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown('<div class="section-title" style="margin-top:14px;">六层结构</div>', unsafe_allow_html=True)
    labels = ["真实趋势", "真假资金", "异动干预", "诱多诱空", "博弈反身", "中期确认"]
    values = [
        float(sector_data.get("第1层_真实趋势", 50)),
        float(sector_data.get("第2层_真假资金", 50)),
        float(sector_data.get("第3层_异动干预", 50)),
        float(sector_data.get("第4层_诱多诱空", 50)),
        float(sector_data.get("第5层_博弈反身", 50)),
        float(sector_data.get("第6层_中期确认", 50)),
    ]
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]], fill="toself", line=dict(color="#1f5eff", width=2), fillcolor="rgba(31,94,255,.16)", name=selected_sector))
    radar.add_trace(go.Scatterpolar(r=[50] * 7, theta=labels + [labels[0]], mode="lines", line=dict(color="#d0d5dd", width=1, dash="dot"), name="中枢"))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=330, margin=dict(l=18, r=18, t=12, b=12), showlegend=False)
    st.plotly_chart(radar, width="stretch", config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown('<div class="panel"><div class="section-title">近期因子</div>', unsafe_allow_html=True)
if has_history:
    factor_cols = [
        "数据日期", "涨跌幅", "动态水位", "第1层_真实趋势", "第2层_真假资金",
        "第4层_诱多诱空", "第5层_博弈反身", "第6层_中期确认", "综合博弈得分",
    ]
    factor_cols = [c for c in factor_cols if c in hist_df.columns]
    show = hist_df[factor_cols].tail(10).copy()
    st.dataframe(
        show.style.format({
            "涨跌幅": "{:+.2f}%",
            "动态水位": "{:.1f}",
            "第1层_真实趋势": "{:.1f}",
            "第2层_真假资金": "{:.1f}",
            "第4层_诱多诱空": "{:.1f}",
            "第5层_博弈反身": "{:.1f}",
            "第6层_中期确认": "{:.1f}",
            "综合博弈得分": "{:.1f}",
        }),
        hide_index=True,
        width="stretch",
    )
else:
    st.caption("暂无可展示序列。")
st.markdown("</div>", unsafe_allow_html=True)
