from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from snapshot_store import load_snapshot_frame
from ui_theme import clean_signal, inject_theme, page_header


st.set_page_config(page_title="板块轮动地图", layout="wide")
inject_theme()

DATA_DIR = Path("./data")
HISTORY_FILE = DATA_DIR / "sw_board_history.csv"


st.markdown(
    """
<style>
  :root {
    --bg: #eef6fb;
    --panel: #ffffff;
    --panel-soft: #f7fbfe;
    --ink: #0b1f33;
    --muted: #5f7285;
    --line: #dce9f2;
    --line-strong: #bed4e4;
    --brand: #1d7ff2;
    --brand-soft: #eef8ff;
    --danger: #d0703a;
    --danger-soft: #fff1ec;
    --radius: 18px;
    --shadow: 0 12px 30px rgba(11, 31, 51, 0.08);
    --gap: 12px;
  }
  .stApp { background: var(--bg); color: var(--ink); }
  .block-container { max-width: 1500px; padding-top: 1rem; padding-bottom: 2rem; }

  .ui-panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 16px 18px;
  }
  .ui-head {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: flex-start;
    margin-bottom: 14px;
  }
  .ui-title { font-size: 24px; line-height: 1.25; font-weight: 850; color: var(--ink); margin: 0; }
  .ui-sub { font-size: 13px; color: var(--muted); line-height: 1.7; margin-top: 6px; }
  .ui-chip {
    display: inline-flex;
    align-items: center;
    height: 30px;
    padding: 0 10px;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: var(--panel-soft);
    color: var(--muted);
    font-size: 12px;
    font-weight: 750;
    white-space: nowrap;
  }
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: var(--gap);
  }
  .metric-cell {
    background: var(--panel-soft);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 12px;
    transition: border-color 140ms ease, box-shadow 140ms ease;
  }
  .metric-cell:hover { border-color: var(--line-strong); box-shadow: 0 2px 8px rgba(15,23,42,.08); }
  .metric-label { color: var(--muted); font-size: 12px; font-weight: 700; margin-bottom: 6px; }
  .metric-value { color: var(--ink); font-size: 22px; font-weight: 850; letter-spacing: 0; }
  .metric-note { color: var(--muted); font-size: 12px; margin-top: 4px; }
  .section-title {
    font-size: 15px;
    font-weight: 850;
    color: var(--ink);
    margin: 0 0 10px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--line);
  }
  .stage-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    padding: 9px 0;
    border-bottom: 1px solid var(--line);
    font-size: 13px;
  }
  .stage-row:last-child { border-bottom: 0; }
  .stage-name { font-weight: 750; color: var(--ink); }
  .stage-meta { color: var(--muted); font-size: 12px; }
  .stage-count {
    min-width: 32px;
    text-align: center;
    padding: 3px 8px;
    border-radius: 999px;
    background: var(--panel-soft);
    border: 1px solid var(--line);
    color: var(--ink);
    font-weight: 800;
  }
  div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
  @media (max-width: 900px) {
    .ui-head { display: block; }
    .ui-chip { margin-top: 10px; }
    .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
</style>
""",
    unsafe_allow_html=True,
)


def _num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _rank100(s: pd.Series, neutral: float = 50.0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    valid = x.dropna()
    if valid.empty or valid.nunique() <= 1:
        return pd.Series(neutral, index=s.index, dtype=float)
    return x.rank(pct=True, method="average") * 100


def _first_existing(df: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return _num(df[name], default)
    return pd.Series(default, index=df.index, dtype=float)


def _classify_stage(row: pd.Series) -> str:
    if row["risk_score"] >= 78:
        return "风险压制"
    if row["trend_axis"] >= 60 and row["momentum_axis"] >= 60:
        return "主升确认"
    if row["trend_axis"] < 50 and row["momentum_axis"] >= 60:
        return "轮入观察"
    if row["trend_axis"] >= 60 and row["momentum_axis"] < 45:
        return "动能衰减"
    if row["trend_axis"] < 45 and row["momentum_axis"] < 45:
        return "弱势退潮"
    return "均衡震荡"


STAGE_COLOR = {
    "主升确认": "#2563eb",
    "轮入观察": "#3b82f6",
    "均衡震荡": "#94a3b8",
    "动能衰减": "#f97316",
    "风险压制": "#dc2626",
    "弱势退潮": "#475569",
}

STAGE_ORDER = ["主升确认", "轮入观察", "均衡震荡", "动能衰减", "风险压制", "弱势退潮"]

RIVER_PALETTE = [
    "#1d7ff2", "#2f6fed", "#3b82f6", "#0ea5b7",
    "#64748b", "#7c8da1", "#94a3b8", "#b6c2cf",
]


def add_rotation_features(df: pd.DataFrame) -> pd.DataFrame:
    res = df.copy()
    res["sector"] = res["板块名称"].astype(str)
    res["score"] = _first_existing(res, ["综合博弈得分"])
    res["risk_score"] = _first_existing(res, ["逃顶风险分"], 50.0).clip(0, 100)
    res["entry_score"] = _first_existing(res, ["入场共振分"], 50.0).clip(0, 100)
    res["pct"] = _first_existing(res, ["涨跌幅"])
    res["amount"] = _first_existing(res, ["成交额", "last_amount"])
    res["dyn_pos"] = _first_existing(res, ["动态水位", "dyn_pos"], 50.0).clip(0, 100)
    res["accel"] = _first_existing(res, ["趋势加速度"])
    res["flow"] = _first_existing(res, ["金额流向", "资金流向", "fund_raw"])
    res["breadth"] = _first_existing(res, ["上涨占比"], 50.0).clip(0, 100)
    res["layer2"] = _first_existing(res, ["第2层_真假资金"], 50.0).clip(0, 100)
    res["layer6"] = _first_existing(res, ["第6层_中期确认"], 50.0).clip(0, 100)
    res["signal"] = res["终极信号"].map(clean_signal) if "终极信号" in res.columns else "未分类"

    trend = (
        _rank100(res["score"]) * 0.42
        + res["layer6"] * 0.22
        + res["dyn_pos"] * 0.20
        + _rank100(res["pct"]) * 0.16
    )
    momentum = (
        _rank100(res["entry_score"]) * 0.30
        + _rank100(res["accel"]) * 0.24
        + _rank100(res["flow"]) * 0.20
        + res["layer2"] * 0.16
        + _rank100(res["pct"]) * 0.10
    )
    res["trend_axis"] = trend.clip(0, 100).round(1)
    res["momentum_axis"] = momentum.clip(0, 100).round(1)
    res["rotation_power"] = (res["momentum_axis"] - res["risk_score"] * 0.35 + res["trend_axis"] * 0.25).round(1)
    res["stage"] = res.apply(_classify_stage, axis=1)
    res["bubble_size"] = (_rank100(res["amount"]) * 0.42 + 16).clip(18, 58)
    return res


@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_current_data() -> pd.DataFrame:
    return add_rotation_features(load_snapshot_frame())


@st.cache_data(ttl=120, show_spinner=False)
def load_history_data() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    try:
        hist = pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame()
    if hist.empty or "板块名称" not in hist.columns:
        return pd.DataFrame()

    date_col = next((c for c in ["snapshot_time", "snapshot_date", "date", "日期"] if c in hist.columns), None)
    if date_col is None:
        return pd.DataFrame()
    hist = hist.copy()
    hist["snapshot_dt"] = pd.to_datetime(hist[date_col], errors="coerce")
    hist = hist.dropna(subset=["snapshot_dt"])
    if hist.empty:
        return pd.DataFrame()

    keep_cols = [
        "snapshot_dt", "板块名称", "综合博弈得分", "逃顶风险分", "入场共振分", "涨跌幅",
        "成交额", "last_amount", "动态水位", "dyn_pos", "趋势加速度", "金额流向",
        "资金流向", "fund_raw", "上涨占比", "第2层_真假资金", "第6层_中期确认", "终极信号",
    ]
    hist = hist[[c for c in keep_cols if c in hist.columns]].copy()
    hist["snapshot_day"] = hist["snapshot_dt"].dt.date
    hist = hist.sort_values("snapshot_dt").drop_duplicates(["snapshot_day", "板块名称"], keep="last")
    hist = add_rotation_features(hist)
    return hist.sort_values("snapshot_dt").reset_index(drop=True)


def _metric(label: str, value: str, note: str) -> str:
    return f"""
    <div class="metric-cell">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-note">{note}</div>
    </div>
    """


def build_momentum_river(history_df: pd.DataFrame, current_df: pd.DataFrame, days: int = 20, top_n: int = 8) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()

    recent_days = sorted(history_df["snapshot_day"].dropna().unique())[-days:]
    if len(recent_days) < 2:
        return pd.DataFrame()

    river = history_df[history_df["snapshot_day"].isin(recent_days)].copy()
    if river.empty:
        return pd.DataFrame()

    river["river_strength"] = (
        river["score"] * 0.42
        + river["entry_score"] * 0.22
        + river["momentum_axis"] * 0.18
        + river["trend_axis"] * 0.12
        - river["risk_score"] * 0.18
    ).clip(lower=0)

    top_by_history = (
        river.groupby("sector")["river_strength"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )
    top_by_current = current_df.sort_values("rotation_power", ascending=False)["sector"].head(3).tolist()
    focus = []
    for sector in top_by_current + top_by_history:
        if sector not in focus:
            focus.append(sector)
    focus = focus[:top_n]

    river = river[river["sector"].isin(focus)].copy()
    river["snapshot_label"] = pd.to_datetime(river["snapshot_dt"]).dt.strftime("%Y-%m-%d")
    return river.sort_values(["snapshot_dt", "river_strength"], ascending=[True, False])


with st.spinner("正在计算板块轮动坐标..."):
    current_df = load_current_data()
    history_df = load_history_data()

if current_df.empty:
    st.error("暂无本地板块快照。请先由云端任务生成 data/latest_snapshot.json。")
    st.stop()


market_env = current_df["市场环境"].iloc[0] if "市场环境" in current_df.columns else "未知"
heat = float(current_df["market_heat"].iloc[0]) if "market_heat" in current_df.columns else 0.0
top_in = current_df.sort_values("rotation_power", ascending=False).iloc[0]
top_risk = current_df.sort_values("risk_score", ascending=False).iloc[0]
strong_count = int(current_df["stage"].isin(["主升确认", "轮入观察"]).sum())
risk_count = int((current_df["stage"] == "风险压制").sum())

page_header("板块轮动地图", "趋势强度、动能变化与资金迁移路径", f"{market_env} · Heat {heat:.3f}")
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="ui-panel">
      <div class="metric-grid">
        {_metric("轮动强度最高", str(top_in["sector"]), f'Power {top_in["rotation_power"]:.1f}')}
        {_metric("高风险板块", str(top_risk["sector"]), f'Risk {top_risk["risk_score"]:.1f}')}
        {_metric("进攻候选数", f"{strong_count}", "主升确认 + 轮入观察")}
        {_metric("风险压制数", f"{risk_count}", "逃顶风险分较高")}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

trail_days = 10
label_mode = "仅重点板块"
focus_sectors = current_df.sort_values("rotation_power", ascending=False)["sector"].head(5).tolist()
plot_df = current_df.copy()
river_df = build_momentum_river(history_df, current_df)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

st.markdown('<div class="ui-panel"><div class="section-title">动能河流图</div>', unsafe_allow_html=True)
if river_df.empty:
    st.caption("历史快照不足，暂无法生成动能河流图。")
else:
    river_fig = go.Figure()
    for idx, (sector, sub) in enumerate(river_df.groupby("sector", sort=False)):
        sub = sub.sort_values("snapshot_dt")
        color = RIVER_PALETTE[idx % len(RIVER_PALETTE)]
        river_fig.add_trace(go.Scatter(
            x=sub["snapshot_label"],
            y=sub["river_strength"],
            mode="lines",
            line=dict(width=0.8, color=color, shape="spline", smoothing=0.65),
            stackgroup="one",
            groupnorm="percent",
            name=sector,
            customdata=np.stack([
                sub["sector"], sub["river_strength"], sub["score"], sub["risk_score"],
                sub["entry_score"], sub["stage"],
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "日期 %{x}<br>"
                "动能占比 %{y:.1f}%<br>"
                "动能强度 %{customdata[1]:.1f}<br>"
                "综合分 %{customdata[2]:.1f}<br>"
                "风险分 %{customdata[3]:.1f}<br>"
                "入场共振 %{customdata[4]:.1f}<br>"
                "状态 %{customdata[5]}<extra></extra>"
            ),
        ))

    river_fig.update_layout(
        height=360,
        template="plotly_white",
        margin=dict(l=6, r=6, t=10, b=4),
        hovermode="x unified",
        xaxis=dict(title="", gridcolor="#eef2f7", showline=False, tickfont=dict(color="#5f7285")),
        yaxis=dict(title="动能占比", range=[0, 100], ticksuffix="%", gridcolor="#eef2f7", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11, color="#334155")),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#cbd5e1", font_size=12),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    st.plotly_chart(river_fig, width="stretch", config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

left, right = st.columns([2.3, 1], gap="medium")

with left:
    st.markdown('<div class="ui-panel"><div class="section-title">轮动象限气泡图</div>', unsafe_allow_html=True)
    fig = go.Figure()

    fig.add_shape(type="rect", x0=50, x1=100, y0=50, y1=100, fillcolor="#eff6ff", opacity=0.55, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=50, y0=50, y1=100, fillcolor="#f8fafc", opacity=0.95, line_width=0)
    fig.add_shape(type="rect", x0=50, x1=100, y0=0, y1=50, fillcolor="#fff7ed", opacity=0.55, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=50, y0=0, y1=50, fillcolor="#f1f5f9", opacity=0.75, line_width=0)

    if not history_df.empty and focus_sectors:
        recent_days = sorted(history_df["snapshot_day"].dropna().unique())[-trail_days:]
        trail_df = history_df[
            history_df["snapshot_day"].isin(recent_days)
            & history_df["sector"].isin(focus_sectors)
        ].copy()
        for sector, sub in trail_df.groupby("sector", sort=False):
            sub = sub.sort_values("snapshot_dt")
            if len(sub) < 2:
                continue
            color = STAGE_COLOR.get(current_df.loc[current_df["sector"] == sector, "stage"].iloc[0], "#64748b")
            fig.add_trace(go.Scatter(
                x=sub["trend_axis"],
                y=sub["momentum_axis"],
                mode="lines",
                line=dict(color=color, width=1.8, dash="dot"),
                opacity=0.55,
                hoverinfo="skip",
                showlegend=False,
            ))

    for stage in STAGE_ORDER:
        sub = plot_df[plot_df["stage"] == stage]
        if sub.empty:
            continue
        if label_mode == "全部板块":
            labels = sub["sector"]
        elif label_mode == "仅重点板块":
            focus_label = set(
                current_df.sort_values("rotation_power", ascending=False)["sector"].head(8).tolist()
                + current_df.sort_values("risk_score", ascending=False)["sector"].head(4).tolist()
            )
            labels = sub["sector"].where(sub["sector"].isin(focus_label), "")
        else:
            labels = ""
        fig.add_trace(go.Scatter(
            x=sub["trend_axis"],
            y=sub["momentum_axis"],
            mode="markers+text" if label_mode != "不显示" else "markers",
            text=labels,
            textposition="top center",
            textfont=dict(size=11, color="#334155"),
            marker=dict(
                size=sub["bubble_size"],
                color=STAGE_COLOR.get(stage, "#64748b"),
                opacity=0.78,
                line=dict(color="#ffffff", width=1.4),
            ),
            name=stage,
            customdata=np.stack([
                sub["sector"], sub["score"], sub["risk_score"], sub["entry_score"],
                sub["pct"], sub["signal"], sub["amount"],
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "趋势强度 %{x:.1f}<br>"
                "动能变化 %{y:.1f}<br>"
                "综合分 %{customdata[1]:.1f}<br>"
                "逃顶风险 %{customdata[2]:.1f}<br>"
                "入场共振 %{customdata[3]:.1f}<br>"
                "涨跌幅 %{customdata[4]:+.2f}%<br>"
                "信号 %{customdata[5]}<extra></extra>"
            ),
        ))

    fig.add_vline(x=50, line_width=1, line_dash="dash", line_color="#cbd5e1")
    fig.add_hline(y=50, line_width=1, line_dash="dash", line_color="#cbd5e1")
    fig.add_annotation(x=75, y=97, text="主升确认", showarrow=False, font=dict(size=12, color="#1d4ed8"))
    fig.add_annotation(x=25, y=97, text="轮入观察", showarrow=False, font=dict(size=12, color="#475569"))
    fig.add_annotation(x=75, y=4, text="动能衰减", showarrow=False, font=dict(size=12, color="#9a3412"))
    fig.add_annotation(x=25, y=4, text="弱势退潮", showarrow=False, font=dict(size=12, color="#475569"))
    fig.update_layout(
        height=650,
        template="plotly_white",
        margin=dict(l=6, r=6, t=10, b=8),
        xaxis=dict(title="趋势强度", range=[0, 100], gridcolor="#eef2f7", zeroline=False),
        yaxis=dict(title="动能变化", range=[0, 100], gridcolor="#eef2f7", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#cbd5e1", font_size=12),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="ui-panel"><div class="section-title">轮动分组</div>', unsafe_allow_html=True)
    stage_summary = (
        current_df.groupby("stage", as_index=False)
        .agg(count=("sector", "count"), avg_power=("rotation_power", "mean"), avg_risk=("risk_score", "mean"))
    )
    stage_summary["stage"] = pd.Categorical(stage_summary["stage"], categories=STAGE_ORDER, ordered=True)
    stage_summary = stage_summary.sort_values("stage")
    for _, row in stage_summary.iterrows():
        st.markdown(
            f"""
            <div class="stage-row">
              <div>
                <div class="stage-name">{row["stage"]}</div>
                <div class="stage-meta">Power {row["avg_power"]:.1f} · Risk {row["avg_risk"]:.1f}</div>
              </div>
              <div class="stage-count">{int(row["count"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

bottom_left, bottom_right = st.columns(2, gap="medium")

with bottom_left:
    st.markdown('<div class="ui-panel"><div class="section-title">轮入候选</div>', unsafe_allow_html=True)
    in_df = current_df[current_df["stage"].isin(["主升确认", "轮入观察"])].sort_values("rotation_power", ascending=False)
    show = in_df[["sector", "stage", "rotation_power", "score", "entry_score", "risk_score", "signal"]].head(10)
    st.dataframe(
        show.rename(columns={
            "sector": "板块",
            "stage": "状态",
            "rotation_power": "轮动强度",
            "score": "综合分",
            "entry_score": "入场共振",
            "risk_score": "风险",
            "signal": "终极信号",
        }).style.format({
            "轮动强度": "{:.1f}",
            "综合分": "{:.1f}",
            "入场共振": "{:.1f}",
            "风险": "{:.1f}",
        }),
        hide_index=True,
        width="stretch",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with bottom_right:
    st.markdown('<div class="ui-panel"><div class="section-title">退潮与风险</div>', unsafe_allow_html=True)
    risk_df = current_df[current_df["stage"].isin(["风险压制", "动能衰减", "弱势退潮"])].copy()
    risk_df = risk_df.sort_values(["risk_score", "rotation_power"], ascending=[False, True]).head(10)
    st.dataframe(
        risk_df[["sector", "stage", "risk_score", "trend_axis", "momentum_axis", "pct", "signal"]]
        .rename(columns={
            "sector": "板块",
            "stage": "状态",
            "risk_score": "风险",
            "trend_axis": "趋势",
            "momentum_axis": "动能",
            "pct": "涨跌幅",
            "signal": "终极信号",
        })
        .style.format({
            "风险": "{:.1f}",
            "趋势": "{:.1f}",
            "动能": "{:.1f}",
            "涨跌幅": "{:+.2f}%",
        }),
        hide_index=True,
        width="stretch",
    )
    st.markdown("</div>", unsafe_allow_html=True)

if history_df.empty:
    st.caption("历史轨迹依赖 data/sw_board_history.csv。当前文件字段不足或不可读时，页面只展示当日轮动地图。")
