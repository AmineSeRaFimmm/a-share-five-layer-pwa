from __future__ import annotations

import re

import streamlit as st


SIGNAL_RE = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u27bf"
    "]+"
)


def clean_signal(value: object) -> str:
    text = SIGNAL_RE.sub("", str(value)).strip()
    return re.sub(r"\s+", " ", text)


def signal_tone(value: object) -> str:
    text = clean_signal(value)
    if any(key in text for key in ["强制清仓", "崩盘", "鱼尾", "诱多", "强弩"]):
        return "danger"
    if any(key in text for key in ["战术减仓", "冲顶", "控仓", "能量耗尽", "预警"]):
        return "warn"
    if any(key in text for key in ["满仓", "底仓", "顺势", "低位", "安全"]):
        return "good"
    return "neutral"


def inject_theme() -> None:
    st.markdown(
        """
<style>
  :root {
    --bg: #eef6fb;
    --surface: #ffffff;
    --surface-2: #f7fbfe;
    --ink: #0b1f33;
    --muted: #5f7285;
    --line: #dce9f2;
    --line-strong: #bed4e4;
    --brand: #1d7ff2;
    --brand-2: #2bb6a8;
    --danger: #d0703a;
    --danger-2: #b42318;
    --warn: #c88719;
    --good: #179b84;
    --radius: 18px;
    --shadow: 0 12px 30px rgba(11, 31, 51, .08);
  }
  .stApp { background: var(--bg); color: var(--ink); }
  .block-container { max-width: 1480px; padding-top: 1rem; padding-bottom: 2rem; }
  div[data-testid="stSidebar"] {
    display: none;
  }
  div[data-testid="collapsedControl"] {
    display: none;
  }
  section[data-testid="stSidebar"] {
    display: none;
  }
  .main .block-container {
    padding-left: 2rem;
    padding-right: 2rem;
  }
  .top-nav {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid var(--line);
  }
  .top-nav [data-testid="stPageLink"] a {
    border: 1px solid var(--line);
    background: #f7fbfe;
    border-radius: 999px;
    padding: 6px 12px;
    min-height: 30px;
    color: var(--muted);
    font-size: 13px;
    font-weight: 760;
    transition: border-color 140ms ease, background 140ms ease, color 140ms ease;
  }
  .top-nav [data-testid="stPageLink"] a:hover {
    border-color: var(--line-strong);
    background: #eef8ff;
    color: var(--brand);
  }
  .app-shell {
    background: linear-gradient(180deg, #ffffff 0%, #f8fcff 100%);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 22px 24px;
  }
  .page-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 14px;
  }
  .page-title {
    margin: 0;
    font-size: 28px;
    line-height: 1.2;
    font-weight: 850;
    letter-spacing: 0;
    color: var(--ink);
  }
  .page-subtitle {
    margin-top: 6px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
  }
  .status-chip {
    display: inline-flex;
    align-items: center;
    height: 30px;
    border-radius: 999px;
    padding: 0 11px;
    border: 1px solid var(--line);
    background: #eef8ff;
    color: #226796;
    font-size: 12px;
    font-weight: 750;
    white-space: nowrap;
  }
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
  }
  .metric-card {
    background: linear-gradient(180deg, #ffffff 0%, #f7fbfe 100%);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 15px 16px;
    box-shadow: 0 8px 24px rgba(11, 31, 51, .05);
  }
  .metric-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 720;
    margin-bottom: 6px;
  }
  .metric-value {
    color: var(--ink);
    font-size: 25px;
    line-height: 1.2;
    font-weight: 850;
  }
  .metric-note {
    margin-top: 5px;
    color: var(--muted);
    font-size: 12px;
  }
  .panel {
    background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 18px 20px;
  }
  .section-title {
    margin: 0 0 11px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--line);
    font-size: 16px;
    font-weight: 850;
    color: var(--ink);
  }
  .signal-row {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 10px;
    align-items: start;
    padding: 10px 0;
    border-bottom: 1px solid var(--line);
  }
  .signal-row:last-child { border-bottom: 0; }
  .signal-name { font-weight: 820; color: var(--ink); font-size: 14px; }
  .signal-meta { margin-top: 4px; color: var(--muted); font-size: 12px; line-height: 1.5; }
  .badge {
    display: inline-flex;
    align-items: center;
    min-height: 24px;
    border-radius: 999px;
    padding: 2px 9px;
    font-size: 12px;
    font-weight: 800;
    border: 1px solid var(--line);
    background: var(--surface-2);
    color: var(--muted);
    white-space: nowrap;
  }
  .badge.good { background: #e8fbf7; color: var(--good); border-color: #aee9df; }
  .badge.warn { background: #fff8e8; color: var(--warn); border-color: #f4d491; }
  .badge.danger { background: #fff1ec; color: var(--danger-2); border-color: #f7c5b8; }
  .badge.neutral { background: var(--surface-2); color: var(--muted); }
  .recommend-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
  }
  .recommend-card {
    position: relative;
    overflow: hidden;
    min-height: 168px;
    border-radius: 18px;
    border: 1px solid var(--line);
    background: linear-gradient(180deg, #ffffff 0%, #f7fbfe 100%);
    box-shadow: 0 10px 26px rgba(11, 31, 51, .07);
    padding: 16px;
  }
  .recommend-card::after {
    content: "";
    position: absolute;
    inset: auto -32px -42px auto;
    width: 110px;
    height: 110px;
    border-radius: 999px;
    background: rgba(29, 127, 242, .08);
  }
  .recommend-card.sell::after { background: rgba(208, 112, 58, .10); }
  .recommend-kicker {
    color: var(--muted);
    font-size: 12px;
    font-weight: 820;
    margin-bottom: 7px;
  }
  .recommend-title {
    color: var(--ink);
    font-size: 20px;
    line-height: 1.2;
    font-weight: 880;
    margin-bottom: 10px;
  }
  .recommend-meta {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin-top: 12px;
  }
  .recommend-stat {
    background: rgba(255, 255, 255, .72);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 8px;
  }
  .recommend-stat-label {
    color: var(--muted);
    font-size: 11px;
    font-weight: 720;
    margin-bottom: 3px;
  }
  .recommend-stat-value {
    color: var(--ink);
    font-size: 14px;
    font-weight: 850;
  }
  .empty-state {
    border: 1px dashed var(--line-strong);
    border-radius: 18px;
    background: #fbfdff;
    padding: 28px;
    text-align: center;
    color: var(--muted);
    font-weight: 760;
  }
  div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 12px;
    color: var(--muted);
    font-weight: 750;
  }
  .stTabs [aria-selected="true"] { color: var(--brand); background: #eef4ff; }
  @media (max-width: 900px) {
    .page-head { display: block; }
    .status-chip { margin-top: 10px; }
    .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .recommend-grid { grid-template-columns: 1fr; }
  }
</style>
""",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", status: str = "") -> None:
    st.markdown(
        f"""
<div class="app-shell">
  <div class="page-head">
    <div>
      <h1 class="page-title">{title}</h1>
      <div class="page-subtitle">{subtitle}</div>
    </div>
    <div class="status-chip">{status}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="top-nav">', unsafe_allow_html=True)
    nav_cols = st.columns([1, 1, 1, 1, 1, 5], gap="small")
    nav_items = [
        ("app.py", "今日复盘"),
        ("pages/4_板块轮动地图.py", "轮动地图"),
        ("pages/1_板块详情页.py", "板块详情"),
        ("pages/3_回测透明页.py", "模型回测"),
        ("pages/2_轮动分析页.py", "资金诊断"),
    ]
    for col, (path, label) in zip(nav_cols, nav_items):
        with col:
            st.page_link(path, label=label)
    st.markdown("</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str = "") -> str:
    return f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{value}</div>
  <div class="metric-note">{note}</div>
</div>
"""


def signal_badge(signal: object) -> str:
    text = clean_signal(signal)
    return f'<span class="badge {signal_tone(signal)}">{text}</span>'
