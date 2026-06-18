from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_theme import inject_theme, metric_card, page_header


st.set_page_config(page_title="模型回测", layout="wide")
inject_theme()

page_header("模型回测", "300日完整风险分主策略回测", "Full-risk grid")

BACKTEST_DIR = Path("data/backtest")
FULLRISK_RESULTS_PATH = BACKTEST_DIR / "top1_fullrisk_grid_300_results.csv"
FULLRISK_PRIMARY_PATH = BACKTEST_DIR / "top1_fullrisk_grid_300_primary_path.csv"
FULLRISK_STRATEGY_COMPARISON_PATH = BACKTEST_DIR / "top1_fullrisk_grid_300_strategy_comparison.csv"
FULLRISK_WINDOW_ROBUSTNESS_PATH = BACKTEST_DIR / "top1_fullrisk_grid_300_window_robustness.csv"
FULLRISK_RECENT_SIGNALS_PATH = BACKTEST_DIR / "top1_fullrisk_grid_300_recent_signals.csv"
FULLRISK_METADATA_PATH = BACKTEST_DIR / "top1_fullrisk_grid_300_metadata.json"
FULLRISK_COMPARE_PATH = BACKTEST_DIR / "top1_fullrisk_grid_300_compare_report.json"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col not in ["策略", "窗口", "date", "持有板块", "动作"]:
            out[col] = pd.to_numeric(out[col], errors="ignore")
    return out


def _format_date_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


fullrisk_results = _coerce_numeric(_read_csv(FULLRISK_RESULTS_PATH))
primary_path = _coerce_numeric(_read_csv(FULLRISK_PRIMARY_PATH))
strategy_cmp = _coerce_numeric(_read_csv(FULLRISK_STRATEGY_COMPARISON_PATH))
window_robust = _coerce_numeric(_read_csv(FULLRISK_WINDOW_ROBUSTNESS_PATH))
recent_signals = _coerce_numeric(_read_csv(FULLRISK_RECENT_SIGNALS_PATH))
fullrisk_meta = _read_json(FULLRISK_METADATA_PATH)
fullrisk_compare = _read_json(FULLRISK_COMPARE_PATH)

if fullrisk_results.empty:
    st.error("未找到完整风险分 300 日正式结果表：data/backtest/top1_fullrisk_grid_300_results.csv")
    st.stop()

for col in ["买入广度", "卖出广度", "综合分阈值", "风险分阈值"]:
    if col in fullrisk_results.columns:
        fullrisk_results[col] = pd.to_numeric(fullrisk_results[col], errors="coerce")

mask = (
    (fullrisk_results["买入广度"].round(4) == 0.7000)
    & (fullrisk_results["卖出广度"].round(4) == 0.3500)
    & (fullrisk_results["综合分阈值"].round(4) == 54.0000)
    & (fullrisk_results["风险分阈值"].round(4) == 45.0000)
)
primary_rows = fullrisk_results[mask].copy()
if primary_rows.empty:
    st.error("未找到 300 日完整风险分主策略行：买入70% / 卖出35% / 综合分54 / 风险45。")
    st.stop()
primary = primary_rows.iloc[0]

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown(
    f"""
<div class="metric-grid">
  {metric_card("累计收益", pct(primary.get('累计收益')), "300日完整风险分")}
  {metric_card("年化收益", pct(primary.get('年化收益')), "300日折算")}
  {metric_card("最大回撤", pct(primary.get('最大回撤')), "回测路径内")}
  {metric_card("盈亏因子", number(primary.get('profit_factor')), "profit factor")}
</div>
<div class="metric-grid" style="margin-top:12px;">
  {metric_card("交易胜率", pct_plain(primary.get('交易胜率')), "单笔交易")}
  {metric_card("日胜率", pct_plain(primary.get('日胜率')), "持仓日/信号日")}
  {metric_card("相对胜率", pct_plain(primary.get('相对胜率')), "跑赢行业等权")}
  {metric_card("交易次数", count_value(primary.get('交易次数')), "300日窗口")}
</div>
<div class="metric-grid" style="margin-top:12px;">
  {metric_card("持仓占比", pct_plain(primary.get('持仓占比')), "300日暴露率")}
  {metric_card("最长连亏", count_value(primary.get('最长连续亏损')), "连续亏损交易")}
  {metric_card("买入/卖出广度", f"{float(primary.get('买入广度')):.0%} / {float(primary.get('卖出广度')):.0%}", "广度过滤")}
  {metric_card("分数/风险阈值", f"{count_value(primary.get('综合分阈值'))} / {count_value(primary.get('风险分阈值'))}", "综合分 ≥ / 风险 <")}
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown('<div class="panel"><div class="section-title">净值曲线</div>', unsafe_allow_html=True)
if primary_path.empty:
    st.info("主策略净值路径文件尚未生成。下一次 GitHub Actions 成功运行后会生成 data/backtest/top1_fullrisk_grid_300_primary_path.csv。")
else:
    curve = primary_path.copy()
    if "date" in curve.columns:
        curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
    fig = go.Figure()
    if "strategy_nav" in curve.columns:
        fig.add_trace(go.Scatter(x=curve["date"], y=curve["strategy_nav"], name="Top1 + 广度过滤", line=dict(color="#1f5eff", width=2.5)))
    if "benchmark_nav" in curve.columns:
        fig.add_trace(go.Scatter(x=curve["date"], y=curve["benchmark_nav"], name="行业等权", line=dict(color="#667085", width=1.8)))
    if "hs300_nav" in curve.columns:
        fig.add_trace(go.Scatter(x=curve["date"], y=curve["hs300_nav"], name="沪深300", line=dict(color="#f79009", width=1.8)))
    fig.update_layout(
        height=440,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=8, b=0),
        yaxis_title="净值",
        xaxis=dict(gridcolor="#eef2f6"),
        yaxis=dict(gridcolor="#eef2f6"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("主策略净值曲线同源于 300 日完整风险分正式表；比较线为行业等权与沪深300指数。")
st.markdown("</div>", unsafe_allow_html=True)

left, right = st.columns(2, gap="medium")

with left:
    st.markdown('<div class="panel"><div class="section-title">策略对比</div>', unsafe_allow_html=True)
    if strategy_cmp.empty:
        st.info("完整风险分策略对比文件尚未生成。下一次 GitHub Actions 成功运行后会生成。")
    else:
        st.dataframe(
            strategy_cmp.style.format({
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
                "最长连续亏损": "{:.0f}",
                "持仓占比": "{:.1%}",
            }, na_rep="-"),
            hide_index=True,
            width="stretch",
        )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><div class="section-title">窗口稳健性</div>', unsafe_allow_html=True)
    if window_robust.empty:
        st.info("完整风险分窗口稳健性文件尚未生成。下一次 GitHub Actions 成功运行后会生成。")
    else:
        st.dataframe(
            window_robust.style.format({
                "累计收益": "{:+.2%}",
                "行业等权收益": "{:+.2%}",
                "沪深300收益": "{:+.2%}",
                "年化收益": "{:+.2%}",
                "最大回撤": "{:+.2%}",
                "交易次数": "{:.0f}",
                "交易胜率": "{:.1%}",
                "日胜率": "{:.1%}",
                "相对胜率": "{:.1%}",
                "profit_factor": "{:.2f}",
                "持仓占比": "{:.1%}",
            }, na_rep="-"),
            hide_index=True,
            width="stretch",
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown('<div class="panel"><div class="section-title">最近信号</div>', unsafe_allow_html=True)
if recent_signals.empty:
    st.info("完整风险分最近信号文件尚未生成。下一次 GitHub Actions 成功运行后会生成。")
else:
    recent_show = _format_date_col(recent_signals, "date")
    if "date" in recent_show.columns:
        recent_show = recent_show.sort_values("date", ascending=False)
    st.dataframe(
        recent_show.style.format({
            "综合博弈得分": "{:.1f}",
            "逃顶风险分": "{:.1f}",
            "入场共振分": "{:.1f}",
            "市场广度": "{:.1%}",
            "strategy_ret": "{:+.2%}",
            "benchmark_ret": "{:+.2%}",
            "hs300_ret": "{:+.2%}",
        }, na_rep="-"),
        hide_index=True,
        width="stretch",
    )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown('<div class="panel"><div class="section-title">生成口径与对账</div>', unsafe_allow_html=True)
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
    st.success(f"完整风险分正式表 metadata：{meta_generated} · {meta_method}")
    checks = fullrisk_meta.get("strict_checks") or []
    if checks:
        st.caption("严格检查：" + "；".join(str(x) for x in checks))
else:
    st.warning("当前正式 300 日完整风险分表尚未由新脚本生成 metadata。正式表仍作为你认可的 baseline 使用。")

if fullrisk_compare:
    comparison = fullrisk_compare.get("comparison", {}) or {}
    compare_status = comparison.get("status", fullrisk_compare.get("compare_status", "-"))
    compare_message = comparison.get("message", "")
    compare_generated = fullrisk_compare.get("generated_at", "-")
    if compare_status == "match_within_tolerance":
        st.success(f"影子对账：candidate 与正式表差异在容忍范围内 · {compare_generated}")
    elif compare_status == "mismatch_requires_review":
        st.error(f"影子对账：candidate 与正式表差异较大，暂不应 promote · {compare_generated}")
    else:
        st.warning(f"影子对账：{compare_status} · {compare_generated}")
    if compare_message:
        st.caption(compare_message)
    diffs = comparison.get("metric_diffs", {}) or {}
    if diffs:
        diff_rows = []
        for metric, values in diffs.items():
            diff_rows.append({"指标": metric, "正式表": values.get("official"), "candidate": values.get("candidate"), "差异": values.get("diff")})
        with st.expander("查看影子对账差异", expanded=False):
            st.dataframe(pd.DataFrame(diff_rows), hide_index=True, width="stretch")
else:
    st.info("尚未生成影子对账报告。下一次 GitHub Actions 成功运行后会写入 candidate 表和 compare_report。")

st.warning("该主卡使用 300 日完整风险分正式表作为基准回测。当前版本按你的要求暂不加入 ETF 成本，因此主卡不展示成本后收益。")
st.markdown("</div>", unsafe_allow_html=True)
