# -*- coding: utf-8 -*-
"""Web 服务层：Streamlit 交互式回测界面。

散户视角：浏览器打开即用，选股票、选策略、点按钮出结果，
无需命令行。本模块只做展示，业务逻辑全部复用既有模块。

启动方式：
    python -m streamlit run retailquant/webapp.py
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from retailquant.advisor import advise_portfolio
from retailquant.backtest import BacktestEngine, BacktestResult
from retailquant.config import BacktestConfig, RISK_PROFILES, RiskConfig
from retailquant.data import load_daily, validate_ohlcv
from retailquant.portfolio import Portfolio, Position, load_portfolio, save_portfolio
from retailquant.report import PerformanceMetrics, compute_metrics
from retailquant.stockmeta import (
    all_code_names,
    build_external_links,
    get_stock_info,
)
from retailquant.strategy import ALL_STRATEGIES

# ---------------------------------------------------------------- 页面配置
st.set_page_config(page_title="retailquant 散户量化回测", page_icon="📈", layout="wide")

_STRATEGY_LABELS = {
    "dual_ma": "双均线 (MA5/MA20)",
    "macd_cross": "MACD 金叉（零轴过滤）",
    "rsi_reversal": "RSI 超卖反转",
    "donchian": "海龟/唐奇安通道突破",
    "boll_reversion": "布林带均值回归",
    "momentum": "20日动量择时",
    "buy_and_hold": "买入持有（基准）",
}
_PROFILE_LABELS = {
    "纪律版": "纪律版（蓝筹：止损8%/止盈20%）",
    "宽松版": "宽松版（高波动：止损15%/止盈50%）",
    "趋势版": "趋势版（热门股：止损15%/不止盈）",
}
# 策略中文名 -> 键的反向映射（持仓助手下拉回写用）
_LABEL_TO_STRATEGY = {v: k for k, v in _STRATEGY_LABELS.items()}
# 持仓助手可选策略（中文名，排除买入持有基准）
_ADVISOR_STRATEGY_LABELS = [_STRATEGY_LABELS[k] for k in ALL_STRATEGIES if k != "buy_and_hold"]
# 持仓助手取数回看天数：覆盖策略预热期且留节假日余量
_ADVISOR_LOOKBACK_DAYS = 400


@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def _load_daily_cached(symbol: str, start: str, end: str) -> pd.DataFrame:
    """行情加载（Streamlit 层缓存，叠加 data 模块的磁盘缓存）。

    max_entries 限制常驻内存的行情条数，避免多次不同区间查询后内存膨胀。
    """
    df = load_daily(symbol, start, end)
    validate_ohlcv(df)
    return df


@st.cache_resource(show_spinner=False)
def _code_name_map_cached() -> dict[str, str]:
    """全 A 股代码 -> 名称映射（进程级共享，只读不复制，省内存）。

    用 cache_resource 而非 cache_data：后者每次调用都会深拷贝这份
    5000+ 条的字典，前者返回同一对象，显著降低重复渲染的开销。
    """
    return all_code_names()


@st.cache_resource(show_spinner=False)
def _snapshot_options() -> tuple[list[str], int]:
    """个股速览下拉选项（「代码 名称」列表 + 默认项下标），只构建一次。"""
    code_names = _code_name_map_cached()
    options = [f"{c} {n}" for c, n in code_names.items()]
    default_idx = next((i for i, o in enumerate(options)
                        if o.startswith("600900")), 0)
    return options, default_idx


@st.cache_data(show_spinner=False, ttl=1800, max_entries=256)
def _stock_info_cached(symbol: str) -> dict:
    """个股基本情况（缓存半小时，限量 256 条）。"""
    return get_stock_info(symbol)


def _name_of(symbol: str) -> str:
    """代码 -> 股票简称（查共享映射，查不到返回空串）。"""
    return _code_name_map_cached().get(symbol, "")


def _named(symbol: str) -> str:
    """「代码 名称」拼接，未识别时仅返回代码。"""
    name = _name_of(symbol)
    return f"{symbol} {name}" if name else symbol


def _fmt_yi(value) -> str:
    """将市值等大数格式化为「X 亿」，无法解析时原样返回。"""
    try:
        return f"{float(value) / 1e8:,.2f} 亿"
    except (TypeError, ValueError):
        return str(value) if value else "-"


def _metrics_row(res: BacktestResult, m: PerformanceMetrics) -> dict:
    return {
        "标的": res.symbol,
        "策略": _STRATEGY_LABELS.get(res.strategy_name, res.strategy_name),
        "总收益%": m.total_return_pct,
        "年化%": m.annual_return_pct,
        "最大回撤%": m.max_drawdown_pct,
        "夏普": m.sharpe,
        "交易数": m.num_trades,
        "胜率%": m.win_rate_pct,
        "盈亏比": None if m.profit_factor == float("inf") else m.profit_factor,
        "费用(元)": m.total_costs,
    }


def _trades_df(res: BacktestResult) -> pd.DataFrame:
    return pd.DataFrame([{
        "买入日": t.entry_date.date(), "买入价": round(t.entry_price, 3),
        "股数": t.shares,
        "卖出日": t.exit_date.date() if t.exit_date is not None else "",
        "卖出价": round(t.exit_price, 3) if t.exit_price is not None else "",
        "离场原因": t.exit_reason, "盈亏(元)": round(t.pnl, 2),
        "收益率%": round(t.ret_pct * 100, 2),
    } for t in res.trades])


def _sidebar() -> dict:
    """侧边栏参数收集（回测页签）。"""
    st.sidebar.header("⚙️ 回测参数")
    symbols_text = st.sidebar.text_input(
        "股票代码（6 位，空格分隔）", "600900 000858 601318 600036",
        help="示例：600900 长江电力 / 601318 中国平安")
    symbols_preview = [s.strip() for s in symbols_text.split() if s.strip()]
    if symbols_preview:
        names = [
            f"{s} {nm}" if (nm := _name_of(s)) else f"{s}（未识别）"
            for s in symbols_preview
        ]
        st.sidebar.caption("识别到：" + " · ".join(names))
    col1, col2 = st.sidebar.columns(2)
    start = col1.date_input("开始日期", date(2022, 1, 1))
    end = col2.date_input("结束日期", date(2025, 12, 31))
    strategies = st.sidebar.multiselect(
        "策略（可多选对比）", options=list(ALL_STRATEGIES),
        default=list(ALL_STRATEGIES),
        format_func=lambda k: _STRATEGY_LABELS.get(k, k))
    capital = st.sidebar.number_input(
        "本金（元）", min_value=10_000, max_value=10_000_000,
        value=60_000, step=10_000)
    st.sidebar.subheader("风控纪律")
    stop_loss = st.sidebar.slider("硬止损 %", 3, 20, 8)
    take_profit = st.sidebar.slider("止盈 %", 5, 50, 20)
    run = st.sidebar.button("🚀 开始回测", use_container_width=True, type="primary")
    return {
        "symbols": [s.strip() for s in symbols_text.split() if s.strip()],
        "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
        "strategies": strategies, "capital": float(capital),
        "stop_loss": stop_loss / 100.0, "take_profit": take_profit / 100.0,
        "run": run,
    }


def _run_backtests(p: dict) -> tuple[list, list[str]]:
    """执行回测，返回 (结果列表, 错误消息列表)。"""
    cfg = BacktestConfig(
        initial_capital=p["capital"],
        risk=RiskConfig(stop_loss_pct=p["stop_loss"], take_profit_pct=p["take_profit"]),
    )
    engine = BacktestEngine(cfg)
    results, errors = [], []
    progress = st.progress(0.0, text="回测中…")
    total = max(len(p["symbols"]) * len(p["strategies"]), 1)
    done = 0
    for symbol in p["symbols"]:
        try:
            df = _load_daily_cached(symbol, p["start"], p["end"])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{symbol}：数据获取失败（{exc}）")
            done += len(p["strategies"])
            progress.progress(min(done / total, 1.0))
            continue
        for name in p["strategies"]:
            res = engine.run(df, ALL_STRATEGIES[name](), symbol=symbol)
            results.append((res, compute_metrics(res)))
            done += 1
            progress.progress(min(done / total, 1.0),
                              text=f"回测中… {symbol} × {_STRATEGY_LABELS.get(name, name)}")
    progress.empty()
    return results, errors


def _render_results(results: list, capital: float) -> None:
    """结果展示：绩效总表 + 净值曲线 + 交易明细。"""
    st.subheader("📊 绩效对比总表")
    table = pd.DataFrame([_metrics_row(r, m) for r, m in results])
    st.dataframe(
        table.style
        .format({"总收益%": "{:.2f}", "年化%": "{:.2f}", "最大回撤%": "{:.2f}",
                 "夏普": "{:.3f}", "胜率%": "{:.2f}", "盈亏比": "{:.2f}",
                 "费用(元)": "{:.0f}"}, na_rep="∞")
        .background_gradient(subset=["总收益%"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True)

    best = table.loc[table["总收益%"].idxmax()]
    st.success(f"🏆 最佳组合：**{best['标的']} × {best['策略']}**，"
               f"总收益 **{best['总收益%']:.2f}%**，最大回撤 {best['最大回撤%']:.2f}%")

    st.subheader("📈 净值曲线对比")
    symbols = sorted({r.symbol for r, _ in results})
    sel_symbol = st.selectbox("选择标的", symbols)
    curves = {
        _STRATEGY_LABELS.get(r.strategy_name, r.strategy_name): r.equity_curve / capital
        for r, _ in results if r.symbol == sel_symbol
    }
    if curves:
        st.line_chart(pd.DataFrame(curves), height=380)
        st.caption("纵轴为净值（期初 = 1.0），已含佣金/印花税/过户费/滑点等全部费用。")

    st.subheader("🧾 交易明细（逐笔复盘）")
    for res, m in results:
        if res.symbol != sel_symbol or not res.trades:
            continue
        label = _STRATEGY_LABELS.get(res.strategy_name, res.strategy_name)
        with st.expander(f"{label} — {m.num_trades} 笔，胜率 {m.win_rate_pct:.1f}%"):
            tdf = _trades_df(res)
            st.dataframe(tdf, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ 下载 CSV", tdf.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"trades_{res.symbol}_{res.strategy_name}.csv",
                mime="text/csv", key=f"dl_{res.symbol}_{res.strategy_name}")


@st.fragment
def _render_backtest_tab() -> None:
    """页签一：策略回测。"""
    params = _sidebar()

    if params["run"]:
        if not params["symbols"]:
            st.error("请至少输入一个股票代码")
            st.stop()
        if not params["strategies"]:
            st.error("请至少选择一个策略")
            st.stop()
        results, errors = _run_backtests(params)
        for msg in errors:
            st.warning(msg)
        if results:
            st.session_state["results"] = results
            st.session_state["capital"] = params["capital"]

    if "results" in st.session_state:
        _render_results(st.session_state["results"], st.session_state["capital"])
    else:
        st.info("👈 在左侧设置参数后点击「开始回测」。默认组合：长江电力/五粮液/中国平安/招商银行，2022–2025。")


# ================================================================ 持仓助手

def _positions_editor(pf: Portfolio) -> tuple[Portfolio, bool]:
    """持仓编辑表格，返回 (编辑后的账户, 是否点击保存)。"""
    st.markdown("把券商 App 里的持仓抄进来；**股数填 0 表示自选观察股**（只等买入信号）。")
    cash = st.number_input("可用现金（元）", min_value=0.0, value=float(pf.cash),
                           step=1000.0, format="%.2f")
    rows = [{
        "代码": p.symbol,
        "名称": _name_of(p.symbol),
        "股数": p.shares, "成本价": p.cost_price,
        "策略": _STRATEGY_LABELS.get(p.strategy, p.strategy),
        "风控档位": p.risk_profile,
    } for p in pf.positions]
    edited = st.data_editor(
        pd.DataFrame(rows, columns=["代码", "名称", "股数", "成本价", "策略", "风控档位"]),
        num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={
            "代码": st.column_config.TextColumn("代码", help="6 位股票代码", max_chars=6),
            "名称": st.column_config.TextColumn("名称", disabled=True,
                                                help="根据代码自动识别的股票名称"),
            "股数": st.column_config.NumberColumn("股数", min_value=0, step=100),
            "成本价": st.column_config.NumberColumn("成本价", min_value=0.0, format="%.3f"),
            "策略": st.column_config.SelectboxColumn(
                "策略", options=_ADVISOR_STRATEGY_LABELS, required=True),
            "风控档位": st.column_config.SelectboxColumn(
                "风控档位", options=list(RISK_PROFILES), required=True,
                help=" / ".join(_PROFILE_LABELS.values())),
        },
        key="positions_editor")
    saved = st.button("💾 保存持仓", type="secondary")

    positions = []
    for _, row in edited.iterrows():
        symbol = str(row["代码"] or "").strip()
        if not symbol:
            continue
        positions.append(Position(
            symbol=symbol,
            shares=int(row["股数"] or 0),
            cost_price=float(row["成本价"] or 0.0),
            strategy=_LABEL_TO_STRATEGY.get(str(row["策略"] or ""), "donchian"),
            risk_profile=str(row["风控档位"] or "纪律版"),
        ))
    recognized = [_named(p.symbol) for p in positions]
    if recognized:
        st.caption("识别到：" + " · ".join(recognized))
    return Portfolio(cash=float(cash), positions=positions), saved


def _advices_table(advices: list, data: dict[str, pd.DataFrame], pf: Portfolio) -> None:
    """建议结果展示：账户总览 + 逐只建议。"""
    pos_by_symbol = {p.symbol: p for p in pf.positions}
    market_value = sum(
        p.shares * float(data[p.symbol]["close"].iloc[-1])
        for p in pf.positions if p.shares > 0 and p.symbol in data)
    total_cost = sum(p.shares * p.cost_price for p in pf.positions if p.shares > 0)
    col1, col2, col3 = st.columns(3)
    col1.metric("持仓市值", f"{market_value:,.0f} 元",
                f"{market_value - total_cost:+,.0f} 元" if total_cost else None)
    col2.metric("可用现金", f"{pf.cash:,.0f} 元")
    col3.metric("账户总资产", f"{market_value + pf.cash:,.0f} 元")

    rows = []
    for adv in advices:
        pos = pos_by_symbol[adv.symbol]
        rows.append({
            "代码": adv.symbol,
            "名称": _name_of(adv.symbol),
            "持仓": f"{pos.shares} 股" if pos.shares else "观察",
            "策略": _STRATEGY_LABELS.get(pos.strategy, pos.strategy),
            "行情日期": adv.last_date.date(),
            "最新收盘": round(adv.last_close, 3),
            "浮动盈亏%": round(adv.pnl_pct * 100, 2) if pos.shares else None,
            "止损参考价": round(adv.stop_price, 3) if adv.stop_price else None,
            "止盈参考价": round(adv.target_price, 3) if adv.target_price else None,
            "建议买入": f"{adv.suggest_shares} 股" if adv.suggest_shares else "",
            "操作建议": adv.action,
            "理由": adv.reason,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("信号基于最新收盘数据，按 T+1 规则建议「明日开盘」执行；"
               "盘中若跌破止损参考价请直接执行纪律，不必等收盘。仅供研究，不构成投资建议。")


@st.fragment
def _render_advisor_tab() -> None:
    """页签二：持仓助手（手动录入持仓 + 策略执行建议）。"""
    st.subheader("💼 我的持仓")
    try:
        stored = load_portfolio()
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"portfolio.json 损坏：{exc}，已重置为空账户")
        stored = Portfolio()
    pf, saved = _positions_editor(stored)

    if saved:
        try:
            save_portfolio(pf)
            st.success("持仓已保存到 portfolio.json")
        except ValueError as exc:
            st.error(f"保存失败：{exc}")
            return

    st.divider()
    if st.button("🧭 生成今日操作建议", type="primary",
                 disabled=not pf.positions):
        try:
            pf.validate()
        except ValueError as exc:
            st.error(f"持仓数据有误：{exc}")
            return
        end = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=_ADVISOR_LOOKBACK_DAYS)).strftime("%Y%m%d")
        data: dict[str, pd.DataFrame] = {}
        with st.spinner("获取最新行情…"):
            for p in pf.positions:
                try:
                    data[p.symbol] = _load_daily_cached(p.symbol, start, end)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"{p.symbol}：行情获取失败（{exc}）")
        if not data:
            st.error("没有可用行情数据")
            return
        try:
            advices = advise_portfolio(pf, data)
        except ValueError as exc:
            st.error(f"生成建议失败：{exc}")
            return
        st.subheader("🧭 今日操作建议")
        _advices_table(advices, data, pf)
    elif not pf.positions:
        st.info("先在上方表格添加持仓或观察股（点表格左下角 + 号），保存后生成建议。")


# ================================================================ 个股速览

def _select_symbol() -> str:
    """个股速览选股：优先用全市场搜索下拉（代码/名称均可筛），
    名称表不可用时降级为手录代码。返回 6 位代码。"""
    if _code_name_map_cached():
        options, default_idx = _snapshot_options()
        choice = st.selectbox(
            "搜索股票（输入代码或名称筛选）", options, index=default_idx)
        return choice.split()[0]
    # 名称表拉取失败（离线/网络异常）：回退到手录
    return st.text_input("股票代码（6 位）", "600900", max_chars=6).strip()


@st.fragment
def _render_snapshot_tab() -> None:
    """页签三：个股速览（基本情况 + 公告/资料外链）。"""
    st.subheader("🔎 个股速览")
    symbol = _select_symbol()
    if not (symbol.isdigit() and len(symbol) == 6):
        st.warning("请选择或输入合法的 6 位股票代码")
        return

    name = _name_of(symbol)
    if name:
        st.success(f"已选中：**{symbol} {name}**")
    else:
        st.info(f"已选中：**{symbol}**（未在代码表识别到名称，仍可尝试查询）")

    # ---- 基本情况 ----
    try:
        info = _stock_info_cached(symbol)
    except Exception as exc:  # noqa: BLE001 网络异常降级展示
        info = {}
        st.warning(f"基本情况获取失败（{exc}），可直接点下方外链查看")
    if info:
        c1, c2, c3 = st.columns(3)
        c1.metric("最新价", f"{info.get('price', '') or '-'}")
        c2.metric("所属行业", info.get("industry", "") or "-")
        c3.metric("上市时间", info.get("list_date", "") or "-")
        c4, c5 = st.columns(2)
        c4.metric("总市值", _fmt_yi(info.get("total_mv", "")))
        c5.metric("流通市值", _fmt_yi(info.get("float_mv", "")))

    # ---- 公告 / 资料外链 ----
    st.divider()
    st.markdown("##### 🔗 基本情况与公告（外部站点）")
    links = build_external_links(symbol)
    cols = st.columns(len(links))
    for col, (label, url) in zip(cols, links.items()):
        col.link_button(label, url, use_container_width=True)
    st.caption("外链跳转东方财富/同花顺/巨潮资讯，查看行情、公告全文与 F10 资料。"
               "数据仅供研究参考，不构成投资建议。")


_VIEWS = {
    "📊 策略回测": _render_backtest_tab,
    "💼 持仓助手": _render_advisor_tab,
    "🔎 个股速览": _render_snapshot_tab,
}


def main() -> None:
    st.title("📈 retailquant — 散户量化工具")
    st.caption("A 股真实规则模拟：T+1 · 整手 · 涨跌停 · 佣金/印花税/滑点 · 硬止损纪律 | "
               "仅供研究，不构成投资建议")

    # 单视图渲染：只执行当前选中页，避免每次交互都重算三个页签
    #（省内存、提速）；默认落地在回测页，不会预先拉取个股速览的全市场名称表。
    view = st.segmented_control(
        "功能导航", list(_VIEWS), default="📊 策略回测",
        label_visibility="collapsed")
    _VIEWS.get(view, _render_backtest_tab)()


main()
