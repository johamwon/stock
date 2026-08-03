# -*- coding: utf-8 -*-
"""绩效统计与报告输出：把回测结果翻译成散户看得懂的指标。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retailquant.backtest import BacktestResult
from retailquant.config import OUTPUT_DIR
from retailquant.logger import get_logger

log = get_logger("report")

TRADING_DAYS_PER_YEAR = 244   # A 股年均交易日
RISK_FREE_RATE = 0.02         # 无风险利率近似（1年期存款/国债）


@dataclass(frozen=True)
class PerformanceMetrics:
    """核心绩效指标。"""

    total_return_pct: float       # 总收益率 %
    annual_return_pct: float      # 年化收益率 %
    max_drawdown_pct: float       # 最大回撤 %（负值）
    sharpe: float                 # 夏普比率
    volatility_pct: float         # 年化波动率 %
    num_trades: int               # 完整交易笔数
    win_rate_pct: float           # 胜率 %
    avg_win: float                # 平均盈利（元）
    avg_loss: float               # 平均亏损（元）
    profit_factor: float          # 盈亏比（总盈利/总亏损）
    total_costs: float            # 总交易费用（元）


def compute_metrics(result: BacktestResult) -> PerformanceMetrics:
    """从回测结果计算绩效指标。"""
    curve = result.equity_curve
    if curve.empty:
        raise ValueError("净值曲线为空")

    total_ret = curve.iloc[-1] / result.initial_capital - 1.0
    n_days = len(curve)
    years = n_days / TRADING_DAYS_PER_YEAR
    annual_ret = (1.0 + total_ret) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    daily_ret = curve.pct_change().dropna()
    vol = float(daily_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(daily_ret) > 1 else 0.0
    sharpe = (annual_ret - RISK_FREE_RATE) / vol if vol > 1e-12 else 0.0

    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    max_dd = float(drawdown.min())

    pnls = [t.pnl for t in result.trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_win / gross_loss if gross_loss > 1e-12 else float("inf") if gross_win > 0 else 0.0

    return PerformanceMetrics(
        total_return_pct=round(total_ret * 100, 2),
        annual_return_pct=round(annual_ret * 100, 2),
        max_drawdown_pct=round(max_dd * 100, 2),
        sharpe=round(sharpe, 3),
        volatility_pct=round(vol * 100, 2),
        num_trades=len(pnls),
        win_rate_pct=round(win_rate * 100, 2),
        avg_win=round(float(np.mean(wins)) if wins else 0.0, 2),
        avg_loss=round(float(np.mean(losses)) if losses else 0.0, 2),
        profit_factor=round(profit_factor, 3) if profit_factor != float("inf") else float("inf"),
        total_costs=round(result.total_costs, 2),
    )


def render_text_report(results: list[tuple[BacktestResult, PerformanceMetrics]],
                       title: str = "回归测试报告") -> str:
    """生成对齐的纯文本报告（多标的 × 多策略对比）。"""
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(f"  retailquant {title}")
    lines.append("=" * 100)
    header = (f"{'标的':<10}{'策略':<16}{'总收益%':>9}{'年化%':>9}{'回撤%':>9}"
              f"{'夏普':>8}{'交易数':>7}{'胜率%':>8}{'盈亏比':>8}{'费用(元)':>11}")
    lines.append(header)
    lines.append("-" * 100)
    for res, m in results:
        pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
        lines.append(
            f"{res.symbol:<12}{res.strategy_name:<18}{m.total_return_pct:>9.2f}"
            f"{m.annual_return_pct:>10.2f}{m.max_drawdown_pct:>10.2f}{m.sharpe:>9.3f}"
            f"{m.num_trades:>7d}{m.win_rate_pct:>9.2f}{pf:>9}{m.total_costs:>12.2f}"
        )
    lines.append("=" * 100)
    return "\n".join(lines)


def save_report(text: str, filename: str) -> str:
    """保存报告到 output 目录，返回文件路径。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    path.write_text(text, encoding="utf-8")
    log.info("报告已保存：%s", path)
    return str(path)


def save_trades_csv(result: BacktestResult, filename: str) -> str:
    """导出交易明细 CSV，便于散户逐笔复盘。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{
        "标的": t.symbol, "买入日": t.entry_date.date(), "买入价": round(t.entry_price, 3),
        "股数": t.shares, "卖出日": t.exit_date.date() if t.exit_date is not None else "",
        "卖出价": round(t.exit_price, 3) if t.exit_price is not None else "",
        "离场原因": t.exit_reason, "盈亏(元)": round(t.pnl, 2),
        "收益率%": round(t.ret_pct * 100, 2),
    } for t in result.trades]
    path = OUTPUT_DIR / filename
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)
