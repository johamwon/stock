# -*- coding: utf-8 -*-
"""高热度股票的风控参数对照实验。

问题：默认纪律（止损8%/止盈20%）为低波动蓝筹设计，
高热度 AI 股年化波动 50%~72%，止损易被噪声打掉、止盈截断大趋势。
本实验对比 3 组风控参数下各策略的表现。

用法：
    python scripts/ai_chain_risk_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from retailquant.backtest import BacktestEngine
from retailquant.config import BacktestConfig, RiskConfig
from retailquant.data import load_daily
from retailquant.report import compute_metrics
from retailquant.strategy import ALL_STRATEGIES

AI_CHAIN = ["300308", "000977", "002463", "002230", "300418"]
START, END = "20250101", "20260728"
STRATS = ["dual_ma", "donchian", "momentum", "macd_cross"]

# 三组风控：纪律版（默认）/ 宽松版 / 趋势版（几乎不主动止盈，靠信号离场）
RISK_PROFILES = {
    "纪律版 8/20": RiskConfig(stop_loss_pct=0.08, take_profit_pct=0.20),
    "宽松版 15/50": RiskConfig(stop_loss_pct=0.15, take_profit_pct=0.50),
    "趋势版 15/∞": RiskConfig(stop_loss_pct=0.15, take_profit_pct=10.0),
}


def main() -> None:
    data = {s: load_daily(s, START, END) for s in AI_CHAIN}
    rows = []
    for profile_name, risk in RISK_PROFILES.items():
        engine = BacktestEngine(BacktestConfig(risk=risk))
        for strat_name in STRATS:
            rets, dds = [], []
            for symbol, df in data.items():
                res = engine.run(df, ALL_STRATEGIES[strat_name](), symbol=symbol)
                m = compute_metrics(res)
                rets.append(m.total_return_pct)
                dds.append(m.max_drawdown_pct)
            rows.append({
                "风控": profile_name, "策略": strat_name,
                "平均收益%": round(sum(rets) / len(rets), 1),
                "最差标的%": round(min(rets), 1),
                "最好标的%": round(max(rets), 1),
                "平均回撤%": round(sum(dds) / len(dds), 1),
                "5标的中盈利数": sum(1 for r in rets if r > 0),
            })
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.unicode.east_asian_width", True):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
