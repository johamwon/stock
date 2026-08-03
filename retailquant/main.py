# -*- coding: utf-8 -*-
"""命令行入口：一条命令完成"取数 -> 回测 -> 报告"全流程。

用法示例：
    python -m retailquant.main --symbols 600519 000858 --start 20220101 --end 20241231
"""
from __future__ import annotations

import argparse
from datetime import datetime

from retailquant.backtest import BacktestEngine
from retailquant.config import DEFAULT_BACKTEST_CONFIG
from retailquant.data import load_daily, validate_ohlcv
from retailquant.logger import get_logger
from retailquant.report import compute_metrics, render_text_report, save_report, save_trades_csv
from retailquant.strategy import ALL_STRATEGIES

log = get_logger("main")

DEFAULT_SYMBOLS = ["600900", "000858", "601318", "600036"]  # 长江电力/五粮液/平安/招行（6万本金均可买足一手）
DEFAULT_START = "20220101"
DEFAULT_END = "20251231"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="retailquant 散户量化回测工具")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="6 位股票代码列表")
    p.add_argument("--start", default=DEFAULT_START, help="开始日期 YYYYMMDD")
    p.add_argument("--end", default=DEFAULT_END, help="结束日期 YYYYMMDD")
    p.add_argument("--strategies", nargs="+", default=list(ALL_STRATEGIES),
                   choices=list(ALL_STRATEGIES), help="要运行的策略")
    return p.parse_args()


def run(symbols: list[str], start: str, end: str, strategy_names: list[str]) -> str:
    """执行多标的 × 多策略回归测试，返回报告文本。"""
    engine = BacktestEngine()
    results = []
    for symbol in symbols:
        try:
            df = load_daily(symbol, start, end)
            validate_ohlcv(df)
        except Exception as exc:  # noqa: BLE001
            log.error("跳过 %s：%s", symbol, exc)
            continue
        for name in strategy_names:
            strat = ALL_STRATEGIES[name]()
            res = engine.run(df, strat, symbol=symbol)
            metrics = compute_metrics(res)
            results.append((res, metrics))
            if res.trades:
                save_trades_csv(res, f"trades_{symbol}_{name}.csv")

    if not results:
        raise RuntimeError("没有任何标的成功回测")

    capital_wan = DEFAULT_BACKTEST_CONFIG.initial_capital / 10_000
    report = render_text_report(
        results,
        title=f"回归测试报告（{start} ~ {end}，本金 {capital_wan:.0f} 万，T+1/整手/含费用）")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_report(report, f"regression_report_{stamp}.txt")
    return report


def main() -> None:
    args = parse_args()
    report = run(args.symbols, args.start, args.end, args.strategies)
    print()
    print(report)


if __name__ == "__main__":
    main()
