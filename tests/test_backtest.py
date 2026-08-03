# -*- coding: utf-8 -*-
"""回测引擎单元测试：验证 A 股交易规则被正确执行。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retailquant.backtest import BacktestEngine
from retailquant.config import BacktestConfig, RiskConfig, TradeCostConfig, price_limit_for
from retailquant.strategy import BuyAndHold, Strategy
from tests.conftest import make_ohlcv


class _FixedSignal(Strategy):
    """测试辅助策略：按预设序列给信号。"""

    name = "fixed"

    def __init__(self, signals: list[int]) -> None:
        self._sig = signals

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        vals = (self._sig + [0] * len(df))[: len(df)]
        return pd.Series(vals, index=df.index, dtype=int)


def _no_cost_config(**risk_kw) -> BacktestConfig:
    """零费用零滑点配置，便于精确断言。"""
    return BacktestConfig(
        initial_capital=100_000.0,
        cost=TradeCostConfig(commission_rate=0.0, commission_min=0.0,
                             stamp_tax_rate=0.0, transfer_fee_rate=0.0,
                             slippage_rate=0.0),
        risk=RiskConfig(**risk_kw) if risk_kw else RiskConfig(),
    )


class TestBasicRules:
    def test_too_few_bars_raises(self):
        df = make_ohlcv([10.0])
        with pytest.raises(ValueError):
            BacktestEngine().run(df, BuyAndHold(), "TEST")

    def test_lot_size_rounding(self):
        """10 万本金、开盘价 333 元：只能买 200 股（2 手），不能买 285 股。"""
        df = make_ohlcv([333.0] * 10)
        cfg = _no_cost_config()
        res = BacktestEngine(cfg).run(df, _FixedSignal([1]), "TEST")
        assert res.trades, "应有一笔期末强平交易"
        assert res.trades[0].shares % 100 == 0
        assert res.trades[0].shares == 200  # 95000/333=285.28 -> 2手

    def test_t_plus_1_execution(self):
        """T 日信号 T+1 开盘成交：首日信号，成交日应为第 2 根K线。"""
        df = make_ohlcv([10.0] * 10)
        cfg = _no_cost_config()
        res = BacktestEngine(cfg).run(df, _FixedSignal([1]), "TEST")
        assert res.trades[0].entry_date == df.index[1]

    def test_signal_roundtrip_pnl(self):
        """无费用时：10 元开盘买、12 元开盘卖，盈亏应精确等于价差×股数。"""
        closes = [10.0] * 5 + [12.0] * 5
        df = make_ohlcv(closes)
        # 平滑 open，避免 fixture 的默认扰动影响断言
        df.loc[:, "open"] = [10.0] * 6 + [12.0] * 4
        df.loc[:, "high"] = df[["open", "close"]].max(axis=1) * 1.001
        df.loc[:, "low"] = df[["open", "close"]].min(axis=1) * 0.999
        cfg = _no_cost_config(stop_loss_pct=0.99, take_profit_pct=9.9)
        sig = [1, 0, 0, 0, 0, 0, -1]        # 第1根收盘买 -> 第2根开盘成交；第7根收盘卖 -> 第8根开盘
        res = BacktestEngine(cfg).run(df, _FixedSignal(sig), "TEST")
        t = res.trades[0]
        assert t.entry_price == pytest.approx(10.0)
        assert t.exit_price == pytest.approx(12.0)
        assert t.pnl == pytest.approx((12.0 - 10.0) * t.shares)


class TestRiskControl:
    def test_stop_loss_triggered(self):
        """买入后暴跌 20%，应触发 -8% 硬止损，单笔亏损接近 -8%。"""
        closes = [10.0, 10.0, 10.0, 8.0, 7.5, 7.0, 7.0, 7.0]
        df = make_ohlcv(closes)
        cfg = _no_cost_config()
        res = BacktestEngine(cfg).run(df, _FixedSignal([1]), "TEST")
        stop_trades = [t for t in res.trades if t.exit_reason == "stop_loss"]
        assert stop_trades, "应触发止损"
        # 止损含跳空：亏损不应显著超过跳空后的开盘价折算
        assert stop_trades[0].ret_pct < -0.05

    def test_take_profit_triggered(self):
        """买入后大涨 30%，应触发 +20% 止盈。"""
        closes = [10.0, 10.0, 10.5, 11.5, 12.5, 13.5, 13.5, 13.5]
        df = make_ohlcv(closes)
        cfg = _no_cost_config()
        res = BacktestEngine(cfg).run(df, _FixedSignal([1]), "TEST")
        tp = [t for t in res.trades if t.exit_reason == "take_profit"]
        assert tp, "应触发止盈"
        assert tp[0].ret_pct >= 0.15

    def test_t_plus_1_no_same_day_stop(self):
        """当日买入当日暴跌，不允许当日止损（T+1），最早次日离场。"""
        closes = [10.0, 9.0, 8.5, 8.5, 8.5]
        df = make_ohlcv(closes)
        cfg = _no_cost_config()
        res = BacktestEngine(cfg).run(df, _FixedSignal([1]), "TEST")
        stop_trades = [t for t in res.trades if t.exit_reason == "stop_loss"]
        if stop_trades:
            assert stop_trades[0].exit_date > stop_trades[0].entry_date


class TestCosts:
    def test_commission_minimum(self):
        """小额成交佣金按最低 5 元收取。"""
        engine = BacktestEngine(BacktestConfig())
        # 1000 元成交额：按万2.5 是 0.25 元 -> 应收 5 元 + 过户费
        cost = engine._buy_cost(1000.0)
        assert cost == pytest.approx(5.0 + 1000.0 * 1e-5)

    def test_sell_cost_includes_stamp_tax(self):
        engine = BacktestEngine(BacktestConfig())
        amount = 100_000.0
        cost = engine._sell_cost(amount)
        expected = amount * 2.5e-4 + amount * 5e-4 + amount * 1e-5
        assert cost == pytest.approx(expected)

    def test_costs_reduce_equity(self):
        """同样行情下，含费用的期末净值应低于零费用。"""
        closes = [10.0 + 0.05 * i for i in range(30)]
        df = make_ohlcv(closes)
        res_free = BacktestEngine(_no_cost_config()).run(df, BuyAndHold(), "T")
        res_real = BacktestEngine(BacktestConfig()).run(df, BuyAndHold(), "T")
        assert res_real.final_equity < res_free.final_equity


class TestPriceLimit:
    def test_price_limit_for_by_board(self):
        """主板 ±10%，创业板/科创板 ±20%。"""
        assert price_limit_for("600036") == pytest.approx(0.099)
        assert price_limit_for("000858") == pytest.approx(0.099)
        assert price_limit_for("300308") == pytest.approx(0.199)
        assert price_limit_for("688256") == pytest.approx(0.199)

    def test_limit_up_open_blocks_buy(self):
        """次日开盘一字涨停（+10%）时不追买。"""
        closes = [10.0, 11.0, 12.1, 13.3, 13.3]
        df = make_ohlcv(closes)
        # 手工构造：第2根开盘 = 前收 * 1.10（一字涨停开盘）
        df.loc[:, "open"] = [10.0, 11.0, 12.1, 13.3, 13.3]
        df.loc[:, "high"] = df["open"] * 1.0
        df.loc[:, "low"] = df["open"] * 1.0
        df.loc[:, "close"] = df["open"]
        cfg = _no_cost_config()
        res = BacktestEngine(cfg).run(df, _FixedSignal([1, 0, 0, 0, 0]), "TEST")
        # 第2根开盘 11.0 = 10.0*1.10 >= 涨停阈值 -> 放弃买入；后续无信号
        assert all(t.entry_date != df.index[1] for t in res.trades)

    def test_chinext_10pct_gap_does_not_block_buy(self):
        """同样 +10% 高开，创业板（±20%）不算涨停，应正常买入。"""
        df = make_ohlcv([10.0, 11.0, 12.1, 13.3, 13.3])
        df.loc[:, "open"] = [10.0, 11.0, 12.1, 13.3, 13.3]
        df.loc[:, "high"] = df["open"]
        df.loc[:, "low"] = df["open"]
        df.loc[:, "close"] = df["open"]
        cfg = _no_cost_config()
        res = BacktestEngine(cfg).run(df, _FixedSignal([1, 0, 0, 0, 0]), "300308")
        assert any(t.entry_date == df.index[1] for t in res.trades)


class TestEquityCurve:
    def test_curve_length_and_start(self, trend_up_df):
        res = BacktestEngine(_no_cost_config()).run(trend_up_df, BuyAndHold(), "T")
        assert len(res.equity_curve) == len(trend_up_df)
        assert res.equity_curve.iloc[0] == pytest.approx(100_000.0)

    def test_cash_conservation_no_cost(self, trend_up_df):
        """零费用时资金守恒：期末净值 = 本金 + 全部交易盈亏。"""
        res = BacktestEngine(_no_cost_config()).run(trend_up_df, BuyAndHold(), "T")
        total_pnl = sum(t.pnl for t in res.trades)
        assert res.final_equity == pytest.approx(100_000.0 + total_pnl, rel=1e-9)
