# -*- coding: utf-8 -*-
"""策略模块单元测试：信号值域、防未来函数、经典场景验证。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retailquant.strategy import (
    ALL_STRATEGIES, BollingerReversion, BuyAndHold, DonchianBreakout,
    DualMovingAverage, MacdCross, Momentum, RsiReversal,
)
from tests.conftest import make_ohlcv


def _v_shape_df() -> pd.DataFrame:
    """先跌 30 日再涨 30 日的 V 形行情，双均线应先死叉后金叉。"""
    down = [20.0 * (1 - 0.015) ** i for i in range(30)]
    up = [down[-1] * (1 + 0.02) ** i for i in range(1, 31)]
    return make_ohlcv(down + up)


class TestSignalContract:
    """所有策略的公共契约。"""

    @pytest.mark.parametrize("name", list(ALL_STRATEGIES))
    def test_signal_domain_and_alignment(self, name, trend_up_df):
        strat = ALL_STRATEGIES[name]()
        sig = strat.generate_signals(trend_up_df)
        assert sig.index.equals(trend_up_df.index)
        assert set(sig.unique()).issubset({-1, 0, 1})

    @pytest.mark.parametrize("name", ["dual_ma", "macd_cross", "rsi_reversal",
                                      "donchian", "boll_reversion", "momentum"])
    def test_no_lookahead(self, name):
        """防未来函数：截断后半段数据，前半段信号不应改变。"""
        df = _v_shape_df()
        strat = ALL_STRATEGIES[name]()
        full = strat.generate_signals(df)
        half = strat.generate_signals(df.iloc[:40])
        pd.testing.assert_series_equal(full.iloc[:40], half, check_names=False)


class TestDualMovingAverage:
    def test_invalid_params(self):
        with pytest.raises(ValueError):
            DualMovingAverage(fast=20, slow=5)

    def test_v_shape_has_buy_after_bottom(self):
        df = _v_shape_df()
        sig = DualMovingAverage(5, 20).generate_signals(df)
        buys = sig[sig == 1]
        assert len(buys) >= 1
        # 金叉应出现在 V 形底部（第 30 根）之后
        assert buys.index[0] > df.index[30]

    def test_no_signal_before_warmup(self):
        df = _v_shape_df()
        sig = DualMovingAverage(5, 20).generate_signals(df)
        assert (sig.iloc[:19] == 0).all()


class TestMacdCross:
    def test_downtrend_no_buy_with_zero_filter(self, trend_down_df):
        sig = MacdCross(require_above_zero=True).generate_signals(trend_down_df)
        assert (sig != 1).all()


class TestRsiReversal:
    def test_invalid_thresholds(self):
        with pytest.raises(ValueError):
            RsiReversal(oversold=80, overbought=30)

    def test_v_shape_buy_near_bottom(self):
        df = _v_shape_df()
        sig = RsiReversal().generate_signals(df)
        buys = sig[sig == 1]
        assert len(buys) >= 1


class TestDonchianBreakout:
    def test_invalid_params(self):
        with pytest.raises(ValueError):
            DonchianBreakout(entry_n=0)

    def test_uptrend_breakout_buy(self, trend_up_df):
        """持续创新高的行情应出现买入信号，且不早于通道形成。"""
        sig = DonchianBreakout(20, 10).generate_signals(trend_up_df)
        buys = sig[sig == 1]
        assert len(buys) >= 1
        assert buys.index[0] >= trend_up_df.index[20]

    def test_downtrend_no_buy(self, trend_down_df):
        sig = DonchianBreakout(20, 10).generate_signals(trend_down_df)
        assert (sig != 1).all()


class TestBollingerReversion:
    def test_invalid_params(self):
        with pytest.raises(ValueError):
            BollingerReversion(window=1)

    def test_crash_recovery_buy(self):
        """横盘→急跌破下轨→修复：收回下轨时应触发买入。"""
        # 小幅震荡横盘 25 日，第 26 日急跌 8% 破下轨，随后逐步修复
        base = [10.0 + 0.05 * ((-1) ** i) for i in range(25)]
        crash_recover = [9.2, 9.3, 9.6, 9.9, 10.0]
        df = make_ohlcv(base + crash_recover)
        sig = BollingerReversion().generate_signals(df)
        buys = sig[sig == 1]
        assert len(buys) >= 1
        assert buys.index[0] > df.index[25]


class TestMomentum:
    def test_invalid_params(self):
        with pytest.raises(ValueError):
            Momentum(window=0)

    def test_pure_downtrend_no_buy(self, trend_down_df):
        """单边下跌中动量始终为负，不应出现买入信号。"""
        sig = Momentum(20).generate_signals(trend_down_df)
        assert (sig != 1).all()

    def test_v_shape_buy_after_turn(self):
        """V 形反转后动量由负转正，应出现买入信号。"""
        df = _v_shape_df()
        sig = Momentum(20).generate_signals(df)
        buys = sig[sig == 1]
        assert len(buys) >= 1
        assert buys.index[0] > df.index[30]


class TestBuyAndHold:
    def test_only_first_day_buy(self, trend_up_df):
        sig = BuyAndHold().generate_signals(trend_up_df)
        assert sig.iloc[0] == 1
        assert (sig.iloc[1:] == 0).all()
