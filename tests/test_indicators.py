# -*- coding: utf-8 -*-
"""指标模块单元测试：用手工可验证的小样本核对数值。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retailquant import indicators as ind


class TestSma:
    def test_basic_value(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = ind.sma(s, 3)
        assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
        assert out.iloc[2] == pytest.approx(2.0)
        assert out.iloc[4] == pytest.approx(4.0)

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            ind.sma(pd.Series([1.0]), 0)


class TestEma:
    def test_first_value_equals_input(self):
        s = pd.Series([10.0, 11.0, 12.0])
        out = ind.ema(s, 5)
        assert out.iloc[0] == pytest.approx(10.0)

    def test_converges_to_constant(self):
        s = pd.Series([7.0] * 50)
        out = ind.ema(s, 10)
        assert out.iloc[-1] == pytest.approx(7.0)


class TestMacd:
    def test_columns_and_hist_relation(self):
        s = pd.Series(np.linspace(10, 20, 60))
        m = ind.macd(s)
        assert list(m.columns) == ["dif", "dea", "hist"]
        # hist = (dif - dea) * 2
        np.testing.assert_allclose(m["hist"], (m["dif"] - m["dea"]) * 2.0)

    def test_uptrend_dif_positive(self):
        s = pd.Series(np.linspace(10, 30, 80))
        m = ind.macd(s)
        assert m["dif"].iloc[-1] > 0


class TestRsi:
    def test_all_up_is_100(self):
        s = pd.Series(np.arange(1.0, 40.0))
        out = ind.rsi(s, 14)
        assert out.iloc[-1] == pytest.approx(100.0)

    def test_range_bounds(self):
        rng = np.random.default_rng(42)
        s = pd.Series(100 + rng.normal(0, 1, 200).cumsum())
        out = ind.rsi(s, 14).dropna()
        assert ((out >= 0) & (out <= 100)).all()

    def test_warmup_is_nan(self):
        s = pd.Series(np.arange(1.0, 40.0))
        out = ind.rsi(s, 14)
        assert out.iloc[:13].isna().all()


class TestBollinger:
    def test_band_order(self):
        rng = np.random.default_rng(0)
        s = pd.Series(50 + rng.normal(0, 2, 100).cumsum())
        bb = ind.bollinger(s).dropna()
        assert (bb["upper"] >= bb["mid"]).all()
        assert (bb["mid"] >= bb["lower"]).all()

    def test_constant_series_zero_width(self):
        s = pd.Series([10.0] * 30)
        bb = ind.bollinger(s).dropna()
        np.testing.assert_allclose(bb["upper"], bb["lower"])


class TestAtr:
    def test_positive(self, trend_up_df):
        out = ind.atr(trend_up_df).dropna()
        assert (out > 0).all()
