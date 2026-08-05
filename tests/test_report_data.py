# -*- coding: utf-8 -*-
"""绩效与数据校验模块单元测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retailquant.backtest import BacktestEngine, BacktestResult, Trade
from retailquant.data import ensure_domestic_no_proxy, validate_ohlcv
from retailquant.report import compute_metrics, render_text_report
from retailquant.strategy import BuyAndHold
from tests.conftest import make_ohlcv


def _make_result(equity: list[float], trades: list[Trade] | None = None) -> BacktestResult:
    idx = pd.bdate_range("2024-01-01", periods=len(equity))
    return BacktestResult(
        symbol="TEST", strategy_name="unit",
        equity_curve=pd.Series(equity, index=idx),
        trades=trades or [],
        initial_capital=equity[0], final_equity=equity[-1],
    )


class TestMetrics:
    def test_flat_curve(self):
        m = compute_metrics(_make_result([100_000.0] * 50))
        assert m.total_return_pct == 0.0
        assert m.max_drawdown_pct == 0.0
        assert m.num_trades == 0

    def test_total_return(self):
        m = compute_metrics(_make_result([100_000.0, 105_000.0, 110_000.0]))
        assert m.total_return_pct == pytest.approx(10.0)

    def test_max_drawdown(self):
        # 100k -> 120k -> 90k：最大回撤 = 90/120-1 = -25%
        m = compute_metrics(_make_result([100_000.0, 120_000.0, 90_000.0, 95_000.0]))
        assert m.max_drawdown_pct == pytest.approx(-25.0)

    def test_win_rate(self):
        idx = pd.Timestamp("2024-01-02")
        trades = [
            Trade("T", idx, 10, 100, exit_date=idx, exit_price=11, pnl=100.0),
            Trade("T", idx, 10, 100, exit_date=idx, exit_price=9, pnl=-50.0),
        ]
        m = compute_metrics(_make_result([100_000.0, 100_050.0], trades))
        assert m.win_rate_pct == pytest.approx(50.0)
        assert m.profit_factor == pytest.approx(2.0)

    def test_empty_curve_raises(self):
        res = BacktestResult(symbol="T", strategy_name="u")
        with pytest.raises(ValueError):
            compute_metrics(res)


class TestReportRender:
    def test_render_contains_rows(self, trend_up_df):
        res = BacktestEngine().run(trend_up_df, BuyAndHold(), "600000")
        m = compute_metrics(res)
        text = render_text_report([(res, m)])
        assert "600000" in text
        assert "buy_and_hold" in text


class TestValidateOhlcv:
    def test_valid_passes(self, trend_up_df):
        validate_ohlcv(trend_up_df)  # 不应抛异常

    def test_missing_column(self, trend_up_df):
        with pytest.raises(ValueError, match="缺少必需列"):
            validate_ohlcv(trend_up_df.drop(columns=["volume"]))

    def test_nan_rejected(self, trend_up_df):
        df = trend_up_df.copy()
        df.iloc[3, df.columns.get_loc("close")] = np.nan
        with pytest.raises(ValueError, match="缺失值"):
            validate_ohlcv(df)

    def test_bad_high_low(self, trend_up_df):
        df = trend_up_df.copy()
        df.iloc[2, df.columns.get_loc("high")] = 0.01
        with pytest.raises(ValueError, match="非法K线"):
            validate_ohlcv(df)


class TestEnsureDomesticNoProxy:
    def test_appends_and_keeps_existing(self, monkeypatch):
        """追加国内域名且保留用户原有条目。

        注意：Windows 环境变量不区分大小写，需先清除再设置。
        """
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
        ensure_domestic_no_proxy()
        import os
        parts = os.environ["NO_PROXY"].split(",")
        assert "localhost" in parts and "127.0.0.1" in parts
        assert "qt.gtimg.cn" in parts and "cninfo.com.cn" in parts

    def test_idempotent(self, monkeypatch):
        """重复调用不产生重复条目。"""
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.setenv("NO_PROXY", "")
        ensure_domestic_no_proxy()
        import os
        first = os.environ["NO_PROXY"]
        ensure_domestic_no_proxy()
        assert os.environ["NO_PROXY"] == first
        assert first.split(",").count("qt.gtimg.cn") == 1
