# -*- coding: utf-8 -*-
"""advisor / portfolio 模块单元测试。"""
from __future__ import annotations

import pytest

from retailquant.advisor import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_NEAR_STOP,
    ACTION_SELL_SIGNAL,
    ACTION_STOP_LOSS,
    ACTION_TAKE_PROFIT,
    ACTION_WATCH,
    MIN_BARS_FOR_SIGNAL,
    advise_portfolio,
    advise_position,
)
from retailquant.portfolio import Portfolio, Position, load_portfolio, save_portfolio
from tests.conftest import make_ohlcv


def _flat_df(price: float = 10.0, n: int = MIN_BARS_FOR_SIGNAL + 10):
    """恒定价格长横盘：所有策略均无信号（交替扰动会造成均线假穿越）。"""
    return make_ohlcv([price] * n)


class TestPositionValidation:
    def test_bad_symbol_rejected(self):
        with pytest.raises(ValueError, match="代码非法"):
            Position(symbol="6009", shares=100, cost_price=10.0).validate()

    def test_holding_requires_cost(self):
        with pytest.raises(ValueError, match="成本价"):
            Position(symbol="600900", shares=100, cost_price=0.0).validate()

    def test_bad_risk_profile_rejected(self):
        with pytest.raises(ValueError, match="风控档位"):
            Position(symbol="600900", risk_profile="不存在").validate()

    def test_duplicate_symbol_rejected(self):
        pf = Portfolio(cash=0, positions=[Position("600900"), Position("600900")])
        with pytest.raises(ValueError, match="重复"):
            pf.validate()


class TestPortfolioIO:
    def test_json_roundtrip(self, tmp_path):
        pf = Portfolio(cash=12345.6, positions=[
            Position("600900", 1000, 28.5, "dual_ma", "纪律版"),
            Position("300308", 0, 0.0, "donchian", "趋势版"),
        ])
        path = tmp_path / "pf.json"
        save_portfolio(pf, path)
        loaded = load_portfolio(path)
        assert loaded == pf

    def test_missing_file_returns_empty(self, tmp_path):
        pf = load_portfolio(tmp_path / "nope.json")
        assert pf.cash == 0.0 and pf.positions == []


class TestAdvisePosition:
    def test_insufficient_bars_raises(self):
        pos = Position("600900", 100, 10.0)
        with pytest.raises(ValueError, match="数据不足"):
            advise_position(pos, _flat_df(n=MIN_BARS_FOR_SIGNAL - 1))

    def test_stop_loss_has_top_priority(self):
        # 成本 12，现价 10：-16.7% 破纪律版 8% 止损线
        pos = Position("600900", 100, 12.0, "dual_ma", "纪律版")
        adv = advise_position(pos, _flat_df(10.0))
        assert adv.action == ACTION_STOP_LOSS
        assert adv.pnl_pct < -0.08

    def test_take_profit_triggered(self):
        # 成本 8，现价 10：+25% 达纪律版 20% 止盈线
        pos = Position("600900", 100, 8.0, "dual_ma", "纪律版")
        adv = advise_position(pos, _flat_df(10.0))
        assert adv.action == ACTION_TAKE_PROFIT

    def test_trend_profile_never_takes_profit(self):
        # 趋势版：浮盈 25% 也不止盈，横盘无信号 -> 继续持有
        pos = Position("600900", 100, 8.0, "dual_ma", "趋势版")
        adv = advise_position(pos, _flat_df(10.0))
        assert adv.action == ACTION_HOLD
        assert adv.target_price == 0.0

    def test_near_stop_warning(self):
        # 纪律版止损线 = 10.8*0.92 = 9.936，现价 10.0 在 3% 预警区内且未破线
        pos = Position("600900", 100, 10.8, "dual_ma", "纪律版")
        adv = advise_position(pos, _flat_df(10.0))
        assert adv.action == ACTION_NEAR_STOP

    def test_sell_signal_from_strategy(self):
        # 长升后急跌必现 MA5/MA20 死叉；截断到死叉日，使离场信号落在最后一根K线
        from retailquant.strategy import DualMovingAverage
        closes = [10.0 + 0.05 * i for i in range(70)] + \
                 [13.45 - 0.4 * i for i in range(1, 11)]
        df = make_ohlcv(closes)
        sigs = DualMovingAverage().generate_signals(df)
        sell_days = sigs[sigs == -1].index
        assert len(sell_days) > 0, "夹具应产生死叉"
        df = df.loc[:sell_days[-1]]
        pos = Position("600900", 100, 11.0, "dual_ma", "趋势版")
        adv = advise_position(pos, df)
        assert adv.action == ACTION_SELL_SIGNAL

    def test_watch_stock_buy_signal_with_lot_sizing(self):
        # 观察股 + 上涨突破 -> donchian 买入信号，按现金测算整手股数
        closes = [10.0] * 60 + [10.2, 10.5, 10.9, 11.4, 12.0]
        pos = Position("600900", 0, 0.0, "donchian", "趋势版")
        adv = advise_position(pos, make_ohlcv(closes), cash=60_000.0)
        assert adv.action == ACTION_BUY
        assert adv.suggest_shares > 0
        assert adv.suggest_shares % 100 == 0

    def test_watch_stock_no_signal(self):
        pos = Position("600900", 0, 0.0, "donchian", "趋势版")
        adv = advise_position(pos, _flat_df(10.0), cash=60_000.0)
        assert adv.action == ACTION_WATCH

    def test_buy_signal_but_cash_insufficient(self):
        closes = [10.0] * 60 + [10.2, 10.5, 10.9, 11.4, 12.0]
        pos = Position("600900", 0, 0.0, "donchian", "趋势版")
        adv = advise_position(pos, make_ohlcv(closes), cash=500.0)
        assert adv.action == ACTION_WATCH
        assert "不足一手" in adv.reason


class TestAdvisePortfolio:
    def test_missing_data_skipped(self):
        pf = Portfolio(cash=0, positions=[
            Position("600900", 100, 10.0), Position("000858", 100, 10.0),
        ])
        advices = advise_portfolio(pf, {"600900": _flat_df(10.0)})
        assert len(advices) == 1
        assert advices[0].symbol == "600900"
