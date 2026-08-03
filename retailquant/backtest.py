# -*- coding: utf-8 -*-
"""回测引擎：严格模拟 A 股散户交易环境的事件式回测。

关键规则：
    1. T 日收盘生成信号，T+1 日开盘价成交（防未来函数）；
    2. T+1 制度：当日买入的股票当日不可卖出（含止损）；
    3. 整手交易：买入数量向下取整到 100 股；
    4. 涨跌停约束：开盘涨停不追买、开盘跌停不杀卖；
    5. 真实成本：佣金（最低5元）、卖出印花税、过户费、双边滑点；
    6. 风控纪律：持仓期间盘中触发硬止损/止盈，按触发价离场。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from retailquant.config import BacktestConfig, DEFAULT_BACKTEST_CONFIG, price_limit_for
from retailquant.logger import get_logger
from retailquant.strategy import Strategy

log = get_logger("backtest")


@dataclass
class Trade:
    """一笔完整交易（开仓 -> 平仓）。"""

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""          # signal / stop_loss / take_profit / eod
    pnl: float = 0.0               # 已扣除全部费用
    ret_pct: float = 0.0


@dataclass
class BacktestResult:
    """回测结果容器。"""

    symbol: str
    strategy_name: str
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: list[Trade] = field(default_factory=list)
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_costs: float = 0.0


class BacktestEngine:
    """单标的、只做多、全仓进出的日线回测引擎。"""

    def __init__(self, config: BacktestConfig = DEFAULT_BACKTEST_CONFIG) -> None:
        self.cfg = config

    # ------------------------------------------------------------ 费用
    def _buy_cost(self, amount: float) -> float:
        """买入费用：佣金（含最低收费）+ 过户费。"""
        c = self.cfg.cost
        return max(amount * c.commission_rate, c.commission_min) + amount * c.transfer_fee_rate

    def _sell_cost(self, amount: float) -> float:
        """卖出费用：佣金 + 印花税 + 过户费。"""
        c = self.cfg.cost
        commission = max(amount * c.commission_rate, c.commission_min)
        return commission + amount * c.stamp_tax_rate + amount * c.transfer_fee_rate

    # ------------------------------------------------------------ 涨跌停判断
    def _is_limit_up(self, price: float, prev_close: float, limit_pct: float) -> bool:
        return prev_close > 0 and price >= prev_close * (1 + limit_pct)

    def _is_limit_down(self, price: float, prev_close: float, limit_pct: float) -> bool:
        return prev_close > 0 and price <= prev_close * (1 - limit_pct)

    # ------------------------------------------------------------ 主流程
    def run(self, df: pd.DataFrame, strategy: Strategy, symbol: str = "") -> BacktestResult:
        """执行回测。

        Args:
            df: OHLCV 日线（date 索引升序）。
            strategy: 策略实例。
            symbol: 标的代码（仅用于报告展示）。
        """
        if len(df) < 2:
            raise ValueError("数据不足 2 根K线，无法回测")

        signals = strategy.generate_signals(df)
        # 按板块自适应涨跌停：创业板/科创板 ±20%，主板 ±10%
        limit_pct = price_limit_for(symbol, self.cfg.price_limit_pct)
        cash = self.cfg.initial_capital
        shares = 0
        entry_price = 0.0
        entry_date: pd.Timestamp | None = None
        pending: int = 0                # 前一日收盘信号，今日开盘执行
        total_costs = 0.0
        trades: list[Trade] = []
        equity: list[float] = []
        slip = self.cfg.cost.slippage_rate

        dates = df.index
        opens = df["open"].to_numpy()
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()

        def close_position(i: int, price: float, reason: str) -> None:
            nonlocal cash, shares, entry_price, entry_date, total_costs
            amount = shares * price
            cost = self._sell_cost(amount)
            cash += amount - cost
            total_costs += cost
            buy_amount = shares * entry_price
            buy_cost_est = self._buy_cost(buy_amount)
            pnl = amount - cost - buy_amount - buy_cost_est
            trades.append(Trade(
                symbol=symbol, entry_date=entry_date, entry_price=entry_price,
                shares=shares, exit_date=dates[i], exit_price=price,
                exit_reason=reason, pnl=pnl,
                ret_pct=pnl / (buy_amount + buy_cost_est) if buy_amount else 0.0,
            ))
            shares = 0
            entry_price = 0.0
            entry_date = None

        for i in range(len(df)):
            today = dates[i]
            prev_close = closes[i - 1] if i > 0 else 0.0

            # ---- 1) 开盘执行前一日信号 ----
            if pending == 1 and shares == 0:
                exec_price = opens[i] * (1 + slip)
                if self._is_limit_up(opens[i], prev_close, limit_pct):
                    log.debug("%s 开盘涨停，放弃买入", today.date())
                else:
                    budget = cash * self.cfg.risk.max_position_pct
                    lots = int(budget / (exec_price * self.cfg.lot_size))
                    buy_shares = lots * self.cfg.lot_size
                    if buy_shares == 0:
                        # 散户现实约束：本金不足一手（如高价股茅台），只能放弃
                        log.warning("[%s] %s 资金不足一手（价格 %.2f，预算 %.2f），放弃买入",
                                    symbol, today.date(), exec_price, budget)
                    else:
                        amount = buy_shares * exec_price
                        cost = self._buy_cost(amount)
                        if amount + cost <= cash:
                            cash -= amount + cost
                            total_costs += cost
                            shares = buy_shares
                            entry_price = exec_price
                            entry_date = today
            elif pending == -1 and shares > 0:
                if self._is_limit_down(opens[i], prev_close, limit_pct):
                    log.debug("%s 开盘跌停，无法卖出", today.date())
                else:
                    close_position(i, opens[i] * (1 - slip), "signal")

            # ---- 2) 盘中止损/止盈（T+1：当日买入不触发） ----
            if shares > 0 and entry_date is not None and entry_date < today:
                stop_price = entry_price * (1 - self.cfg.risk.stop_loss_pct)
                target_price = entry_price * (1 + self.cfg.risk.take_profit_pct)
                if lows[i] <= stop_price and not self._is_limit_down(lows[i], prev_close, limit_pct):
                    # 若开盘即低于止损价，按开盘价成交更贴近现实
                    fill = min(opens[i], stop_price) * (1 - slip)
                    close_position(i, fill, "stop_loss")
                elif highs[i] >= target_price:
                    fill = max(opens[i], target_price) * (1 - slip)
                    close_position(i, fill, "take_profit")

            # ---- 3) 收盘记录净值、生成次日待执行信号 ----
            equity.append(cash + shares * closes[i])
            pending = int(signals.iloc[i])

        # 期末若仍持仓，按最后收盘价强制平仓结算
        if shares > 0:
            close_position(len(df) - 1, closes[-1], "eod")
            equity[-1] = cash

        curve = pd.Series(equity, index=dates, name="equity")
        result = BacktestResult(
            symbol=symbol, strategy_name=strategy.name,
            equity_curve=curve, trades=trades,
            initial_capital=self.cfg.initial_capital,
            final_equity=float(curve.iloc[-1]),
            total_costs=total_costs,
        )
        log.info("[%s|%s] 回测完成：期末 %.2f / 交易 %d 笔 / 总费用 %.2f",
                 symbol, strategy.name, result.final_equity, len(trades), total_costs)
        return result
