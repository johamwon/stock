# -*- coding: utf-8 -*-
"""操作建议引擎：基于最新行情，对每只持仓给出明日操作建议。

决策优先级（与回测引擎的风控/信号规则一致，T+1 语义）：
    1. 已跌破止损线      -> 明日卖出（止损）
    2. 已达到止盈线      -> 明日卖出（止盈，趋势版不触发）
    3. 策略离场信号      -> 明日开盘卖出
    4. 接近止损（3%内）  -> 持有但警示
    5. 其他              -> 继续持有
观察股（shares=0）：策略买入信号 -> 给出按现金可买的整手股数建议。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from retailquant.config import (
    DEFAULT_BACKTEST_CONFIG,
    RISK_PROFILES,
    TAKE_PROFIT_DISABLED,
    RiskConfig,
)
from retailquant.logger import get_logger
from retailquant.portfolio import Portfolio, Position
from retailquant.strategy import ALL_STRATEGIES

log = get_logger("advisor")

# 距止损线 3% 以内给出预警
_NEAR_STOP_BUFFER = 0.03
# 信号计算所需的最少K线（覆盖 MACD 慢线 26+9、唐奇安 20 等预热期）
MIN_BARS_FOR_SIGNAL = 60

ACTION_STOP_LOSS = "⛔ 明日卖出（止损）"
ACTION_TAKE_PROFIT = "💰 明日卖出（止盈）"
ACTION_SELL_SIGNAL = "🔽 明日开盘卖出（策略离场）"
ACTION_NEAR_STOP = "⚠️ 持有（逼近止损线）"
ACTION_HOLD = "✊ 继续持有"
ACTION_BUY = "🟢 明日开盘买入"
ACTION_WATCH = "👀 观望（无信号）"


@dataclass(frozen=True)
class Advice:
    """单只标的的操作建议。"""

    symbol: str
    action: str
    reason: str
    last_date: pd.Timestamp
    last_close: float
    pnl_pct: float          # 持仓浮动盈亏（观察股为 0）
    stop_price: float       # 止损参考价（观察股为 0）
    target_price: float     # 止盈参考价（不止盈/观察股为 0）
    suggest_shares: int     # 建议买入股数（仅买入建议时 > 0）


def _latest_signal(pos: Position, df: pd.DataFrame) -> int:
    strategy = ALL_STRATEGIES[pos.strategy]()
    return int(strategy.generate_signals(df).iloc[-1])


def advise_position(pos: Position, df: pd.DataFrame, cash: float = 0.0) -> Advice:
    """对单只持仓/观察股生成建议。

    Args:
        pos:  持仓记录（shares=0 表示观察股）。
        df:   截至最新交易日的日线数据（建议 >= MIN_BARS_FOR_SIGNAL 根）。
        cash: 可用现金，仅用于观察股的买入股数测算。
    """
    if len(df) < MIN_BARS_FOR_SIGNAL:
        raise ValueError(f"{pos.symbol} 数据不足 {MIN_BARS_FOR_SIGNAL} 根K线，信号不可靠")
    if pos.strategy not in ALL_STRATEGIES:
        raise ValueError(f"{pos.symbol} 策略非法：{pos.strategy!r}")

    risk: RiskConfig = RISK_PROFILES[pos.risk_profile]
    last_date = df.index[-1]
    last_close = float(df["close"].iloc[-1])
    sig = _latest_signal(pos, df)

    # ---- 观察股：只看买入信号 ----
    if pos.shares == 0:
        if sig == 1:
            cfg = DEFAULT_BACKTEST_CONFIG
            budget = cash * risk.max_position_pct
            est_price = last_close * (1 + cfg.cost.slippage_rate)
            lots = int(budget / (est_price * cfg.lot_size)) if est_price > 0 else 0
            shares = lots * cfg.lot_size
            if shares <= 0:
                return Advice(pos.symbol, ACTION_WATCH,
                              f"策略给出买入信号，但现金 {cash:.0f} 元不足一手（约需 "
                              f"{est_price * cfg.lot_size:.0f} 元）",
                              last_date, last_close, 0.0, 0.0, 0.0, 0)
            return Advice(pos.symbol, ACTION_BUY,
                          f"{pos.strategy} 于 {last_date.date()} 收盘触发买入信号",
                          last_date, last_close, 0.0, 0.0, 0.0, shares)
        return Advice(pos.symbol, ACTION_WATCH,
                      f"{pos.strategy} 无买入信号", last_date, last_close,
                      0.0, 0.0, 0.0, 0)

    # ---- 持仓股：止损 > 止盈 > 离场信号 > 预警 > 持有 ----
    pnl_pct = last_close / pos.cost_price - 1.0
    stop_price = pos.cost_price * (1 - risk.stop_loss_pct)
    no_take_profit = risk.take_profit_pct >= TAKE_PROFIT_DISABLED
    target_price = 0.0 if no_take_profit else pos.cost_price * (1 + risk.take_profit_pct)

    if last_close <= stop_price:
        return Advice(pos.symbol, ACTION_STOP_LOSS,
                      f"现价已跌破止损线 {stop_price:.2f}（浮亏 {pnl_pct:.1%}），纪律优先",
                      last_date, last_close, pnl_pct, stop_price, target_price, 0)
    if not no_take_profit and last_close >= target_price:
        return Advice(pos.symbol, ACTION_TAKE_PROFIT,
                      f"现价已达止盈线 {target_price:.2f}（浮盈 {pnl_pct:.1%}）",
                      last_date, last_close, pnl_pct, stop_price, target_price, 0)
    if sig == -1:
        return Advice(pos.symbol, ACTION_SELL_SIGNAL,
                      f"{pos.strategy} 于 {last_date.date()} 收盘触发离场信号",
                      last_date, last_close, pnl_pct, stop_price, target_price, 0)
    if last_close <= stop_price * (1 + _NEAR_STOP_BUFFER):
        return Advice(pos.symbol, ACTION_NEAR_STOP,
                      f"现价距止损线 {stop_price:.2f} 不足 {_NEAR_STOP_BUFFER:.0%}，"
                      "明日若继续下跌请果断执行",
                      last_date, last_close, pnl_pct, stop_price, target_price, 0)
    return Advice(pos.symbol, ACTION_HOLD,
                  f"无离场信号（浮动盈亏 {pnl_pct:+.1%}）",
                  last_date, last_close, pnl_pct, stop_price, target_price, 0)


def advise_portfolio(pf: Portfolio, data: dict[str, pd.DataFrame]) -> list[Advice]:
    """对整个账户生成建议列表。

    Args:
        pf:   账户（含现金与持仓）。
        data: symbol -> 最新日线数据（由调用方负责获取，便于测试与缓存）。
    """
    advices: list[Advice] = []
    for pos in pf.positions:
        df = data.get(pos.symbol)
        if df is None:
            log.warning("%s 缺少行情数据，跳过", pos.symbol)
            continue
        advices.append(advise_position(pos, df, cash=pf.cash))
    return advices
