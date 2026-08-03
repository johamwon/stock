# -*- coding: utf-8 -*-
"""策略模块：面向散户的低频日线策略。

设计原则（散户特点）：
    1. 低频：日线级别，收盘后 10 分钟即可完成决策，不影响上班；
    2. 规则简单透明：均线/MACD/RSI 人人看得懂，能坚持执行；
    3. 防未来函数：信号在 T 日收盘后生成，T+1 日开盘价成交；
    4. 只做多：散户无融券条件，不设计做空信号。

信号约定：
    signal 列取值 {1: 买入, -1: 卖出, 0: 观望}，表示 T 日收盘后的决策。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from retailquant import indicators as ind


class Strategy(ABC):
    """策略基类：输入 OHLCV，输出信号序列。"""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成信号序列（与 df 索引对齐，值域 {1, -1, 0}）。"""

    @staticmethod
    def _cross_up(fast: pd.Series, slow: pd.Series) -> pd.Series:
        """fast 上穿 slow（金叉），只用 T 日及以前的数据。"""
        return (fast > slow) & (fast.shift(1) <= slow.shift(1))

    @staticmethod
    def _cross_down(fast: pd.Series, slow: pd.Series) -> pd.Series:
        """fast 下穿 slow（死叉）。"""
        return (fast < slow) & (fast.shift(1) >= slow.shift(1))


class DualMovingAverage(Strategy):
    """双均线策略：短均线金叉买入、死叉卖出。

    散户最经典的趋势跟随入门策略，默认 MA5/MA20。
    """

    name = "dual_ma"

    def __init__(self, fast: int = 5, slow: int = 20) -> None:
        if fast >= slow:
            raise ValueError("fast 必须小于 slow")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ma_fast = ind.sma(df["close"], self.fast)
        ma_slow = ind.sma(df["close"], self.slow)
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[self._cross_up(ma_fast, ma_slow)] = 1
        sig[self._cross_down(ma_fast, ma_slow)] = -1
        # 均线未形成前不给信号
        sig[ma_slow.isna()] = 0
        return sig


class MacdCross(Strategy):
    """MACD 策略：DIF 金叉 DEA 且柱线转正买入；死叉卖出。

    加入零轴过滤：只在 DIF > 0（多头区）金叉才买，减少震荡市假信号。
    """

    name = "macd_cross"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 require_above_zero: bool = True) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.require_above_zero = require_above_zero

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        m = ind.macd(df["close"], self.fast, self.slow, self.signal)
        buy = self._cross_up(m["dif"], m["dea"])
        if self.require_above_zero:
            buy &= m["dif"] > 0
        sell = self._cross_down(m["dif"], m["dea"])
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy] = 1
        sig[sell] = -1
        return sig


class RsiReversal(Strategy):
    """RSI 超卖反转策略：RSI 上穿超卖线买入，上穿超买线后回落卖出。

    适合震荡市，配合回测引擎的硬止损控制单笔风险。
    """

    name = "rsi_reversal"

    def __init__(self, window: int = 14, oversold: float = 30.0,
                 overbought: float = 70.0) -> None:
        if not 0 < oversold < overbought < 100:
            raise ValueError("需满足 0 < oversold < overbought < 100")
        self.window = window
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        r = ind.rsi(df["close"], self.window)
        buy = (r > self.oversold) & (r.shift(1) <= self.oversold)
        sell = (r < self.overbought) & (r.shift(1) >= self.overbought)
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig


class DonchianBreakout(Strategy):
    """海龟/唐奇安通道突破策略（经典趋势跟踪，聚宽/优矿平台常青策略）。

    收盘突破前 entry_n 日最高价买入；跌破前 exit_n 日最低价卖出。
    靠多次小止损捕捉大趋势，胜率低但盈亏比高，适合单边趋势行情。
    """

    name = "donchian"

    def __init__(self, entry_n: int = 20, exit_n: int = 10) -> None:
        if entry_n <= 0 or exit_n <= 0:
            raise ValueError("entry_n/exit_n 必须为正整数")
        self.entry_n = entry_n
        self.exit_n = exit_n

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # shift(1)：通道只用“昨日及以前”的极值，避免用当日高低点自比
        upper = df["high"].rolling(self.entry_n, min_periods=self.entry_n).max().shift(1)
        lower = df["low"].rolling(self.exit_n, min_periods=self.exit_n).min().shift(1)
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[(df["close"] > upper).fillna(False)] = 1
        sig[(df["close"] < lower).fillna(False)] = -1
        return sig


class BollingerReversion(Strategy):
    """布林带均值回归策略（米筐官方示例同源思路）。

    收盘从下轨下方收回下轨上方时买入（超跌修复），回归中轨卖出。
    震荡市胜率高；单边下跌靠引擎 -8% 硬止损兑底。
    """

    name = "boll_reversion"

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        if window <= 1 or num_std <= 0:
            raise ValueError("window 需 >1 且 num_std 需 >0")
        self.window = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        bb = ind.bollinger(df["close"], self.window, self.num_std)
        buy = self._cross_up(df["close"], bb["lower"])
        sell = self._cross_up(df["close"], bb["mid"])
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig


class Momentum(Strategy):
    """动量择时策略（ETF 轮动类策略的单标的简化版）。

    N 日动量（涨幅）由负转正买入，由正转负卖出：只在“涨势中”持股，
    下跌期空仓规避，是二八轮动/ETF 轮动的核心择时因子。
    """

    name = "momentum"

    def __init__(self, window: int = 20) -> None:
        if window <= 0:
            raise ValueError("window 必须为正整数")
        self.window = window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        mom = df["close"].pct_change(self.window)
        buy = (mom > 0) & (mom.shift(1) <= 0)
        sell = (mom < 0) & (mom.shift(1) >= 0)
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig


class BuyAndHold(Strategy):
    """买入持有基准：首日买入不再操作，用于对比策略是否真的跑赢"拿着不动"。"""

    name = "buy_and_hold"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index, dtype=int)
        if len(sig) > 0:
            sig.iloc[0] = 1
        return sig


ALL_STRATEGIES: dict[str, type[Strategy]] = {
    DualMovingAverage.name: DualMovingAverage,
    MacdCross.name: MacdCross,
    RsiReversal.name: RsiReversal,
    DonchianBreakout.name: DonchianBreakout,
    BollingerReversion.name: BollingerReversion,
    Momentum.name: Momentum,
    BuyAndHold.name: BuyAndHold,
}
