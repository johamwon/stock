# -*- coding: utf-8 -*-
"""技术指标模块：纯函数 + 向量化实现，不修改入参 DataFrame。

全部基于历史窗口计算，天然无未来函数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均线。"""
    if window <= 0:
        raise ValueError("window 必须为正整数")
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均线。"""
    if span <= 0:
        raise ValueError("span 必须为正整数")
    return series.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD 指标（国内软件常见参数 12/26/9）。

    Returns:
        DataFrame，含 dif / dea / hist 三列（hist 已按国内惯例 ×2）。
    """
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    hist = (dif - dea) * 2.0
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist})


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI 相对强弱指标（Wilder 平滑）。取值范围 0~100。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # 全涨无跌时 avg_loss=0，RSI 定义为 100
    out = out.where(avg_loss != 0.0, 100.0)
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """布林带：mid / upper / lower。"""
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    return pd.DataFrame({
        "mid": mid,
        "upper": mid + num_std * std,
        "lower": mid - num_std * std,
    })


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """平均真实波幅 ATR，需要 high/low/close 三列。"""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
