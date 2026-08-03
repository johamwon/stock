# -*- coding: utf-8 -*-
"""共享测试夹具。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    """由收盘价序列构造合法 OHLCV（open=前收，high/low 包住 open/close）。"""
    closes_arr = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes_arr[0]], closes_arr[:-1]])
    highs = np.maximum(opens, closes_arr) * 1.01
    lows = np.minimum(opens, closes_arr) * 0.99
    idx = pd.bdate_range(start=start, periods=len(closes_arr))
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes_arr, "volume": 1_000_000.0,
    }, index=pd.Index(idx, name="date"))


@pytest.fixture
def trend_up_df() -> pd.DataFrame:
    """60 日温和上涨行情。"""
    closes = [10.0 * (1.0 + 0.01) ** i for i in range(60)]
    return make_ohlcv(closes)


@pytest.fixture
def trend_down_df() -> pd.DataFrame:
    """60 日温和下跌行情。"""
    closes = [10.0 * (1.0 - 0.01) ** i for i in range(60)]
    return make_ohlcv(closes)
