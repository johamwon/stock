# -*- coding: utf-8 -*-
"""AI 产业链行情特征分析：量化 2025 年至今各环节的趋势/波动/回撤。

用法：
    python scripts/ai_chain_trend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from retailquant.data import load_daily
from retailquant.report import TRADING_DAYS_PER_YEAR

AI_CHAIN = {
    "300308": "中际旭创(上游·光模块)",
    "000977": "浪潮信息(上游·服务器)",
    "002463": "沪电股份(上游·PCB)",
    "002230": "科大讯飞(中游·大模型)",
    "300418": "昆仑万维(下游·AI应用)",
}
START, END = "20250101", "20260728"


def analyze(symbol: str, name: str) -> dict:
    df = load_daily(symbol, START, END)
    close = df["close"]
    ret = close.pct_change().dropna()
    running_max = close.cummax()
    dd = (close / running_max - 1.0).min()
    # 半年度涨幅：观察热度节奏
    halves = close.resample("6ME").last()
    seg = close.iloc[-1] / close.iloc[0] - 1
    return {
        "标的": f"{symbol} {name}",
        "区间涨幅%": round(seg * 100, 1),
        "年化波动%": round(float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) * 100, 1),
        "最大回撤%": round(float(dd) * 100, 1),
        "单日涨超5%天数": int((ret > 0.05).sum()),
        "单日跌超5%天数": int((ret < -0.05).sum()),
        "最新价": round(float(close.iloc[-1]), 2),
        "半年度节奏": " -> ".join(f"{v:.0f}" for v in halves.round(0)),
    }


def main() -> None:
    rows = [analyze(s, n) for s, n in AI_CHAIN.items()]
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.unicode.east_asian_width", True):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
