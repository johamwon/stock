# -*- coding: utf-8 -*-
"""retailquant —— 面向 A 股散户的轻量级量化交易工具。

模块分层：
    config    全局配置（交易成本、风控参数）
    data      行情数据获取与本地缓存（akshare）
    indicators 常用技术指标（纯函数，向量化）
    strategy  交易策略（信号生成，防未来函数）
    backtest  回测引擎（T+1、整手、涨跌停、真实费用）
    report    绩效统计与报告输出
"""

__version__ = "0.1.0"
