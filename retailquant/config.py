# -*- coding: utf-8 -*-
"""全局配置：A 股散户真实交易环境的参数集中管理。

工程规约：所有"魔法数字"集中在此，代码中不允许散落硬编码。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data_cache"
OUTPUT_DIR = ROOT_DIR / "output"
LOG_DIR = ROOT_DIR / "logs"


@dataclass(frozen=True)
class TradeCostConfig:
    """A 股散户真实交易成本。

    - 佣金：万 2.5，单笔最低 5 元（散户主流费率）
    - 印花税：卖出 0.05%（2023-08-28 减半后费率）
    - 过户费：成交金额的 0.001%（沪深均收）
    - 滑点：按成交价的 0.1% 估计（小资金冲击成本低，但集合竞价/开盘波动需要留量）
    """

    commission_rate: float = 2.5e-4
    commission_min: float = 5.0
    stamp_tax_rate: float = 5e-4       # 仅卖出
    transfer_fee_rate: float = 1e-5
    slippage_rate: float = 1e-3


@dataclass(frozen=True)
class RiskConfig:
    """散户风控纪律：止损止盈是散户长期存活的第一要务。"""

    stop_loss_pct: float = 0.08        # 硬性止损 -8%
    take_profit_pct: float = 0.20      # 止盈 +20%
    max_position_pct: float = 0.95     # 最大仓位（留出手续费余量）


@dataclass(frozen=True)
class BacktestConfig:
    """回测环境参数。"""

    initial_capital: float = 60_000.0   # 散户典型本金 6 万
    lot_size: int = 100                 # A 股整手 100 股
    price_limit_pct: float = 0.099      # 主板涨跌停阈值（近似 ±10%，留精度余量）
    cost: TradeCostConfig = field(default_factory=TradeCostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


DEFAULT_BACKTEST_CONFIG = BacktestConfig()

# ---------------------------------------------------------------- 板块规则
# 创业板（300/301）、科创板（688/689）涨跌停 ±20%；主板 ±10%
_WIDE_LIMIT_PREFIXES = ("300", "301", "688", "689")
WIDE_PRICE_LIMIT_PCT = 0.199


def price_limit_for(symbol: str, default: float = 0.099) -> float:
    """按股票代码前缀返回涨跌停阈值（留精度余量）。"""
    if symbol and symbol.startswith(_WIDE_LIMIT_PREFIXES):
        return WIDE_PRICE_LIMIT_PCT
    return default


# ---------------------------------------------------------------- 风控档位
# 来自 AI 产业链对照实验结论（scripts/ai_chain_risk_experiment.py）：
#   纪律版：低波动蓝筹；宽松版：高波动股；趋势版：热门趋势股（不主动止盈，靠信号离场）
TAKE_PROFIT_DISABLED = 10.0  # take_profit_pct >= 此值视为不止盈
RISK_PROFILES: dict[str, RiskConfig] = {
    "纪律版": RiskConfig(stop_loss_pct=0.08, take_profit_pct=0.20),
    "宽松版": RiskConfig(stop_loss_pct=0.15, take_profit_pct=0.50),
    "趋势版": RiskConfig(stop_loss_pct=0.15, take_profit_pct=TAKE_PROFIT_DISABLED),
}
