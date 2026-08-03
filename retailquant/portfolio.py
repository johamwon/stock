# -*- coding: utf-8 -*-
"""持仓管理：用户手动录入的真实持仓 + JSON 持久化。

散户视角：把券商 App 里的持仓抄进来（代码/股数/成本价），
为每只票指定策略与风控档位，工具据此给出每日操作建议。
股数为 0 的记录视为"自选观察股"，用于产生买入信号。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from retailquant.config import RISK_PROFILES, ROOT_DIR
from retailquant.logger import get_logger

log = get_logger("portfolio")

PORTFOLIO_PATH = ROOT_DIR / "portfolio.json"
_DEFAULT_STRATEGY = "donchian"      # 实验结论：高热股首选通道突破
_DEFAULT_PROFILE = "纪律版"


@dataclass
class Position:
    """单只持仓（shares=0 表示仅观察，等待买入信号）。"""

    symbol: str
    shares: int = 0
    cost_price: float = 0.0
    strategy: str = _DEFAULT_STRATEGY
    risk_profile: str = _DEFAULT_PROFILE

    def validate(self) -> None:
        if not (self.symbol.isdigit() and len(self.symbol) == 6):
            raise ValueError(f"股票代码非法：{self.symbol!r}（应为 6 位数字）")
        if self.shares < 0:
            raise ValueError(f"{self.symbol} 股数不能为负")
        if self.shares > 0 and self.cost_price <= 0:
            raise ValueError(f"{self.symbol} 持仓中，成本价必须大于 0")
        if self.risk_profile not in RISK_PROFILES:
            raise ValueError(f"{self.symbol} 风控档位非法：{self.risk_profile!r}，"
                             f"可选 {list(RISK_PROFILES)}")


@dataclass
class Portfolio:
    """账户快照：可用现金 + 持仓/观察列表。"""

    cash: float = 0.0
    positions: list[Position] = field(default_factory=list)

    def validate(self) -> None:
        if self.cash < 0:
            raise ValueError("现金不能为负")
        seen: set[str] = set()
        for pos in self.positions:
            pos.validate()
            if pos.symbol in seen:
                raise ValueError(f"重复的股票代码：{pos.symbol}")
            seen.add(pos.symbol)


def save_portfolio(pf: Portfolio, path: Path = PORTFOLIO_PATH) -> None:
    """校验后保存到 JSON（utf-8）。"""
    pf.validate()
    payload = {"cash": pf.cash, "positions": [asdict(p) for p in pf.positions]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("持仓已保存：%d 条记录，现金 %.2f -> %s", len(pf.positions), pf.cash, path.name)


def load_portfolio(path: Path = PORTFOLIO_PATH) -> Portfolio:
    """从 JSON 加载；文件不存在时返回空账户。"""
    if not path.exists():
        return Portfolio()
    payload = json.loads(path.read_text(encoding="utf-8"))
    pf = Portfolio(
        cash=float(payload.get("cash", 0.0)),
        positions=[Position(**p) for p in payload.get("positions", [])],
    )
    pf.validate()
    return pf
