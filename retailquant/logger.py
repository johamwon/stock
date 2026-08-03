# -*- coding: utf-8 -*-
"""统一日志：控制台 + 文件双通道，避免各模块各自 print。"""
from __future__ import annotations

import logging
import sys
from datetime import datetime

from retailquant.config import LOG_DIR

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """获取带控制台与文件输出的 logger（幂等，重复调用不重复挂 handler）。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(console)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"retailquant_{datetime.now():%Y%m%d}.log"
    fileh = logging.FileHandler(logfile, encoding="utf-8")
    fileh.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(fileh)
    return logger
