# -*- coding: utf-8 -*-
"""行情数据模块：akshare 获取 A 股日线（前复权）+ 本地 CSV 缓存。

散户视角：
    - 只用免费公开数据源（akshare / 腾讯、新浪接口），无需付费行情
    - 本地缓存避免重复请求被限流
    - 前复权价格保证回测收益率连续性
"""
from __future__ import annotations

import os
import time

import pandas as pd

from retailquant.config import DATA_DIR
from retailquant.logger import get_logger

log = get_logger("data")

# akshare 各数据源返回的中文列 -> 标准英文列
_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

_RETRY_TIMES = 3
_RETRY_WAIT_SEC = 2.0

# 国内数据源域名：无需代理，直连更稳（本机代理故障时仍可用）
_DIRECT_DOMAINS = (
    "qt.gtimg.cn",     # 腾讯行情
    "tencent.com",     # 腾讯
    "sse.com.cn",      # 上交所
    "szse.cn",         # 深交所
    "bse.cn",          # 北交所
    "cninfo.com.cn",   # 巨潮
)


def ensure_domestic_no_proxy() -> None:
    """把国内数据源域名追加进 NO_PROXY，绕开系统代理直连。

    背景：本机若配置了代理（HTTP_PROXY 等），requests/akshare 会自动跟随；
    代理进程故障时会抛 ProxyError，而腾讯等国内接口直连即可访问。
    幂等：已存在的条目不会重复追加，不覆盖用户原有 NO_PROXY 设置。
    """
    for var in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(var, "")
        parts = [p.strip() for p in current.split(",") if p.strip()]
        added = [d for d in _DIRECT_DOMAINS if d not in parts]
        if added:
            os.environ[var] = ",".join(parts + added)
            log.info("%s 追加国内直连域名：%s", var, ", ".join(added))


def _cache_path(symbol: str, start: str, end: str) -> "pd.io.common.Path":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{symbol}_{start}_{end}_qfq.csv"


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """列名标准化、按日期升序索引、数值化（兼容东财中文列/新浪英文列）。"""
    df = raw.rename(columns=_COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[[c for c in REQUIRED_COLUMNS if c in df.columns]]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df


def _fetch_sina(symbol: str, start: str, end: str) -> pd.DataFrame:
    """备用数据源：新浪前复权日线。

    新浪接口需带交易所前缀（sh/sz/bj）；成交量单位为股，
    这里换算为手（/100）与腾讯口径保持一致。
    """
    import akshare as ak

    from retailquant.stockmeta import market_of  # 函数内导入避免循环依赖

    raw = ak.stock_zh_a_daily(
        symbol=f"{market_of(symbol)}{symbol}",
        start_date=start, end_date=end, adjust="qfq",
    )
    if raw is not None and not raw.empty and "volume" in raw.columns:
        raw = raw.copy()
        raw["volume"] = raw["volume"] / 100.0
    return raw


def _fetch_tencent(symbol: str, start: str, end: str) -> pd.DataFrame:
    """主数据源：腾讯前复权日线（数据更新及时，网络可达性好）。"""
    import akshare as ak

    return ak.stock_zh_a_hist_tx(
        symbol=symbol, start_date=start, end_date=end, adjust="qfq",
    )


# 数据源优先级：腾讯（主，及时/可达）-> 新浪（备用）
_SOURCES: tuple[tuple[str, object], ...] = (
    ("腾讯", _fetch_tencent),
    ("新浪", _fetch_sina),
)


def _today_cst() -> "date":
    """返回北京时间(UTC+8)的“今天”日期。

    Streamlit Cloud 等部署平台的服务器默认使用 UTC，直接调用
    ``date.today()`` 会得到 UTC 日期，比北京时间晚最多 8 小时；
    在北京时间的 00:00~08:00 时段会“少算一天”，导致行情请求的
    end 日期错位、拿到的是前天的陈旧数据。统一用北京时间计算，
    可消除该时区 bug。
    """
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc) + timedelta(hours=8)).date()


def load_daily(symbol: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """加载单只 A 股前复权日线。

    Args:
        symbol: 6 位股票代码，如 "600519"。
        start:  开始日期 "YYYYMMDD"。
        end:    结束日期 "YYYYMMDD"。
        use_cache: 是否优先读取本地缓存。

    Returns:
        DataFrame，索引为 date，列为 open/high/low/close/volume。

    Raises:
        RuntimeError: 所有数据源多次尝试均失败或返回空数据。
    """
    cache = _cache_path(symbol, start, end)
    if use_cache and cache.exists():
        log.info("命中缓存 %s", cache.name)
        return pd.read_csv(cache, index_col="date", parse_dates=["date"])

    ensure_domestic_no_proxy()

    last_err: Exception | None = None
    for source_name, fetch in _SOURCES:
        for attempt in range(1, _RETRY_TIMES + 1):
            try:
                raw = fetch(symbol, start, end)
                if raw is None or raw.empty:
                    raise RuntimeError(f"{symbol} 返回空数据")
                df = _normalize(raw)
                df.to_csv(cache, encoding="utf-8")
                log.info("下载 %s 完成（%s）：%d 根K线（%s ~ %s）",
                         symbol, source_name, len(df),
                         df.index[0].date(), df.index[-1].date())
                return df
            except Exception as exc:  # noqa: BLE001 网络异常类型繁杂，统一重试
                last_err = exc
                log.warning("%s 第 %d/%d 次获取 %s 失败：%s",
                            source_name, attempt, _RETRY_TIMES, symbol, exc)
                time.sleep(_RETRY_WAIT_SEC * attempt)
        log.warning("数据源【%s】不可用，尝试下一个备用源", source_name)
    raise RuntimeError(f"获取 {symbol} 日线失败（所有数据源）：{last_err}")


def validate_ohlcv(df: pd.DataFrame) -> None:
    """数据质量校验：列齐全、无缺失、high/low 关系合法。"""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必需列：{missing}")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("数据存在缺失值")
    bad = df[(df["high"] < df["low"]) | (df["high"] < df["close"]) | (df["low"] > df["close"])]
    if not bad.empty:
        raise ValueError(f"存在 {len(bad)} 根非法K线（high/low 关系错误）")
