# -*- coding: utf-8 -*-
"""个股元信息：名称查询、基本情况、外部链接构建。

散户视角：输入 6 位代码就能看到股票叫什么、属于哪个板块，
并一键跳转到东方财富/同花顺/巨潮看详细资料和公告。

网络接口（akshare）集中在此，纯逻辑函数（板块判定、外链构建）
不依赖网络，便于单元测试。
"""
from __future__ import annotations

import functools

from retailquant.data import ensure_domestic_no_proxy
from retailquant.logger import get_logger

log = get_logger("stockmeta")


def market_of(symbol: str) -> str:
    """按代码前缀判定交易所：sh / sz / bj。"""
    if symbol.startswith(("60", "68", "9")):        # 沪主板/科创板/沪B
        return "sh"
    if symbol.startswith(("0", "2", "30", "31")):   # 深主板/中小板/创业板/深B
        return "sz"
    if symbol.startswith(("4", "8")):               # 北交所
        return "bj"
    return "sh"


def build_external_links(symbol: str) -> dict[str, str]:
    """构建个股外部链接（行情、公告、F10 资料）。

    Args:
        symbol: 6 位股票代码。

    Returns:
        名称 -> URL 的有序字典。
    """
    mkt = market_of(symbol)
    return {
        "东方财富·行情": f"https://quote.eastmoney.com/{mkt}{symbol}.html",
        "东方财富·公告": f"https://data.eastmoney.com/notices/stock/{symbol}.html",
        "同花顺·F10 档案": f"http://basic.10jqka.com.cn/{symbol}/",
        "巨潮资讯·公告全文": f"http://www.cninfo.com.cn/new/fulltextSearch?keyWord={symbol}",
    }


@functools.lru_cache(maxsize=1)
def _code_name_map() -> dict[str, str]:
    """全 A 股代码->名称映射（进程内缓存，首次调用联网）。"""
    ensure_domestic_no_proxy()
    import akshare as ak  # 延迟导入，离线单测不触发

    try:
        df = ak.stock_info_a_code_name()
        return dict(zip(df["code"].astype(str), df["name"].astype(str)))
    except Exception as exc:  # noqa: BLE001 网络异常统一降级
        log.warning("获取 A 股代码名称表失败：%s", exc)
        return {}


def all_code_names() -> dict[str, str]:
    """返回全 A 股代码->名称映射（可能为空，调用方需容错）。"""
    return _code_name_map()


def get_stock_name(symbol: str) -> str:
    """按代码返回股票简称，查不到返回空串。"""
    return _code_name_map().get(symbol, "")


def get_stock_info(symbol: str) -> dict[str, str]:
    """获取个股基本情况（最新价/市值/行业/上市时间等）。

    Args:
        symbol: 6 位股票代码。

    Returns:
        标准化后的信息字典；字段缺失时值为空串。

    Raises:
        RuntimeError: 网络请求失败或返回异常。
    """
    ensure_domestic_no_proxy()
    import akshare as ak

    try:
        df = ak.stock_individual_info_em(symbol=symbol)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"获取 {symbol} 基本信息失败：{exc}") from exc

    raw = dict(zip(df["item"].astype(str), df["value"]))
    return {
        "name": str(raw.get("股票简称", "")),
        "price": raw.get("最新", ""),
        "industry": str(raw.get("行业", "")),
        "total_mv": raw.get("总市值", ""),
        "float_mv": raw.get("流通市值", ""),
        "total_share": raw.get("总股本", ""),
        "list_date": str(raw.get("上市时间", "")),
    }
