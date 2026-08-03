# -*- coding: utf-8 -*-
"""stockmeta 纯逻辑单元测试（板块判定 + 外链构建，均不联网）。"""
from __future__ import annotations

from retailquant.stockmeta import build_external_links, market_of


class TestMarketOf:
    def test_shanghai_main_and_star(self):
        assert market_of("600900") == "sh"   # 沪主板
        assert market_of("601318") == "sh"   # 沪主板
        assert market_of("688256") == "sh"   # 科创板

    def test_shenzhen_main_and_chinext(self):
        assert market_of("000858") == "sz"   # 深主板
        assert market_of("002463") == "sz"   # 中小板
        assert market_of("300308") == "sz"   # 创业板

    def test_beijing(self):
        assert market_of("830799") == "bj"   # 北交所
        assert market_of("430047") == "bj"


class TestBuildExternalLinks:
    def test_keys_present(self):
        links = build_external_links("300308")
        assert set(links) == {
            "东方财富·行情", "东方财富·公告",
            "同花顺·F10 档案", "巨潮资讯·公告全文",
        }

    def test_market_prefix_in_quote_url(self):
        # 创业板 300 -> sz 前缀
        assert "sz300308" in build_external_links("300308")["东方财富·行情"]
        # 沪主板 600 -> sh 前缀
        assert "sh600900" in build_external_links("600900")["东方财富·行情"]

    def test_symbol_embedded_in_every_url(self):
        symbol = "601318"
        for url in build_external_links(symbol).values():
            assert symbol in url

    def test_urls_are_http(self):
        for url in build_external_links("000858").values():
            assert url.startswith("http")
