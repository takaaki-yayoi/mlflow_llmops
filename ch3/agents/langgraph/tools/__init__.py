"""エージェントが使用するツールのパッケージ。"""
import os

doc_search = None
web_search = None

if os.environ.get("ENABLE_DOC_SEARCH", "true").lower() == "true":
    from .doc_search import doc_search

if os.environ.get("ENABLE_WEB_SEARCH", "true").lower() == "true":
    from .web_search import web_search

from .open_url import open_url

__all__ = ["doc_search", "web_search", "open_url"]
