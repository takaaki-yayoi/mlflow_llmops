#!/usr/bin/env python3
"""MLflowドキュメントをMilvusベクトルデータベースに取り込むスクリプト。

このスクリプトは以下の処理を行います:
1. Scrapyを使用してMLflowドキュメントをクロール
2. テキストコンテンツを抽出・クリーニング
3. テキストを適切なサイズにチャンク分割
4. 埋め込みを生成してMilvusに保存

使用例:
    デフォルトのMLflowドキュメントを取り込む::

        $ python scripts/web_ingest.py

    カスタムURLを指定::

        $ python scripts/web_ingest.py --base-url https://example.com/docs
"""

import argparse
import json
import logging
import os
import re
import uuid
from pathlib import Path
from urllib.parse import urlparse

import html2text
import scrapy
import tiktoken
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from scrapy.crawler import CrawlerProcess

# ノイズの多いロガーを抑制
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# デフォルトで取り込むMLflowドキュメントのURL
DEFAULT_URL = "https://mlflow.org/docs/latest/"

# デフォルトで除外するURLパターン（API referenceは量が多く、ノイズになりやすい）
DEFAULT_IGNORE_PATTERNS = [
    r"/api_reference/",
]


class DocsSpider(scrapy.Spider):
    """ドキュメントをクロールするScrapyスパイダー。"""

    name = "docs_spider"

    def __init__(
        self,
        base_url,
        max_pages=None,
        max_crawl_threads=32,
        ignore_url_patterns=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base_url = base_url
        self.start_urls = [self.base_url]

        # base_urlからドメインを抽出
        parsed = urlparse(self.base_url)
        self.allowed_domains = [parsed.netloc]

        self.max_pages = int(max_pages) if max_pages else None
        self.pages_scraped = 0

        # URL除外パターンをコンパイル
        self.ignore_patterns = []
        if ignore_url_patterns:
            self.ignore_patterns = [re.compile(pattern) for pattern in ignore_url_patterns]

        # カスタム設定（並行リクエスト数など）
        self.custom_settings = {
            "USER_AGENT": "MLflowDocsBot/1.0",
            "ROBOTSTXT_OBEY": True,
            "CONCURRENT_REQUESTS": int(max_crawl_threads),
            "CONCURRENT_REQUESTS_PER_DOMAIN": int(max_crawl_threads),
            "DOWNLOAD_DELAY": 0,
            "HTTPCACHE_ENABLED": True,
            "HTTPCACHE_EXPIRATION_SECS": 86400,  # 24時間キャッシュ
            "HTTPCACHE_DIR": ".scrapy_cache",
            "HTTPCACHE_IGNORE_HTTP_CODES": [500, 502, 503, 504, 522, 524, 408, 429],
        }

    def should_ignore_url(self, url):
        """URLが除外パターンに一致するかチェック。"""
        for pattern in self.ignore_patterns:
            if pattern.search(url):
                return True
        return False

    def parse(self, response):
        """ドキュメントページをパースしてコンテンツを抽出。"""
        if self.max_pages and self.pages_scraped >= self.max_pages:
            return

        # ページタイトルを取得
        title = response.css("title::text").get() or response.url

        # メインコンテンツを抽出（一般的なセレクターを試行）
        content_selectors = ["main", "article", ".document", ".content", "#content", "body"]

        content_html = None
        for selector in content_selectors:
            content = response.css(selector).get()
            if content:
                content_html = content
                break

        if not content_html:
            content_html = response.text

        # スクレイピング結果を返す
        self.pages_scraped += 1
        print(f"取得: {response.url} ({self.pages_scraped} ページ)")

        yield {"url": response.url, "title": title, "content_html": content_html}

        # base_url内のリンクをフォロー
        if self.max_pages is None or self.pages_scraped < self.max_pages:
            for link in response.css("a::attr(href)").getall():
                full_link = response.urljoin(link)
                if full_link.startswith(self.base_url) and not self.should_ignore_url(full_link):
                    yield response.follow(link, callback=self.parse)


def clean_html_to_text(html_content: str) -> str:
    """HTMLコンテンツをクリーンなテキストに変換。"""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.ignore_emphasis = False
    h.body_width = 0  # 行折り返しを無効化

    text = h.handle(html_content)

    # 過剰な改行を削除
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    text = "\n".join(lines)

    return text


def chunk_text(text: str, url: str, title: str, max_tokens: int = 512) -> list[Document]:
    """テキストを約max_tokensサイズのLangChain Documentsに分割。"""
    encoding = tiktoken.get_encoding("cl100k_base")

    # まず段落で分割
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(encoding.encode(para))

        # 単一の段落がmax_tokensを超える場合は文で分割
        if para_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # 長い段落を文で分割
            sentences = para.split(". ")
            for sentence in sentences:
                sentence_tokens = len(encoding.encode(sentence))
                if current_tokens + sentence_tokens > max_tokens:
                    if current_chunk:
                        chunks.append(". ".join(current_chunk))
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
        else:
            # 段落を追加すると制限を超える場合
            if current_tokens + para_tokens > max_tokens:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

    # 最後のチャンクを追加
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # LangChain Documentsを作成
    documents = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "chunk_id": uuid.uuid4().hex,
                    "url": url,
                    "title": title,
                },
            )
        )

    return documents


def ingest_url(base_url: str, db_path: Path, chunk_size: int, max_pages: int | None,
               max_crawl_threads: int, ignore_patterns: list[str] | None) -> int:
    """単一のURLをクロールしてMilvusに保存。

    Returns:
        保存されたドキュメント数
    """
    print(f"\n{'=' * 60}")
    print(f"クロール開始: {base_url}")
    print(f"{'=' * 60}")

    # Scrapyでクロール
    process = CrawlerProcess(
        settings={
            "LOG_LEVEL": "WARNING",
            "FEEDS": {
                "scraped_data.json": {
                    "format": "json",
                    "overwrite": True,
                },
            },
        }
    )

    spider_kwargs = {
        "base_url": base_url,
        "max_crawl_threads": max_crawl_threads,
    }
    if max_pages:
        spider_kwargs["max_pages"] = max_pages
    if ignore_patterns:
        spider_kwargs["ignore_url_patterns"] = ignore_patterns

    process.crawl(DocsSpider, **spider_kwargs)
    process.start(stop_after_crawl=False)

    # スクレイピングデータを読み込み
    try:
        with open("scraped_data.json", "r") as f:
            scraped_data = json.load(f)
        os.remove("scraped_data.json")
    except FileNotFoundError:
        scraped_data = []

    print(f"取得ページ数: {len(scraped_data)}")

    if not scraped_data:
        print("警告: ページが取得できませんでした")
        return 0

    # テキストをクリーニングしてチャンク分割
    all_documents = []
    for page in scraped_data:
        clean_text = clean_html_to_text(page["content_html"])
        if not clean_text.strip():
            continue

        documents = chunk_text(
            text=clean_text,
            url=page["url"],
            title=page["title"],
            max_tokens=chunk_size,
        )
        all_documents.extend(documents)

    print(f"作成チャンク数: {len(all_documents)}")

    if not all_documents:
        print("警告: チャンクが作成されませんでした")
        return 0

    # Milvusに保存
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Milvus.from_documents(
        documents=all_documents,
        embedding=embeddings,
        collection_name="mlflow_docs",
        connection_args={"uri": str(db_path)},
    )

    print(f"保存完了: {len(all_documents)} ドキュメント")
    return len(all_documents)


def main():
    parser = argparse.ArgumentParser(
        description="MLflowドキュメントをMilvusベクトルデータベースに取り込む"
    )
    parser.add_argument(
        "--base-url",
        help="クロールするベースURL（指定しない場合はデフォルトのMLflowドキュメントを取り込み）",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Milvusデータベースの保存先ディレクトリ（デフォルト: data）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="チャンクあたりの最大トークン数（デフォルト: 512）",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="クロールする最大ページ数（テスト用）",
    )
    parser.add_argument(
        "--max-crawl-threads",
        type=int,
        default=32,
        help="同時クロールスレッド数（デフォルト: 32）",
    )
    parser.add_argument(
        "--ignore-url-regex",
        action="append",
        help="除外するURLの正規表現パターン（複数指定可）",
    )

    args = parser.parse_args()

    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "milvus.db"

    print("=" * 60)
    print("MLflowドキュメント取り込み")
    print("=" * 60)
    print(f"Milvus DB: {db_path}")

    if args.base_url:
        # 単一URLを指定した場合
        docs = ingest_url(
            base_url=args.base_url,
            db_path=db_path,
            chunk_size=args.chunk_size,
            max_pages=args.max_pages,
            max_crawl_threads=args.max_crawl_threads,
            ignore_patterns=args.ignore_url_regex,
        )
    else:
        # デフォルトのMLflowドキュメントを取り込み
        print(f"\nデフォルトURLを取り込みます")

        # デフォルトの除外パターンとユーザー指定のパターンをマージ
        ignore_patterns = DEFAULT_IGNORE_PATTERNS.copy()
        if args.ignore_url_regex:
            ignore_patterns.extend(args.ignore_url_regex)

        docs = ingest_url(
            base_url=DEFAULT_URL,
            db_path=db_path,
            chunk_size=args.chunk_size,
            max_pages=args.max_pages,
            max_crawl_threads=args.max_crawl_threads,
            ignore_patterns=ignore_patterns,
        )

    # サマリー
    print("\n" + "=" * 60)
    print("取り込み完了")
    print("=" * 60)
    print(f"総ドキュメント数: {docs}")
    print(f"Milvus DB: {db_path.absolute()}")


if __name__ == "__main__":
    main()
