"""Milvusベクトルストアを使用したドキュメント検索ツール。

このモジュールは、事前にインジェストされたMLflowドキュメントから
関連情報を検索するツールを提供します。
"""
import logging
from pathlib import Path

from langchain.tools import tool
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from pymilvus import connections

import os

logger = logging.getLogger(__name__)

# AsyncMilvusClient初期化時の警告を抑制（同期処理のみ使用するため影響なし）
logging.getLogger("langchain_milvus").setLevel(logging.ERROR)

DB_PATH = Path("data") / "milvus.db"


@tool
def doc_search(query: str) -> str:
    """MLflowドキュメントから関連情報を検索する。

    MLflowの機能、API、ガイド、ベストプラクティスに関する
    詳細情報を見つけるために使用します。

    Args:
        query: 検索クエリ

    Returns:
        検索結果のテキスト、またはエラーメッセージ
    """
    # データベースが存在しない場合
    if not DB_PATH.exists():
        return "ドキュメント検索は利用できません。先に 'make ingest' を実行してください。"

    # リトライロジック（接続エラー対策）
    max_retries = 2
    for attempt in range(max_retries):
        try:
            retriever = _get_retriever()
            if retriever is None:
                return "ドキュメント検索は利用できません。先に 'make ingest' を実行してください。"

            # 検索を実行
            docs = retriever.invoke(query)
            # 結果を結合して返す
            return "\n\n".join([doc.page_content for doc in docs])

        except Exception as e:
            # 接続エラーの場合はリトライ
            if "Connection refused" in str(e) or "connect" in str(e).lower():
                logger.warning(f"Milvus接続に失敗しました（試行 {attempt + 1}回目）、接続をリセット中...")
                _reset_milvus_connection()
                if attempt == max_retries - 1:
                    return "接続の問題により、ドキュメント検索が一時的に利用できません。もう一度お試しください。"
            else:
                raise


def _reset_milvus_connection():
    """Milvusの接続をリセットして古いソケットをクリアする。"""
    try:
        for alias in list(connections.list_connections()):
            connections.disconnect(alias[0])
    except Exception:
        pass


def _get_retriever():
    """Milvusデータベースからリトリーバーを取得する。"""
    if not DB_PATH.exists():
        return None

    embeddings = OpenAIEmbeddings(model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))

    # Milvusベクトルストアに接続
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": str(DB_PATH)},
        collection_name="mlflow_docs",
    )
    # 上位5件を返すリトリーバーを作成
    return vectorstore.as_retriever(search_kwargs={"k": 5})
