"""Exa APIを使用したWeb検索ツール。

このモジュールは、最新のWeb情報を検索するツールを提供します。
ドキュメントにない最新情報が必要な場合に使用します。
"""
import json
import os

from langchain.tools import tool


@tool
def web_search(query: str) -> str:
    """Webから最新の情報を検索する。

    MLflow、Databricks、または関連トピックについて、
    ドキュメントにない最新情報が必要な場合に使用します。

    Args:
        query: 検索クエリ

    Returns:
        検索結果のJSON文字列
    """
    # Exa APIクライアントをインポート（遅延インポートで依存関係を軽減）
    from exa_py import Exa

    # APIキーを環境変数から取得
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return json.dumps({"error": "EXA_API_KEYが環境変数に設定されていません"})

    try:
        # Exaクライアントを初期化
        exa = Exa(api_key=api_key)

        # 検索を実行（コンテンツとハイライト付き）
        response = exa.search_and_contents(
            query=query,
            type="auto",  # 検索タイプを自動選択
            num_results=5,  # 上位5件を取得
            text={"max_characters": 1000},  # テキストは最大1000文字
            highlights={"num_sentences": 3},  # ハイライトは3文
        )

        # 結果を整形
        results = []
        for result in response.results:
            entry = {
                "title": result.title,
                "url": result.url,
            }
            # テキストコンテンツがあれば追加
            if hasattr(result, "text") and result.text:
                entry["content"] = result.text
            # ハイライトがあれば追加
            if hasattr(result, "highlights") and result.highlights:
                entry["highlights"] = result.highlights
            results.append(entry)

        return json.dumps({"results": results, "total": len(results)}, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})
